"""
GMF (Generalized Matrix Factorization) training script.

Trains on implicit feedback using BinaryCrossEntropy loss with negative sampling.
Validates after each epoch using HR@10 on the val split.
Saves the best checkpoint to models/gmf_best.pt.

Usage:
    python -m src.training.train_gmf
    python -m src.training.train_gmf --epochs 30 --lr 0.001 --emb-dim 64
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.models.matrix_factorization import GMF
from src.data.dataset import SteamDataset, collate_fn
from src.evaluation.metrics import ndcg_at_k, hit_rate_at_k, aggregate_metrics
from src.utils.config import (
    TRAIN_PATH, VAL_PATH, TEST_PATH, ITEMS_META_PATH,
    EMBEDDING_DIM, BATCH_SIZE, LEARNING_RATE, MAX_EPOCHS,
    NUM_NEGATIVES, TOP_K, MODELS_DIR, DATA_PROCESSED_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

EXPERIMENTS_PATH = DATA_PROCESSED_DIR / "experiments.json"
CHECKPOINT_PATH = MODELS_DIR / "gmf_best.pt"
VAL_K = 10
EARLY_STOP_PATIENCE = 5


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--emb-dim", type=int, default=EMBEDDING_DIM)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--num-negatives", type=int, default=NUM_NEGATIVES)
    p.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    return p.parse_args()


def build_user_train_items(train: pd.DataFrame) -> dict[int, set[int]]:
    return train.groupby("user_idx")["item_idx"].apply(set).to_dict()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_df: pd.DataFrame,
    user_train_items: dict[int, set[int]],
    num_items: int,
    device: torch.device,
    k: int = VAL_K,
) -> tuple[float, float]:
    """
    Fast evaluation: score all items for each user, exclude train items,
    check if the single val item appears in top-k.
    Returns (HR@k, NDCG@k).
    """
    model.eval()
    hit_rates, ndcgs = [], []

    for row in val_df.itertuples():
        user_idx = int(row.user_idx)
        val_item = int(row.item_idx)
        exclude = user_train_items.get(user_idx, set())

        # Score all items in batches
        user_tensor = torch.tensor([user_idx], dtype=torch.long, device=device)
        all_items = torch.arange(num_items, dtype=torch.long, device=device)

        scores = []
        for start in range(0, num_items, 2048):
            batch = all_items[start:start + 2048]
            u = user_tensor.expand(len(batch))
            s = model(u, batch)
            scores.append(s.cpu())
        scores = torch.cat(scores).numpy()

        # Mask out training items
        for item in exclude:
            if item < len(scores):
                scores[item] = -1e9

        top_k = np.argsort(scores)[::-1][:k].tolist()
        relevant = {val_item}
        hit_rates.append(hit_rate_at_k(top_k, relevant, k))
        ndcgs.append(ndcg_at_k(top_k, relevant, k))

    return float(np.mean(hit_rates)), float(np.mean(ndcgs))


def save_experiment(metrics: dict, args, duration_s: float) -> None:
    runs = []
    if EXPERIMENTS_PATH.exists():
        with open(EXPERIMENTS_PATH) as f:
            runs = json.load(f)
    runs.append({
        "model": "GMF",
        "emb_dim": args.emb_dim,
        "lr": args.lr,
        "epochs_trained": metrics.get("epochs_trained", 0),
        "metrics": {k: round(float(v), 6) for k, v in metrics.items() if k != "epochs_trained"},
        "duration_s": round(duration_s, 2),
        "k_values": TOP_K,
    })
    with open(EXPERIMENTS_PATH, "w") as f:
        json.dump(runs, f, indent=2)
    log.info("Saved experiment results to %s", EXPERIMENTS_PATH)


def run(args=None):
    if args is None:
        args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    if device.type == "cuda":
        log.info("GPU: %s (%.1f GB VRAM)", torch.cuda.get_device_name(0),
                 torch.cuda.get_device_properties(0).total_memory / 1e9)

    # ── Load data ─────────────────────────────────────────────────────────────
    train_df = pd.read_parquet(TRAIN_PATH)
    val_df = pd.read_parquet(VAL_PATH)
    test_df = pd.read_parquet(TEST_PATH)
    meta = pd.read_parquet(ITEMS_META_PATH)

    num_users = int(train_df["user_idx"].max()) + 1
    num_items = int(train_df["item_idx"].max()) + 1
    log.info("Users: %d | Items: %d", num_users, num_items)

    user_train_items = build_user_train_items(train_df)

    # ── Dataset & DataLoader ──────────────────────────────────────────────────
    dataset = SteamDataset(
        train_df, num_items=num_items,
        num_negatives=args.num_negatives,
        use_confidence=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,          # 0 = main process, avoids Windows multiproc issues
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    log.info("Batches per epoch: %d", len(loader))

    # ── Model, optimizer, loss ────────────────────────────────────────────────
    model = GMF(num_users, num_items, emb_dim=args.emb_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    scaler = GradScaler("cuda", enabled=(device.type == "cuda" and not args.no_amp))

    total_params = sum(p.numel() for p in model.parameters())
    log.info("GMF parameters: %s", f"{total_params:,}")

    # ── Training loop ─────────────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_hr = 0.0
    best_ndcg = 0.0
    patience_counter = 0
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t_ep = time.time()

        for batch in loader:
            users = batch["user"].to(device)
            items = batch["item"].to(device)
            labels = batch["label"].to(device)
            weights = batch["weight"].to(device)

            optimizer.zero_grad()
            with autocast("cuda", enabled=(device.type == "cuda" and not args.no_amp)):
                logits = model(users, items)
                loss = (criterion(logits, labels) * weights).mean()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        ep_time = time.time() - t_ep

        # ── Validation ────────────────────────────────────────────────────────
        hr, ndcg = evaluate(model, val_df, user_train_items, num_items, device)
        log.info(
            "Epoch %2d/%d | loss=%.4f | val HR@%d=%.4f | NDCG@%d=%.4f | %.1fs",
            epoch, args.epochs, avg_loss, VAL_K, hr, VAL_K, ndcg, ep_time
        )

        # ── Checkpoint ────────────────────────────────────────────────────────
        if hr > best_hr:
            best_hr = hr
            best_ndcg = ndcg
            patience_counter = 0
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            log.info("  -> New best! Saved checkpoint (HR@%d=%.4f)", VAL_K, best_hr)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                log.info("Early stopping at epoch %d (no improvement for %d epochs)", epoch, EARLY_STOP_PATIENCE)
                break

    total_time = time.time() - t_start
    log.info("Training complete. Best val HR@%d=%.4f | NDCG@%d=%.4f", VAL_K, best_hr, VAL_K, best_ndcg)

    # ── Final test evaluation ─────────────────────────────────────────────────
    log.info("Loading best checkpoint for test evaluation...")
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))

    # Build full user_train_items including val for test eval (standard protocol)
    val_items_per_user = val_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
    user_test_exclude = {
        u: user_train_items.get(u, set()) | val_items_per_user.get(u, set())
        for u in test_df["user_idx"].unique()
    }

    log.info("Evaluating on test set...")
    test_hr, test_ndcg = evaluate(model, test_df, user_test_exclude, num_items, device)

    # Also compute @5 and @20
    test_hr5, test_ndcg5 = evaluate(model, test_df, user_test_exclude, num_items, device, k=5)
    test_hr20, test_ndcg20 = evaluate(model, test_df, user_test_exclude, num_items, device, k=20)

    metrics = {
        "hit_rate@5": test_hr5,
        "ndcg@5": test_ndcg5,
        "hit_rate@10": test_hr,
        "ndcg@10": test_ndcg,
        "hit_rate@20": test_hr20,
        "ndcg@20": test_ndcg20,
        "best_val_hr@10": best_hr,
        "epochs_trained": epoch,
    }

    print("\n=== GMF Test Results ===")
    print(f"  HR@5 =  {test_hr5:.4f}  |  NDCG@5 =  {test_ndcg5:.4f}")
    print(f"  HR@10 = {test_hr:.4f}  |  NDCG@10 = {test_ndcg:.4f}")
    print(f"  HR@20 = {test_hr20:.4f}  |  NDCG@20 = {test_ndcg20:.4f}")
    print(f"\n  vs Popularity baseline:  HR@10=0.2203  NDCG@10=0.1352")

    save_experiment(metrics, args, total_time)
    log.info("Done. Checkpoint at %s", CHECKPOINT_PATH)


if __name__ == "__main__":
    run()
