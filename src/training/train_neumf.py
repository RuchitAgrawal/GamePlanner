"""
NeuMF (Neural Collaborative Filtering) training script.

Follows the 3-stage protocol from He et al. (WWW 2017):
  Stage 1: Pre-train MLP branch (MLPBranch) independently
  Stage 2: Load GMF (already trained) + MLP weights into NeuMF
  Stage 3: Fine-tune NeuMF jointly at a lower learning rate

Pre-training consistently beats random initialization on this task.
GMF checkpoint must already exist at models/gmf_best.pt.

Usage:
    python -m src.training.train_neumf
    python -m src.training.train_neumf --mlp-epochs 20 --finetune-epochs 20
"""

import argparse
import json
import logging
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.models.neural_cf import NeuMF, MLPBranch
from src.models.matrix_factorization import GMF
from src.data.dataset import SteamDataset, collate_fn
from src.evaluation.metrics import ndcg_at_k, hit_rate_at_k
from src.utils.config import (
    TRAIN_PATH, VAL_PATH, TEST_PATH, ITEMS_META_PATH,
    EMBEDDING_DIM, BATCH_SIZE, MLP_LAYERS, DROPOUT,
    LEARNING_RATE, FINETUNE_LR, MAX_EPOCHS,
    NUM_NEGATIVES, TOP_K, MODELS_DIR, DATA_PROCESSED_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

EXPERIMENTS_PATH = DATA_PROCESSED_DIR / "experiments.json"
GMF_CHECKPOINT = MODELS_DIR / "gmf_best.pt"
MLP_CHECKPOINT = MODELS_DIR / "mlp_best.pt"
NEUMF_CHECKPOINT = MODELS_DIR / "neumf_best.pt"

VAL_K = 10
EARLY_STOP_PATIENCE = 5


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mlp-epochs",      type=int,   default=30,         help="MLP pre-training epochs")
    p.add_argument("--finetune-epochs", type=int,   default=30,         help="NeuMF joint fine-tuning epochs")
    p.add_argument("--mlp-lr",          type=float, default=LEARNING_RATE)
    p.add_argument("--finetune-lr",     type=float, default=FINETUNE_LR)
    p.add_argument("--emb-dim",         type=int,   default=EMBEDDING_DIM)
    p.add_argument("--batch-size",      type=int,   default=BATCH_SIZE)
    p.add_argument("--num-negatives",   type=int,   default=NUM_NEGATIVES)
    p.add_argument("--no-amp",          action="store_true")
    p.add_argument("--skip-mlp-pretrain", action="store_true",
                   help="Skip MLP pre-training (use existing mlp_best.pt)")
    return p.parse_args()


# ── Shared helpers ─────────────────────────────────────────────────────────────

def build_user_train_items(train: pd.DataFrame) -> dict[int, set[int]]:
    return train.groupby("user_idx")["item_idx"].apply(set).to_dict()


def make_loader(train_df: pd.DataFrame, num_items: int, args) -> DataLoader:
    dataset = SteamDataset(
        train_df, num_items=num_items,
        num_negatives=args.num_negatives,
        use_confidence=True,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_df: pd.DataFrame,
    user_train_items: dict[int, set[int]],
    num_items: int,
    device: torch.device,
    k: int = VAL_K,
) -> tuple[float, float]:
    model.eval()
    hrs, ndcgs = [], []
    all_items = torch.arange(num_items, dtype=torch.long, device=device)

    for row in val_df.itertuples():
        user_idx = int(row.user_idx)
        val_item = int(row.item_idx)
        exclude = user_train_items.get(user_idx, set())

        user_t = torch.tensor([user_idx], dtype=torch.long, device=device)
        scores = []
        for start in range(0, num_items, 2048):
            batch_items = all_items[start:start + 2048]
            u = user_t.expand(len(batch_items))
            scores.append(model(u, batch_items).cpu())
        scores = torch.cat(scores).numpy()

        for item in exclude:
            if item < len(scores):
                scores[item] = -1e9

        top_k = np.argsort(scores)[::-1][:k].tolist()
        relevant = {val_item}
        hrs.append(hit_rate_at_k(top_k, relevant, k))
        ndcgs.append(ndcg_at_k(top_k, relevant, k))

    return float(np.mean(hrs)), float(np.mean(ndcgs))


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        users  = batch["user"].to(device)
        items  = batch["item"].to(device)
        labels = batch["label"].to(device)
        weights = batch["weight"].to(device)

        optimizer.zero_grad()
        with autocast("cuda", enabled=use_amp):
            logits = model(users, items)
            loss = (criterion(logits, labels) * weights).mean()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

    return total_loss / len(loader)


# ── Stage 1: MLP pre-training ─────────────────────────────────────────────────

def pretrain_mlp(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    num_users: int,
    num_items: int,
    user_train_items: dict[int, set[int]],
    device: torch.device,
    args,
) -> MLPBranch:
    log.info("=== Stage 1: MLP Branch Pre-training ===")
    use_amp = device.type == "cuda" and not args.no_amp

    model = MLPBranch(
        num_users, num_items,
        emb_dim=args.emb_dim,
        layers=MLP_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    loader    = make_loader(train_df, num_items, args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.mlp_lr, weight_decay=1e-6)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    scaler    = GradScaler("cuda", enabled=use_amp)

    log.info("MLPBranch parameters: %s", f"{sum(p.numel() for p in model.parameters()):,}")

    best_hr   = 0.0
    patience  = 0

    for epoch in range(1, args.mlp_epochs + 1):
        t0 = time.time()
        loss = train_one_epoch(model, loader, optimizer, criterion, scaler, device, use_amp)
        hr, ndcg = evaluate(model, val_df, user_train_items, num_items, device)

        log.info(
            "MLP Epoch %2d/%d | loss=%.4f | val HR@10=%.4f | NDCG@10=%.4f | %.1fs",
            epoch, args.mlp_epochs, loss, hr, ndcg, time.time() - t0
        )

        if hr > best_hr:
            best_hr = hr
            patience = 0
            torch.save(model.state_dict(), MLP_CHECKPOINT)
            log.info("  -> New best MLP! HR@10=%.4f", best_hr)
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                log.info("MLP early stopping at epoch %d", epoch)
                break

    log.info("MLP pre-training done. Best val HR@10=%.4f", best_hr)
    model.load_state_dict(torch.load(MLP_CHECKPOINT, map_location=device))
    return model


# ── Stage 2+3: NeuMF init + fine-tuning ──────────────────────────────────────

def finetune_neumf(
    mlp_model: MLPBranch,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_users: int,
    num_items: int,
    user_train_items: dict[int, set[int]],
    device: torch.device,
    args,
) -> dict:
    log.info("=== Stage 2: Initialize NeuMF from pre-trained weights ===")
    use_amp = device.type == "cuda" and not args.no_amp

    if not GMF_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"GMF checkpoint not found at {GMF_CHECKPOINT}. "
            "Run train_gmf.py first."
        )

    neumf = NeuMF(
        num_users, num_items,
        emb_dim=args.emb_dim,
        layers=MLP_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    gmf_state = torch.load(GMF_CHECKPOINT, map_location=device)
    mlp_state  = mlp_model.state_dict()
    neumf.load_pretrained_weights(gmf_state, mlp_state)
    log.info("Loaded GMF weights from %s", GMF_CHECKPOINT)
    log.info("Loaded MLP weights from pre-trained model")
    log.info("NeuMF parameters: %s", f"{sum(p.numel() for p in neumf.parameters()):,}")

    log.info("=== Stage 3: Joint Fine-tuning ===")
    loader    = make_loader(train_df, num_items, args)
    optimizer = torch.optim.Adam(neumf.parameters(), lr=args.finetune_lr, weight_decay=1e-6)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    scaler    = GradScaler("cuda", enabled=use_amp)

    best_hr  = 0.0
    best_ndcg = 0.0
    patience = 0
    t_start  = time.time()

    for epoch in range(1, args.finetune_epochs + 1):
        t0 = time.time()
        loss = train_one_epoch(neumf, loader, optimizer, criterion, scaler, device, use_amp)
        hr, ndcg = evaluate(neumf, val_df, user_train_items, num_items, device)

        log.info(
            "NeuMF Epoch %2d/%d | loss=%.4f | val HR@10=%.4f | NDCG@10=%.4f | %.1fs",
            epoch, args.finetune_epochs, loss, hr, ndcg, time.time() - t0
        )

        if hr > best_hr:
            best_hr   = hr
            best_ndcg = ndcg
            patience  = 0
            torch.save(neumf.state_dict(), NEUMF_CHECKPOINT)
            log.info("  -> New best NeuMF! HR@10=%.4f", best_hr)
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                log.info("NeuMF early stopping at epoch %d", epoch)
                break

    total_time = time.time() - t_start
    log.info("Fine-tuning done. Best val HR@10=%.4f | NDCG@10=%.4f", best_hr, best_ndcg)

    # ── Test evaluation ────────────────────────────────────────────────────────
    log.info("Loading best NeuMF checkpoint for test evaluation...")
    neumf.load_state_dict(torch.load(NEUMF_CHECKPOINT, map_location=device))

    val_items_per_user = val_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
    user_test_exclude = {
        u: user_train_items.get(u, set()) | val_items_per_user.get(u, set())
        for u in test_df["user_idx"].unique()
    }

    log.info("Evaluating on test set...")
    hr5,  ndcg5  = evaluate(neumf, test_df, user_test_exclude, num_items, device, k=5)
    hr10, ndcg10 = evaluate(neumf, test_df, user_test_exclude, num_items, device, k=10)
    hr20, ndcg20 = evaluate(neumf, test_df, user_test_exclude, num_items, device, k=20)

    metrics = {
        "hit_rate@5":  hr5,   "ndcg@5":  ndcg5,
        "hit_rate@10": hr10,  "ndcg@10": ndcg10,
        "hit_rate@20": hr20,  "ndcg@20": ndcg20,
        "best_val_hr@10": best_hr,
        "epochs_finetuned": epoch,
    }

    print("\n=== NeuMF Test Results ===")
    print(f"  HR@5  = {hr5:.4f}  |  NDCG@5  = {ndcg5:.4f}")
    print(f"  HR@10 = {hr10:.4f}  |  NDCG@10 = {ndcg10:.4f}")
    print(f"  HR@20 = {hr20:.4f}  |  NDCG@20 = {ndcg20:.4f}")
    print(f"\n  vs GMF:        HR@10=0.3096  NDCG@10=0.1841")
    print(f"  vs Popularity: HR@10=0.2203  NDCG@10=0.1352")

    return metrics, total_time, epoch


# ── Save experiment ────────────────────────────────────────────────────────────

def save_experiment(metrics: dict, args, duration_s: float, epochs: int) -> None:
    runs = []
    if EXPERIMENTS_PATH.exists():
        with open(EXPERIMENTS_PATH) as f:
            runs = json.load(f)
    runs.append({
        "model": "NeuMF",
        "emb_dim": args.emb_dim,
        "mlp_layers": MLP_LAYERS,
        "mlp_lr": args.mlp_lr,
        "finetune_lr": args.finetune_lr,
        "epochs_finetuned": epochs,
        "metrics": {k: round(float(v), 6) for k, v in metrics.items()
                    if k not in ("epochs_finetuned",)},
        "duration_s": round(duration_s, 2),
        "k_values": TOP_K,
    })
    with open(EXPERIMENTS_PATH, "w") as f:
        json.dump(runs, f, indent=2)
    log.info("Saved experiment results to %s", EXPERIMENTS_PATH)


# ── Entry point ────────────────────────────────────────────────────────────────

def run(args=None):
    if args is None:
        args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    if device.type == "cuda":
        log.info("GPU: %s (%.1f GB VRAM)",
                 torch.cuda.get_device_name(0),
                 torch.cuda.get_device_properties(0).total_memory / 1e9)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet(TRAIN_PATH)
    val_df   = pd.read_parquet(VAL_PATH)
    test_df  = pd.read_parquet(TEST_PATH)

    num_users = int(train_df["user_idx"].max()) + 1
    num_items = int(train_df["item_idx"].max()) + 1
    log.info("Users: %d | Items: %d", num_users, num_items)

    user_train_items = build_user_train_items(train_df)

    # Stage 1
    if args.skip_mlp_pretrain:
        log.info("Skipping MLP pre-training -- loading existing %s", MLP_CHECKPOINT)
        mlp = MLPBranch(num_users, num_items, emb_dim=args.emb_dim,
                        layers=MLP_LAYERS, dropout=DROPOUT).to(device)
        mlp.load_state_dict(torch.load(MLP_CHECKPOINT, map_location=device))
    else:
        mlp = pretrain_mlp(train_df, val_df, num_users, num_items,
                           user_train_items, device, args)

    # Stages 2 + 3
    metrics, duration, epochs = finetune_neumf(
        mlp, train_df, val_df, test_df, num_users, num_items,
        user_train_items, device, args
    )

    save_experiment(metrics, args, duration, epochs)
    log.info("Done. NeuMF checkpoint at %s", NEUMF_CHECKPOINT)


if __name__ == "__main__":
    run()
