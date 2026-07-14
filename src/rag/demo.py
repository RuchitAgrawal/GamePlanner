"""
End-to-end demo of the full recommendation + explanation pipeline.

Loads the GMF model, picks a random user from the test set, generates
recommendations, then uses the RAG pipeline to explain each one.

Usage:
    python -m src.rag.demo
    python -m src.rag.demo --user-idx 42 --top-k 5
"""

import argparse
import logging
import numpy as np
import pandas as pd
import torch

from src.models.matrix_factorization import GMF
from src.rag.embeddings import load_item_embeddings
from src.rag.vector_store import VectorStore
from src.rag.explainability import ExplainabilityPipeline
from src.rag.llm_client import LLMClient
from src.utils.cache import ExplanationCache
from src.utils.config import (
    TRAIN_PATH, TEST_PATH, ITEMS_META_PATH,
    MODELS_DIR, CACHE_PATH,
    EMBEDDING_DIM,
)

logging.basicConfig(level=logging.WARNING, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

GMF_CHECKPOINT = MODELS_DIR / "gmf_best.pt"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--user-idx", type=int, default=None, help="User index (random if not set)")
    p.add_argument("--top-k",    type=int, default=5,    help="Number of recommendations to show")
    return p.parse_args()


@torch.no_grad()
def get_recommendations(
    model: GMF,
    user_idx: int,
    num_items: int,
    exclude_items: set[int],
    top_k: int,
    device: torch.device,
) -> list[int]:
    model.eval()
    user_t = torch.tensor([user_idx], dtype=torch.long, device=device)
    all_items = torch.arange(num_items, dtype=torch.long, device=device)

    scores = []
    for start in range(0, num_items, 2048):
        batch = all_items[start:start + 2048]
        u = user_t.expand(len(batch))
        scores.append(model(u, batch).cpu())
    scores = torch.cat(scores).numpy()

    for item in exclude_items:
        if item < len(scores):
            scores[item] = -1e9

    return np.argsort(scores)[::-1][:top_k].tolist()


def run():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load data ──────────────────────────────────────────────────────────────
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df  = pd.read_parquet(TEST_PATH)
    meta     = pd.read_parquet(ITEMS_META_PATH)

    num_users = int(train_df["user_idx"].max()) + 1
    num_items = int(train_df["item_idx"].max()) + 1

    # idx -> game_name lookup
    idx_to_game: dict[int, str] = {}
    for _, row in meta.iterrows():
        idx_to_game[int(row["item_idx"])] = str(row.get("title", row.get("item_id", "Unknown")))

    # ── Pick user ──────────────────────────────────────────────────────────────
    available_users = test_df["user_idx"].unique().tolist()
    if args.user_idx is not None and args.user_idx in available_users:
        user_idx = args.user_idx
    else:
        user_idx = int(np.random.choice(available_users))

    print(f"\n{'='*60}")
    print(f"  Demo User Index: {user_idx}")
    print(f"{'='*60}")

    # User's training history (games played)
    user_train = train_df[train_df["user_idx"] == user_idx]
    train_item_idxs = set(user_train["item_idx"].astype(int).tolist())

    print(f"\nGames played (training history, up to 10):")
    for _, row in user_train.head(10).iterrows():
        game = idx_to_game.get(int(row["item_idx"]), "Unknown")
        print(f"  - {game}")

    test_item = int(test_df[test_df["user_idx"] == user_idx]["item_idx"].iloc[0])
    print(f"\nHeld-out test game: {idx_to_game.get(test_item, 'Unknown')}")

    # ── Load GMF model ─────────────────────────────────────────────────────────
    model = GMF(num_users, num_items, emb_dim=EMBEDDING_DIM).to(device)
    model.load_state_dict(torch.load(GMF_CHECKPOINT, map_location=device))
    model.eval()

    # ── Get recommendations ────────────────────────────────────────────────────
    recs = get_recommendations(model, user_idx, num_items, train_item_idxs, args.top_k, device)
    hit = test_item in recs

    print(f"\nTop-{args.top_k} Recommendations:")
    for rank, item_idx in enumerate(recs, 1):
        game = idx_to_game.get(item_idx, f"item_{item_idx}")
        marker = " << TEST HIT" if item_idx == test_item else ""
        print(f"  {rank}. {game}{marker}")
    print(f"\nTest game in top-{args.top_k}: {'YES' if hit else 'NO'}")

    # ── RAG Explanation pipeline ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Generating Explanations (RAG + LLM)...")
    print(f"{'='*60}")

    vectors, idx_to_itemid_faiss = load_item_embeddings()
    store = VectorStore()
    store.build(vectors, idx_to_itemid_faiss)

    llm = LLMClient()
    cache = ExplanationCache(str(CACHE_PATH))

    pipeline = ExplainabilityPipeline(
        vector_store=store,
        llm_client=llm,
        cache=cache,
        items_meta=meta,
    )

    # Build user history in the format the pipeline expects
    user_history = []
    for _, row in user_train.iterrows():
        item_id = str(idx_to_itemid_faiss.get(int(row["item_idx"]), row["item_idx"]))
        game = idx_to_game.get(int(row["item_idx"]), "Unknown")
        user_history.append({
            "item_id": item_id,
            "title": game,
            "playtime": float(row.get("playtime", 1.0)),
        })

    for rank, item_idx in enumerate(recs, 1):
        item_id = str(idx_to_itemid_faiss.get(item_idx, item_idx))
        game = idx_to_game.get(item_idx, f"item_{item_idx}")
        print(f"\n{rank}. {game}")

        explanation = pipeline.explain(
            user_id=str(user_idx),
            item_id=item_id,
            user_history=user_history,
        )
        print(f"   Explanation: {explanation}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    run()
