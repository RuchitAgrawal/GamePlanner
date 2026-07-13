"""
Baseline model evaluation script.

Trains and evaluates PopularityRecommender and ContentBasedRecommender
on the processed Steam dataset splits.

Evaluation protocol:
- Leave-one-last: each user has exactly 1 test item
- Metrics computed at K = 5, 10, 20
- Results saved to data/processed/experiments.json for API serving

Usage:
    python -m src.training.train_baseline
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.baseline import PopularityRecommender, ContentBasedRecommender
from src.evaluation.metrics import compute_all_metrics, aggregate_metrics
from src.utils.config import (
    TRAIN_PATH, TEST_PATH, ITEMS_META_PATH,
    DATA_PROCESSED_DIR, TOP_K,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

EXPERIMENTS_PATH = DATA_PROCESSED_DIR / "experiments.json"
K_VALUES = [5, 10, 20]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    log.info("Loading processed splits...")
    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)
    meta = pd.read_parquet(ITEMS_META_PATH)
    log.info("Train: %d rows | Test: %d rows | Items: %d", len(train), len(test), len(meta))
    return train, test, meta


def build_user_train_items(train: pd.DataFrame) -> dict[int, set[int]]:
    """Per-user set of item_idxs seen in training (to exclude from recommendations)."""
    return (
        train.groupby("user_idx")["item_idx"]
        .apply(set)
        .to_dict()
    )


def build_user_train_item_names(train: pd.DataFrame) -> dict[int, list[str]]:
    """Per-user list of game names seen in training (for ContentBased)."""
    return (
        train.groupby("user_idx")["item_id"]
        .apply(list)
        .to_dict()
    )


def evaluate_model(
    model_name: str,
    get_recs_fn,
    test: pd.DataFrame,
    user_train_items: dict[int, set[int]],
    k_values: list[int],
    num_items: int,
) -> dict:
    """
    Run leave-one-last evaluation for a single model.

    get_recs_fn(user_idx, exclude_items) -> list[int] of recommended item_idxs
    Returns aggregated metric dict.
    """
    log.info("Evaluating %s...", model_name)
    user_metrics = []
    skipped = 0

    for row in test.itertuples():
        user_idx = int(row.user_idx)
        test_item = int(row.item_idx)
        exclude = user_train_items.get(user_idx, set())

        try:
            recommended = get_recs_fn(user_idx, exclude)
        except Exception as exc:
            log.warning("Recommendation failed for user %d: %s", user_idx, exc)
            skipped += 1
            continue

        relevant = {test_item}
        user_metrics.append(compute_all_metrics(recommended, relevant, k_values))

    if skipped:
        log.warning("Skipped %d users due to errors", skipped)

    agg = aggregate_metrics(user_metrics)
    log.info("%s results:", model_name)
    for k in k_values:
        log.info(
            "  @%2d  HR=%.4f  NDCG=%.4f  Recall=%.4f  Prec=%.4f",
            k,
            agg.get(f"hit_rate@{k}", 0),
            agg.get(f"ndcg@{k}", 0),
            agg.get(f"recall@{k}", 0),
            agg.get(f"precision@{k}", 0),
        )
    return agg


def save_experiment(model_name: str, metrics: dict, duration_s: float) -> None:
    """Append results to experiments.json."""
    EXPERIMENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    runs = []
    if EXPERIMENTS_PATH.exists():
        with open(EXPERIMENTS_PATH) as f:
            runs = json.load(f)

    runs.append({
        "model": model_name,
        "metrics": {k: round(float(v), 6) for k, v in metrics.items()},
        "duration_s": round(duration_s, 2),
        "k_values": K_VALUES,
    })

    with open(EXPERIMENTS_PATH, "w") as f:
        json.dump(runs, f, indent=2)
    log.info("Saved experiment results to %s", EXPERIMENTS_PATH)


def run() -> None:
    train, test, meta = load_data()
    num_items = meta["item_idx"].max() + 1
    user_train_items = build_user_train_items(train)
    user_train_item_names = build_user_train_item_names(train)

    # ── Popularity Baseline ───────────────────────────────────────────────────
    t0 = time.time()
    pop = PopularityRecommender()
    pop.fit(train, meta)

    def pop_recs(user_idx: int, exclude: set[int]) -> list[int]:
        return pop.recommend(user_idx=user_idx, k=max(K_VALUES), exclude_items=exclude)

    pop_metrics = evaluate_model(
        "Popularity", pop_recs, test, user_train_items, K_VALUES, num_items
    )
    save_experiment("Popularity", pop_metrics, time.time() - t0)

    # ── Content-Based Baseline ────────────────────────────────────────────────
    t0 = time.time()
    cb = ContentBasedRecommender(max_features=2000)
    cb.fit(meta)

    # Build item_idx -> row index map for ContentBased
    item_idx_to_row: dict[int, int] = {}
    for row_i, item_idx in enumerate(meta["item_idx"].astype(int).values):
        item_idx_to_row[item_idx] = row_i

    def cb_recs(user_idx: int, exclude: set[int]) -> list[int]:
        played_item_ids = user_train_item_names.get(user_idx, [])
        # Map item_id strings to item_idx integers
        played_idxs = []
        for item_id in played_item_ids:
            match = meta[meta["item_id"].astype(str) == str(item_id)]
            if not match.empty:
                played_idxs.append(int(match.iloc[0]["item_idx"]))
        if not played_idxs:
            # Fall back to popularity if no mapped items
            return pop.recommend(user_idx=user_idx, k=max(K_VALUES), exclude_items=exclude)
        return cb.recommend(
            user_item_idxs=played_idxs,
            k=max(K_VALUES),
            exclude_items=exclude,
        )

    cb_metrics = evaluate_model(
        "ContentBased", cb_recs, test, user_train_items, K_VALUES, num_items
    )
    save_experiment("ContentBased", cb_metrics, time.time() - t0)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n=== Baseline Results Summary ===")
    print(f"{'Model':<15} {'HR@10':>8} {'NDCG@10':>9} {'Recall@10':>11}")
    print("-" * 46)
    for name, m in [("Popularity", pop_metrics), ("ContentBased", cb_metrics)]:
        print(
            f"{name:<15} {m.get('hit_rate@10', 0):>8.4f} "
            f"{m.get('ndcg@10', 0):>9.4f} "
            f"{m.get('recall@10', 0):>11.4f}"
        )


if __name__ == "__main__":
    run()
