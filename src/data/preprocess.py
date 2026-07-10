"""
Data preprocessing pipeline for the Steam dataset.

Handles loading, cleaning, filtering, splitting, and encoding of
user-item interaction data. Saves processed splits as parquet files.

Usage:
    python -m src.data.preprocess
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

from src.utils.config import (
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    MIN_USER_INTERACTIONS,
    MIN_ITEM_INTERACTIONS,
    CONFIDENCE_ALPHA,
    USER2IDX_PATH,
    ITEM2IDX_PATH,
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    ITEMS_META_PATH,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_kaggle_csv(path: Path) -> pd.DataFrame:
    """
    Load the Kaggle Steam CSV (tamber/steam-video-games).
    Columns: user_id, game_name, behavior, value, 0
    Only keep 'play' rows; 'value' is playtime in hours.
    """
    df = pd.read_csv(path, header=None,
                     names=["user_id", "game_name", "behavior", "playtime", "drop"])
    df = df[df["behavior"] == "play"].copy()
    df = df.drop(columns=["behavior", "drop"])
    df["item_id"] = df["game_name"]   # use game name as item ID for Kaggle version
    df["playtime"] = df["playtime"].astype(float)
    log.info("Loaded Kaggle CSV: %d rows", len(df))
    return df


def load_ucsd_reviews(reviews_path: Path, games_path: Path) -> pd.DataFrame:
    """
    Load the UCSD Steam dataset (McAuley et al.).
    reviews_path: steam_reviews.json.gz
    games_path:   steam_games.json.gz
    """
    log.info("Loading UCSD reviews...")
    reviews = pd.read_json(reviews_path, lines=True, compression="gzip")
    reviews = reviews.rename(columns={
        "username": "user_id",
        "product_id": "item_id",
        "hours": "playtime",
        "date": "timestamp",
    })
    keep_cols = [c for c in ["user_id", "item_id", "playtime", "timestamp"] if c in reviews.columns]
    reviews = reviews[keep_cols]

    log.info("Loading UCSD game metadata...")
    games = pd.read_json(games_path, lines=True, compression="gzip")
    games = games.rename(columns={"id": "item_id"})

    df = reviews.merge(games[["item_id", "title", "tags", "genres", "description", "developer"]],
                       on="item_id", how="left")
    log.info("Loaded UCSD dataset: %d rows, %d unique games", len(df), df["item_id"].nunique())
    return df


# ── Cleaning and filtering ─────────────────────────────────────────────────────

def filter_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Remove low-activity users and items to reduce sparsity."""
    before = len(df)

    # iterate until stable (users and items affect each other after filtering)
    for _ in range(5):
        user_counts = df["user_id"].value_counts()
        item_counts = df["item_id"].value_counts()
        valid_users = user_counts[user_counts >= MIN_USER_INTERACTIONS].index
        valid_items = item_counts[item_counts >= MIN_ITEM_INTERACTIONS].index
        df = df[df["user_id"].isin(valid_users) & df["item_id"].isin(valid_items)]

    log.info("Filtered interactions: %d -> %d rows", before, len(df))
    log.info("  Users: %d | Items: %d", df["user_id"].nunique(), df["item_id"].nunique())
    return df.reset_index(drop=True)


def compute_confidence(playtime: pd.Series) -> pd.Series:
    """
    Log-scaled confidence score based on playtime.
    Standard weighting used in implicit feedback models (ALS/WARP).
    confidence = 1 + alpha * log(1 + playtime)
    """
    return 1.0 + CONFIDENCE_ALPHA * np.log1p(playtime)


def report_sparsity(df: pd.DataFrame) -> None:
    """Print sparsity stats for the interaction matrix."""
    n_users = df["user_id"].nunique()
    n_items = df["item_id"].nunique()
    n_interactions = len(df)
    sparsity = 1.0 - n_interactions / (n_users * n_items)
    log.info("Matrix stats:")
    log.info("  Users: %d | Items: %d | Interactions: %d", n_users, n_items, n_interactions)
    log.info("  Sparsity: %.4f%%", sparsity * 100)
    log.info("  Avg interactions per user: %.1f", n_interactions / n_users)
    log.info("  Avg interactions per item: %.1f", n_interactions / n_items)


# ── Splitting ─────────────────────────────────────────────────────────────────

def leave_one_last_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Time-based leave-one-last split.
    For each user: most recent interaction = test, second most recent = val, rest = train.

    This protocol is standard for temporal recommendation evaluation and simulates
    real deployment conditions much better than a random split.
    """
    if "timestamp" in df.columns and df["timestamp"].notna().any():
        df = df.sort_values(["user_id", "timestamp"])
        log.info("Splitting by timestamp")
    else:
        # Kaggle CSV has no timestamps; sort by row order as a proxy
        log.warning("No timestamp column found. Using row order for split.")
        df = df.sort_values("user_id")

    df["rank"] = df.groupby("user_id").cumcount(ascending=False)

    test_df = df[df["rank"] == 0].drop(columns=["rank"])
    val_df = df[df["rank"] == 1].drop(columns=["rank"])
    train_df = df[df["rank"] >= 2].drop(columns=["rank"])

    log.info("Split sizes: train=%d | val=%d | test=%d",
             len(train_df), len(val_df), len(test_df))
    return train_df, val_df, test_df


# ── Encoding ──────────────────────────────────────────────────────────────────

def encode_ids(train_df: pd.DataFrame,
               val_df: pd.DataFrame,
               test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict]:
    """
    Fit LabelEncoders on training users/items, then transform all splits.
    Saves mappings to JSON for use at inference time.
    """
    user_enc = LabelEncoder()
    item_enc = LabelEncoder()

    user_enc.fit(train_df["user_id"])
    item_enc.fit(train_df["item_id"])

    for split in [train_df, val_df, test_df]:
        split["user_idx"] = user_enc.transform(
            split["user_id"].where(split["user_id"].isin(user_enc.classes_), other=user_enc.classes_[0])
        )
        split["item_idx"] = item_enc.transform(
            split["item_id"].where(split["item_id"].isin(item_enc.classes_), other=item_enc.classes_[0])
        )

    user2idx = {str(u): int(i) for i, u in enumerate(user_enc.classes_)}
    item2idx = {str(it): int(i) for i, it in enumerate(item_enc.classes_)}

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(USER2IDX_PATH, "w") as f:
        json.dump(user2idx, f)
    with open(ITEM2IDX_PATH, "w") as f:
        json.dump(item2idx, f)

    log.info("Encoded %d users and %d items", len(user2idx), len(item2idx))
    return train_df, val_df, test_df, user2idx, item2idx


# ── Metadata ──────────────────────────────────────────────────────────────────

def build_items_metadata(df: pd.DataFrame, item2idx: dict) -> pd.DataFrame:
    """
    Build a per-item metadata table used by the RAG embedding pipeline.
    Deduplicates on item_id and keeps available text fields.
    """
    meta_cols = ["item_id"]
    for col in ["game_name", "title", "tags", "genres", "description", "developer"]:
        if col in df.columns:
            meta_cols.append(col)

    meta = df[meta_cols].drop_duplicates(subset=["item_id"]).copy()
    meta["item_idx"] = meta["item_id"].astype(str).map(item2idx)
    meta = meta.dropna(subset=["item_idx"])
    meta["item_idx"] = meta["item_idx"].astype(int)

    # Kaggle version uses game_name as title
    if "title" not in meta.columns and "game_name" in meta.columns:
        meta = meta.rename(columns={"game_name": "title"})

    return meta.reset_index(drop=True)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(dataset: str = "kaggle") -> None:
    """
    Full preprocessing pipeline.
    dataset: 'kaggle' or 'ucsd'
    """
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if dataset == "kaggle":
        csv_candidates = list(DATA_RAW_DIR.glob("*.csv"))
        if not csv_candidates:
            raise FileNotFoundError(f"No CSV file found in {DATA_RAW_DIR}. "
                                    "Download from: https://www.kaggle.com/datasets/tamber/steam-video-games")
        df = load_kaggle_csv(csv_candidates[0])
    elif dataset == "ucsd":
        reviews_path = DATA_RAW_DIR / "steam_reviews.json.gz"
        games_path = DATA_RAW_DIR / "steam_games.json.gz"
        if not reviews_path.exists():
            raise FileNotFoundError(f"steam_reviews.json.gz not found in {DATA_RAW_DIR}")
        df = load_ucsd_reviews(reviews_path, games_path)
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Use 'kaggle' or 'ucsd'.")

    df["playtime"] = pd.to_numeric(df["playtime"], errors="coerce").fillna(0.0)
    df = df[df["playtime"] > 0].copy()
    df["confidence"] = compute_confidence(df["playtime"])

    df = filter_interactions(df)
    report_sparsity(df)

    train_df, val_df, test_df = leave_one_last_split(df)
    train_df, val_df, test_df, user2idx, item2idx = encode_ids(train_df, val_df, test_df)

    train_df.to_parquet(TRAIN_PATH, index=False)
    val_df.to_parquet(VAL_PATH, index=False)
    test_df.to_parquet(TEST_PATH, index=False)
    log.info("Saved train/val/test parquet files to %s", DATA_PROCESSED_DIR)

    meta = build_items_metadata(df, item2idx)
    meta.to_parquet(ITEMS_META_PATH, index=False)
    log.info("Saved items_metadata.parquet (%d items)", len(meta))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["kaggle", "ucsd"], default="kaggle")
    args = parser.parse_args()
    run(dataset=args.dataset)
