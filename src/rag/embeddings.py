"""
Item embedding generation using sentence-transformers.

Encodes game metadata (title + tags + genres + description) into dense vectors
that are stored in the FAISS index for semantic retrieval.

This is a one-time precomputation step. Run before starting any RAG feature.
"""

import json
import logging
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.utils.config import (
    SENTENCE_TRANSFORMER_MODEL,
    EMBEDDING_BATCH_SIZE,
    ITEMS_META_PATH,
    ITEM_VECTORS_PATH,
    IDX_TO_ITEMID_PATH,
    DATA_EMBEDDINGS_DIR,
)

log = logging.getLogger(__name__)


def build_item_text(row: pd.Series) -> str:
    """
    Concatenate available text fields for a game into a single input string.
    Keeps description short to avoid token limits and focus on key signals.
    """
    parts = []
    if pd.notna(row.get("title")):
        parts.append(str(row["title"]))
    if pd.notna(row.get("tags")):
        parts.append(str(row["tags"]))
    if pd.notna(row.get("genres")):
        parts.append(str(row["genres"]))
    if pd.notna(row.get("description")):
        parts.append(str(row["description"])[:200])
    return " | ".join(parts)


def generate_item_embeddings(
    items_meta: pd.DataFrame | None = None,
    model_name: str = SENTENCE_TRANSFORMER_MODEL,
    batch_size: int = EMBEDDING_BATCH_SIZE,
    device: str | None = None,
) -> tuple[np.ndarray, dict[int, str]]:
    """
    Encode all items in items_metadata.parquet into dense vectors.

    Args:
        items_meta: optional pre-loaded metadata DataFrame. Loaded from disk if None.
        model_name: sentence-transformers model name.
        batch_size: encoding batch size (64 is safe for 4GB VRAM).
        device: 'cuda', 'cpu', or None (auto-detect).

    Returns:
        item_vectors: np.ndarray of shape (num_items, dim), L2-normalized.
        idx_to_itemid: dict mapping row index -> item_id string.
    """
    if items_meta is None:
        items_meta = pd.read_parquet(ITEMS_META_PATH)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    log.info("Loading sentence-transformer model: %s on %s", model_name, device)
    model = SentenceTransformer(model_name, device=device)

    texts = items_meta.apply(build_item_text, axis=1).tolist()
    item_idxs = items_meta["item_idx"].astype(int).tolist()

    log.info("Encoding %d items in batches of %d...", len(texts), batch_size)
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,   # L2-normalize so inner product = cosine similarity
        convert_to_numpy=True,
    )

    idx_to_itemid = {row: str(items_meta.iloc[row]["item_id"]) for row in range(len(items_meta))}

    DATA_EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(ITEM_VECTORS_PATH, vectors)
    log.info("Saved item vectors: %s  shape=%s", ITEM_VECTORS_PATH, vectors.shape)

    with open(IDX_TO_ITEMID_PATH, "w") as f:
        json.dump(idx_to_itemid, f)
    log.info("Saved idx->item_id mapping: %s", IDX_TO_ITEMID_PATH)

    return vectors, idx_to_itemid


def load_item_embeddings() -> tuple[np.ndarray, dict[int, str]]:
    """Load precomputed embeddings from disk."""
    vectors = np.load(ITEM_VECTORS_PATH)
    with open(IDX_TO_ITEMID_PATH) as f:
        idx_to_itemid = {int(k): v for k, v in json.load(f).items()}
    return vectors, idx_to_itemid


def embed_texts(texts: list[str], model_name: str = SENTENCE_TRANSFORMER_MODEL,
                device: str | None = None) -> np.ndarray:
    """
    Embed a small list of texts on the fly (for query embedding, cold-start, etc.).
    Returns L2-normalized vectors.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
    generate_item_embeddings()
