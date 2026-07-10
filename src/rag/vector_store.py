"""
FAISS vector store wrapper for semantic item retrieval.

Builds and queries a FAISS IndexFlatIP (inner product) index over
L2-normalized item embeddings. Inner product on normalized vectors
equals cosine similarity, which is what we want for semantic search.

The index is built once and loaded from disk on subsequent runs.
At ~10K items, CPU FAISS queries are sub-millisecond.
"""

import json
import logging
import numpy as np
import faiss

from src.utils.config import (
    FAISS_INDEX_PATH,
    IDX_TO_ITEMID_PATH,
    ITEM_VECTORS_PATH,
)

log = logging.getLogger(__name__)


class VectorStore:
    """
    Thin FAISS wrapper for building and querying the item embedding index.

    Stores item_id as metadata alongside the index so retrieval results
    can be mapped back to item IDs and metadata.
    """

    def __init__(self):
        self.index: faiss.Index | None = None
        self.idx_to_itemid: dict[int, str] = {}
        self.itemid_to_idx: dict[str, int] = {}
        self.dim: int = 0

    def build(self, item_vectors: np.ndarray, idx_to_itemid: dict[int, str]) -> None:
        """
        Build the FAISS index from pre-normalized item vectors.

        Args:
            item_vectors: (num_items, dim) float32 array, L2-normalized.
            idx_to_itemid: row index -> item_id string mapping.
        """
        vectors = item_vectors.astype(np.float32)
        if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5):
            log.warning("Item vectors are not L2-normalized. Normalizing now.")
            faiss.normalize_L2(vectors)

        self.dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(vectors)

        self.idx_to_itemid = {int(k): v for k, v in idx_to_itemid.items()}
        self.itemid_to_idx = {v: int(k) for k, v in self.idx_to_itemid.items()}

        faiss.write_index(self.index, str(FAISS_INDEX_PATH))
        with open(IDX_TO_ITEMID_PATH, "w") as f:
            json.dump({str(k): v for k, v in self.idx_to_itemid.items()}, f)

        log.info("Built FAISS index: %d vectors, dim=%d", self.index.ntotal, self.dim)

    def load(self, index_path: str | None = None) -> None:
        """Load index and ID mapping from disk."""
        path = index_path or str(FAISS_INDEX_PATH)
        self.index = faiss.read_index(path)
        with open(IDX_TO_ITEMID_PATH) as f:
            raw = json.load(f)
        self.idx_to_itemid = {int(k): v for k, v in raw.items()}
        self.itemid_to_idx = {v: int(k) for k, v in self.idx_to_itemid.items()}
        self.dim = self.index.d
        log.info("Loaded FAISS index: %d vectors, dim=%d", self.index.ntotal, self.dim)

    def retrieve_similar_items(
        self, query_vector: np.ndarray, k: int, exclude_ids: set[str] | None = None
    ) -> list[tuple[str, float]]:
        """
        Retrieve top-k most similar items by cosine similarity.

        Args:
            query_vector: (dim,) or (1, dim) float32, should be L2-normalized.
            k: number of results to return.
            exclude_ids: item_id strings to remove from results.

        Returns:
            List of (item_id, score) tuples, descending by score.
        """
        if self.index is None:
            raise RuntimeError("Index not built or loaded. Call build() or load() first.")

        q = query_vector.astype(np.float32).reshape(1, -1)
        # Retrieve extra candidates so we still have k after filtering
        fetch_k = k + len(exclude_ids or []) + 10
        scores, indices = self.index.search(q, fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            item_id = self.idx_to_itemid.get(int(idx))
            if item_id is None:
                continue
            if exclude_ids and item_id in exclude_ids:
                continue
            results.append((item_id, float(score)))
            if len(results) >= k:
                break

        return results

    def retrieve_by_text(
        self, query_text: str, k: int, exclude_ids: set[str] | None = None
    ) -> list[tuple[str, float]]:
        """
        Embed a text query and retrieve similar items.
        Convenience wrapper around retrieve_similar_items.
        """
        from src.rag.embeddings import embed_texts
        query_vector = embed_texts([query_text])[0]
        return self.retrieve_similar_items(query_vector, k, exclude_ids)

    def get_item_vector(self, item_id: str) -> np.ndarray | None:
        """Return the stored embedding for a single item by item_id."""
        if self.index is None:
            return None
        idx = self.itemid_to_idx.get(item_id)
        if idx is None:
            return None
        vectors = np.zeros((1, self.dim), dtype=np.float32)
        self.index.reconstruct(idx, vectors[0])
        return vectors[0]
