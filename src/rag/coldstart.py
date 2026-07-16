"""
Cold-start recommendation handler.

Handles two cases:
  1. New user: no interaction history. User provides a list of liked game titles,
     which are embedded and used to query the FAISS index for similar games.
  2. New item: a game with too few interactions for the CF model to have a good
     embedding. Falls back to FAISS neighbors to borrow their interaction signal.

Both are wired into the main /recommend endpoint as automatic fallbacks.
"""

import logging
import numpy as np
import pandas as pd

from src.rag.vector_store import VectorStore
from src.rag.embeddings import embed_texts
from src.utils.config import COLD_START_MULTIPLIER

log = logging.getLogger(__name__)

# ── Semantic note templates ────────────────────────────────────────────────────
_SEED_NOTE_TMPL = "Recommended because it's semantically similar to {seed} in your profile."
_EMBED_NOTE_TMPL = "Recommended based on the overall style and tone of your game preferences."


class ColdStartHandler:
    """
    Provides recommendations for new users and new items.

    For new users: builds a pseudo-user vector from liked game titles and
    retrieves nearest neighbors from the FAISS index.

    For new items: finds the item's nearest neighbors in embedding space and
    averages their interaction popularity as a proxy ranking score.
    """

    def __init__(self, vector_store: VectorStore, items_meta: pd.DataFrame):
        self.vs = vector_store
        self.items_meta = items_meta
        self._title_to_item_id = self._build_title_index(items_meta)

    def _build_title_index(self, items_meta: pd.DataFrame) -> dict[str, str]:
        """Build a lowercase title -> item_id lookup for matching user-provided game names."""
        index = {}
        if "title" in items_meta.columns and "item_id" in items_meta.columns:
            for _, row in items_meta.iterrows():
                if pd.notna(row["title"]):
                    index[str(row["title"]).lower()] = str(row["item_id"])
        return index

    def get_new_user_recommendations(
        self, liked_game_titles: list[str], k: int = 10
    ) -> tuple[list[dict], list[str]]:
        """
        Recommend games for a new user who provided a list of titles they've enjoyed.

        Returns:
            (results, matched_seeds) where matched_seeds is the subset of
            liked_game_titles that were found in our catalog.
        """
        if not liked_game_titles:
            log.warning("No liked games provided for cold-start")
            return [], []

        # Match titles to item_ids
        matched_ids = []   # item_ids found in catalog
        matched_titles = []  # corresponding display titles
        unmatched = []
        for title in liked_game_titles:
            item_id = self._title_to_item_id.get(title.lower())
            if item_id:
                matched_ids.append(item_id)
                matched_titles.append(title)
            else:
                unmatched.append(title)

        if unmatched:
            log.info("Could not match these titles to catalog: %s", unmatched)

        # Build per-seed vectors for F-E1 closest-seed computation
        seed_vecs: list[np.ndarray] = []
        if matched_ids:
            for iid in matched_ids:
                v = self.vs.get_item_vector(iid)
                if v is not None:
                    seed_vecs.append(v)

        if matched_ids and seed_vecs:
            pseudo_vector = np.mean(seed_vecs, axis=0)
            pseudo_vector = pseudo_vector / (np.linalg.norm(pseudo_vector) + 1e-9)
        elif liked_game_titles:
            pseudo_vector = self._embed_titles(liked_game_titles)
        else:
            pseudo_vector = None

        if pseudo_vector is None:
            return [], matched_titles

        exclude = set(matched_ids)
        candidates = self.vs.retrieve_similar_items(
            pseudo_vector, k=k * COLD_START_MULTIPLIER, exclude_ids=exclude
        )

        results = []
        for item_id, score in candidates[:k]:
            meta = self._get_meta(item_id)
            title_str = meta.get("title", item_id) if meta else item_id

            # F-E1: find which seed this result is closest to
            closest_seed, semantic_note = self._compute_closest_seed(
                item_id, matched_titles, seed_vecs
            )

            results.append({
                "item_id": item_id,
                "title": title_str,
                "tags": meta.get("tags", "") if meta else "",
                "score": round(score, 4),
                "closest_seed": closest_seed,
                "semantic_note": semantic_note,
            })

        return results, matched_titles

    def _compute_closest_seed(
        self,
        item_id: str,
        seed_titles: list[str],
        seed_vecs: list[np.ndarray],
    ) -> tuple[str | None, str | None]:
        """
        F-E1: Find which seed game this result item is most similar to.

        Computes cosine similarity between the result's FAISS vector and each
        seed vector. Returns the seed title + a human-readable semantic note.
        Returns (None, None) if no seed vectors are available.
        """
        if not seed_vecs:
            return None, _EMBED_NOTE_TMPL

        result_vec = self.vs.get_item_vector(item_id)
        if result_vec is None:
            return None, None

        # Cosine similarity (vectors are already L2-normalized in the index)
        similarities = [
            float(np.dot(result_vec, sv))
            for sv in seed_vecs
        ]
        best_idx = int(np.argmax(similarities))
        best_seed = seed_titles[best_idx] if best_idx < len(seed_titles) else None
        best_score = similarities[best_idx]

        if best_seed is None:
            return None, None

        note = _SEED_NOTE_TMPL.format(seed=best_seed)
        log.debug("Item %s closest seed: %s (sim=%.3f)", item_id, best_seed, best_score)
        return best_seed, note

    def get_new_item_proxy_score(
        self, item_id: str, item_popularity: dict[str, int], k_neighbors: int = 5
    ) -> float:
        """
        Estimate a score for a cold-start item by averaging the popularity of its
        nearest neighbors in the FAISS index. Used to rank new items against known ones.

        Args:
            item_id: the new item's ID.
            item_popularity: dict mapping item_id -> interaction count in training data.
            k_neighbors: number of neighbors to average over.

        Returns:
            Proxy popularity score (float).
        """
        vec = self.vs.get_item_vector(item_id)
        if vec is None:
            return 0.0

        neighbors = self.vs.retrieve_similar_items(vec, k=k_neighbors + 1, exclude_ids={item_id})
        scores = [item_popularity.get(nid, 0) for nid, _ in neighbors[:k_neighbors]]
        return float(np.mean(scores)) if scores else 0.0

    def _embed_titles(self, titles: list[str]) -> np.ndarray | None:
        """Embed a list of title strings and return their mean vector."""
        try:
            vecs = embed_texts(titles)
            mean_vec = np.mean(vecs, axis=0)
            return mean_vec / (np.linalg.norm(mean_vec) + 1e-9)
        except Exception as exc:
            log.error("Failed to embed titles: %s", exc)
            return None

    def _get_meta(self, item_id: str) -> dict | None:
        subset = self.items_meta[self.items_meta["item_id"].astype(str) == str(item_id)]
        if subset.empty:
            return None
        return subset.iloc[0].to_dict()
