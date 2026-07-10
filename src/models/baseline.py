"""
Baseline recommenders used as performance benchmarks.

PopularityRecommender: ranks items by overall interaction count.
ContentBasedRecommender: builds TF-IDF user profiles from played game text.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class PopularityRecommender:
    """
    Recommends the most interacted-with items across all users.
    Serves as the baseline floor -- any CF model should clearly outperform this.

    Optionally filters recommendations by a genre to create a genre-aware variant.
    """

    def __init__(self):
        self.item_scores: pd.Series | None = None
        self.item_meta: pd.DataFrame | None = None

    def fit(self, train_df: pd.DataFrame, items_meta: pd.DataFrame | None = None) -> None:
        self.item_scores = train_df.groupby("item_idx")["item_idx"].count().rename("count")
        self.item_meta = items_meta

    def recommend(self, user_idx: int, k: int = 10,
                  exclude_items: set[int] | None = None,
                  genre_filter: str | None = None) -> list[int]:
        """
        Returns top-k item indices by popularity.
        exclude_items: items to remove from results (e.g. user's train interactions).
        genre_filter: optional genre string to subset the catalog.
        """
        scores = self.item_scores.copy()

        if exclude_items:
            scores = scores[~scores.index.isin(exclude_items)]

        if genre_filter and self.item_meta is not None and "genres" in self.item_meta.columns:
            genre_items = self.item_meta[
                self.item_meta["genres"].str.contains(genre_filter, case=False, na=False)
            ]["item_idx"].values
            scores = scores[scores.index.isin(genre_items)]

        return scores.nlargest(k).index.tolist()


class ContentBasedRecommender:
    """
    TF-IDF content-based recommender.

    Builds a bag-of-words item representation from title + tags + genres,
    then scores unplayed items by cosine similarity to a user's average profile.

    Also useful as a cold-start fallback since it requires no interaction history.
    """

    def __init__(self, max_features: int = 5000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            ngram_range=(1, 2),
        )
        self.item_vectors: np.ndarray | None = None
        self.item_idx_to_row: dict[int, int] = {}
        self.row_to_item_idx: dict[int, int] = {}

    def fit(self, items_meta: pd.DataFrame) -> None:
        """
        Build TF-IDF vectors from item metadata.
        items_meta must have: item_idx, and at least one of title, tags, genres.
        """
        text_cols = [c for c in ["title", "tags", "genres", "description"] if c in items_meta.columns]
        items_meta = items_meta.dropna(subset=["item_idx"]).copy()
        items_meta["text"] = items_meta[text_cols].fillna("").agg(" ".join, axis=1)

        self.item_vectors = self.vectorizer.fit_transform(items_meta["text"]).toarray()
        for row, item_idx in enumerate(items_meta["item_idx"].astype(int).values):
            self.item_idx_to_row[item_idx] = row
            self.row_to_item_idx[row] = item_idx

    def recommend(self, user_item_idxs: list[int], k: int = 10,
                  exclude_items: set[int] | None = None) -> list[int]:
        """
        Recommend items based on the user's play history (list of item_idxs).
        Returns top-k item indices sorted by cosine similarity to the user profile.
        """
        if self.item_vectors is None:
            raise RuntimeError("Call fit() before recommend()")

        # Build user profile: average TF-IDF of played items
        rows = [self.item_idx_to_row[i] for i in user_item_idxs if i in self.item_idx_to_row]
        if not rows:
            return []

        user_profile = self.item_vectors[rows].mean(axis=0, keepdims=True)
        scores = cosine_similarity(user_profile, self.item_vectors)[0]

        exclude = exclude_items or set()
        ranked = np.argsort(scores)[::-1]

        results = []
        for row in ranked:
            item_idx = self.row_to_item_idx[row]
            if item_idx not in exclude and len(results) < k:
                results.append(item_idx)
            if len(results) >= k:
                break

        return results
