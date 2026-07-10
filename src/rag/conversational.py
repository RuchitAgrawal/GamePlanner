"""
Conversational recommender.

Single-turn natural language query -> ranked game recommendations.

Pipeline:
1. Embed the user query with the same sentence-transformer used for items.
2. Retrieve top-N semantic candidates from the FAISS index.
3. If the user is known, re-rank candidates by blending CF model scores with
   semantic similarity scores (0.6 CF + 0.4 semantic by default).
4. Generate a short 2-sentence summary tying the results back to the query.

Multi-turn memory and intent parsing are out of scope for now (documented
in Future Scope in the README).
"""

import logging
import numpy as np
import pandas as pd
import torch

from src.rag.vector_store import VectorStore
from src.rag.llm_client import LLMClient
from src.rag.embeddings import embed_texts
from src.utils.config import FAISS_CANDIDATES

log = logging.getLogger(__name__)

CF_WEIGHT = 0.6
SEMANTIC_WEIGHT = 0.4


class ConversationalRecommender:
    """
    Natural language query to re-ranked game recommendations.

    Args:
        vector_store: loaded VectorStore instance.
        llm_client: initialized LLMClient.
        items_meta: DataFrame with item_id, title, tags, genres, description.
        cf_model: trained NeuMF or GMF model (can be None for anonymous users).
        num_items: total item catalog size (needed for CF scoring).
        device: torch device for CF model inference.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        llm_client: LLMClient,
        items_meta: pd.DataFrame,
        cf_model: torch.nn.Module | None = None,
        num_items: int = 0,
        device: str = "cpu",
    ):
        self.vs = vector_store
        self.llm = llm_client
        self.items_meta = items_meta
        self.cf_model = cf_model
        self.num_items = num_items
        self.device = torch.device(device)

    def recommend(
        self,
        query: str,
        user_idx: int | None = None,
        k: int = 10,
        n_candidates: int = FAISS_CANDIDATES,
    ) -> dict:
        """
        Run the full query -> recommend -> summarize pipeline.

        Args:
            query: natural language query string.
            user_idx: encoded user index for CF scoring (None for anonymous users).
            k: number of final recommendations to return.
            n_candidates: number of semantic candidates to retrieve before re-ranking.

        Returns:
            dict with keys: 'recommendations' (list of item dicts) and 'summary' (str).
        """
        if not query.strip():
            return {"recommendations": [], "summary": ""}

        # 1. Embed query
        query_vector = embed_texts([query])[0]

        # 2. Semantic retrieval
        candidates = self.vs.retrieve_similar_items(query_vector, k=n_candidates)
        if not candidates:
            return {"recommendations": [], "summary": "No matching games found."}

        # 3. Re-rank
        ranked = self._rerank(candidates, user_idx)
        top_k = ranked[:k]

        # 4. Build result list
        results = []
        for item_id, score in top_k:
            meta = self._get_meta(item_id)
            results.append({
                "item_id": item_id,
                "title": meta.get("title", item_id) if meta else item_id,
                "tags": meta.get("tags", "") if meta else "",
                "score": round(float(score), 4),
            })

        # 5. Generate summary
        summary = self._generate_summary(query, results[:5])

        return {"recommendations": results, "summary": summary}

    def _rerank(
        self, candidates: list[tuple[str, float]], user_idx: int | None
    ) -> list[tuple[str, float]]:
        """
        Blend semantic scores with CF model scores for known users.
        For anonymous users, rank by semantic score + normalize by position.
        """
        if user_idx is None or self.cf_model is None:
            return candidates

        item_ids = [iid for iid, _ in candidates]
        semantic_scores = {iid: s for iid, s in candidates}

        # Get CF scores for candidate items only
        cf_scores = self._get_cf_scores(user_idx, item_ids)

        blended = []
        for item_id in item_ids:
            sem = semantic_scores.get(item_id, 0.0)
            cf = cf_scores.get(item_id, 0.0)
            blended.append((item_id, CF_WEIGHT * cf + SEMANTIC_WEIGHT * sem))

        blended.sort(key=lambda x: x[1], reverse=True)
        return blended

    def _get_cf_scores(self, user_idx: int, item_ids: list[str]) -> dict[str, float]:
        """
        Score a specific set of items for a user using the CF model.
        Returns raw sigmoid probabilities (already in [0, 1]).
        """
        scores = {}
        if self.cf_model is None:
            return scores

        self.cf_model.eval()
        with torch.no_grad():
            for item_id in item_ids:
                item_meta = self._get_meta(item_id)
                if item_meta is None:
                    continue
                item_idx = item_meta.get("item_idx")
                if item_idx is None:
                    continue
                u = torch.tensor([user_idx], dtype=torch.long, device=self.device)
                i = torch.tensor([int(item_idx)], dtype=torch.long, device=self.device)
                logit = self.cf_model(u, i)
                scores[item_id] = float(torch.sigmoid(logit).item())

        # Min-max normalize to [0, 1]
        if scores:
            vals = list(scores.values())
            min_s, max_s = min(vals), max(vals)
            if max_s > min_s:
                scores = {k: (v - min_s) / (max_s - min_s) for k, v in scores.items()}

        return scores

    def _generate_summary(self, query: str, top_results: list[dict]) -> str:
        """Generate a 2-sentence summary connecting the query to the top results."""
        if not top_results:
            return ""

        titles = ", ".join(r["title"] for r in top_results)
        prompt = (
            f"A user searched for game recommendations with this query: '{query}'. "
            f"The top matches are: {titles}. "
            f"Write 2 sentences explaining why these games match the query. Be specific."
        )
        return self.llm.generate(prompt=prompt, context="")

    def _get_meta(self, item_id: str) -> dict | None:
        subset = self.items_meta[self.items_meta["item_id"].astype(str) == str(item_id)]
        return subset.iloc[0].to_dict() if not subset.empty else None
