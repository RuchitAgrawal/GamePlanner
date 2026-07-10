"""
Explainable recommendations feature.

Generates a short, grounded natural-language explanation for each recommendation
by retrieving similar items from the user's play history and feeding that context
to the LLM. Explanations are cached to avoid redundant API calls.

A lightweight groundedness check verifies that the explanation doesn't reference
game names that were not provided as context (basic hallucination detection).
"""

import logging
import re
import pandas as pd

from src.rag.llm_client import LLMClient
from src.rag.vector_store import VectorStore
from src.utils.cache import ExplanationCache

log = logging.getLogger(__name__)


def build_explanation_prompt(
    recommended_title: str,
    recommended_tags: str,
    recommended_description: str,
    similar_past_games: list[tuple[str, float]],  # [(title, playtime_hours), ...]
    review_snippets: list[str],
) -> str:
    """
    Construct the prompt sent to the LLM.
    Keeping prompts small keeps latency low and stays within token limits.
    """
    past_games_str = ", ".join(
        f"{title} ({hours:.0f}h)" for title, hours in similar_past_games
    ) or "no similar past games found"

    reviews_str = " ".join(f'"{s}"' for s in review_snippets[:2]) or "no reviews available"

    return (
        f"You are a game recommendation assistant. "
        f"Explain in 1-2 sentences why a player who enjoyed {past_games_str} "
        f"would like '{recommended_title}'. "
        f"Be specific and reference their history. Do not invent game titles.\n\n"
        f"Game tags: {recommended_tags}\n"
        f"Description: {recommended_description[:150]}\n"
        f"Sample reviews: {reviews_str}"
    )


def groundedness_check(explanation: str, allowed_titles: set[str]) -> bool:
    """
    Simple check: any quoted or title-cased name in the explanation that is not
    in allowed_titles might be a hallucination. Returns False if a suspicious
    name is detected.

    This is heuristic -- it catches obvious cases like fabricated game names
    but is not a full factual verification system.
    """
    # Extract quoted strings as potential title references
    quoted = re.findall(r'"([^"]+)"', explanation)
    for phrase in quoted:
        if len(phrase) > 3 and not any(
            phrase.lower() in title.lower() for title in allowed_titles
        ):
            log.warning("Possible hallucinated reference in explanation: '%s'", phrase)
            return False
    return True


def fallback_explanation(
    recommended_title: str, tags: str, similar_past_games: list[tuple[str, float]]
) -> str:
    """Template explanation used when LLM call fails or groundedness check fails."""
    if similar_past_games:
        ref = similar_past_games[0][0]
        return (
            f"Recommended based on your interest in {ref} and similar {tags} games."
        )
    return f"Recommended based on your interest in {tags} games."


class ExplainabilityPipeline:
    """
    Orchestrates the full explanation generation workflow for a recommendation.

    Steps:
    1. Check explanation cache (return immediately on hit).
    2. Get the recommended item's metadata.
    3. Find semantically similar games in the user's history via FAISS.
    4. Retrieve review snippets (if available).
    5. Build prompt and call LLM.
    6. Run groundedness check.
    7. Cache and return.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        llm_client: LLMClient,
        cache: ExplanationCache,
        items_meta: pd.DataFrame,
    ):
        self.vs = vector_store
        self.llm = llm_client
        self.cache = cache
        self.items_meta = items_meta.set_index("item_id") if "item_id" in items_meta.columns else items_meta

    def explain(
        self,
        user_id: str,
        item_id: str,
        user_history: list[dict],   # [{"item_id": ..., "title": ..., "playtime": ...}, ...]
    ) -> str:
        """
        Generate or retrieve a cached explanation for (user_id, item_id).

        Args:
            user_id: string user identifier.
            item_id: string item identifier for the recommended game.
            user_history: list of the user's interacted items with playtime.

        Returns:
            A 1-2 sentence explanation string.
        """
        # 1. Cache check
        cached = self.cache.get(user_id, item_id)
        if cached:
            return cached

        # 2. Get recommended item metadata
        meta = self._get_item_meta(item_id)
        if meta is None:
            log.warning("No metadata for item %s, using fallback", item_id)
            return f"Recommended game (id: {item_id})"

        title = meta.get("title", item_id)
        tags = str(meta.get("tags", ""))
        description = str(meta.get("description", ""))

        # 3. Find semantically similar items from user history
        item_vec = self.vs.get_item_vector(item_id)
        similar_past = []
        if item_vec is not None and user_history:
            history_ids = {str(h["item_id"]) for h in user_history}
            candidates = self.vs.retrieve_similar_items(item_vec, k=10, exclude_ids={item_id})
            for cand_id, _ in candidates:
                if cand_id in history_ids:
                    cand_meta = self._get_item_meta(cand_id)
                    if cand_meta:
                        playtime = next(
                            (h["playtime"] for h in user_history if str(h["item_id"]) == cand_id), 0.0
                        )
                        similar_past.append((str(cand_meta.get("title", cand_id)), float(playtime)))
                    if len(similar_past) >= 2:
                        break

        # 4. Review snippets (best-effort -- only available in UCSD dataset)
        review_snippets = []   # extend here if review text is embedded in the vector store

        # 5. Build prompt and call LLM
        prompt = build_explanation_prompt(title, tags, description, similar_past, review_snippets)
        explanation = self.llm.generate(prompt=prompt, context="")

        # 6. Groundedness check
        if explanation:
            allowed = {title} | {g for g, _ in similar_past}
            if not groundedness_check(explanation, allowed):
                log.info("Groundedness check failed for %s, using fallback", item_id)
                explanation = fallback_explanation(title, tags, similar_past)
        else:
            explanation = fallback_explanation(title, tags, similar_past)

        # 7. Cache and return
        self.cache.set(user_id, item_id, explanation)
        return explanation

    def _get_item_meta(self, item_id: str) -> dict | None:
        try:
            row = self.items_meta.loc[str(item_id)]
            return row.to_dict() if not isinstance(row, dict) else row
        except KeyError:
            return None
