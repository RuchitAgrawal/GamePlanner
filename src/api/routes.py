"""
FastAPI route definitions.

All routes are registered under /api/v1/ prefix.
Core endpoints (/recommend, /metrics, /health) are built on the CF model.
Extension endpoints (/explain, /coldstart, /chat) layer on the RAG pipeline.
"""

import json
import logging

import pandas as pd
import torch
from fastapi import APIRouter, Depends, HTTPException, Request

from src.api.schemas import (
    RecommendResponse, ExplainResponse, ItemResult,
    ColdStartRequest, ColdStartResponse,
    ChatRequest, ChatResponse,
    HealthResponse, MetricsResponse,
)
from src.utils.config import DEFAULT_TOP_K, TOP_K, EXPERIMENTS_PATH

log = logging.getLogger(__name__)
router = APIRouter()


def get_state(request: Request):
    """Dependency: pulls shared state loaded at app startup from app.state."""
    return request.app.state


# ── Core endpoints ─────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
def health(state=Depends(get_state)):
    return HealthResponse(
        status="ok",
        model=state.model_name,
        num_users=state.num_users,
        num_items=state.num_items,
        cache_size=len(state.cache),
    )


@router.get("/metrics", response_model=MetricsResponse)
def metrics(state=Depends(get_state)):
    if not EXPERIMENTS_PATH.exists():
        raise HTTPException(status_code=404, detail="No evaluation results found. Run evaluate.py first.")
    with open(EXPERIMENTS_PATH) as f:
        runs = json.load(f)
    if not runs:
        raise HTTPException(status_code=404, detail="experiments.json is empty")
    # Return the best run by HR@10 (production model)
    best = max(runs, key=lambda r: r.get("metrics", {}).get("hit_rate@10", 0))
    return MetricsResponse(
        model=best.get("model", "unknown"),
        metrics=best.get("metrics", {}),
        k_values=TOP_K,
    )


@router.get("/recommend/{user_id}", response_model=RecommendResponse)
def recommend(user_id: str, k: int = DEFAULT_TOP_K, state=Depends(get_state)):
    """
    Return top-k recommendations for a known user.
    Automatically falls back to cold-start if the user is not in the training set.
    """
    user_idx = state.user2idx.get(user_id)

    if user_idx is None:
        # User not in training set: fall back to popularity-based recs
        log.info("User %s not in training set, using popularity fallback", user_id)
        item_idxs = state.popularity_model.recommend(user_idx=-1, k=k)
        recs = _idxs_to_results(item_idxs, state.items_meta, state.item_popularity)
        return RecommendResponse(user_id=user_id, recommendations=recs,
                                 source="cold_start", count=len(recs))

    # CF model scoring
    user_tensor = torch.tensor([user_idx], dtype=torch.long, device=state.device)
    all_scores = state.model.score_all_items(user_tensor, state.num_items, state.device)

    # Exclude items already in training set
    train_items = state.user_train_items.get(str(user_idx), set())
    all_scores_np = all_scores.cpu().numpy()
    for item_idx in train_items:
        if item_idx < len(all_scores_np):
            all_scores_np[item_idx] = -1e9

    top_k_idxs = all_scores_np.argsort()[::-1][:k].tolist()
    recs = _idxs_to_results(top_k_idxs, state.items_meta, state.item_popularity, scores=all_scores_np)

    return RecommendResponse(user_id=user_id, recommendations=recs,
                             source="collaborative_filtering", count=len(recs))


# ── RAG extension endpoints ────────────────────────────────────────────────────

@router.get("/recommend/{user_id}/explain", response_model=ExplainResponse)
def recommend_with_explanations(user_id: str, k: int = DEFAULT_TOP_K, state=Depends(get_state)):
    """
    Same as /recommend but each result includes a generated explanation.
    Explanations are cached so repeated calls for the same user/item are free.
    """
    base = recommend(user_id=user_id, k=k, state=state)
    user_history = _get_user_history(str(state.user2idx.get(user_id, "")), state)

    explained_recs = []
    for rec in base.recommendations:
        explanation = state.explainability.explain(
            user_id=user_id,
            item_id=rec.item_id,
            user_history=user_history,
        )
        explained_recs.append(ItemResult(**rec.model_dump(exclude={"explanation"}), explanation=explanation))

    return ExplainResponse(user_id=user_id, recommendations=explained_recs,
                           source=base.source, count=len(explained_recs))


@router.post("/coldstart", response_model=ColdStartResponse)
def coldstart(body: ColdStartRequest, state=Depends(get_state)):
    """New user recommendations from a list of liked game titles.

    F-E1: Each result includes closest_seed + semantic_note (computed from FAISS cosine sim).
    F-E2: Response includes an optional llm_summary paragraph (1 LLM call, cached by seed hash).
    """
    results_raw, matched_seeds = state.coldstart_handler.get_new_user_recommendations(
        liked_game_titles=body.liked_games, k=body.k
    )

    results = [
        ItemResult(
            item_id=r["item_id"],
            title=r["title"],
            tags=str(r.get("tags", "")),
            score=r["score"],
            closest_seed=r.get("closest_seed"),
            semantic_note=r.get("semantic_note"),
        )
        for r in results_raw
    ]

    # F-E2: generate one LLM summary paragraph (1 call, cached)
    llm_summary: str | None = None
    if results and matched_seeds:
        result_titles = [r.title for r in results]
        try:
            summary = state.coldstart_handler.generate_summary(
                seed_titles=matched_seeds,
                result_titles=result_titles,
                llm_client=state.llm,
            )
            llm_summary = summary if summary else None
        except Exception as exc:
            log.warning("Could not generate coldstart summary: %s", exc)

    return ColdStartResponse(
        recommendations=results,
        count=len(results),
        llm_summary=llm_summary,
        matched_seeds=matched_seeds,
    )


@router.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest, state=Depends(get_state)):
    """Natural language query to re-ranked game recommendations."""
    user_idx = state.user2idx.get(body.user_id) if body.user_id else None
    result = state.conversational.recommend(
        query=body.query, user_idx=user_idx, k=body.k
    )
    recs = [
        ItemResult(
            item_id=r["item_id"],
            title=r["title"],
            tags=str(r.get("tags", "")),
            score=r["score"],
        )
        for r in result["recommendations"]
    ]
    return ChatResponse(
        query=body.query,
        recommendations=recs,
        summary=result.get("summary", ""),
        count=len(recs),
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _idxs_to_results(
    item_idxs: list[int],
    items_meta: pd.DataFrame,
    item_popularity: dict[int, int],
    scores: list | None = None,
) -> list[ItemResult]:
    results = []
    for rank, idx in enumerate(item_idxs):
        row = items_meta[items_meta["item_idx"] == idx]
        if row.empty:
            continue
        row = row.iloc[0]
        score = float(scores[idx]) if scores is not None else float(item_popularity.get(idx, 0))
        results.append(ItemResult(
            item_id=str(row.get("item_id", idx)),
            title=str(row.get("title", f"item_{idx}")),
            tags=str(row.get("tags", "")),
            score=round(score, 4),
        ))
    return results


def _get_user_history(user_idx_str: str, state) -> list[dict]:
    """Return user's interaction history as a list of dicts with item_id and playtime."""
    if not user_idx_str:
        return []
    train_items = state.user_train_items.get(user_idx_str, set())
    history = []
    for item_idx in list(train_items)[:20]:
        row = state.items_meta[state.items_meta["item_idx"] == int(item_idx)]
        if not row.empty:
            row = row.iloc[0]
            history.append({
                "item_id": str(row.get("item_id", item_idx)),
                "title": str(row.get("title", "")),
                "playtime": float(state.user_train_playtime.get(f"{user_idx_str}_{item_idx}", 0.0)),
            })
    return history
