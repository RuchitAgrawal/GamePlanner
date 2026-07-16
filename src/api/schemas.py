"""
Pydantic request and response schemas for the FastAPI layer.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field


class ItemResult(BaseModel):
    item_id: str
    title: str
    tags: str = ""
    score: float
    explanation: Optional[str] = None
    # F-E1: closest seed game (for cold-start semantic explanations)
    closest_seed: Optional[str] = None
    # F-E1: human-readable semantic note generated from closest_seed
    semantic_note: Optional[str] = None


class RecommendResponse(BaseModel):
    user_id: str
    recommendations: list[ItemResult]
    source: Literal["collaborative_filtering", "cold_start"]
    count: int


class ExplainResponse(BaseModel):
    user_id: str
    recommendations: list[ItemResult]   # explanation field populated on each item
    source: Literal["collaborative_filtering", "cold_start"]
    count: int


class ColdStartRequest(BaseModel):
    liked_games: list[str] = Field(
        ..., min_length=1, description="List of game titles the user has enjoyed"
    )
    k: int = Field(default=10, ge=1, le=50)


class ColdStartResponse(BaseModel):
    recommendations: list[ItemResult]
    count: int
    # F-E2: optional LLM summary paragraph for the full result set
    llm_summary: Optional[str] = None
    # F-E1: seed games actually matched in the catalog
    matched_seeds: list[str] = []


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language query for game recommendations")
    user_id: str | None = Field(default=None, description="Optional user ID for CF re-ranking")
    k: int = Field(default=10, ge=1, le=50)


class ChatResponse(BaseModel):
    query: str
    recommendations: list[ItemResult]
    summary: str
    count: int


class HealthResponse(BaseModel):
    status: str
    model: str
    num_users: int
    num_items: int
    cache_size: int


class MetricsResponse(BaseModel):
    model: str
    metrics: dict[str, float]
    k_values: list[int]
