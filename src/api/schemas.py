"""
Pydantic request and response schemas for the FastAPI layer.
"""

from typing import Literal
from pydantic import BaseModel, Field


class ItemResult(BaseModel):
    item_id: str
    title: str
    tags: str = ""
    score: float
    explanation: str | None = None


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
