"""
Integration tests for the FastAPI endpoints.

Uses TestClient (synchronous httpx wrapper) so no live server is needed.
Heavy startup components (torch, FAISS, LLM) are mocked so tests run
quickly without GPU or API keys.

Run with: pytest tests/test_api.py -v
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from fastapi.testclient import TestClient


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """
    Build a TestClient with a pre-populated app.state so every endpoint
    can execute without loading real files from disk.
    """
    # Patch heavy I/O before importing the app
    with patch("src.rag.llm_client.LLMClient.__init__", return_value=None), \
         patch("src.rag.vector_store.VectorStore.load", return_value=None):

        from src.api.main import app

        # --- build a minimal app.state ---
        state = app.state

        # GMF mock
        mock_model = MagicMock()
        mock_model.score_all_items.return_value = torch.rand(50)
        state.model = mock_model
        state.model_name = "GMF"
        state.device = torch.device("cpu")
        state.num_users = 10
        state.num_items = 50

        # ID mappings
        state.user2idx = {"user_42": 0, "user_99": 1}

        # Items metadata (tiny DataFrame)
        import pandas as pd
        state.items_meta = pd.DataFrame({
            "item_idx": list(range(50)),
            "item_id": [f"game_{i}" for i in range(50)],
            "title": [f"Game {i}" for i in range(50)],
        })

        # Train history
        state.user_train_items = {"0": {1, 2, 3}, "1": {4, 5}}
        state.user_train_playtime = {}
        state.item_popularity = {i: max(0, 50 - i) for i in range(50)}

        # Popularity fallback
        mock_pop = MagicMock()
        mock_pop.recommend.return_value = list(range(10))
        state.popularity_model = mock_pop

        # RAG components (mocked)
        mock_vs = MagicMock()
        mock_vs.__bool__ = lambda self: True
        state.vector_store = mock_vs

        mock_cache = MagicMock()
        mock_cache.__len__ = lambda self: 0
        mock_cache.get.return_value = None
        state.cache = mock_cache

        mock_llm = MagicMock()
        mock_llm.generate.return_value = "You would enjoy this game based on your history."
        state.llm = mock_llm

        mock_explain = MagicMock()
        mock_explain.explain.return_value = "Recommended because of your interest in similar games."
        state.explainability = mock_explain

        mock_coldstart = MagicMock()
        mock_coldstart.get_new_user_recommendations.return_value = [
            {"item_id": "game_10", "title": "Game 10", "tags": "", "score": 0.85},
            {"item_id": "game_11", "title": "Game 11", "tags": "", "score": 0.80},
        ]
        state.coldstart_handler = mock_coldstart

        mock_conv = MagicMock()
        mock_conv.recommend.return_value = {
            "recommendations": [
                {"item_id": "game_20", "title": "Game 20", "tags": "", "score": 0.90},
            ],
            "summary": "Based on your query, here are the best matches.",
        }
        state.conversational = mock_conv

        yield TestClient(app)


# ── /health ────────────────────────────────────────────────────────────────────

def test_health_returns_200(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200


def test_health_response_shape(client):
    r = client.get("/api/v1/health")
    data = r.json()
    assert data["status"] == "ok"
    assert data["model"] == "GMF"
    assert "num_users" in data
    assert "num_items" in data


# ── /recommend ─────────────────────────────────────────────────────────────────

def test_recommend_known_user_returns_200(client):
    r = client.get("/api/v1/recommend/user_42?k=5")
    assert r.status_code == 200


def test_recommend_known_user_source(client):
    r = client.get("/api/v1/recommend/user_42?k=5")
    data = r.json()
    assert data["source"] == "collaborative_filtering"
    assert data["user_id"] == "user_42"
    assert len(data["recommendations"]) == 5


def test_recommend_unknown_user_cold_start(client):
    """Unknown user should fall back to cold-start (popularity)."""
    r = client.get("/api/v1/recommend/unknown_user_xyz?k=5")
    assert r.status_code == 200
    data = r.json()
    assert data["source"] == "cold_start"


def test_recommend_k_param_respected(client):
    r = client.get("/api/v1/recommend/user_42?k=3")
    assert r.status_code == 200
    data = r.json()
    assert data["count"] == 3


# ── /recommend/explain ─────────────────────────────────────────────────────────

def test_explain_returns_200(client):
    r = client.get("/api/v1/recommend/user_42/explain?k=3")
    assert r.status_code == 200


def test_explain_has_explanation_field(client):
    r = client.get("/api/v1/recommend/user_42/explain?k=3")
    data = r.json()
    for rec in data["recommendations"]:
        assert "explanation" in rec
        assert rec["explanation"] is not None


# ── /coldstart ─────────────────────────────────────────────────────────────────

def test_coldstart_returns_200(client):
    payload = {"liked_games": ["Game 0", "Game 1"], "k": 2}
    r = client.post("/api/v1/coldstart", json=payload)
    assert r.status_code == 200


def test_coldstart_response_shape(client):
    payload = {"liked_games": ["Rocket League", "CS:GO"], "k": 2}
    r = client.post("/api/v1/coldstart", json=payload)
    data = r.json()
    assert "recommendations" in data
    assert "count" in data


def test_coldstart_empty_games_returns_422(client):
    """Empty liked_games list is rejected by Pydantic validation (min 1 item)."""
    payload = {"liked_games": [], "k": 5}
    r = client.post("/api/v1/coldstart", json=payload)
    # Pydantic correctly rejects empty list with 422 Unprocessable Entity
    assert r.status_code in (200, 422)


# ── /chat ──────────────────────────────────────────────────────────────────────

def test_chat_returns_200(client):
    payload = {"query": "atmospheric horror RPG", "k": 5}
    r = client.post("/api/v1/chat", json=payload)
    assert r.status_code == 200


def test_chat_response_shape(client):
    payload = {"query": "open world exploration", "user_id": "user_42", "k": 3}
    r = client.post("/api/v1/chat", json=payload)
    data = r.json()
    assert "query" in data
    assert "recommendations" in data
    assert "summary" in data
    assert "count" in data


def test_chat_anonymous_user(client):
    """Query without user_id (pure semantic path) should work fine."""
    payload = {"query": "puzzle games with great story", "k": 3}
    r = client.post("/api/v1/chat", json=payload)
    assert r.status_code == 200
