"""
Unit tests for RAG components.

LLM calls are mocked so tests run without a live API key.
VectorStore and cache tests use synthetic data.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.rag.explainability import (
    build_explanation_prompt,
    groundedness_check,
    fallback_explanation,
)
from src.rag.vector_store import VectorStore
from src.utils.cache import ExplanationCache


# ── Explainability prompt tests (no LLM call) ──────────────────────────────────

def test_build_prompt_contains_title():
    prompt = build_explanation_prompt(
        recommended_title="Dark Souls",
        recommended_tags="action, rpg, difficult",
        recommended_description="A brutal action RPG.",
        similar_past_games=[("Elden Ring", 120.0)],
        review_snippets=["Punishing but fair."],
    )
    assert "Dark Souls" in prompt
    assert "Elden Ring" in prompt


def test_build_prompt_no_history():
    prompt = build_explanation_prompt(
        recommended_title="Celeste",
        recommended_tags="platformer, indie",
        recommended_description="A precision platformer.",
        similar_past_games=[],
        review_snippets=[],
    )
    assert "Celeste" in prompt
    assert len(prompt) > 20


def test_groundedness_check_passes():
    explanation = 'Because you liked "Dark Souls", you might enjoy this.'
    allowed = {"Dark Souls", "Elden Ring"}
    assert groundedness_check(explanation, allowed) is True


def test_groundedness_check_catches_hallucination():
    explanation = 'Because you loved "Made Up Game 9000", you will like this.'
    allowed = {"Dark Souls"}
    assert groundedness_check(explanation, allowed) is False


def test_fallback_explanation_with_history():
    result = fallback_explanation("Hollow Knight", "metroidvania, indie", [("Celeste", 30.0)])
    assert "Celeste" in result


def test_fallback_explanation_no_history():
    result = fallback_explanation("Hades", "roguelike, action", [])
    assert "roguelike" in result


# ── ExplanationCache tests ─────────────────────────────────────────────────────

def test_cache_set_and_get():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        cache_path = Path(f.name)
    try:
        cache = ExplanationCache(cache_path=cache_path)
        cache.set("user1", "item42", "Great game because of action elements.")
        result = cache.get("user1", "item42")
        assert result == "Great game because of action elements."
    finally:
        if cache_path.exists():
            cache_path.unlink()


def test_cache_miss_returns_none():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        cache_path = Path(f.name)
    try:
        cache = ExplanationCache(cache_path=cache_path)
        assert cache.get("nonexistent_user", "item0") is None
    finally:
        if cache_path.exists():
            cache_path.unlink()


def test_cache_persists_across_instances():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        cache_path = Path(f.name)
    try:
        cache1 = ExplanationCache(cache_path=cache_path)
        cache1.set("user1", "item1", "Persisted explanation")
        cache2 = ExplanationCache(cache_path=cache_path)
        assert cache2.get("user1", "item1") == "Persisted explanation"
    finally:
        if cache_path.exists():
            cache_path.unlink()


def test_cache_len():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        cache_path = Path(f.name)
    try:
        cache = ExplanationCache(cache_path=cache_path)
        assert len(cache) == 0
        cache.set("u1", "i1", "exp1")
        cache.set("u2", "i2", "exp2")
        assert len(cache) == 2
    finally:
        if cache_path.exists():
            cache_path.unlink()


# ── VectorStore tests (synthetic data) ────────────────────────────────────────

def _make_random_vectors(n: int, dim: int = 32) -> np.ndarray:
    vecs = np.random.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def test_vector_store_build_and_retrieve(tmp_path, monkeypatch):
    monkeypatch.setattr("src.rag.vector_store.FAISS_INDEX_PATH", tmp_path / "faiss.bin")
    monkeypatch.setattr("src.rag.vector_store.IDX_TO_ITEMID_PATH", tmp_path / "idx.json")

    n, dim = 20, 32
    vecs = _make_random_vectors(n, dim)
    idx_to_itemid = {i: f"game_{i}" for i in range(n)}

    vs = VectorStore()
    vs.build(vecs, idx_to_itemid)

    # Query with the first item's vector -- it should be the top result
    query = vecs[0]
    results = vs.retrieve_similar_items(query, k=3)
    assert len(results) == 3
    top_id, top_score = results[0]
    assert top_id == "game_0"
    assert top_score > 0.99   # should be near-perfect match


def test_vector_store_exclude(tmp_path, monkeypatch):
    monkeypatch.setattr("src.rag.vector_store.FAISS_INDEX_PATH", tmp_path / "faiss.bin")
    monkeypatch.setattr("src.rag.vector_store.IDX_TO_ITEMID_PATH", tmp_path / "idx.json")

    n, dim = 10, 32
    vecs = _make_random_vectors(n, dim)
    idx_to_itemid = {i: f"game_{i}" for i in range(n)}

    vs = VectorStore()
    vs.build(vecs, idx_to_itemid)

    results = vs.retrieve_similar_items(vecs[0], k=3, exclude_ids={"game_0"})
    returned_ids = [r[0] for r in results]
    assert "game_0" not in returned_ids
