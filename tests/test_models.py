"""
Unit tests for model architectures and evaluation metrics.
Run with: pytest tests/test_models.py -v
"""

import numpy as np
import pytest
import torch

from src.models.matrix_factorization import GMF
from src.models.neural_cf import NeuMF, MLPBranch
from src.evaluation.metrics import (
    precision_at_k, recall_at_k, ndcg_at_k,
    hit_rate_at_k, coverage, compute_all_metrics,
)

NUM_USERS = 50
NUM_ITEMS = 100
EMB_DIM = 16
BATCH = 8


# ── GMF tests ──────────────────────────────────────────────────────────────────

def test_gmf_forward_shape():
    model = GMF(NUM_USERS, NUM_ITEMS, emb_dim=EMB_DIM)
    users = torch.randint(0, NUM_USERS, (BATCH,))
    items = torch.randint(0, NUM_ITEMS, (BATCH,))
    out = model(users, items)
    assert out.shape == (BATCH,), f"Expected ({BATCH},), got {out.shape}"


def test_gmf_output_is_finite():
    model = GMF(NUM_USERS, NUM_ITEMS, emb_dim=EMB_DIM)
    users = torch.randint(0, NUM_USERS, (BATCH,))
    items = torch.randint(0, NUM_ITEMS, (BATCH,))
    out = model(users, items)
    assert torch.isfinite(out).all(), "GMF output contains NaN or Inf"


# ── NeuMF tests ────────────────────────────────────────────────────────────────

def test_neumf_forward_shape():
    model = NeuMF(NUM_USERS, NUM_ITEMS, emb_dim=EMB_DIM, layers=[32, 16])
    users = torch.randint(0, NUM_USERS, (BATCH,))
    items = torch.randint(0, NUM_ITEMS, (BATCH,))
    out = model(users, items)
    assert out.shape == (BATCH,)


def test_neumf_output_is_finite():
    model = NeuMF(NUM_USERS, NUM_ITEMS, emb_dim=EMB_DIM, layers=[32, 16])
    users = torch.randint(0, NUM_USERS, (BATCH,))
    items = torch.randint(0, NUM_ITEMS, (BATCH,))
    out = model(users, items)
    assert torch.isfinite(out).all()


def test_neumf_score_all_items():
    model = NeuMF(NUM_USERS, NUM_ITEMS, emb_dim=EMB_DIM, layers=[32, 16])
    model.eval()
    user = torch.tensor([0], dtype=torch.long)
    scores = model.score_all_items(user, NUM_ITEMS, torch.device("cpu"), batch_size=32)
    assert scores.shape == (NUM_ITEMS,)
    assert torch.isfinite(scores).all()


def test_mlp_branch_forward():
    model = MLPBranch(NUM_USERS, NUM_ITEMS, emb_dim=EMB_DIM, layers=[32, 16])
    users = torch.randint(0, NUM_USERS, (BATCH,))
    items = torch.randint(0, NUM_ITEMS, (BATCH,))
    out = model(users, items)
    assert out.shape == (BATCH,)


# ── Metric tests ───────────────────────────────────────────────────────────────

def test_precision_at_k_perfect():
    recs = [0, 1, 2, 3, 4]
    relevant = {0, 1, 2, 3, 4}
    assert precision_at_k(recs, relevant, k=5) == 1.0


def test_precision_at_k_none():
    recs = [5, 6, 7]
    relevant = {0, 1, 2}
    assert precision_at_k(recs, relevant, k=3) == 0.0


def test_recall_at_k_partial():
    recs = [0, 1, 5, 6, 7]
    relevant = {0, 1, 2, 3}
    assert recall_at_k(recs, relevant, k=5) == 0.5   # 2 out of 4 relevant found


def test_ndcg_perfect():
    recs = [0, 1, 2]
    relevant = {0, 1, 2}
    assert abs(ndcg_at_k(recs, relevant, k=3) - 1.0) < 1e-6


def test_ndcg_no_hits():
    recs = [3, 4, 5]
    relevant = {0, 1, 2}
    assert ndcg_at_k(recs, relevant, k=3) == 0.0


def test_hit_rate_hit():
    assert hit_rate_at_k([0, 1, 2], {2, 3}, k=3) == 1.0


def test_hit_rate_miss():
    assert hit_rate_at_k([0, 1, 2], {5, 6}, k=3) == 0.0


def test_coverage_full():
    all_recs = [[0, 1], [2, 3], [4]]
    assert coverage(all_recs, catalog_size=5) == 1.0


def test_coverage_partial():
    all_recs = [[0, 1], [0, 1]]
    assert coverage(all_recs, catalog_size=10) == 0.2


def test_compute_all_metrics_keys():
    recs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    relevant = {0, 3}
    result = compute_all_metrics(recs, relevant, k_values=[5, 10])
    expected_keys = [
        "precision@5", "precision@10",
        "recall@5", "recall@10",
        "ndcg@5", "ndcg@10",
        "hit_rate@5", "hit_rate@10",
    ]
    for key in expected_keys:
        assert key in result, f"Missing key: {key}"
        assert 0.0 <= result[key] <= 1.0, f"Out of range: {key} = {result[key]}"
