"""
Ranking metrics for evaluating recommendation quality.

All functions take lists/arrays of recommended item indices and
relevant (ground truth) item indices and return a float in [0, 1].

Evaluation is done at one or more cutoffs K (e.g. K=5, K=10).
"""

import numpy as np


def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Fraction of the top-k recommended items that are relevant."""
    if not recommended or k <= 0:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Fraction of all relevant items that appear in the top-k."""
    if not recommended or not relevant or k <= 0:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain at K.
    Rewards placing relevant items higher in the ranking.
    IDCG is computed assuming the best possible ordering.
    """
    if not recommended or not relevant or k <= 0:
        return 0.0

    top_k = recommended[:k]
    dcg = sum(
        1.0 / np.log2(rank + 2)
        for rank, item in enumerate(top_k)
        if item in relevant
    )
    # Ideal DCG: all relevant items at the top positions
    ideal_len = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(rank + 2) for rank in range(ideal_len))

    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """1 if at least one relevant item appears in the top-k, otherwise 0."""
    if not recommended or not relevant or k <= 0:
        return 0.0
    top_k = set(recommended[:k])
    return 1.0 if top_k & relevant else 0.0


def coverage(all_recommendations: list[list[int]], catalog_size: int) -> float:
    """
    Fraction of the total item catalog that ever appears in any recommendation list.
    Low coverage means the model is stuck recommending the same popular items.
    """
    if catalog_size <= 0:
        return 0.0
    recommended_items: set[int] = set()
    for recs in all_recommendations:
        recommended_items.update(recs)
    return len(recommended_items) / catalog_size


def diversity_at_k(recommended: list[int], item_vectors: np.ndarray, k: int) -> float:
    """
    Average pairwise cosine distance between items in the top-k list.
    Higher = more diverse recommendations (less repetitive in content).
    Requires pre-normalized item vectors (e.g. from sentence-transformers).

    Returns 0.0 if fewer than 2 items are recommended.
    """
    top_k = recommended[:k]
    if len(top_k) < 2:
        return 0.0

    valid = [i for i in top_k if i < len(item_vectors)]
    if len(valid) < 2:
        return 0.0

    vecs = item_vectors[valid]   # (n, dim), assumed L2-normalized
    sim_matrix = vecs @ vecs.T   # cosine similarity matrix
    n = len(valid)

    # Average pairwise distance (distance = 1 - similarity)
    total_dist = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_dist += 1.0 - float(sim_matrix[i, j])
            count += 1

    return total_dist / count if count > 0 else 0.0


def compute_all_metrics(
    recommended: list[int],
    relevant: set[int],
    k_values: list[int],
    item_vectors: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Compute all metrics for a single user at multiple K cutoffs.
    Returns a flat dict like {'precision@5': 0.2, 'ndcg@10': 0.35, ...}.
    """
    results = {}
    for k in k_values:
        results[f"precision@{k}"] = precision_at_k(recommended, relevant, k)
        results[f"recall@{k}"] = recall_at_k(recommended, relevant, k)
        results[f"ndcg@{k}"] = ndcg_at_k(recommended, relevant, k)
        results[f"hit_rate@{k}"] = hit_rate_at_k(recommended, relevant, k)
        if item_vectors is not None:
            results[f"diversity@{k}"] = diversity_at_k(recommended, item_vectors, k)
    return results


def aggregate_metrics(user_metrics: list[dict[str, float]]) -> dict[str, float]:
    """Average per-user metric dicts into a single summary dict."""
    if not user_metrics:
        return {}
    keys = user_metrics[0].keys()
    return {k: float(np.mean([m[k] for m in user_metrics])) for k in keys}
