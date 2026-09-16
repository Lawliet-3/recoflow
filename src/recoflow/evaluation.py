from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def ground_truth_by_user(
    transactions: pd.DataFrame,
    allowed_users: set[str],
    allowed_items: set[str],
) -> dict[str, set[str]]:
    filtered = transactions[
        transactions["customer_id"].astype(str).isin(allowed_users)
        & transactions["article_id"].astype(str).isin(allowed_items)
    ]
    return {
        str(user): set(group["article_id"].astype(str))
        for user, group in filtered.groupby("customer_id")
    }


def seen_items_by_user(transactions: pd.DataFrame) -> dict[str, set[str]]:
    return {
        str(user): set(group["article_id"].astype(str))
        for user, group in transactions.groupby("customer_id")
    }


def exact_recommendations(
    user_ids: list[str],
    user_vectors: np.ndarray,
    item_ids: list[str],
    item_vectors: np.ndarray,
    seen: dict[str, set[str]],
    k: int,
    batch_size: int = 128,
) -> dict[str, list[str]]:
    """Exact dot-product retrieval used as the offline ANN quality reference."""
    if not item_ids:
        return {user_id: [] for user_id in user_ids}
    item_positions = {item_id: index for index, item_id in enumerate(item_ids)}
    effective_k = min(k, len(item_ids))
    recommendations: dict[str, list[str]] = {}
    for start in range(0, len(user_ids), batch_size):
        batch_ids = user_ids[start : start + batch_size]
        scores = user_vectors[start : start + batch_size] @ item_vectors.T
        for row, user_id in enumerate(batch_ids):
            for item_id in seen.get(user_id, set()):
                position = item_positions.get(item_id)
                if position is not None:
                    scores[row, position] = -np.inf
            candidates = np.argpartition(scores[row], -effective_k)[-effective_k:]
            ordered = candidates[np.argsort(scores[row, candidates])[::-1]]
            recommendations[user_id] = [item_ids[index] for index in ordered]
    return recommendations


def popularity_recommendations(
    user_ids: Iterable[str],
    ranked_items: list[str],
    seen: dict[str, set[str]],
    k: int,
) -> dict[str, list[str]]:
    return {
        user_id: [item for item in ranked_items if item not in seen.get(user_id, set())][:k]
        for user_id in user_ids
    }


def ranking_metrics(
    recommendations: dict[str, list[str]],
    ground_truth: dict[str, set[str]],
    k: int,
) -> dict[str, float | int]:
    recalls: list[float] = []
    ndcgs: list[float] = []
    reciprocal_ranks: list[float] = []
    for user_id, relevant in ground_truth.items():
        ranked = recommendations.get(user_id, [])[:k]
        hits = np.asarray([item in relevant for item in ranked], dtype=np.float32)
        recalls.append(float(hits.sum() / len(relevant)))
        discounts = 1.0 / np.log2(np.arange(2, len(hits) + 2))
        dcg = float((hits * discounts).sum())
        ideal_length = min(len(relevant), k)
        ideal_dcg = float(discounts[:ideal_length].sum())
        ndcgs.append(dcg / ideal_dcg if ideal_dcg else 0.0)
        hit_positions = np.flatnonzero(hits)
        reciprocal_ranks.append(
            1.0 / (int(hit_positions[0]) + 1) if len(hit_positions) else 0.0
        )
    return {
        f"recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        "evaluated_users": len(recalls),
    }
