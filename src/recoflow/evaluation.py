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
    include_items: dict[str, set[str]] | None = None,
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
            excluded = seen.get(user_id, set())
            for item_id in excluded:
                position = item_positions.get(item_id)
                if position is not None:
                    scores[row, position] = -np.inf

            allowed = None if include_items is None else include_items.get(user_id, set())
            if allowed is not None:
                allowed_positions = {
                    item_positions[item_id]
                    for item_id in allowed
                    if item_id in item_positions and item_id not in excluded
                }
                if not allowed_positions:
                    recommendations[user_id] = []
                    continue
                blocked = np.ones(len(item_ids), dtype=bool)
                blocked[list(allowed_positions)] = False
                scores[row, blocked] = -np.inf
                user_k = min(effective_k, len(allowed_positions))
            else:
                available_count = len(item_ids) - len(excluded.intersection(item_positions))
                user_k = min(effective_k, available_count)
            if user_k == 0:
                recommendations[user_id] = []
                continue
            candidates = np.argpartition(scores[row], -user_k)[-user_k:]
            ordered = candidates[np.argsort(scores[row, candidates])[::-1]]
            recommendations[user_id] = [item_ids[index] for index in ordered]
    return recommendations


def popularity_recommendations(
    user_ids: Iterable[str],
    ranked_items: list[str],
    seen: dict[str, set[str]],
    k: int,
    include_items: dict[str, set[str]] | None = None,
) -> dict[str, list[str]]:
    return {
        user_id: [
            item
            for item in ranked_items
            if item not in seen.get(user_id, set())
            and (include_items is None or item in include_items.get(user_id, set()))
        ][:k]
        for user_id in user_ids
    }


def per_user_ranking_metrics(
    recommendations: dict[str, list[str]],
    ground_truth: dict[str, set[str]],
    k: int,
) -> dict[str, dict[str, float]]:
    """Return user-level metrics so segments and uncertainty remain auditable."""
    results: dict[str, dict[str, float]] = {}
    for user_id, relevant in ground_truth.items():
        if not relevant:
            continue
        ranked = recommendations.get(user_id, [])[:k]
        hits = np.asarray([item in relevant for item in ranked], dtype=np.float32)
        discounts = 1.0 / np.log2(np.arange(2, len(hits) + 2))
        dcg = float((hits * discounts).sum())
        ideal_length = min(len(relevant), k)
        ideal_discounts = 1.0 / np.log2(np.arange(2, ideal_length + 2))
        hit_positions = np.flatnonzero(hits)
        results[user_id] = {
            f"recall@{k}": float(hits.sum() / len(relevant)),
            f"ndcg@{k}": dcg / float(ideal_discounts.sum()) if ideal_length else 0.0,
            "mrr": 1.0 / (int(hit_positions[0]) + 1) if len(hit_positions) else 0.0,
        }
    return results


def metrics_with_confidence(
    recommendations: dict[str, list[str]],
    ground_truth: dict[str, set[str]],
    k: int,
    bootstrap_samples: int = 500,
    seed: int = 42,
) -> dict[str, object]:
    """Aggregate metrics with a deterministic 95% user-bootstrap interval."""
    per_user = per_user_ranking_metrics(recommendations, ground_truth, k)
    metric_names = (f"recall@{k}", f"ndcg@{k}", "mrr")
    result: dict[str, object] = {"evaluated_users": len(per_user)}
    if not per_user:
        result.update(
            {name: {"mean": 0.0, "ci95_lower": 0.0, "ci95_upper": 0.0} for name in metric_names}
        )
        return result

    values = np.asarray(
        [[user_metrics[name] for name in metric_names] for user_metrics in per_user.values()],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    if bootstrap_samples > 0:
        sampled = rng.integers(0, len(values), size=(bootstrap_samples, len(values)))
        bootstrap_means = values[sampled].mean(axis=1)
        lower = np.quantile(bootstrap_means, 0.025, axis=0)
        upper = np.quantile(bootstrap_means, 0.975, axis=0)
    else:
        lower = upper = values.mean(axis=0)
    means = values.mean(axis=0)
    for index, name in enumerate(metric_names):
        result[name] = {
            "mean": float(means[index]),
            "ci95_lower": float(lower[index]),
            "ci95_upper": float(upper[index]),
        }
    return result


def ranking_metrics(
    recommendations: dict[str, list[str]],
    ground_truth: dict[str, set[str]],
    k: int,
) -> dict[str, float | int]:
    per_user = per_user_ranking_metrics(recommendations, ground_truth, k)
    metric_names = (f"recall@{k}", f"ndcg@{k}", "mrr")
    return {
        **{
            name: float(np.mean([values[name] for values in per_user.values()]))
            if per_user
            else 0.0
            for name in metric_names
        },
        "evaluated_users": len(per_user),
    }
