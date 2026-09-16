import numpy as np

from recoflow.evaluation import (
    exact_recommendations,
    metrics_with_confidence,
    popularity_recommendations,
    ranking_metrics,
)


def test_exact_retrieval_excludes_seen_items_and_metrics_are_correct():
    recommendations = exact_recommendations(
        user_ids=["u1"],
        user_vectors=np.array([[1.0, 0.0]], dtype=np.float32),
        item_ids=["seen", "target", "other"],
        item_vectors=np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32),
        seen={"u1": {"seen"}},
        k=2,
    )
    assert recommendations["u1"][0] == "target"
    metrics = ranking_metrics(recommendations, {"u1": {"target"}}, k=2)
    assert metrics["recall@2"] == 1.0
    assert metrics["ndcg@2"] == 1.0
    assert metrics["mrr"] == 1.0


def test_retrieval_can_limit_candidates_for_repeat_purchase_evaluation():
    recommendations = exact_recommendations(
        user_ids=["u1"],
        user_vectors=np.array([[1.0, 0.0]], dtype=np.float32),
        item_ids=["repeat", "discovery", "other"],
        item_vectors=np.array([[0.8, 0.2], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        seen={},
        include_items={"u1": {"repeat"}},
        k=2,
    )
    assert recommendations == {"u1": ["repeat"]}
    assert popularity_recommendations(
        ["u1"],
        ["discovery", "repeat", "other"],
        {},
        2,
        include_items={"u1": {"repeat"}},
    ) == {"u1": ["repeat"]}


def test_confidence_metrics_are_reproducible_and_keep_empty_users_out():
    recommendations = {"u1": ["a"], "u2": ["x"], "empty": ["a"]}
    truth = {"u1": {"a"}, "u2": {"b"}, "empty": set()}
    first = metrics_with_confidence(recommendations, truth, 1, bootstrap_samples=100, seed=7)
    second = metrics_with_confidence(recommendations, truth, 1, bootstrap_samples=100, seed=7)
    assert first == second
    assert first["evaluated_users"] == 2
    assert first["recall@1"]["mean"] == 0.5
    assert first["recall@1"]["ci95_lower"] <= 0.5 <= first["recall@1"]["ci95_upper"]
