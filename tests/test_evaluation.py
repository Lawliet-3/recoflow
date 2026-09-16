import numpy as np

from recoflow.evaluation import exact_recommendations, ranking_metrics


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
