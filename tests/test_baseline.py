import pandas as pd

from recoflow.baseline import PopularityRecommender


def test_popularity_is_ranked_and_can_exclude_seen_items():
    frame = pd.DataFrame({"article_id": ["b", "a", "b", "c", "a", "a"]})
    model = PopularityRecommender().fit(frame)
    assert model.recommend(limit=2) == ["a", "b"]
    assert model.recommend(limit=2, exclude=["a"]) == ["b", "c"]

