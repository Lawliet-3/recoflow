from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


class PopularityRecommender:
    """Deterministic global-popularity baseline and cold-start fallback."""

    def __init__(self) -> None:
        self._ranked_items: list[str] = []

    def fit(self, transactions: pd.DataFrame) -> PopularityRecommender:
        counts = transactions["article_id"].astype(str).value_counts()
        self._ranked_items = sorted(counts.index, key=lambda item: (-counts[item], item))
        return self

    def recommend(self, limit: int = 20, exclude: Iterable[str] = ()) -> list[str]:
        if limit < 1:
            raise ValueError("limit must be positive")
        excluded = {str(item) for item in exclude}
        return [item for item in self._ranked_items if item not in excluded][:limit]

    @property
    def ranked_items(self) -> list[str]:
        return self._ranked_items.copy()
