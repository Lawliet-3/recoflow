from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_TRANSACTION_COLUMNS = {"t_dat", "customer_id", "article_id"}
ITEM_FEATURE_COLUMNS = (
    "product_type_no",
    "colour_group_code",
    "department_no",
    "index_group_no",
)


@dataclass(frozen=True)
class TemporalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class TrainingExamples:
    user_indices: np.ndarray
    item_indices: np.ndarray
    history_indices: np.ndarray
    history_mask: np.ndarray
    latest_histories: np.ndarray
    latest_history_mask: np.ndarray


def load_transactions(path: str | Path, max_rows: int | None = None) -> pd.DataFrame:
    """Load H&M transactions and enforce the minimum V1 schema."""
    path = Path(path)
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
        if max_rows is not None:
            frame = frame.head(max_rows)
    else:
        frame = pd.read_csv(path, nrows=max_rows, dtype={"article_id": str})
    missing = REQUIRED_TRANSACTION_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"transactions are missing columns: {sorted(missing)}")

    frame = frame.copy()
    frame["customer_id"] = frame["customer_id"].astype(str)
    frame["article_id"] = frame["article_id"].astype(str).str.zfill(10)
    frame["t_dat"] = pd.to_datetime(frame["t_dat"], errors="raise")
    if frame[list(REQUIRED_TRANSACTION_COLUMNS)].isnull().any().any():
        raise ValueError("required transaction fields cannot be null")
    return frame.sort_values("t_dat").reset_index(drop=True)


def load_articles(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"article_id": str})
    if "article_id" not in frame:
        raise ValueError("articles are missing column: article_id")
    frame = frame.copy()
    frame["article_id"] = frame["article_id"].astype(str).str.zfill(10)
    for column in ITEM_FEATURE_COLUMNS:
        if column not in frame:
            frame[column] = "unknown"
    return frame


def temporal_split(
    transactions: pd.DataFrame,
    validation_days: int = 7,
    test_days: int = 7,
) -> TemporalSplit:
    """Split by event time, preventing future purchases from leaking into training."""
    if transactions.empty:
        raise ValueError("cannot split an empty transaction set")
    if validation_days < 1 or test_days < 1:
        raise ValueError("validation_days and test_days must be positive")

    dates = pd.to_datetime(transactions["t_dat"])
    end = dates.max().normalize() + pd.Timedelta(days=1)
    test_start = end - pd.Timedelta(days=test_days)
    validation_start = test_start - pd.Timedelta(days=validation_days)

    train = transactions.loc[dates < validation_start].copy()
    validation = transactions.loc[(dates >= validation_start) & (dates < test_start)].copy()
    test = transactions.loc[dates >= test_start].copy()
    if train.empty:
        raise ValueError("date range is too short to create a non-empty training split")
    return TemporalSplit(train=train, validation=validation, test=test)


def encode_item_metadata(
    articles: pd.DataFrame,
    item_ids: list[str],
) -> tuple[np.ndarray, list[int]]:
    """Create integer metadata codes aligned with the internal item indices."""
    indexed = articles.drop_duplicates("article_id").set_index("article_id")
    matrix = np.zeros((len(item_ids) + 1, len(ITEM_FEATURE_COLUMNS)), dtype=np.int64)
    cardinalities: list[int] = []
    for column_index, column in enumerate(ITEM_FEATURE_COLUMNS):
        values = indexed[column].fillna("unknown").astype(str)
        vocabulary = {value: index + 1 for index, value in enumerate(sorted(values.unique()))}
        cardinalities.append(len(vocabulary) + 1)
        for item_index, item_id in enumerate(item_ids, start=1):
            if item_id in indexed.index:
                matrix[item_index, column_index] = vocabulary.get(
                    str(indexed.at[item_id, column]), 0
                )
    return matrix, cardinalities


def build_training_examples(
    transactions: pd.DataFrame,
    user_to_index: dict[str, int],
    item_to_index: dict[str, int],
    max_history: int,
) -> TrainingExamples:
    """Build prefix histories: a target purchase never appears in its own input history."""
    histories: dict[int, list[int]] = {index: [] for index in user_to_index.values()}
    users: list[int] = []
    items: list[int] = []
    example_histories: list[list[int]] = []

    for row in transactions.itertuples(index=False):
        user_index = user_to_index.get(str(row.customer_id))
        item_index = item_to_index.get(str(row.article_id))
        if user_index is None or item_index is None:
            continue
        previous = histories[user_index][-max_history:]
        users.append(user_index)
        items.append(item_index)
        example_histories.append(previous.copy())
        histories[user_index].append(item_index)

    history_array, mask_array = _pad_histories(example_histories, max_history)
    latest = [histories[index][-max_history:] for index in range(len(user_to_index))]
    latest_array, latest_mask = _pad_histories(latest, max_history)
    return TrainingExamples(
        user_indices=np.asarray(users, dtype=np.int64),
        item_indices=np.asarray(items, dtype=np.int64),
        history_indices=history_array,
        history_mask=mask_array,
        latest_histories=latest_array,
        latest_history_mask=latest_mask,
    )


def _pad_histories(histories: list[list[int]], length: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros((len(histories), length), dtype=np.int64)
    mask = np.zeros((len(histories), length), dtype=bool)
    for row, history in enumerate(histories):
        trimmed = history[-length:]
        if trimmed:
            values[row, -len(trimmed):] = trimmed
            mask[row, -len(trimmed):] = True
    return values, mask
