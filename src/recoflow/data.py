from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REQUIRED_TRANSACTION_COLUMNS = {"t_dat", "customer_id", "article_id"}


@dataclass(frozen=True)
class TemporalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def load_transactions(path: str | Path) -> pd.DataFrame:
    """Load H&M transactions and enforce the minimum V1 schema."""
    path = Path(path)
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    missing = REQUIRED_TRANSACTION_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"transactions are missing columns: {sorted(missing)}")

    frame = frame.copy()
    frame["t_dat"] = pd.to_datetime(frame["t_dat"], errors="raise")
    if frame[list(REQUIRED_TRANSACTION_COLUMNS)].isnull().any().any():
        raise ValueError("required transaction fields cannot be null")
    return frame.sort_values("t_dat").reset_index(drop=True)


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

