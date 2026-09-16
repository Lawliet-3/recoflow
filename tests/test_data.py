import pandas as pd
import pytest

from recoflow.data import temporal_split


def test_temporal_split_has_ordered_non_overlapping_windows():
    frame = pd.DataFrame(
        {
            "t_dat": pd.date_range("2024-01-01", periods=21, freq="D"),
            "customer_id": ["u1"] * 21,
            "article_id": [str(index) for index in range(21)],
        }
    )
    split = temporal_split(frame, validation_days=7, test_days=7)
    assert len(split.train) == len(split.validation) == len(split.test) == 7
    assert split.train.t_dat.max() < split.validation.t_dat.min() < split.test.t_dat.min()


def test_temporal_split_rejects_short_history():
    frame = pd.DataFrame(
        {"t_dat": [pd.Timestamp("2024-01-01")], "customer_id": ["u"], "article_id": ["i"]}
    )
    with pytest.raises(ValueError, match="too short"):
        temporal_split(frame)

