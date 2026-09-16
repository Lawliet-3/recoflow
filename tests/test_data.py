import pandas as pd
import pytest

from recoflow.data import build_training_examples, encode_item_metadata, temporal_split


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


def test_training_histories_only_contain_previous_items():
    frame = pd.DataFrame(
        {
            "customer_id": ["u", "u", "u"],
            "article_id": ["a", "b", "c"],
        }
    )
    encoded = build_training_examples(frame, {"u": 0}, {"a": 1, "b": 2, "c": 3}, 2)
    assert encoded.history_indices.tolist() == [[0, 0], [0, 1], [1, 2]]
    assert encoded.latest_histories.tolist() == [[2, 3]]


def test_item_metadata_is_aligned_with_item_index():
    articles = pd.DataFrame(
        {
            "article_id": ["a", "b"],
            "product_type_no": [10, 20],
            "colour_group_code": [1, 2],
            "department_no": [5, 5],
            "index_group_no": [3, 3],
        }
    )
    matrix, cardinalities = encode_item_metadata(articles, ["b", "a"])
    assert matrix.shape == (3, 4)
    assert matrix[1].tolist() != matrix[2].tolist()
    assert all(value >= 2 for value in cardinalities)
