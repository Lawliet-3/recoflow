import numpy as np

from recoflow.artifacts import load_embeddings, save_embeddings


def test_embedding_archive_round_trip(tmp_path):
    path = tmp_path / "embeddings.npz"
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    save_embeddings(path, ["a", "b"], vectors)
    ids, restored = load_embeddings(path)
    assert ids == ["a", "b"]
    np.testing.assert_array_equal(restored, vectors)
