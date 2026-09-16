import numpy as np
from fastapi.testclient import TestClient

from recoflow.api import Runtime, create_app
from recoflow.artifacts import save_embeddings
from recoflow.config import Settings


class FakeStore:
    def search(self, vector, limit):
        assert vector == [1.0, 0.0]
        return [{"article_id": "900", "score": 0.91}][:limit]


class FakeRuntime:
    def __init__(self):
        self.popular = ["100", "200", "300"]
        self.user_embeddings = {"known-user": np.array([1.0, 0.0], dtype=np.float32)}
        self.store = FakeStore()


def test_unknown_user_receives_popularity_fallback():
    app = create_app(Settings(), runtime=FakeRuntime())
    with TestClient(app) as client:
        response = client.get("/v1/recommendations/new-user?limit=2")
    assert response.status_code == 200
    assert response.json()["source"] == "popularity"
    assert [item["article_id"] for item in response.json()["items"]] == ["100", "200"]


def test_known_user_queries_vector_store():
    app = create_app(Settings(), runtime=FakeRuntime())
    with TestClient(app) as client:
        response = client.get("/v1/recommendations/known-user?limit=1")
    assert response.status_code == 200
    assert response.json()["source"] == "two_tower"
    assert response.json()["items"] == [{"article_id": "900", "score": 0.91}]


def test_exported_artifact_qdrant_and_api_work_together(tmp_path):
    user_path = tmp_path / "users.npz"
    popularity_path = tmp_path / "popularity.json"
    save_embeddings(user_path, ["known-user"], np.array([[1.0, 0.0]], dtype=np.float32))
    popularity_path.write_text('["fallback"]')
    settings = Settings(
        qdrant_url=":memory:",
        qdrant_collection="products",
        embedding_dim=2,
        user_embeddings_path=str(user_path),
        popularity_path=str(popularity_path),
    )
    runtime = Runtime(settings)
    runtime.store.ensure_collection()
    runtime.store.upsert(["nearest", "farther"], [[1.0, 0.0], [0.0, 1.0]])

    app = create_app(settings, runtime=runtime)
    with TestClient(app) as client:
        response = client.get("/v1/recommendations/known-user?limit=1")
    assert response.status_code == 200
    assert response.json()["source"] == "two_tower"
    assert response.json()["items"][0]["article_id"] == "nearest"
