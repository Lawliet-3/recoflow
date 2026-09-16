from fastapi.testclient import TestClient

from recoflow.api import Runtime, create_app
from recoflow.config import Settings


class FakeRuntime:
    popular = ["100", "200", "300"]
    user_embeddings = {}
    store = None


def test_unknown_user_receives_popularity_fallback():
    app = create_app(Settings(), runtime=FakeRuntime())
    with TestClient(app) as client:
        response = client.get("/v1/recommendations/new-user?limit=2")
    assert response.status_code == 200
    assert response.json() == {
        "user_id": "new-user",
        "source": "popularity",
        "items": [
            {"article_id": "100", "score": None},
            {"article_id": "200", "score": None},
        ],
    }

