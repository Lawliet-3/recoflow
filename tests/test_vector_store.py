from recoflow.vector_store import ProductVectorStore


def test_qdrant_round_trip_in_memory():
    store = ProductVectorStore(":memory:", "products", dimension=2)
    store.ensure_collection()
    store.upsert(["a", "b"], [[1.0, 0.0], [0.0, 1.0]])
    result = store.search([0.9, 0.1], limit=1)
    assert result[0]["article_id"] == "a"
    assert result[0]["score"] > 0.9
