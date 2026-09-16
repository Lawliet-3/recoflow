from __future__ import annotations

from collections.abc import Sequence

from qdrant_client import QdrantClient, models


class ProductVectorStore:
    def __init__(self, url: str, collection: str, dimension: int) -> None:
        self.client = (
            QdrantClient(location=":memory:") if url == ":memory:" else QdrantClient(url=url)
        )
        self.collection = collection
        self.dimension = dimension

    def ensure_collection(self, recreate: bool = False) -> None:
        collections = {item.name for item in self.client.get_collections().collections}
        if recreate and self.collection in collections:
            self.client.delete_collection(self.collection)
            collections.remove(self.collection)
        if self.collection not in collections:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=self.dimension, distance=models.Distance.COSINE
                ),
            )

    def upsert(
        self,
        article_ids: Sequence[str],
        vectors: Sequence[Sequence[float]],
        start_index: int = 0,
    ) -> None:
        if len(article_ids) != len(vectors):
            raise ValueError("article_ids and vectors must have equal lengths")
        points = [
            models.PointStruct(
                id=start_index + index,
                vector=list(vector),
                payload={"article_id": article_id},
            )
            for index, (article_id, vector) in enumerate(zip(article_ids, vectors, strict=True))
        ]
        self.client.upsert(collection_name=self.collection, points=points, wait=True)

    def search(self, vector: Sequence[float], limit: int) -> list[dict[str, object]]:
        response = self.client.query_points(
            collection_name=self.collection,
            query=list(vector),
            limit=limit,
            with_payload=True,
        )
        return [
            {"article_id": str(point.payload["article_id"]), "score": float(point.score)}
            for point in response.points
        ]
