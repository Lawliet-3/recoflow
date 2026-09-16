from __future__ import annotations

import argparse

from recoflow.artifacts import load_embeddings
from recoflow.config import get_settings
from recoflow.vector_store import ProductVectorStore


def index_embeddings(
    artifact_path: str,
    qdrant_url: str,
    collection: str,
    batch_size: int = 512,
    recreate: bool = False,
) -> int:
    article_ids, vectors = load_embeddings(artifact_path)
    store = ProductVectorStore(qdrant_url, collection, vectors.shape[1])
    store.ensure_collection(recreate=recreate)
    for start in range(0, len(article_ids), batch_size):
        end = start + batch_size
        store.upsert(article_ids[start:end], vectors[start:end].tolist(), start_index=start)
    return len(article_ids)


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Index trained product vectors in Qdrant")
    parser.add_argument("--artifact", default="artifacts/item_embeddings.npz")
    parser.add_argument("--qdrant-url", default=settings.qdrant_url)
    parser.add_argument("--collection", default=settings.qdrant_collection)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args()
    count = index_embeddings(
        args.artifact,
        args.qdrant_url,
        args.collection,
        args.batch_size,
        args.recreate,
    )
    print(f"Indexed {count} product embeddings into {args.collection}")


if __name__ == "__main__":
    main()
