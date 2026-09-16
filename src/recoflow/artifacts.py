from __future__ import annotations

from pathlib import Path

import numpy as np


def save_embeddings(path: str | Path, ids: list[str], vectors: np.ndarray) -> None:
    if len(ids) != len(vectors):
        raise ValueError("ids and vectors must have equal lengths")
    np.savez_compressed(
        path,
        ids=np.asarray(ids, dtype=str),
        vectors=np.asarray(vectors, dtype=np.float32),
    )


def load_embeddings(path: str | Path) -> tuple[list[str], np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        ids = archive["ids"].astype(str).tolist()
        vectors = archive["vectors"].astype(np.float32)
    if len(ids) != len(vectors):
        raise ValueError("embedding artifact has mismatched ids and vectors")
    return ids, vectors
