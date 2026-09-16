from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel

from recoflow.config import Settings, get_settings
from recoflow.vector_store import ProductVectorStore


class Recommendation(BaseModel):
    article_id: str
    score: float | None = None


class RecommendationResponse(BaseModel):
    user_id: str
    source: str
    items: list[Recommendation]


class Runtime:
    def __init__(self, settings: Settings) -> None:
        path = Path(settings.popularity_path)
        self.popular = json.loads(path.read_text()) if path.exists() else []
        embedding_path = Path(settings.user_embeddings_path)
        self.user_embeddings: dict[str, np.ndarray] = {}
        if embedding_path.exists():
            archive = np.load(embedding_path)
            self.user_embeddings = {key: archive[key] for key in archive.files}
        self.store = ProductVectorStore(
            settings.qdrant_url, settings.qdrant_collection, settings.embedding_dim
        )


def create_app(settings: Settings | None = None, runtime: Runtime | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.runtime = runtime or Runtime(settings)
        yield

    app = FastAPI(title="RecoFlow", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/recommendations/{user_id}", response_model=RecommendationResponse)
    def recommend(
        request: Request,
        user_id: str,
        limit: int = Query(default=settings.default_limit, ge=1, le=100),
    ) -> RecommendationResponse:
        state: Runtime = request.app.state.runtime
        vector = state.user_embeddings.get(user_id)
        if vector is not None:
            try:
                matches = state.store.search(vector.tolist(), limit)
                return RecommendationResponse(user_id=user_id, source="two_tower", items=matches)
            except Exception:
                # Qdrant can be unavailable during startup; keep cold-start serving alive.
                pass
        items = [Recommendation(article_id=item) for item in state.popular[:limit]]
        if not items:
            raise HTTPException(status_code=503, detail="no recommendation artifacts are loaded")
        return RecommendationResponse(user_id=user_id, source="popularity", items=items)

    return app


app = create_app()

