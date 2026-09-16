from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from recoflow.artifacts import load_embeddings
from recoflow.config import Settings, get_settings
from recoflow.vector_store import ProductVectorStore

logger = logging.getLogger(__name__)


class Recommendation(BaseModel):
    article_id: str
    score: float | None = None


class RecommendationResponse(BaseModel):
    user_id: str
    model_version: str
    source: str
    items: list[Recommendation]


class Runtime:
    def __init__(self, settings: Settings) -> None:
        path = Path(settings.popularity_path)
        self.popular = json.loads(path.read_text()) if path.exists() else []
        self.user_embeddings: dict[str, np.ndarray] = {}
        embedding_path = Path(settings.user_embeddings_path)
        if embedding_path.exists():
            user_ids, vectors = load_embeddings(embedding_path)
            self.user_embeddings = dict(zip(user_ids, vectors, strict=True))
        self.store = ProductVectorStore(
            settings.qdrant_url, settings.qdrant_collection, settings.embedding_dim
        )


def create_app(settings: Settings | None = None, runtime: Runtime | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.runtime = runtime or Runtime(settings)
        yield

    app = FastAPI(title="RecoFlow", version=settings.model_version, lifespan=lifespan)

    @app.get("/health")
    def health(request: Request) -> dict[str, str | int]:
        state: Runtime = request.app.state.runtime
        return {
            "status": "ok",
            "model_version": settings.model_version,
            "personalized_users": len(state.user_embeddings),
            "fallback_items": len(state.popular),
        }

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
                return RecommendationResponse(
                    user_id=user_id,
                    model_version=settings.model_version,
                    source="two_tower",
                    items=matches,
                )
            except (ResponseHandlingException, UnexpectedResponse, OSError) as error:
                logger.warning("Qdrant unavailable; using popularity fallback: %s", error)
        items = [Recommendation(article_id=item) for item in state.popular[:limit]]
        if not items:
            raise HTTPException(status_code=503, detail="no recommendation artifacts are loaded")
        return RecommendationResponse(
            user_id=user_id,
            model_version=settings.model_version,
            source="popularity",
            items=items,
        )

    return app


app = create_app()
