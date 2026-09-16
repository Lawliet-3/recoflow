from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RECOFLOW_", env_file=".env")

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "products"
    embedding_dim: int = 64
    default_limit: int = 20
    popularity_path: str = "artifacts/popularity.json"
    user_embeddings_path: str = "artifacts/user_embeddings.npz"


@lru_cache
def get_settings() -> Settings:
    return Settings()

