"""Configuration: Pydantic Settings layered over config.yaml and .env.

Layering (lowest → highest priority):
    pydantic defaults  <  config.yaml  <  process env vars  <  .env file
CLI flags in the ingest module override at runtime by mutating the Settings
object before passing it down.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class QdrantConfig(BaseModel):
    url: str = "http://localhost:6333"
    collection: str = "boardgames"
    distance: Literal["Cosine", "Dot", "Euclid", "Manhattan"] = "Cosine"


class EmbeddingConfig(BaseModel):
    model_name: str = "BAAI/bge-small-en-v1.5"
    device: str = "cpu"
    batch_size: int = 32
    normalize: bool = True


class ChunkingConfig(BaseModel):
    target_tokens: int = 512
    overlap_tokens: int = 50
    tokenizer: str = "cl100k_base"


class IngestConfig(BaseModel):
    source_dir: Path = Path("./data/raw")


class Settings(BaseSettings):
    """Application settings. Construct via `load_settings()`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        case_sensitive=False,
    )

    qdrant: QdrantConfig = QdrantConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    ingest: IngestConfig = IngestConfig()

    # Free-tier API keys, populated from .env. Not used until week 4+.
    gemini_api_key: str | None = None
    groq_api_key: str | None = None


def load_settings(config_path: Path | str = "config.yaml") -> Settings:
    """Load Settings, merging values from a YAML file if present."""
    config_path = Path(config_path)
    yaml_data: dict = {}
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}
    return Settings(**yaml_data)
