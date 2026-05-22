"""Configuration: Pydantic Settings layered over config.yaml and .env.

Layering (lowest → highest priority):
    pydantic defaults  <  config.yaml  <  process env vars  <  .env file
CLI flags in the ingest / retrieve modules override at runtime by mutating
the Settings object before passing it down.
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


class BM25Config(BaseModel):
    """rank-bm25 parameters and on-disk index location."""

    index_dir: Path = Path("./data/bm25")
    k1: float = 1.5
    b: float = 0.75


class FusionConfig(BaseModel):
    """How dense + BM25 rankings are combined into a single result list."""

    method: Literal["rrf", "weighted"] = "rrf"
    rrf_k: int = 60
    # Weight assigned to the dense retriever in weighted-score fusion; the
    # BM25 retriever gets (1 - weighted_alpha). Ignored when method == "rrf".
    weighted_alpha: float = 0.5


class RetrievalConfig(BaseModel):
    """Hybrid retrieval settings."""

    top_k: int = 10
    k_per_retriever: int = 20
    bm25: BM25Config = BM25Config()
    fusion: FusionConfig = FusionConfig()


class RerankConfig(BaseModel):
    """Cross-encoder reranking settings."""

    enabled: bool = True
    model_name: str = "BAAI/bge-reranker-base"
    device: str = "cpu"
    # Number of chunks kept after reranking the retrieved candidates.
    top_k: int = 5


class GenerationConfig(BaseModel):
    """LLM answer-generation settings.

    `base_url` points at any OpenAI-compatible endpoint. Ollama exposes one at
    /v1 locally; a hosted free-tier (Groq) can be swapped in later by changing
    `base_url` + `backend` with no code change.
    """

    backend: Literal["ollama"] = "ollama"
    base_url: str = "http://localhost:11434/v1"
    model: str = "llama3.2:3b"
    temperature: float = 0.1
    max_tokens: int = 1024
    timeout_seconds: float = 120.0
    # Number of reranked chunks actually placed in the prompt.
    max_context_chunks: int = 5


class EvalConfig(BaseModel):
    """Ragas evaluation settings (week 4)."""

    testset_path: Path = Path("eval/testset.jsonl")
    results_dir: Path = Path("eval/results")
    # Judge LLM provider for Ragas scoring:
    #   "ollama" — local model, no API quota (slow on CPU but always completes)
    #   "groq"   — hosted free tier (fast, but a daily token quota caps a run)
    #   "gemini" — hosted free tier (heavily rate-limited)
    judge_provider: Literal["ollama", "groq", "gemini"] = "ollama"
    judge_model: str = "llama3.1:latest"
    # Native Ollama API endpoint — used only when judge_provider is "ollama".
    judge_base_url: str = "http://localhost:11434"
    judge_temperature: float = 0.0
    # Ragas RunConfig.
    max_workers: int = 2
    timeout_seconds: int = 180
    max_retries: int = 5


class AgentConfig(BaseModel):
    """Agentic loop (LangGraph) settings — week 5."""

    # Retrieval attempts (initial + reformulated retries) before the agent
    # falls back to web search. 2 = initial try + one critic-driven retry.
    max_retrieval_attempts: int = 2
    # When true, exhausting the attempts triggers a DuckDuckGo web search;
    # when false the agent answers from whatever context it has.
    web_fallback_enabled: bool = True
    # Number of DuckDuckGo results to pull on a web fallback.
    web_results: int = 5


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
    retrieval: RetrievalConfig = RetrievalConfig()
    rerank: RerankConfig = RerankConfig()
    generation: GenerationConfig = GenerationConfig()
    eval: EvalConfig = EvalConfig()
    agent: AgentConfig = AgentConfig()

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
