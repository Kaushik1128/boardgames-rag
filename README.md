# boardgames-rag

Agentic Retrieval-Augmented Generation over board game rulebooks. **Fully
local, $0 budget.** All LLM inference is either local (Ollama) or free-tier
(Gemini AI Studio, Groq). All embeddings are local via sentence-transformers.

The corpus covers ten games spanning family weight to heavy strategy: Catan,
Ticket to Ride, Carcassonne, Azul, 7 Wonders, Wingspan, Scythe, Terraforming
Mars, Brass: Birmingham — plus the *Magic: The Gathering Comprehensive Rules*
as a deeply-numbered plain-text document.

---

## Vision — 7-week scope

| Week | Deliverable |
|------|-------------|
| 1 | Project scaffold, Qdrant via docker-compose, ingestion pipeline (PDF + plain-text), chunker tests. |
| 2 | Hybrid retrieval — BM25 + dense, fused. |
| 3 | Local cross-encoder reranking with `bge-reranker-base`; generator wiring (Ollama). |
| 4 | Ragas evaluation with Gemini 2.0 Flash as judge (free tier). |
| 5 | Agentic loop in LangGraph — planner / retriever / critic / DuckDuckGo fallback. |
| 6 | FastAPI service exposing query + eval endpoints. |
| 7 | Deploy on HuggingFace Spaces (free tier). |

## Status

**Week 1 — in progress.** What's currently usable:

- `uv`-managed project (Python 3.12), `src/` layout.
- Qdrant runs locally via `docker compose up -d`.
- Ingestion CLI parses PDFs (via `pymupdf4llm`) and plain-text rule files,
  chunks them with heading-aware ~512-token windows + 50-token overlap, and
  upserts to Qdrant with content-hash IDs.
- Chunker test suite, ~80% coverage of `ingest.py`.

The other modules (`retrieve`, `generate`, `evaluate`, `agent`, `api`) are
intentional stubs that fill in across weeks 2–7.

## System requirements

| | |
|---|---|
| OS | Linux / macOS / Windows (tested on Windows 11) |
| RAM | 16 GB minimum |
| Disk | ~2 GB for models + Qdrant volume |
| GPU | **Not required.** Embedding runs CPU-only. |
| Tools | [uv](https://docs.astral.sh/uv/), Docker Desktop, Python 3.12 (uv installs it) |

## Quick start

```bash
# 1. Install deps + create venv (uv reads .python-version and pyproject.toml)
uv sync

# 2. Start Qdrant
docker compose up -d

# 3. Drop your rulebook PDFs and the MTG .txt into ./data/raw

# 4. Ingest
uv run python -m boardgames_rag.ingest \
    --source-dir ./data/raw \
    --collection boardgames

# 5. (Optional) Install pre-commit hooks
uv run pre-commit install
```

First ingestion run downloads `BAAI/bge-small-en-v1.5` (~130 MB) from
Hugging Face into the local model cache.

## Configuration

Defaults live in [`config.yaml`](config.yaml). CLI flags override them.
Secrets (Gemini, Groq keys — needed from week 4 onward) live in `.env`;
copy `.env.example` to get started.

## Development

```bash
# Run tests with coverage
uv run pytest --cov

# Lint + format
uv run ruff check --fix .
uv run ruff format .
```

## License

MIT — see [LICENSE](LICENSE).
