# boardgames-rag

Agentic Retrieval-Augmented Generation over board game rulebooks — ask a
natural-language rules question, get a grounded, cited answer.

**Runs fully local at $0.** The RAG pipeline uses local LLM inference
(Ollama) and local embeddings (sentence-transformers) — no API keys, no paid
services. (The *optional* evaluation harness can point its LLM judge at a
hosted model; the pipeline itself never does.)

The corpus is ten games spanning family weight to heavy strategy — Catan,
Ticket to Ride, Carcassonne, Azul, 7 Wonders, Wingspan, Scythe, Terraforming
Mars, Brass: Birmingham — plus the *Magic: The Gathering Comprehensive Rules*
as a deeply-numbered plain-text document.

---

## Vision — 7-week scope

| Week | Status | Deliverable |
|------|--------|-------------|
| 1 | Done | Project scaffold, Qdrant via docker-compose, ingestion pipeline (PDF + plain-text), chunker tests. |
| 2 | Done | Hybrid retrieval — BM25 + dense, fused with Reciprocal Rank Fusion. |
| 3 | Done | Local cross-encoder reranking (`bge-reranker-base`); grounded answer generation (Ollama). |
| 4 | Done | Ragas evaluation — faithfulness, answer relevancy, context precision & recall; rerank on/off ablation. |
| 5 | Planned | Agentic loop in LangGraph — planner / retriever / critic / DuckDuckGo fallback. |
| 6 | Planned | FastAPI service exposing query + eval endpoints. |
| 7 | Planned | Deploy on HuggingFace Spaces (free tier). |

## Status — weeks 1-4 complete

The pipeline runs end-to-end: **ingest → hybrid retrieve → rerank →
grounded generate**, with a **Ragas evaluation harness** on top.

- **Ingest** — parses PDFs (`pymupdf4llm`) and plain-text rules, chunks them
  with heading-aware ~512-token windows + 50-token overlap, embeds with
  `BAAI/bge-small-en-v1.5`, and stores them in Qdrant with stable
  content-hash IDs. A parallel BM25 index is built for lexical search.
- **Retrieve** — hybrid search: BM25 (lexical) + dense (semantic) candidates
  fused via Reciprocal Rank Fusion, with a consistency check between the two
  indexes.
- **Generate** — retrieve → cross-encoder rerank → strict-grounded answer
  from a local Ollama model, with citations. Refuses rather than hallucinate
  when the rulebooks don't cover the question.
- **Evaluate** — runs the pipeline over a curated 30-question test set and
  scores it with Ragas, comparing rerank-on vs rerank-off.

Current corpus: **10 rulebooks → 1,014 indexed chunks**. Test suite: **167
tests passing**.

Weeks 5-7 (agentic loop, FastAPI service, HuggingFace deploy) are next.

## Evaluation results

Week 4's headline experiment: does the cross-encoder reranker earn its cost?

Judge: `gemini-2.5-flash` · 30-question test set (3 per game) · 100% / 99%
score coverage.

| Metric | rerank-on | rerank-off | Δ (on − off) |
|--------|-----------|------------|--------------|
| Faithfulness | 0.94 | 0.85 | +0.10 |
| Answer relevancy | 0.76 | 0.68 | +0.08 |
| Context precision | 0.48 | 0.41 | +0.07 |
| Context recall | 0.63 | 0.54 | +0.09 |

**Reranking improves every metric — it stays in the pipeline.** Faithfulness
is strong (answers are well-grounded; little hallucination); context
precision and recall are the current weak point and the main target for
week 5's agentic retrieval.

The judge is configurable (`--judge-provider`). An earlier run with a weaker
local judge gave a noisier, partly-misleading verdict — a reminder that, in
LLM-as-judge evaluation, judge quality materially affects the conclusion.

## System requirements

| | |
|---|---|
| OS | Linux / macOS / Windows (developed on Windows 11) |
| RAM | 16 GB minimum |
| Disk | ~5 GB — embedding + reranker models, an Ollama LLM, Qdrant volume |
| GPU | **Not required.** Everything runs CPU-only. |
| Tools | [uv](https://docs.astral.sh/uv/), Docker Desktop, [Ollama](https://ollama.com/), Python 3.12 (uv installs it) |

## Quick start

```bash
# 1. Install deps + create the venv (uv reads .python-version + pyproject.toml)
uv sync

# 2. Start Qdrant
docker compose up -d

# 3. Pull a local LLM for answer generation
ollama pull llama3.2:3b

# 4. Drop your rulebook PDFs and the MTG .txt into ./data/raw

# 5. Ingest — builds the Qdrant collection + BM25 index
uv run python -m boardgames_rag.ingest --source-dir ./data/raw --collection boardgames
```

The first ingest downloads `BAAI/bge-small-en-v1.5` (~130 MB); the first
query downloads `bge-reranker-base` (~280 MB) into the local model cache.

## Usage

```bash
# Hybrid retrieval — inspect the ranked chunks for a query
uv run python -m boardgames_rag.retrieve \
    --query "How do I trade resources in Catan?" --collection boardgames

# Full RAG — retrieve → rerank → grounded, cited answer
uv run python -m boardgames_rag.generate \
    --query "How do I trade resources in Catan?" --collection boardgames

# Evaluate — Ragas metrics with a rerank on/off ablation
uv run python -m boardgames_rag.evaluate --collection boardgames
```

## Configuration

Defaults live in [`config.yaml`](config.yaml); CLI flags override them.
Secrets (`.env`, copied from `.env.example`) are **optional** — only needed
to point the evaluation judge at a hosted LLM (`--judge-provider gemini`).
The RAG pipeline needs no keys.

## Development

```bash
# Run tests with coverage
uv run pytest --cov

# Lint + format
uv run ruff check --fix .
uv run ruff format .

# (Optional) install pre-commit hooks
uv run pre-commit install
```

## License

MIT — see [LICENSE](LICENSE).
