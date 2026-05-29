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
| 5 | Done | Agentic loop in LangGraph — planner / retriever / critic, with reformulated-query retry. |
| 6 | Planned | FastAPI service exposing the agent query endpoint. |
| 7 | Planned | Deploy on HuggingFace Spaces (free tier) — agent LLM switches to Groq for hosted inference. |

## Status — weeks 1-5 complete

The pipeline runs end-to-end as both a **linear** RAG path and an
**agentic** path that share retrieval and generation code:

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
- **Agent** — a LangGraph state machine: plan → retrieve → critique →
  (sufficient ⇒ generate, otherwise loop with a reformulated query, bounded
  by a max-attempts cap). Same retrieval + generation primitives as the
  linear path, with an LLM critic gating the loop.
- **Evaluate** — runs either pipeline over a curated 30-question test set
  and scores with Ragas; ablations supported (rerank on/off, agent vs
  linear). The pipeline LLM is config-pluggable between local Ollama and
  Groq via `--llm-provider`.

Current corpus: **10 rulebooks → 1,014 indexed chunks**. Test suite: **202
tests passing**.

Weeks 6-7 (FastAPI service, HuggingFace deploy) are next.

## Evaluation results

Two measured experiments, both judged by `gemini-2.5-flash` against the same
30-question test set (3 per game), 100% score coverage in every reported run.

### Week 4 — does the cross-encoder reranker earn its cost?

| Metric | rerank-on | rerank-off | Δ (on − off) |
|--------|-----------|------------|--------------|
| Faithfulness | 0.94 | 0.85 | +0.10 |
| Answer relevancy | 0.76 | 0.68 | +0.08 |
| Context precision | 0.48 | 0.41 | +0.07 |
| Context recall | 0.63 | 0.54 | +0.09 |

**Reranking improves every metric — it stays in the pipeline.**

### Week 5 — does the agentic loop beat the linear pipeline?

| Metric | agent | linear | Δ (agent − linear) |
|--------|-------|--------|--------------------|
| Faithfulness | 0.946 | 0.944 | +0.003 |
| Answer relevancy | 0.745 | 0.746 | −0.001 |
| Context precision | 0.479 | 0.509 | −0.030 |
| Context recall | 0.592 | 0.644 | −0.053 |

**Agent matches linear on the output-quality metrics** (faithfulness and
answer relevancy are statistical ties) with a small residual gap on the
retrieval-side metrics. Both pipelines remain shipped and configurable; the
agent's critic + retry loop is most valuable as a defensive structure for
ambiguous queries — its retrieval residue should narrow further once the
agent's LLM is a larger hosted model at deploy time.

A first agent run (with a DuckDuckGo web fallback enabled) regressed on
faithfulness by −0.16; per-question diagnosis showed the small local critic
over-flagged good context as insufficient, the web fallback then fed thin
off-domain snippets to the generator, and grounding collapsed. Disabling the
fallback (now the default) and answering from rulebook context only restored
faithfulness and erased the regression — a useful reminder that adding an
escape hatch to a pipeline can do more harm than good when the escape hatch's
data is out-of-distribution.

The judge is configurable (`--judge-provider`). An earlier rerank-ablation
run with a weaker local judge gave a noisier, partly-misleading verdict — a
reminder that, in LLM-as-judge evaluation, judge quality materially affects
the conclusion.

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

# Linear RAG — retrieve → rerank → grounded, cited answer
uv run python -m boardgames_rag.generate \
    --query "How do I trade resources in Catan?" --collection boardgames

# Agentic RAG — plan → retrieve → critique → (retry or generate)
uv run python -m boardgames_rag.agent \
    --query "How do I trade resources in Catan?" --collection boardgames

# Evaluate — Ragas metrics, rerank on/off ablation by default
uv run python -m boardgames_rag.evaluate --collection boardgames

# Evaluate — agent vs linear comparison
uv run python -m boardgames_rag.evaluate --collection boardgames --pipeline agent
```

## Running the service

A FastAPI app wraps the agent behind a `POST /ask` endpoint. Qdrant and
Ollama need to be running (same prerequisites as the CLI).

```bash
uv run uvicorn boardgames_rag.service:app --reload
```

First request is slow (~5 s) as the cross-encoder warms up; subsequent ones
are warm. The easiest way to poke at the service interactively is the
auto-generated OpenAPI playground:

```
http://127.0.0.1:8000/docs
```

Or hit it from a shell. `curl.exe` works the same on every OS (on
PowerShell, the `.exe` matters — bare `curl` is an alias for
`Invoke-WebRequest`, which has a different flag syntax):

```bash
# Liveness check
curl.exe http://127.0.0.1:8000/health

# Ask a question
curl.exe -X POST http://127.0.0.1:8000/ask -H "content-type: application/json" -d "{\"question\":\"How do I trade resources in Catan?\"}"
```

PowerShell-native equivalent:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health

$body = @{ question = "How do I trade resources in Catan?" } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/ask `
    -Method Post -ContentType "application/json" -Body $body
```

## Configuration

Defaults live in [`config.yaml`](config.yaml); CLI flags override them.
Secrets (`.env`, copied from `.env.example`) are **optional** — only needed
to point the evaluation judge at a hosted LLM (`--judge-provider gemini`) or
to run the pipeline on Groq (`--llm-provider groq`, intended for the
HuggingFace deploy where local Ollama isn't available). The default,
$0-to-run path needs no keys.

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
