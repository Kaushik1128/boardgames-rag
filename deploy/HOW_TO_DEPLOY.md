# Deploying to HuggingFace Spaces

The deploy is a Docker-SDK Space. The container image bakes in a
pre-built embedded Qdrant index + BM25 pickle, runs FastAPI on port
7860, and uses Groq for the agent's LLM (the free-tier `GROQ_API_KEY`
is set as a Space secret, not committed).

The full procedure runs locally once, then becomes a `git push`.

## Prerequisites

- A HuggingFace account.
- `huggingface_hub` CLI installed (`uv run huggingface-cli` works).
- A free Groq API key — already in your `.env` from week 5.

## One-time setup

### 1. Build the embedded Qdrant index

The Space's image expects a populated `data/qdrant_local/` directory.
Build it once from the same source PDFs the dev workflow uses:

```bash
uv run python -m boardgames_rag.ingest \
    --source-dir ./data/raw \
    --collection boardgames \
    --qdrant-path ./data/qdrant_local \
    --config config.yaml
```

This produces:

- `data/qdrant_local/` — the embedded Qdrant collection
- `data/bm25/boardgames.pkl` — the BM25 index (already built by your
  normal dev ingest; reuses the same file)

Re-run this whenever the corpus, chunking, or embedding model changes.

### 2. Create the Space

On <https://huggingface.co/new-space>:

- Owner: your account
- Space name: `boardgames-rag`
- License: MIT
- Space SDK: **Docker**
- Hardware: CPU basic (free)
- Visibility: Public

### 3. Set the Groq secret

In the new Space's *Settings → Variables and secrets*:

- Name: `GROQ_API_KEY`
- Value: paste your Groq key
- Mark as **Secret** (not Variable — secrets aren't visible at build time
  but are injected as env vars at runtime, which is what we want).

### 4. Configure git for HuggingFace

```bash
huggingface-cli login   # paste a write-access token from huggingface.co/settings/tokens
git remote add space https://huggingface.co/spaces/<your-user>/boardgames-rag
```

### 5. Pre-push: swap the README

HuggingFace reads the YAML frontmatter from the root `README.md`. We
don't want to clutter the GitHub README with frontmatter, so the
deploy-flavored README lives at `deploy/README.md` and gets copied to
root just before pushing. On a separate `space` branch:

```bash
git checkout -b space
cp deploy/README.md README.md
git add README.md
git commit -m "deploy: swap to Spaces README"
```

### 6. Push to Spaces

```bash
git push space space:main
```

The Space's first build takes ~5-10 min (downloading torch, sentence-
transformers, etc.). Subsequent pushes only rebuild the layers above
the dependency layer.

## Updating

After the one-time setup, deploying a new version is:

```bash
git checkout space
git merge main             # bring main's changes in
cp deploy/README.md README.md   # if README.md was touched on main
git add . && git commit -m "deploy: sync from main"
git push space space:main
```

Or merge main into space, resolve the README conflict (always keep
`deploy/README.md`'s version), push.

## Tradeoffs documented elsewhere

- **Groq daily quota:** the free-tier `llama-3.3-70b-versatile` has a
  100k token/day cap — roughly 16 questions/day before the Space goes
  dark with a 429 from Groq. We surface that as an SSE `error` event
  in the UI rather than a torn stream. Switch to `llama-3.1-8b-instant`
  in `deploy/config.yaml` for 5× the budget at the cost of model quality.

- **Embedded Qdrant size:** with the 10-game corpus, the
  `data/qdrant_local/` directory is roughly 15-25 MB. Comfortable inside
  the Space's image size budget.

- **Model downloads at startup:** the embedder (~130 MB) and reranker
  (~280 MB) download from the HuggingFace Hub on the Space's first
  request. HF's infra caches these in-datacenter so it's fast.
