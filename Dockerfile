# boardgames-rag — HuggingFace Spaces deploy image.
#
# Single-container layout: FastAPI (uvicorn) + the embedded on-disk Qdrant
# baked into the image. The vector store lives at ./data/qdrant_local and
# is built locally before pushing — see deploy/HOW_TO_DEPLOY.md.
#
# Build / run locally:
#     docker build -t boardgames-rag .
#     docker run --rm -p 7860:7860 -e GROQ_API_KEY=... boardgames-rag

FROM python:3.12-slim

# uv is the project's only blessed dependency manager. Pull it from its
# official distroless image — small, fast, no apt installs needed.
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /bin/

WORKDIR /app

# Two-stage `uv sync` for layer caching:
#
#   1. Sync DEPENDENCIES only — `--no-install-project` tells uv to skip
#      building/installing the local package itself, which means hatchling
#      isn't invoked and README.md doesn't need to be present yet. This
#      layer only invalidates when pyproject.toml or uv.lock change, so
#      source-only edits stay fast on rebuild.
#
#   2. Once the project sources (and README.md) are copied in, run
#      `uv sync` again. This time hatchling builds the local package —
#      it reads README.md to satisfy the `readme = "README.md"` field
#      in pyproject.toml, then installs `boardgames_rag` into the venv
#      as an editable install. `uv run uvicorn boardgames_rag.service:app`
#      depends on that install.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Application source + README (the latter needed for hatchling's metadata
# validation on the next `uv sync`).
COPY src/ ./src/
COPY README.md ./

# Pre-built artifacts: embedded Qdrant collection + BM25 pickle. Both are
# produced locally by `uv run python -m boardgames_rag.ingest --qdrant-path
# ./data/qdrant_local ...` before this build runs.
COPY data/qdrant_local/ ./data/qdrant_local/
COPY data/bm25/ ./data/bm25/

# Deploy-flavored config: embedded Qdrant + Groq generation LLM. Overrides
# the dev-time config.yaml.
COPY deploy/config.yaml ./config.yaml

# Stage 2 of the uv sync — installs the local boardgames_rag package now
# that its source + readme are present.
RUN uv sync --frozen --no-dev

# HF Spaces convention: serve on 7860. The Space's frontmatter declares
# the same port so the iframe knows where to point.
EXPOSE 7860
ENV PORT=7860

# `uv run` resolves the venv created during `uv sync` and dispatches.
# --skip-consistency-check is a runtime escape hatch in case the BM25
# pickle and Qdrant collection drift; the lifespan still verifies on the
# default dev path.
CMD ["uv", "run", "uvicorn", "boardgames_rag.service:app", \
     "--host", "0.0.0.0", "--port", "7860"]
