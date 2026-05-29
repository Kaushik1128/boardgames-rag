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

# Dependency layer first so a source-only edit doesn't bust the cache.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Application source.
COPY src/ ./src/

# Pre-built artifacts: embedded Qdrant collection + BM25 pickle. Both are
# produced locally by `deploy/build_indexes.py` before this build runs.
COPY data/qdrant_local/ ./data/qdrant_local/
COPY data/bm25/ ./data/bm25/

# Deploy-flavored config: embedded Qdrant + Groq generation LLM. Overrides
# the dev-time config.yaml.
COPY deploy/config.yaml ./config.yaml

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
