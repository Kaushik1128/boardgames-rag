"""FastAPI service exposing the agentic RAG pipeline.

Two endpoints:

  - ``POST /ask``  — run the agent for a question, return the grounded
    answer with cited sources and the agent trace.
  - ``GET  /health`` — liveness probe (no external calls).

The agent's heavyweight collaborators (hybrid retriever, cross-encoder
reranker, LLM client, compiled LangGraph) are built **once** in the
lifespan handler and reused for every request — the cross-encoder is the
slow one (~3-5 s cold load) and per-request rebuilds would make the
service unusable.

Run locally:

.. code-block:: bash

    uv run uvicorn boardgames_rag.service:app --reload

The default module-level ``app`` is what uvicorn imports; tests build
isolated apps via :func:`create_app` and inject a pre-built graph.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from boardgames_rag.agent import build_agent_graph, run_agent, run_agent_streaming
from boardgames_rag.config import Settings, load_settings
from boardgames_rag.generate import build_generator
from boardgames_rag.retrieve import Reranker, build_hybrid_retriever
from boardgames_rag.utils import setup_logging

logger = logging.getLogger(__name__)


__all__ = [
    "AskRequest",
    "AskResponse",
    "HealthResponse",
    "Source",
    "app",
    "create_app",
]


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class AskRequest(BaseModel):
    """Body of ``POST /ask``."""

    question: str = Field(..., min_length=1, max_length=2000)


class Source(BaseModel):
    """One cited chunk in an answer."""

    source_file: str
    heading: str
    score: float | None = None


class AskResponse(BaseModel):
    """Response of ``POST /ask``."""

    question: str
    answer: str
    sources: list[Source]
    attempts: int
    trace: list[str]


class HealthResponse(BaseModel):
    """Response of ``GET /health`` — liveness only, no external probes."""

    status: str
    pipeline_llm: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _llm_id(settings: Settings) -> str:
    """Human-readable identifier for the active pipeline LLM."""
    backend = settings.generation.backend
    model = settings.generation.groq_model if backend == "groq" else settings.generation.model
    return f"{backend}:{model}"


def _sources_from_chunks(chunks: list[Any]) -> list[Source]:
    """Map RetrievedChunks to the public Source schema."""
    return [
        Source(
            source_file=(c.payload.get("source_file") or "") if c.payload else "",
            heading=(
                (c.payload.get("heading") or c.payload.get("parent_heading") or "")
                if c.payload
                else ""
            ),
            score=c.score,
        )
        for c in chunks
    ]


# ---------------------------------------------------------------------------
# Lifespan — build heavyweight singletons at startup
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the retriever / reranker / LLM / agent graph once at startup.

    Tests bypass this build phase by setting ``app.state.graph`` before the
    lifespan runs (via :func:`create_app`'s ``graph`` kwarg).
    """
    if getattr(app.state, "graph", None) is not None:
        # Test path — a pre-built graph was injected, nothing to load.
        logger.info("Service starting with pre-injected graph (test mode).")
        yield
        return

    setup_logging()
    settings: Settings = app.state.settings

    logger.info("Building hybrid retriever (collection=%s)...", settings.qdrant.collection)
    hybrid = build_hybrid_retriever(settings, settings.qdrant.collection)
    if not getattr(app.state, "skip_consistency_check", False):
        hybrid.verify_consistency()

    reranker = None
    if settings.rerank.enabled:
        logger.info("Loading reranker %s...", settings.rerank.model_name)
        reranker = Reranker(
            model_name=settings.rerank.model_name,
            device=settings.rerank.device,
        )

    logger.info("Building generator (%s)...", _llm_id(settings))
    llm = build_generator(settings)

    logger.info("Compiling agent graph...")
    graph = build_agent_graph(
        llm=llm,
        retriever=hybrid,
        reranker=reranker,
        retrieve_k=settings.retrieval.k_per_retriever,
        rerank_top_k=settings.rerank.top_k,
        web_results=settings.agent.web_results,
    )
    # Stash the graph (used by /ask) and the raw collaborators (used by
    # /ask/stream, which bypasses the graph to yield events between nodes).
    app.state.graph = graph
    app.state.llm = llm
    app.state.retriever = hybrid
    app.state.reranker = reranker
    app.state.llm_id = _llm_id(settings)
    logger.info("Service ready (pipeline LLM: %s).", app.state.llm_id)

    try:
        yield
    finally:
        logger.info("Service shutting down.")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    settings: Settings | None = None,
    *,
    graph: Any = None,
    llm: Any = None,
    retriever: Any = None,
    reranker: Any = None,
    llm_id: str = "unknown",
    skip_consistency_check: bool = False,
) -> FastAPI:
    """Build a FastAPI app.

    Args:
        settings: Loaded ``Settings``. If ``None``, loads from ``config.yaml``.
        graph: Optional pre-built agent graph. Supplying it bypasses the
            heavyweight build inside the lifespan handler — the path tests
            take.
        llm, retriever, reranker: Optional pre-built collaborators used by
            the streaming ``/ask/stream`` endpoint. Only required for tests
            that exercise streaming; production builds populate them in the
            lifespan handler.
        llm_id: Identifier returned by ``/health`` when ``graph`` is injected.
            Ignored on the production build path (computed from settings).
        skip_consistency_check: If True, the BM25-vs-Qdrant consistency check
            is skipped at startup (saves time on warm dev restarts).
    """
    if settings is None:
        settings = load_settings()

    app = FastAPI(
        title="boardgames-rag",
        description="Agentic RAG over board game rulebooks.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.skip_consistency_check = skip_consistency_check
    # When tests inject a graph, the lifespan will see it and skip the build.
    app.state.graph = graph
    app.state.llm = llm
    app.state.retriever = retriever
    app.state.reranker = reranker
    app.state.llm_id = llm_id

    # Permissive CORS — the HF Spaces iframe demo needs this.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        """Liveness probe — no external calls, no Ollama/Qdrant ping."""
        return HealthResponse(
            status="ok",
            pipeline_llm=getattr(request.app.state, "llm_id", "unknown"),
        )

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest, request: Request) -> AskResponse:
        """Run the agent for a single rules question."""
        state = request.app.state
        try:
            result = run_agent(
                req.question,
                graph=state.graph,
                max_attempts=state.settings.agent.max_retrieval_attempts,
                web_fallback_enabled=state.settings.agent.web_fallback_enabled,
            )
        except RuntimeError as exc:
            # Upstream LLM / retriever unreachable — surface as 502 Bad Gateway
            # rather than a generic 500, so callers can distinguish a service
            # bug from a dependency outage.
            logger.exception("Pipeline failure handling /ask")
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return AskResponse(
            question=result.question,
            answer=result.answer,
            sources=_sources_from_chunks(result.sources),
            attempts=result.attempts,
            trace=result.trace,
        )

    @app.post("/ask/stream")
    def ask_stream(req: AskRequest, request: Request) -> StreamingResponse:
        """Run the agent and stream events as Server-Sent Events.

        Each event is a named SSE message: ``plan`` / ``retrieve`` /
        ``critique`` / ``web_fallback`` / ``token`` / ``sources`` / ``done``.
        The browser side reads them via ``EventSource`` to animate the
        agent trace and stream the final answer token by token.

        A pipeline RuntimeError is surfaced as a final ``error`` SSE event
        rather than a 502 — SSE has no clean way to fail a stream mid-flight
        with a non-200 status once headers are sent.
        """
        state = request.app.state
        events = run_agent_streaming(
            req.question,
            llm=state.llm,
            retriever=state.retriever,
            reranker=state.reranker,
            retrieve_k=state.settings.retrieval.k_per_retriever,
            rerank_top_k=state.settings.rerank.top_k,
            max_attempts=state.settings.agent.max_retrieval_attempts,
            web_fallback_enabled=state.settings.agent.web_fallback_enabled,
            web_results=state.settings.agent.web_results,
        )
        return StreamingResponse(
            _sse_from_events(events),
            media_type="text/event-stream",
            headers={
                "cache-control": "no-cache",
                # Disable buffering on proxies (HF Spaces) so tokens reach the
                # browser as they're produced, not in batches.
                "x-accel-buffering": "no",
            },
        )

    return app


def _sse_from_events(events: Iterator[tuple[str, Any]]) -> Iterator[str]:
    """Format ``(name, payload)`` tuples as SSE wire frames.

    SSE frame format: ``event: <name>\\ndata: <json>\\n\\n``. Errors raised
    by the underlying generator are converted into an ``error`` SSE event so
    the browser can render a clean failure state instead of a torn stream.
    """
    try:
        for name, payload in events:
            yield f"event: {name}\ndata: {json.dumps(payload)}\n\n"
    except RuntimeError as exc:
        logger.exception("Streaming pipeline failure")
        yield f"event: error\ndata: {json.dumps({'message': str(exc)})}\n\n"


# Default app instance — what `uvicorn boardgames_rag.service:app` imports.
app = create_app()
