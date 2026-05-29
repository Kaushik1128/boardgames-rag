"""Tests for the FastAPI service.

Heavyweight collaborators (retriever, reranker, LLM, compiled graph) are
bypassed by injecting a fake graph via ``create_app(graph=...)``. The
lifespan handler sees the pre-set ``app.state.graph`` and skips the build
phase, so these tests run with no Qdrant, no Ollama, no model downloads
and no network.
"""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from boardgames_rag.retrieve import RetrievedChunk
from boardgames_rag.service import create_app

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _chunk(source: str = "catan.pdf", heading: str = "Trading") -> RetrievedChunk:
    return RetrievedChunk(
        point_id="p1",
        score=0.95,
        payload={"source_file": source, "heading": heading, "text": "trade text"},
    )


class FakeGraph:
    """Stand-in for the compiled LangGraph agent graph.

    ``run_agent`` invokes the graph and reads the same keys the real graph
    emits — answer, chunks, used_web, attempts, trace.
    """

    def __init__(self, *, answer: str = "A grounded answer [1].", chunks: list | None = None):
        self.answer = answer
        self.chunks = chunks if chunks is not None else [_chunk()]
        self.invocations: list[dict[str, Any]] = []

    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        self.invocations.append(state)
        return {
            "question": state["question"],
            "answer": self.answer,
            "chunks": self.chunks,
            "used_web": False,
            "attempts": 1,
            "trace": ["plan: q", "retrieve: 1 chunks", "critique: sufficient", "generate: ok"],
        }


class FailingGraph:
    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("LLM endpoint unreachable")


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health_returns_ok_and_llm_id():
    with TestClient(create_app(graph=FakeGraph(), llm_id="ollama:llama3.2:3b")) as client:
        response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body == {"status": "ok", "pipeline_llm": "ollama:llama3.2:3b"}


def test_health_does_not_invoke_graph():
    """Liveness must not touch the LLM/retriever — keeps HF Spaces health
    checks fast and free of flapping when Ollama is warming up."""
    graph = FakeGraph()
    with TestClient(create_app(graph=graph, llm_id="x")) as client:
        client.get("/health")
    assert graph.invocations == []


# ---------------------------------------------------------------------------
# /ask — happy path
# ---------------------------------------------------------------------------


def test_ask_returns_grounded_answer_with_sources():
    graph = FakeGraph(
        answer="Trade resources during your turn [1].",
        chunks=[_chunk(source="catan.pdf", heading="Trading")],
    )
    with TestClient(create_app(graph=graph)) as client:
        response = client.post("/ask", json={"question": "How do I trade in Catan?"})
    assert response.status_code == 200
    body = response.json()
    assert body["question"] == "How do I trade in Catan?"
    assert "Trade resources" in body["answer"]
    assert body["sources"][0]["source_file"] == "catan.pdf"
    assert body["sources"][0]["heading"] == "Trading"
    assert body["sources"][0]["score"] == 0.95
    assert body["attempts"] == 1
    assert any("generate" in line for line in body["trace"])


def test_ask_passes_question_to_graph():
    graph = FakeGraph()
    with TestClient(create_app(graph=graph)) as client:
        client.post("/ask", json={"question": "What is a wonder?"})
    assert graph.invocations[0]["question"] == "What is a wonder?"


# ---------------------------------------------------------------------------
# /ask — validation
# ---------------------------------------------------------------------------


def test_ask_rejects_missing_question():
    with TestClient(create_app(graph=FakeGraph())) as client:
        response = client.post("/ask", json={})
    assert response.status_code == 422


def test_ask_rejects_empty_question():
    with TestClient(create_app(graph=FakeGraph())) as client:
        response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422


def test_ask_rejects_overlong_question():
    with TestClient(create_app(graph=FakeGraph())) as client:
        response = client.post("/ask", json={"question": "q" * 2001})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# /ask — error propagation
# ---------------------------------------------------------------------------


def test_ask_surfaces_pipeline_failure_as_502():
    """RuntimeError from the agent (LLM down, etc.) must become a 502,
    distinguishable from a 500 service bug."""
    with TestClient(create_app(graph=FailingGraph())) as client:
        response = client.post("/ask", json={"question": "q"})
    assert response.status_code == 502
    assert "unreachable" in response.json()["detail"]


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------


def test_cors_headers_present_for_options():
    """Preflight from any origin must succeed — the HF Spaces iframe needs it."""
    with TestClient(create_app(graph=FakeGraph())) as client:
        response = client.options(
            "/ask",
            headers={
                "origin": "https://huggingface.co",
                "access-control-request-method": "POST",
            },
        )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "*"
