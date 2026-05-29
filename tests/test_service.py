"""Tests for the FastAPI service.

Heavyweight collaborators (retriever, reranker, LLM, compiled graph) are
bypassed by injecting fakes via ``create_app(graph=..., llm=..., ...)``.
The lifespan handler sees the pre-set ``app.state.graph`` and skips the
build phase, so these tests run with no Qdrant, no Ollama, no model
downloads and no network.
"""

from __future__ import annotations

import json
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


class FakeLLM:
    """Routes generate() by system prompt; generate_stream() yields chunks."""

    def __init__(
        self,
        *,
        answer: str = "the final answer",
        critiques: list[str] | None = None,
    ) -> None:
        self.answer = answer
        self.critiques = critiques or ["VERDICT: SUFFICIENT\nQUERY:"]
        self._ci = 0

    def generate(self, messages: list[dict[str, str]]) -> str:
        system = messages[0]["content"] if messages else ""
        if "judge whether" in system:
            i = min(self._ci, len(self.critiques) - 1)
            self._ci += 1
            return self.critiques[i]
        if "search query" in system:
            return "planned query"
        return self.answer

    def generate_stream(self, messages: list[dict[str, str]]):
        yield self.answer[: len(self.answer) // 2]
        yield self.answer[len(self.answer) // 2 :]


class FakeRetriever:
    def search(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        return [_chunk()]

    def verify_consistency(self) -> None:
        pass


class FakeReranker:
    def rerank(self, query: str, chunks: list, top_k: int) -> list:
        return list(chunks[:top_k])


def _parse_sse(body: str) -> list[tuple[str, dict[str, Any]]]:
    """Decode an SSE stream body into ``(event_name, payload)`` tuples."""
    events: list[tuple[str, dict[str, Any]]] = []
    for raw in body.strip().split("\n\n"):
        name = ""
        data = ""
        for line in raw.splitlines():
            if line.startswith("event: "):
                name = line[len("event: ") :]
            elif line.startswith("data: "):
                data = line[len("data: ") :]
        if name:
            events.append((name, json.loads(data) if data else {}))
    return events


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


# ---------------------------------------------------------------------------
# /ask/stream — Server-Sent Events
# ---------------------------------------------------------------------------


def _streaming_app(llm: FakeLLM | None = None):
    """Build a streaming-capable test app with all collaborators injected."""
    return create_app(
        graph=FakeGraph(),  # /ask uses this; /ask/stream bypasses it
        llm=llm or FakeLLM(answer="hello world"),
        retriever=FakeRetriever(),
        reranker=FakeReranker(),
    )


def test_ask_stream_returns_event_stream_content_type():
    with TestClient(_streaming_app()) as client:
        response = client.post("/ask/stream", json={"question": "How do I trade?"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")


def test_ask_stream_event_sequence_for_sufficient_path():
    with TestClient(_streaming_app(FakeLLM(answer="alpha beta"))) as client:
        response = client.post("/ask/stream", json={"question": "How do I trade?"})
    events = _parse_sse(response.text)
    names = [e[0] for e in events]
    assert names[0] == "plan"
    assert "retrieve" in names
    assert "critique" in names
    assert "token" in names
    assert names[-2] == "sources"
    assert names[-1] == "done"


def test_ask_stream_reassembles_tokens_into_full_answer():
    with TestClient(_streaming_app(FakeLLM(answer="streamed answer"))) as client:
        response = client.post("/ask/stream", json={"question": "q"})
    events = _parse_sse(response.text)
    text = "".join(p["text"] for name, p in events if name == "token")
    assert text == "streamed answer"


def test_ask_stream_done_payload_carries_attempts_and_used_web():
    with TestClient(_streaming_app()) as client:
        response = client.post("/ask/stream", json={"question": "q"})
    events = _parse_sse(response.text)
    done_payload = next(p for name, p in events if name == "done")
    assert done_payload == {"attempts": 1, "used_web": False}


def test_ask_stream_emits_error_event_on_runtime_error():
    """SSE has no clean way to fail with a non-200 mid-stream; a final
    ``error`` event is the contract instead."""

    class BoomLLM(FakeLLM):
        def generate(self, messages):
            raise RuntimeError("model unreachable")

    with TestClient(_streaming_app(BoomLLM())) as client:
        response = client.post("/ask/stream", json={"question": "q"})
    events = _parse_sse(response.text)
    assert events[-1][0] == "error"
    assert "unreachable" in events[-1][1]["message"]


def test_ask_stream_rejects_empty_question():
    with TestClient(_streaming_app()) as client:
        response = client.post("/ask/stream", json={"question": ""})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Static frontend — index.html, app.js, data/*.json
# ---------------------------------------------------------------------------


def test_root_serves_index_html():
    with TestClient(create_app(graph=FakeGraph())) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "boardgames-rag" in response.text


def test_app_js_is_served():
    with TestClient(create_app(graph=FakeGraph())) as client:
        response = client.get("/app.js")
    assert response.status_code == 200
    # No application/javascript mime expected — just confirm we got JS.
    assert "submitQuestion" in response.text or "fetch" in response.text


def test_index_html_has_compare_toggle():
    """Phase 5 contract — the compare toggle must be present in the markup
    and the JS must fire both pipelines when it's checked."""
    with TestClient(create_app(graph=FakeGraph())) as client:
        html = client.get("/").text
        js = client.get("/app.js").text
    assert 'id="compare-toggle"' in html
    # JS reads the checkbox state and conditionally invokes the linear arm.
    assert "compare-toggle" in js
    assert "runLinear" in js
    assert "runAgent" in js


def test_games_json_is_served_and_has_ten_games():
    with TestClient(create_app(graph=FakeGraph())) as client:
        response = client.get("/data/games.json")
    assert response.status_code == 200
    games = response.json()
    assert isinstance(games, list)
    assert len(games) == 10
    assert {g["slug"] for g in games} >= {"catan", "mtg", "wingspan"}


def test_sample_questions_json_has_three_per_game():
    with TestClient(create_app(graph=FakeGraph())) as client:
        response = client.get("/data/sample_questions.json")
    assert response.status_code == 200
    questions = response.json()
    for slug, qs in questions.items():
        assert len(qs) == 3, f"{slug} has {len(qs)} questions, expected 3"


def test_static_mount_does_not_shadow_api_routes():
    """Regression guard: mounting StaticFiles at '/' must keep /ask, /health
    and /docs reachable. They were registered before the mount, so FastAPI
    resolves them first."""
    with TestClient(create_app(graph=FakeGraph(), llm_id="ollama:x")) as client:
        health = client.get("/health")
        docs = client.get("/docs")
    assert health.status_code == 200
    assert health.json()["pipeline_llm"] == "ollama:x"
    assert docs.status_code == 200
    assert "swagger" in docs.text.lower()


# ---------------------------------------------------------------------------
# /eval.html — measured-eval dashboard
# ---------------------------------------------------------------------------


def test_eval_html_is_served():
    with TestClient(create_app(graph=FakeGraph())) as client:
        response = client.get("/eval.html")
    assert response.status_code == 200
    assert "Evaluation" in response.text
    # Pulls in Chart.js + the dashboard script.
    assert "chart.js" in response.text.lower()
    assert "/eval.js" in response.text


def test_eval_js_is_served():
    with TestClient(create_app(graph=FakeGraph())) as client:
        response = client.get("/eval.js")
    assert response.status_code == 200
    assert "Chart" in response.text


def test_eval_summary_json_has_three_experiments():
    """The dashboard's data source — must keep telling the three-act story:
    rerank ablation, first agent regression, agent recovery after the fix."""
    with TestClient(create_app(graph=FakeGraph())) as client:
        response = client.get("/data/eval_summary.json")
    assert response.status_code == 200
    summary = response.json()
    assert summary["judge"] == "gemini-2.5-flash"
    assert summary["n_samples"] == 30
    assert len(summary["experiments"]) == 3
    ids = {e["id"] for e in summary["experiments"]}
    assert ids == {"rerank-ablation", "agent-first-attempt", "agent-fixed"}


def test_eval_summary_arms_have_scores_for_every_metric():
    """Catches data corruption: every arm must have a score for every metric
    declared at the top level — otherwise Chart.js silently renders nothing."""
    with TestClient(create_app(graph=FakeGraph())) as client:
        response = client.get("/data/eval_summary.json")
    summary = response.json()
    metric_keys = {m["key"] for m in summary["metrics"]}
    for exp in summary["experiments"]:
        for arm in exp["arms"]:
            assert set(arm["scores"].keys()) == metric_keys, (
                f"{exp['id']} / {arm['label']} score keys mismatch"
            )
            for v in arm["scores"].values():
                assert 0.0 <= v <= 1.0
