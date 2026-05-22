"""Tests for the LangGraph agentic loop.

Every collaborator is faked — no LLM, no Qdrant, no network. The graph
itself is real (compiled LangGraph), driven by fakes.
"""

from __future__ import annotations

from typing import Any

import typer
from typer.testing import CliRunner

import boardgames_rag.agent as agent_module
import boardgames_rag.generate as generate_module
import boardgames_rag.retrieve as retrieve_module
from boardgames_rag.agent import (
    _parse_critique,
    build_agent_graph,
    critique_node,
    duckduckgo_search,
    generate_node,
    plan_node,
    retrieve_node,
    route_after_critique,
    run_agent,
    web_fallback_node,
)
from boardgames_rag.retrieve import RetrievedChunk

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _chunk(text: str, pid: str = "p") -> RetrievedChunk:
    return RetrievedChunk(
        point_id=pid,
        score=1.0,
        payload={"text": text, "source_file": "x.pdf", "heading": "H"},
    )


class FakeLLM:
    """Routes by system prompt: plan -> query, critique -> verdict, else answer."""

    def __init__(
        self,
        *,
        critiques: list[str] | None = None,
        answer: str = "the grounded answer",
        query: str = "reformulated query",
    ) -> None:
        self.critiques = list(critiques) if critiques else ["VERDICT: SUFFICIENT\nQUERY:"]
        self.answer = answer
        self.query = query
        self.calls: list[Any] = []
        self._ci = 0

    def generate(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        system = messages[0]["content"] if messages else ""
        # Critique must be checked before plan: the critique prompt also
        # contains the phrase "search query", so "judge whether" (unique to
        # the critic) has to be tested first.
        if "judge whether" in system:
            i = min(self._ci, len(self.critiques) - 1)
            self._ci += 1
            return self.critiques[i]
        if "search query" in system:
            return self.query
        return self.answer


class FakeRetriever:
    def __init__(self, chunks: list[RetrievedChunk] | None = None) -> None:
        self.chunks = (
            chunks if chunks is not None else [_chunk("ctx one", "a"), _chunk("ctx two", "b")]
        )

    def search(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        return list(self.chunks[:top_k])

    def verify_consistency(self) -> None:
        pass


class FakeReranker:
    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        return list(chunks[:top_k])


def _state(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "question": "How do I trade?",
        "query": "trade",
        "attempts": 0,
        "max_attempts": 2,
        "web_fallback_enabled": True,
        "chunks": [],
        "verdict": "",
        "used_web": False,
        "answer": "",
        "trace": [],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# _parse_critique
# ---------------------------------------------------------------------------


class TestParseCritique:
    def test_sufficient(self):
        verdict, query = _parse_critique("VERDICT: SUFFICIENT\nQUERY:")
        assert verdict == "sufficient"
        assert query == ""

    def test_insufficient_with_query(self):
        verdict, query = _parse_critique("VERDICT: INSUFFICIENT\nQUERY: catan robber rules")
        assert verdict == "insufficient"
        assert query == "catan robber rules"

    def test_insufficient_without_query_line(self):
        verdict, query = _parse_critique("INSUFFICIENT - need more")
        assert verdict == "insufficient"
        assert query == ""

    def test_unparseable_defaults_to_sufficient(self):
        verdict, _ = _parse_critique("hmm, not sure")
        assert verdict == "sufficient"

    def test_case_insensitive(self):
        verdict, _ = _parse_critique("verdict: insufficient")
        assert verdict == "insufficient"


# ---------------------------------------------------------------------------
# route_after_critique
# ---------------------------------------------------------------------------


class TestRouteAfterCritique:
    def test_sufficient_routes_to_generate(self):
        assert route_after_critique(_state(verdict="sufficient")) == "generate"

    def test_insufficient_with_attempts_left_loops(self):
        state = _state(verdict="insufficient", attempts=1, max_attempts=2)
        assert route_after_critique(state) == "retrieve"

    def test_insufficient_exhausted_routes_to_web_fallback(self):
        state = _state(
            verdict="insufficient", attempts=2, max_attempts=2, web_fallback_enabled=True
        )
        assert route_after_critique(state) == "web_fallback"

    def test_insufficient_exhausted_web_disabled_routes_to_generate(self):
        state = _state(
            verdict="insufficient", attempts=2, max_attempts=2, web_fallback_enabled=False
        )
        assert route_after_critique(state) == "generate"


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


class TestPlanNode:
    def test_sets_query_from_llm(self):
        update = plan_node(
            _state(question="How do I trade in Catan?"),
            llm=FakeLLM(query="catan trading"),
        )
        assert update["query"] == "catan trading"

    def test_falls_back_to_question_when_llm_empty(self):
        update = plan_node(_state(question="Q"), llm=FakeLLM(query="   "))
        assert update["query"] == "Q"


class TestRetrieveNode:
    def test_retrieves_reranks_and_counts_attempt(self):
        update = retrieve_node(
            _state(query="trade", attempts=0),
            retriever=FakeRetriever(),
            reranker=FakeReranker(),
            retrieve_k=10,
            rerank_top_k=1,
        )
        assert len(update["chunks"]) == 1
        assert update["attempts"] == 1

    def test_without_reranker_truncates_to_top_k(self):
        retriever = FakeRetriever([_chunk("a"), _chunk("b"), _chunk("c")])
        update = retrieve_node(
            _state(),
            retriever=retriever,
            reranker=None,
            retrieve_k=10,
            rerank_top_k=2,
        )
        assert len(update["chunks"]) == 2


class TestCritiqueNode:
    def test_sufficient_leaves_query_unchanged(self):
        update = critique_node(
            _state(chunks=[_chunk("x")]),
            llm=FakeLLM(critiques=["VERDICT: SUFFICIENT\nQUERY:"]),
        )
        assert update["verdict"] == "sufficient"
        assert "query" not in update

    def test_insufficient_sets_reformulated_query(self):
        update = critique_node(
            _state(chunks=[_chunk("x")]),
            llm=FakeLLM(critiques=["VERDICT: INSUFFICIENT\nQUERY: better query"]),
        )
        assert update["verdict"] == "insufficient"
        assert update["query"] == "better query"


class TestWebFallbackNode:
    def test_sets_chunks_and_used_web(self):
        update = web_fallback_node(_state(), search=lambda q, n: [_chunk("web result")], n=3)
        assert update["used_web"] is True
        assert len(update["chunks"]) == 1


class TestGenerateNode:
    def test_produces_answer(self):
        update = generate_node(_state(chunks=[_chunk("x")]), llm=FakeLLM(answer="grounded answer"))
        assert update["answer"] == "grounded answer"


# ---------------------------------------------------------------------------
# duckduckgo_search
# ---------------------------------------------------------------------------


class TestDuckDuckGoSearch:
    def test_returns_chunks(self, monkeypatch):
        class FakeDDGS:
            def text(self, query, max_results):
                rows = [{"title": "T", "href": "http://x", "body": "the body"}]
                return rows[:max_results]

        monkeypatch.setattr("ddgs.DDGS", FakeDDGS)
        chunks = duckduckgo_search("q", 5)
        assert len(chunks) == 1
        assert chunks[0].payload["text"] == "the body"
        assert chunks[0].payload["source_file"] == "http://x"

    def test_search_failure_returns_empty(self, monkeypatch):
        class FailingDDGS:
            def text(self, query, max_results):
                raise RuntimeError("network down")

        monkeypatch.setattr("ddgs.DDGS", FailingDDGS)
        assert duckduckgo_search("q", 5) == []


# ---------------------------------------------------------------------------
# Compiled graph — end to end
# ---------------------------------------------------------------------------


def _graph(llm: FakeLLM, web_search=None):
    return build_agent_graph(
        llm=llm,
        retriever=FakeRetriever(),
        reranker=FakeReranker(),
        retrieve_k=10,
        rerank_top_k=2,
        web_search=web_search or (lambda q, n: [_chunk("web", "w")]),
        web_results=3,
    )


class TestAgentGraph:
    def test_sufficient_path(self):
        llm = FakeLLM(critiques=["VERDICT: SUFFICIENT\nQUERY:"], answer="A")
        result = run_agent(
            "How do I trade?",
            graph=_graph(llm),
            max_attempts=2,
            web_fallback_enabled=True,
        )
        assert result.answer == "A"
        assert result.used_web is False
        assert result.attempts == 1

    def test_loop_then_sufficient(self):
        llm = FakeLLM(
            critiques=["VERDICT: INSUFFICIENT\nQUERY: retry q", "VERDICT: SUFFICIENT\nQUERY:"],
            answer="A",
        )
        result = run_agent("q", graph=_graph(llm), max_attempts=2, web_fallback_enabled=True)
        assert result.answer == "A"
        assert result.used_web is False
        assert result.attempts == 2  # looped once

    def test_exhausted_falls_back_to_web(self):
        llm = FakeLLM(critiques=["VERDICT: INSUFFICIENT\nQUERY: x"], answer="A")
        result = run_agent("q", graph=_graph(llm), max_attempts=2, web_fallback_enabled=True)
        assert result.used_web is True
        assert result.answer == "A"

    def test_exhausted_web_disabled_skips_web(self):
        llm = FakeLLM(critiques=["VERDICT: INSUFFICIENT\nQUERY: x"], answer="A")
        result = run_agent("q", graph=_graph(llm), max_attempts=2, web_fallback_enabled=False)
        assert result.used_web is False
        assert result.answer == "A"

    def test_trace_records_each_step(self):
        llm = FakeLLM(critiques=["VERDICT: SUFFICIENT\nQUERY:"], answer="A")
        result = run_agent("q", graph=_graph(llm), max_attempts=2, web_fallback_enabled=True)
        joined = " ".join(result.trace)
        assert "plan" in joined
        assert "retrieve" in joined
        assert "critique" in joined
        assert "generate" in joined


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_smoke(tmp_path, monkeypatch):
    monkeypatch.setattr(retrieve_module, "build_hybrid_retriever", lambda s, c: FakeRetriever())
    monkeypatch.setattr(retrieve_module, "Reranker", lambda model_name, device: FakeReranker())
    monkeypatch.setattr(
        generate_module, "OllamaGenerator", lambda **kwargs: FakeLLM(answer="CLI answer")
    )

    app = typer.Typer()
    app.command()(agent_module.main)
    result = CliRunner().invoke(
        app,
        [
            "--query",
            "How do I trade?",
            "--collection",
            "test",
            "--config",
            str(tmp_path / "nope.yaml"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "CLI answer" in result.output


def test_cli_handles_runtime_error(tmp_path, monkeypatch):
    class FailingLLM:
        def generate(self, messages):
            raise RuntimeError("Could not reach the LLM endpoint")

    monkeypatch.setattr(retrieve_module, "build_hybrid_retriever", lambda s, c: FakeRetriever())
    monkeypatch.setattr(retrieve_module, "Reranker", lambda model_name, device: FakeReranker())
    monkeypatch.setattr(generate_module, "OllamaGenerator", lambda **kwargs: FailingLLM())

    app = typer.Typer()
    app.command()(agent_module.main)
    result = CliRunner().invoke(
        app,
        ["--query", "q", "--collection", "test", "--config", str(tmp_path / "nope.yaml")],
    )
    assert result.exit_code == 1
    assert "Error" in result.output
