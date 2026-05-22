"""Agentic RAG loop in LangGraph — planner / retriever / critic / web fallback.

The agent plans a search query, retrieves, and asks an LLM critic whether the
retrieved context is sufficient. If not, it reformulates and retries (up to a
cap); if local retrieval is exhausted it falls back to a DuckDuckGo web
search. Finally it generates a grounded answer.

Flow — plan -> retrieve -> critique, then the critic routes:

  - sufficient                 -> generate -> END
  - insufficient, retries left -> retrieve (loop back, reformulated query)
  - insufficient, exhausted    -> web_fallback -> generate -> END

CLI:

.. code-block:: bash

    python -m boardgames_rag.agent --query "How do I trade in Catan?" \
        --collection boardgames
"""

from __future__ import annotations

import logging
import operator
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, TypedDict

import typer
from rich.console import Console
from rich.panel import Panel

from boardgames_rag.config import load_settings
from boardgames_rag.generate import build_prompt
from boardgames_rag.retrieve import RetrievedChunk
from boardgames_rag.utils import setup_logging

if TYPE_CHECKING:  # pragma: no cover
    from boardgames_rag.generate import Generator, RerankerProtocol, RetrieverProtocol

logger = logging.getLogger(__name__)
console = Console()


__all__ = [
    "AgentResult",
    "AgentState",
    "build_agent_graph",
    "critique_node",
    "duckduckgo_search",
    "generate_node",
    "main",
    "plan_node",
    "retrieve_node",
    "route_after_critique",
    "run_agent",
    "web_fallback_node",
]


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    """Mutable state threaded through the LangGraph nodes."""

    question: str  # the user's original question
    query: str  # current search query (reformulated on retry)
    attempts: int  # retrieval attempts run so far
    max_attempts: int  # cap before the web fallback fires
    web_fallback_enabled: bool
    chunks: list[RetrievedChunk]  # current retrieved context
    verdict: str  # "sufficient" / "insufficient" / ""
    used_web: bool  # did the web fallback fire?
    answer: str  # final answer
    trace: Annotated[list[str], operator.add]  # human-readable step log


@dataclass
class AgentResult:
    """The outcome of one agent run."""

    question: str
    answer: str
    sources: list[RetrievedChunk]
    used_web: bool
    attempts: int
    trace: list[str]


# ---------------------------------------------------------------------------
# Critic-response parsing (pure)
# ---------------------------------------------------------------------------


def _parse_critique(text: str) -> tuple[str, str]:
    """Parse a critic LLM response into ``(verdict, suggested_query)``.

    Lenient: ``INSUFFICIENT`` anywhere wins; else ``SUFFICIENT``; else default
    to ``sufficient`` (proceed rather than loop on unparseable output). The
    suggested query is read from a ``QUERY:`` line, if present.
    """
    upper = text.upper()
    verdict = "insufficient" if "INSUFFICIENT" in upper else "sufficient"
    query = ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("QUERY:"):
            query = stripped[len("QUERY:") :].strip()
            break
    return verdict, query


# ---------------------------------------------------------------------------
# Web search
# ---------------------------------------------------------------------------


def duckduckgo_search(query: str, n: int) -> list[RetrievedChunk]:
    """Web search via DuckDuckGo. Returns results as RetrievedChunks.

    Best-effort: a search failure yields an empty list rather than raising,
    so the agent degrades gracefully to a "not found" answer.
    """
    from ddgs import DDGS  # lazy

    try:
        results = list(DDGS().text(query, max_results=n))
    except Exception as exc:
        logger.warning("DuckDuckGo search failed: %s", exc)
        return []
    chunks: list[RetrievedChunk] = []
    for i, result in enumerate(results):
        chunks.append(
            RetrievedChunk(
                point_id=f"web-{i}",
                score=0.0,
                payload={
                    "text": result.get("body", "") or "",
                    "source_file": result.get("href", "web"),
                    "heading": result.get("title", ""),
                },
                debug={"source": "duckduckgo"},
            )
        )
    return chunks


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

_PLAN_SYSTEM = (
    "You turn a board-game rules question into a concise search query for a "
    "rulebook index. Reply with ONLY the search query — no preamble, no quotes."
)

_CRITIQUE_SYSTEM = (
    "You judge whether retrieved context can fully and accurately answer a "
    "question. Reply in exactly two lines:\n"
    "VERDICT: SUFFICIENT or INSUFFICIENT\n"
    "QUERY: a better search query if INSUFFICIENT, otherwise leave blank"
)


def plan_node(state: AgentState, *, llm: Generator) -> dict[str, Any]:
    """Reformulate the question into an initial search query."""
    question = state["question"]
    messages = [
        {"role": "system", "content": _PLAN_SYSTEM},
        {"role": "user", "content": question},
    ]
    raw = llm.generate(messages).strip()
    query = raw.splitlines()[0].strip() if raw else question
    query = query or question
    return {"query": query, "trace": [f"plan: query = {query!r}"]}


def retrieve_node(
    state: AgentState,
    *,
    retriever: RetrieverProtocol,
    reranker: RerankerProtocol | None,
    retrieve_k: int,
    rerank_top_k: int,
) -> dict[str, Any]:
    """Hybrid retrieve + rerank for the current query."""
    query = state["query"]
    chunks = retriever.search(query, top_k=retrieve_k)
    if reranker is not None and chunks:
        chunks = reranker.rerank(query, chunks, top_k=rerank_top_k)
    else:
        chunks = chunks[:rerank_top_k]
    attempts = state["attempts"] + 1
    return {
        "chunks": chunks,
        "attempts": attempts,
        "trace": [f"retrieve (attempt {attempts}): {len(chunks)} chunks"],
    }


def critique_node(state: AgentState, *, llm: Generator) -> dict[str, Any]:
    """LLM critic: is the retrieved context sufficient to answer the question?"""
    chunks = state["chunks"]
    context = (
        "\n\n".join(f"[{i}] {c.payload.get('text', '')}" for i, c in enumerate(chunks, start=1))
        or "(no context retrieved)"
    )
    messages = [
        {"role": "system", "content": _CRITIQUE_SYSTEM},
        {
            "role": "user",
            "content": f"Question: {state['question']}\n\nContext:\n{context}",
        },
    ]
    verdict, suggested = _parse_critique(llm.generate(messages))
    note = f" -> {suggested!r}" if suggested else ""
    update: dict[str, Any] = {"verdict": verdict, "trace": [f"critique: {verdict}{note}"]}
    if verdict == "insufficient" and suggested:
        update["query"] = suggested
    return update


def web_fallback_node(
    state: AgentState,
    *,
    search: Callable[[str, int], list[RetrievedChunk]],
    n: int,
) -> dict[str, Any]:
    """Last-resort web search when local retrieval cannot answer."""
    chunks = search(state["question"], n)
    return {
        "chunks": chunks,
        "used_web": True,
        "trace": [f"web_fallback: {len(chunks)} web result(s)"],
    }


def generate_node(state: AgentState, *, llm: Generator) -> dict[str, Any]:
    """Generate the final grounded answer from the current context."""
    messages = build_prompt(state["question"], state["chunks"])
    return {"answer": llm.generate(messages), "trace": ["generate: answer produced"]}


def route_after_critique(state: AgentState) -> str:
    """Conditional edge after the critic: generate / retrieve (loop) / web_fallback."""
    if state["verdict"] == "sufficient":
        return "generate"
    if state["attempts"] < state["max_attempts"]:
        return "retrieve"
    if state["web_fallback_enabled"]:
        return "web_fallback"
    return "generate"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


def build_agent_graph(
    *,
    llm: Generator,
    retriever: RetrieverProtocol,
    reranker: RerankerProtocol | None,
    retrieve_k: int,
    rerank_top_k: int,
    web_search: Callable[[str, int], list[RetrievedChunk]] = duckduckgo_search,
    web_results: int = 5,
) -> Any:
    """Wire and compile the LangGraph agent. Returns the compiled graph."""
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(AgentState)
    graph.add_node("plan", partial(plan_node, llm=llm))
    graph.add_node(
        "retrieve",
        partial(
            retrieve_node,
            retriever=retriever,
            reranker=reranker,
            retrieve_k=retrieve_k,
            rerank_top_k=rerank_top_k,
        ),
    )
    graph.add_node("critique", partial(critique_node, llm=llm))
    graph.add_node("web_fallback", partial(web_fallback_node, search=web_search, n=web_results))
    graph.add_node("generate", partial(generate_node, llm=llm))

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "critique")
    graph.add_conditional_edges(
        "critique",
        route_after_critique,
        {
            "generate": "generate",
            "retrieve": "retrieve",
            "web_fallback": "web_fallback",
        },
    )
    graph.add_edge("web_fallback", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


def run_agent(
    question: str,
    *,
    graph: Any,
    max_attempts: int,
    web_fallback_enabled: bool,
) -> AgentResult:
    """Invoke a compiled agent graph for one question."""
    initial: AgentState = {
        "question": question,
        "query": question,
        "attempts": 0,
        "max_attempts": max_attempts,
        "web_fallback_enabled": web_fallback_enabled,
        "chunks": [],
        "verdict": "",
        "used_web": False,
        "answer": "",
        "trace": [],
    }
    final = graph.invoke(initial)
    return AgentResult(
        question=question,
        answer=final["answer"],
        sources=final["chunks"],
        used_web=final["used_web"],
        attempts=final["attempts"],
        trace=final["trace"],
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_result(result: AgentResult) -> None:
    console.print()
    console.print(Panel(result.answer, title="Answer", border_style="green"))
    if result.used_web:
        console.print(
            "[yellow]Note: answered from a web-search fallback — not the rulebooks.[/yellow]"
        )
    if result.sources:
        console.print("\n[bold]Sources[/bold]")
        for i, chunk in enumerate(result.sources, start=1):
            source = chunk.payload.get("source_file") or "unknown"
            heading = chunk.payload.get("heading") or "—"
            console.print(f"  [{i}] {source} — {heading}")
    console.print(f"\n[dim]agent trace ({result.attempts} retrieval attempt(s)):[/dim]")
    for line in result.trace:
        console.print(f"[dim]  - {line}[/dim]")


def main(
    query: Annotated[str, typer.Option("--query", "-q", help="Natural-language question.")],
    collection: Annotated[
        str | None, typer.Option("--collection", help="Qdrant collection name.")
    ] = None,
    config_path: Annotated[Path, typer.Option("--config", help="Path to YAML config.")] = Path(
        "config.yaml"
    ),
    skip_consistency_check: Annotated[
        bool,
        typer.Option("--skip-consistency-check", help="Don't verify BM25 vs Qdrant."),
    ] = False,
    log_level: Annotated[
        str, typer.Option("--log-level", help="DEBUG / INFO / WARNING / ERROR.")
    ] = "WARNING",
) -> None:
    """Answer a question with the agentic RAG loop."""
    from boardgames_rag.generate import OllamaGenerator
    from boardgames_rag.retrieve import Reranker, build_hybrid_retriever

    setup_logging(level=getattr(logging, log_level.upper(), logging.WARNING))
    settings = load_settings(config_path)
    if collection is None:
        collection = settings.qdrant.collection

    hybrid = build_hybrid_retriever(settings, collection)
    if not skip_consistency_check:
        hybrid.verify_consistency()
    reranker = Reranker(model_name=settings.rerank.model_name, device=settings.rerank.device)
    llm = OllamaGenerator(
        model=settings.generation.model,
        base_url=settings.generation.base_url,
        temperature=settings.generation.temperature,
        max_tokens=settings.generation.max_tokens,
        timeout_seconds=settings.generation.timeout_seconds,
    )
    graph = build_agent_graph(
        llm=llm,
        retriever=hybrid,
        reranker=reranker,
        retrieve_k=settings.retrieval.k_per_retriever,
        rerank_top_k=settings.rerank.top_k,
        web_results=settings.agent.web_results,
    )
    try:
        result = run_agent(
            query,
            graph=graph,
            max_attempts=settings.agent.max_retrieval_attempts,
            web_fallback_enabled=settings.agent.web_fallback_enabled,
        )
    except RuntimeError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    _print_result(result)


if __name__ == "__main__":  # pragma: no cover
    typer.run(main)
