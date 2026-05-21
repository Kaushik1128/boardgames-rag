"""Answer generation: retrieve → rerank → prompt → local LLM → grounded answer.

The LLM is reached through an OpenAI-compatible endpoint. By default that is
Ollama running locally (``ollama pull llama3.2:3b`` first); a hosted free-tier
can be swapped in later by changing ``generation.base_url`` in config.

CLI:

.. code-block:: bash

    python -m boardgames_rag.generate --query "How do I trade in Catan?" \
        --collection boardgames
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Protocol

import typer
from rich.console import Console
from rich.panel import Panel

from boardgames_rag.config import load_settings
from boardgames_rag.retrieve import RetrievedChunk
from boardgames_rag.utils import setup_logging

logger = logging.getLogger(__name__)
console = Console()


# Fixed refusal string. The generator is instructed to emit this verbatim when
# the retrieved context does not contain the answer — keeps answers grounded
# and gives the week-4 faithfulness evals a clean signal.
NO_ANSWER = "I couldn't find this in the provided rulebooks."

SYSTEM_PROMPT = (
    "You are a board game rules assistant. Answer the user's question using "
    "ONLY the numbered context passages provided. Cite the passages you rely "
    "on with their bracketed numbers, e.g. [1] or [2][3]. Do not use any "
    "knowledge beyond the context. If the context does not contain the "
    f'answer, reply with exactly this sentence and nothing else: "{NO_ANSWER}"'
)


__all__ = [
    "NO_ANSWER",
    "SYSTEM_PROMPT",
    "Generator",
    "OllamaGenerator",
    "RagAnswer",
    "answer",
    "build_prompt",
    "format_context",
    "main",
]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class RagAnswer:
    """The end-to-end result of one RAG query."""

    question: str
    answer: str
    sources: list[RetrievedChunk]
    debug: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Prompt construction (pure functions)
# ---------------------------------------------------------------------------


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Render chunks as a numbered context block for the prompt."""
    blocks: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.payload.get("source_file") or "unknown"
        heading = chunk.payload.get("heading") or chunk.payload.get("parent_heading") or "—"
        text = (chunk.payload.get("text") or "").strip()
        blocks.append(f"[{i}] ({source} — {heading})\n{text}")
    return "\n\n".join(blocks)


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> list[dict[str, str]]:
    """Build the OpenAI-style chat messages list for a question + context."""
    context = format_context(chunks) if chunks else "(no relevant passages found)"
    user_content = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


class Generator(Protocol):
    def generate(self, messages: list[dict[str, str]]) -> str: ...


class RetrieverProtocol(Protocol):
    def search(self, query: str, top_k: int = ...) -> list[RetrievedChunk]: ...


class RerankerProtocol(Protocol):
    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]: ...


class OllamaGenerator:
    """LLM generation via an OpenAI-compatible endpoint (Ollama by default)."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        timeout_seconds: float = 120.0,
    ) -> None:
        from openai import OpenAI  # lazy

        # Ollama ignores api_key; the OpenAI client just requires a non-empty
        # string. A hosted backend would pass a real key here.
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_seconds)
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, messages: list[dict[str, str]]) -> str:
        from openai import APIConnectionError  # lazy

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except APIConnectionError as exc:
            raise RuntimeError(
                f"Could not reach the LLM endpoint at {self.base_url}. "
                f"Is Ollama running? Start it, then run "
                f"`ollama pull {self.model}`."
            ) from exc
        content = response.choices[0].message.content
        return (content or "").strip()


# ---------------------------------------------------------------------------
# RAG pipeline
# ---------------------------------------------------------------------------


def answer(
    question: str,
    *,
    retriever: RetrieverProtocol,
    generator: Generator,
    reranker: RerankerProtocol | None = None,
    retrieve_k: int = 20,
    rerank_top_k: int = 5,
    max_context_chunks: int = 5,
) -> RagAnswer:
    """Run the full RAG pipeline: retrieve → (rerank) → prompt → generate.

    When ``reranker`` is ``None`` the retrieval order is used directly,
    truncated to ``rerank_top_k``.
    """
    t0 = time.perf_counter()
    candidates = retriever.search(question, top_k=retrieve_k)
    t_retrieve = time.perf_counter() - t0

    t_rerank: float | None = None
    if reranker is not None and candidates:
        t1 = time.perf_counter()
        ranked = reranker.rerank(question, candidates, top_k=rerank_top_k)
        t_rerank = time.perf_counter() - t1
    else:
        ranked = candidates[:rerank_top_k]

    context_chunks = ranked[:max_context_chunks]
    messages = build_prompt(question, context_chunks)

    t2 = time.perf_counter()
    text = generator.generate(messages)
    t_generate = time.perf_counter() - t2

    return RagAnswer(
        question=question,
        answer=text,
        sources=context_chunks,
        debug={
            "n_retrieved": len(candidates),
            "n_reranked": len(ranked),
            "n_context": len(context_chunks),
            "reranked": reranker is not None,
            "retrieve_seconds": round(t_retrieve, 3),
            "rerank_seconds": round(t_rerank, 3) if t_rerank is not None else None,
            "generate_seconds": round(t_generate, 3),
        },
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_answer(result: RagAnswer) -> None:
    console.print()
    console.print(Panel(result.answer, title="Answer", border_style="green"))

    if result.sources:
        console.print("\n[bold]Sources[/bold]")
        for i, chunk in enumerate(result.sources, start=1):
            source = chunk.payload.get("source_file") or "unknown"
            heading = chunk.payload.get("heading") or chunk.payload.get("parent_heading") or "—"
            rr = chunk.debug.get("rerank_score")
            score_str = f"  [dim](rerank {rr:.3f})[/dim]" if rr is not None else ""
            console.print(f"  [{i}] {source} — {heading}{score_str}")

    d = result.debug
    timing = f"retrieve {d['retrieve_seconds']}s"
    if d.get("rerank_seconds") is not None:
        timing += f" · rerank {d['rerank_seconds']}s"
    timing += f" · generate {d['generate_seconds']}s"
    console.print(
        f"\n[dim]{d['n_retrieved']} retrieved → {d['n_context']} in context · {timing}[/dim]"
    )


def main(
    query: Annotated[str, typer.Option("--query", "-q", help="Natural-language question.")],
    collection: Annotated[
        str | None,
        typer.Option("--collection", help="Qdrant collection name."),
    ] = None,
    config_path: Annotated[
        Path,
        typer.Option("--config", help="Path to YAML config."),
    ] = Path("config.yaml"),
    no_rerank: Annotated[
        bool,
        typer.Option("--no-rerank", help="Skip the cross-encoder reranking stage."),
    ] = False,
    skip_consistency_check: Annotated[
        bool,
        typer.Option("--skip-consistency-check", help="Don't verify BM25 vs Qdrant."),
    ] = False,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="DEBUG / INFO / WARNING / ERROR."),
    ] = "WARNING",
) -> None:
    """Ask a question and get a grounded, cited answer from the rulebooks."""
    from boardgames_rag.retrieve import Reranker, build_hybrid_retriever  # lazy

    setup_logging(level=getattr(logging, log_level.upper(), logging.WARNING))
    settings = load_settings(config_path)
    if collection is None:
        collection = settings.qdrant.collection

    hybrid = build_hybrid_retriever(settings, collection)
    if not skip_consistency_check:
        hybrid.verify_consistency()

    reranker: Reranker | None = None
    if settings.rerank.enabled and not no_rerank:
        reranker = Reranker(
            model_name=settings.rerank.model_name,
            device=settings.rerank.device,
        )

    generator = OllamaGenerator(
        model=settings.generation.model,
        base_url=settings.generation.base_url,
        temperature=settings.generation.temperature,
        max_tokens=settings.generation.max_tokens,
        timeout_seconds=settings.generation.timeout_seconds,
    )

    try:
        result = answer(
            query,
            retriever=hybrid,
            generator=generator,
            reranker=reranker,
            retrieve_k=settings.retrieval.k_per_retriever,
            rerank_top_k=settings.rerank.top_k,
            max_context_chunks=settings.generation.max_context_chunks,
        )
    except RuntimeError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    _print_answer(result)


if __name__ == "__main__":  # pragma: no cover
    typer.run(main)
