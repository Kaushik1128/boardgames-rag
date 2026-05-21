"""RAG evaluation harness — Ragas metrics with an LLM judge.

Runs the answer() pipeline over a curated test set and scores it on
faithfulness, answer relevancy, context precision, and context recall, then
compares rerank-on against rerank-off.

The judge LLM is configurable via `eval.judge_provider`: "groq" (default —
fast, generous free tier) or "gemini" (free tier is heavily rate-limited).

CLI:

.. code-block:: bash

    python -m boardgames_rag.evaluate --testset eval/testset.jsonl \
        --collection boardgames

Use --limit for a quick smoke test against a few questions first.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from boardgames_rag.config import Settings, load_settings
from boardgames_rag.generate import OllamaGenerator, answer
from boardgames_rag.utils import setup_logging

if TYPE_CHECKING:  # pragma: no cover
    from boardgames_rag.generate import RerankerProtocol, RetrieverProtocol

logger = logging.getLogger(__name__)
console = Console()


__all__ = [
    "EvalReport",
    "EvalSample",
    "aggregate_scores",
    "evaluate_samples",
    "load_testset",
    "main",
    "run_rag_over_testset",
    "save_report",
]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class EvalSample:
    """One test-set entry. ``answer`` and ``contexts`` are filled in after a run."""

    id: str
    game: str
    question: str
    reference: str
    difficulty: str = "medium"
    answer: str | None = None
    contexts: list[str] = field(default_factory=list)


@dataclass
class EvalReport:
    """Aggregated result of one evaluation run (one rerank setting)."""

    label: str
    collection: str
    judge_model: str
    n_samples: int
    aggregates: dict[str, float | None]
    per_sample: list[dict[str, Any]]
    timestamp: str


# ---------------------------------------------------------------------------
# Test set loading
# ---------------------------------------------------------------------------

_REQUIRED_FIELDS = ("id", "game", "question", "reference")


def load_testset(path: Path) -> list[EvalSample]:
    """Parse a JSONL test set into EvalSamples.

    Raises ValueError on malformed JSON, missing required fields, or an
    empty file.
    """
    if not path.exists():
        raise FileNotFoundError(f"Test set not found: {path}")

    samples: list[EvalSample] = []
    with path.open(encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON ({exc})") from exc
            missing = [f for f in _REQUIRED_FIELDS if f not in obj]
            if missing:
                raise ValueError(f"{path}:{lineno}: missing field(s): {missing}")
            samples.append(
                EvalSample(
                    id=obj["id"],
                    game=obj["game"],
                    question=obj["question"],
                    reference=obj["reference"],
                    difficulty=obj.get("difficulty", "medium"),
                )
            )
    if not samples:
        raise ValueError(f"Test set is empty: {path}")
    return samples


# ---------------------------------------------------------------------------
# Running the RAG pipeline over the test set
# ---------------------------------------------------------------------------


def run_rag_over_testset(
    samples: list[EvalSample],
    *,
    retriever: RetrieverProtocol,
    generator: Any,
    reranker: RerankerProtocol | None,
    settings: Settings,
) -> list[EvalSample]:
    """Run answer() for each question. Returns new EvalSamples with outputs filled.

    The input samples are left untouched so the same test set can be run twice
    (rerank-on and rerank-off).
    """
    results: list[EvalSample] = []
    for sample in samples:
        rag = answer(
            sample.question,
            retriever=retriever,
            generator=generator,
            reranker=reranker,
            retrieve_k=settings.retrieval.k_per_retriever,
            rerank_top_k=settings.rerank.top_k,
            max_context_chunks=settings.generation.max_context_chunks,
        )
        results.append(
            EvalSample(
                id=sample.id,
                game=sample.game,
                question=sample.question,
                reference=sample.reference,
                difficulty=sample.difficulty,
                answer=rag.answer,
                contexts=[c.payload.get("text", "") for c in rag.sources],
            )
        )
    return results


# ---------------------------------------------------------------------------
# Ragas scoring (network-bound — exercised in the real eval run, not unit tests)
# ---------------------------------------------------------------------------


def build_judge(settings: Settings) -> Any:
    """Wrap the configured judge LLM (Groq or Gemini) as a Ragas LLM."""
    from ragas.llms import LangchainLLMWrapper

    if settings.eval.judge_provider == "groq":
        from langchain_groq import ChatGroq

        if not settings.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to .env — it is required for "
                "the Ragas judge. Get a free key at "
                "https://console.groq.com/keys."
            )
        chat: Any = ChatGroq(
            model=settings.eval.judge_model,
            api_key=settings.groq_api_key,
            temperature=settings.eval.judge_temperature,
        )
    else:
        from langchain_google_genai import ChatGoogleGenerativeAI

        if not settings.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to .env — it is required for the Gemini judge."
            )
        chat = ChatGoogleGenerativeAI(
            model=settings.eval.judge_model,
            google_api_key=settings.gemini_api_key,
            temperature=settings.eval.judge_temperature,
        )
    return LangchainLLMWrapper(chat)


def build_eval_embeddings(settings: Settings) -> Any:
    """Local bge embeddings wrapped as Ragas embeddings (keeps scoring $0)."""
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    hf = HuggingFaceEmbeddings(
        model_name=settings.embedding.model_name,
        model_kwargs={"device": settings.embedding.device},
    )
    return LangchainEmbeddingsWrapper(hf)


def _build_metrics() -> list[Any]:
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )

    return [
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
    ]


def _to_score(value: Any) -> float | None:
    """Coerce a Ragas cell to a plain float, mapping NaN/None to None."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def evaluate_samples(
    samples: list[EvalSample],
    *,
    judge: Any,
    embeddings: Any,
    settings: Settings,
) -> list[dict[str, Any]]:
    """Score filled-in samples with Ragas. Returns per-sample score dicts.

    Output is plain data (no Ragas objects) so the aggregation/report layer
    stays Ragas-independent and unit-testable.
    """
    from ragas import EvaluationDataset, evaluate
    from ragas.run_config import RunConfig

    metrics = _build_metrics()
    dataset = EvaluationDataset.from_list(
        [
            {
                "user_input": s.question,
                "response": s.answer or "",
                "retrieved_contexts": s.contexts,
                "reference": s.reference,
            }
            for s in samples
        ]
    )
    run_config = RunConfig(
        timeout=settings.eval.timeout_seconds,
        max_workers=settings.eval.max_workers,
        max_retries=settings.eval.max_retries,
    )
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge,
        embeddings=embeddings,
        run_config=run_config,
    )
    frame = result.to_pandas()
    metric_columns = [m.name for m in metrics]
    per_sample: list[dict[str, Any]] = []
    for i, sample in enumerate(samples):
        row = frame.iloc[i]
        per_sample.append(
            {
                "id": sample.id,
                "game": sample.game,
                "difficulty": sample.difficulty,
                "scores": {col: _to_score(row[col]) for col in metric_columns},
            }
        )
    return per_sample


# ---------------------------------------------------------------------------
# Aggregation + reporting
# ---------------------------------------------------------------------------


def aggregate_scores(per_sample: list[dict[str, Any]]) -> dict[str, float | None]:
    """Mean of each metric across samples, ignoring None/NaN scores."""
    if not per_sample:
        return {}
    metric_names = list(per_sample[0]["scores"].keys())
    aggregates: dict[str, float | None] = {}
    for name in metric_names:
        values = [s["scores"][name] for s in per_sample if s["scores"].get(name) is not None]
        aggregates[name] = round(statistics.mean(values), 4) if values else None
    return aggregates


def save_report(report: EvalReport, results_dir: Path) -> Path:
    """Write an EvalReport to results_dir as JSON. Returns the file path."""
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = report.timestamp.replace(":", "").replace("-", "")
    path = results_dir / f"eval_{report.label}_{stamp}.json"
    path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return path


def _print_comparison(reports: list[EvalReport]) -> None:
    table = Table(title="Ragas evaluation — rerank ablation")
    table.add_column("Metric")
    for report in reports:
        table.add_column(report.label, justify="right")
    if len(reports) == 2:
        table.add_column("delta (on - off)", justify="right")

    metric_names = list(reports[0].aggregates.keys()) if reports else []
    for name in metric_names:
        values = [r.aggregates.get(name) for r in reports]
        row = [name] + [f"{v:.4f}" if v is not None else "—" for v in values]
        if len(reports) == 2:
            a, b = values
            row.append(f"{a - b:+.4f}" if a is not None and b is not None else "—")
        table.add_row(*row)
    console.print(table)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(
    testset: Annotated[
        Path | None,
        typer.Option("--testset", help="Path to the JSONL test set."),
    ] = None,
    collection: Annotated[
        str | None,
        typer.Option("--collection", help="Qdrant collection name."),
    ] = None,
    config_path: Annotated[
        Path,
        typer.Option("--config", help="Path to YAML config."),
    ] = Path("config.yaml"),
    limit: Annotated[
        int | None,
        typer.Option("--limit", help="Evaluate only the first N questions."),
    ] = None,
    skip_consistency_check: Annotated[
        bool,
        typer.Option("--skip-consistency-check", help="Don't verify BM25 vs Qdrant."),
    ] = False,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="DEBUG / INFO / WARNING / ERROR."),
    ] = "WARNING",
) -> None:
    """Evaluate the RAG pipeline with Ragas, comparing rerank on vs off."""
    from boardgames_rag.retrieve import Reranker, build_hybrid_retriever

    setup_logging(level=getattr(logging, log_level.upper(), logging.WARNING))
    settings = load_settings(config_path)
    if collection is None:
        collection = settings.qdrant.collection
    testset_path = testset or settings.eval.testset_path

    samples = load_testset(testset_path)
    if limit is not None:
        samples = samples[:limit]
    console.print(f"Loaded {len(samples)} test questions from {testset_path}")

    hybrid = build_hybrid_retriever(settings, collection)
    if not skip_consistency_check:
        hybrid.verify_consistency()
    reranker = Reranker(model_name=settings.rerank.model_name, device=settings.rerank.device)
    generator = OllamaGenerator(
        model=settings.generation.model,
        base_url=settings.generation.base_url,
        temperature=settings.generation.temperature,
        max_tokens=settings.generation.max_tokens,
        timeout_seconds=settings.generation.timeout_seconds,
    )
    judge = build_judge(settings)
    embeddings = build_eval_embeddings(settings)

    reports: list[EvalReport] = []
    for label, active_reranker in (("rerank-on", reranker), ("rerank-off", None)):
        console.print(
            f"\n[bold]{label}[/bold]: running the pipeline over {len(samples)} questions ..."
        )
        run_samples = run_rag_over_testset(
            samples,
            retriever=hybrid,
            generator=generator,
            reranker=active_reranker,
            settings=settings,
        )
        console.print(
            f"  scoring with Ragas + {settings.eval.judge_model} (slow on the free tier) ..."
        )
        per_sample = evaluate_samples(
            run_samples, judge=judge, embeddings=embeddings, settings=settings
        )
        report = EvalReport(
            label=label,
            collection=collection,
            judge_model=settings.eval.judge_model,
            n_samples=len(run_samples),
            aggregates=aggregate_scores(per_sample),
            per_sample=per_sample,
            timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        saved = save_report(report, settings.eval.results_dir)
        console.print(f"  saved -> {saved}")
        reports.append(report)

    console.print()
    _print_comparison(reports)


if __name__ == "__main__":  # pragma: no cover
    typer.run(main)
