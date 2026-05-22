"""Tests for the evaluation harness.

Everything is faked — no Gemini calls, no Ragas, no network, no API key
needed to run the suite. The Ragas-bound functions (build_judge,
build_eval_embeddings, evaluate_samples) are exercised only in the real
eval run, not here.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

import boardgames_rag.evaluate as evaluate_module
import boardgames_rag.retrieve as retrieve_module
from boardgames_rag.config import Settings
from boardgames_rag.evaluate import (
    EvalReport,
    _to_score,
    aggregate_scores,
    build_judge,
    load_testset,
    run_agent_over_testset,
    run_rag_over_testset,
    save_report,
    score_coverage,
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


class FakeHybrid:
    def __init__(self, chunks: list[RetrievedChunk] | None = None) -> None:
        self.chunks = chunks or [_chunk("context one"), _chunk("context two")]

    def search(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        return list(self.chunks[:top_k])

    def verify_consistency(self) -> None:
        pass


class FakeReranker:
    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        return list(chunks[:top_k])


class FakeGenerator:
    def __init__(self, response: str = "a grounded answer [1]") -> None:
        self.response = response

    def generate(self, messages: list[dict[str, str]]) -> str:
        return self.response


class FakeGraph:
    """Stand-in for a compiled LangGraph agent — run_agent only needs .invoke."""

    def __init__(self, answer: str = "agent answer") -> None:
        self.answer = answer

    def invoke(self, state: dict) -> dict:
        return {
            **state,
            "answer": self.answer,
            "chunks": [_chunk("agent context")],
            "used_web": False,
            "attempts": 1,
        }


def _write_testset(path, n: int = 2) -> None:
    lines = [
        json.dumps(
            {
                "id": f"q{i}",
                "game": "Catan",
                "difficulty": "easy",
                "question": f"question {i}",
                "reference": f"reference {i}",
            }
        )
        for i in range(n)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# load_testset
# ---------------------------------------------------------------------------


class TestLoadTestset:
    def test_loads_valid_jsonl(self, tmp_path):
        path = tmp_path / "ts.jsonl"
        _write_testset(path, n=3)
        samples = load_testset(path)
        assert len(samples) == 3
        assert samples[0].id == "q0"
        assert samples[0].game == "Catan"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_testset(tmp_path / "nope.jsonl")

    def test_empty_file_raises(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="empty"):
            load_testset(path)

    def test_blank_lines_skipped(self, tmp_path):
        path = tmp_path / "ts.jsonl"
        path.write_text(
            json.dumps({"id": "a", "game": "Azul", "question": "q", "reference": "r"}) + "\n\n\n",
            encoding="utf-8",
        )
        assert len(load_testset(path)) == 1

    def test_malformed_json_raises(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text("{not valid json}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="invalid JSON"):
            load_testset(path)

    def test_missing_required_field_raises(self, tmp_path):
        path = tmp_path / "ts.jsonl"
        path.write_text(
            json.dumps({"id": "a", "game": "Azul", "question": "q"}) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="missing field"):
            load_testset(path)

    def test_difficulty_defaults_to_medium(self, tmp_path):
        path = tmp_path / "ts.jsonl"
        path.write_text(
            json.dumps({"id": "a", "game": "Azul", "question": "q", "reference": "r"}) + "\n",
            encoding="utf-8",
        )
        assert load_testset(path)[0].difficulty == "medium"

    def test_real_project_testset_parses(self):
        # The committed 30-question test set must always parse.
        from pathlib import Path

        samples = load_testset(Path("eval/testset.jsonl"))
        assert len(samples) == 30
        assert len({s.game for s in samples}) == 10


# ---------------------------------------------------------------------------
# run_rag_over_testset
# ---------------------------------------------------------------------------


class TestRunRagOverTestset:
    def test_fills_answer_and_contexts(self, tmp_path):
        path = tmp_path / "ts.jsonl"
        _write_testset(path, n=2)
        samples = load_testset(path)
        run = run_rag_over_testset(
            samples,
            retriever=FakeHybrid(),
            generator=FakeGenerator("the answer"),
            reranker=FakeReranker(),
            settings=Settings(),
        )
        assert len(run) == 2
        assert all(r.answer == "the answer" for r in run)
        assert all(r.contexts for r in run)

    def test_does_not_mutate_input_samples(self, tmp_path):
        path = tmp_path / "ts.jsonl"
        _write_testset(path, n=1)
        samples = load_testset(path)
        run_rag_over_testset(
            samples,
            retriever=FakeHybrid(),
            generator=FakeGenerator(),
            reranker=None,
            settings=Settings(),
        )
        assert samples[0].answer is None
        assert samples[0].contexts == []

    def test_works_without_reranker(self, tmp_path):
        path = tmp_path / "ts.jsonl"
        _write_testset(path, n=1)
        run = run_rag_over_testset(
            load_testset(path),
            retriever=FakeHybrid(),
            generator=FakeGenerator(),
            reranker=None,
            settings=Settings(),
        )
        assert run[0].answer is not None


# ---------------------------------------------------------------------------
# run_agent_over_testset
# ---------------------------------------------------------------------------


class TestRunAgentOverTestset:
    def test_fills_answer_and_contexts(self, tmp_path):
        path = tmp_path / "ts.jsonl"
        _write_testset(path, n=2)
        samples = load_testset(path)
        run = run_agent_over_testset(
            samples,
            graph=FakeGraph(answer="the agent answer"),
            max_attempts=2,
            web_fallback_enabled=True,
        )
        assert len(run) == 2
        assert all(r.answer == "the agent answer" for r in run)
        assert all(r.contexts for r in run)

    def test_does_not_mutate_input_samples(self, tmp_path):
        path = tmp_path / "ts.jsonl"
        _write_testset(path, n=1)
        samples = load_testset(path)
        run_agent_over_testset(
            samples, graph=FakeGraph(), max_attempts=2, web_fallback_enabled=True
        )
        assert samples[0].answer is None


# ---------------------------------------------------------------------------
# _to_score
# ---------------------------------------------------------------------------


class TestToScore:
    def test_float_passes_through(self):
        assert _to_score(0.75) == 0.75

    def test_none_stays_none(self):
        assert _to_score(None) is None

    def test_nan_becomes_none(self):
        assert _to_score(float("nan")) is None

    def test_numeric_string_coerced(self):
        assert _to_score("0.5") == 0.5

    def test_garbage_becomes_none(self):
        assert _to_score("not a number") is None


# ---------------------------------------------------------------------------
# aggregate_scores
# ---------------------------------------------------------------------------


class TestAggregateScores:
    def test_means_per_metric(self):
        per_sample = [
            {"id": "a", "scores": {"faithfulness": 1.0, "recall": 0.5}},
            {"id": "b", "scores": {"faithfulness": 0.0, "recall": 0.5}},
        ]
        agg = aggregate_scores(per_sample)
        assert agg["faithfulness"] == 0.5
        assert agg["recall"] == 0.5

    def test_ignores_none_values(self):
        per_sample = [
            {"id": "a", "scores": {"faithfulness": 1.0}},
            {"id": "b", "scores": {"faithfulness": None}},
        ]
        assert aggregate_scores(per_sample)["faithfulness"] == 1.0

    def test_all_none_metric_aggregates_to_none(self):
        per_sample = [
            {"id": "a", "scores": {"faithfulness": None}},
            {"id": "b", "scores": {"faithfulness": None}},
        ]
        assert aggregate_scores(per_sample)["faithfulness"] is None

    def test_empty_input_returns_empty(self):
        assert aggregate_scores([]) == {}


# ---------------------------------------------------------------------------
# score_coverage
# ---------------------------------------------------------------------------


class TestScoreCoverage:
    def test_full_coverage(self):
        per_sample = [
            {"scores": {"a": 1.0, "b": 0.5}},
            {"scores": {"a": 0.0, "b": 1.0}},
        ]
        assert score_coverage(per_sample) == 1.0

    def test_partial_coverage(self):
        per_sample = [
            {"scores": {"a": 1.0, "b": None}},
            {"scores": {"a": None, "b": None}},
        ]
        assert score_coverage(per_sample) == 0.25

    def test_empty_returns_zero(self):
        assert score_coverage([]) == 0.0


# ---------------------------------------------------------------------------
# save_report
# ---------------------------------------------------------------------------


class TestSaveReport:
    def _report(self) -> EvalReport:
        return EvalReport(
            label="rerank-on",
            collection="boardgames",
            judge_model="llama3.1:latest",
            n_samples=2,
            coverage=1.0,
            aggregates={"faithfulness": 0.9},
            per_sample=[{"id": "a", "scores": {"faithfulness": 0.9}}],
            timestamp="2026-05-21T12:00:00+00:00",
        )

    def test_writes_json_file(self, tmp_path):
        path = save_report(self._report(), tmp_path / "results")
        assert path.exists()
        assert "rerank-on" in path.name

    def test_creates_results_dir(self, tmp_path):
        deep = tmp_path / "nested" / "results"
        save_report(self._report(), deep)
        assert deep.exists()

    def test_content_round_trips(self, tmp_path):
        path = save_report(self._report(), tmp_path / "results")
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["label"] == "rerank-on"
        assert loaded["aggregates"]["faithfulness"] == 0.9
        assert loaded["n_samples"] == 2


# ---------------------------------------------------------------------------
# build_judge
# ---------------------------------------------------------------------------


class TestBuildJudge:
    def test_groq_provider_without_key_raises(self):
        settings = Settings()
        settings.eval.judge_provider = "groq"
        settings.groq_api_key = None
        with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
            build_judge(settings)

    def test_gemini_provider_without_key_raises(self):
        settings = Settings()
        settings.eval.judge_provider = "gemini"
        settings.gemini_api_key = None
        with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
            build_judge(settings)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _fake_evaluate_samples(
    samples: list[Any], *, judge: Any, embeddings: Any, settings: Any
) -> list[dict[str, Any]]:
    return [
        {
            "id": s.id,
            "game": s.game,
            "difficulty": s.difficulty,
            "scores": {"faithfulness": 0.8, "context_recall": 0.7},
        }
        for s in samples
    ]


def _patch_cli_collaborators(monkeypatch, *, build_judge_fn=None) -> None:
    """Patch every network-bound collaborator the evaluate CLI touches."""
    monkeypatch.setattr(retrieve_module, "build_hybrid_retriever", lambda s, c: FakeHybrid())
    monkeypatch.setattr(retrieve_module, "Reranker", lambda model_name, device: FakeReranker())
    monkeypatch.setattr(evaluate_module, "OllamaGenerator", lambda **kwargs: FakeGenerator())
    monkeypatch.setattr(
        evaluate_module,
        "build_judge",
        build_judge_fn or (lambda settings: object()),
    )
    monkeypatch.setattr(evaluate_module, "build_eval_embeddings", lambda settings: object())
    monkeypatch.setattr(evaluate_module, "evaluate_samples", _fake_evaluate_samples)


def _cli_app() -> typer.Typer:
    app = typer.Typer()
    app.command()(evaluate_module.main)
    return app


def test_cli_runs_both_ablations(tmp_path, monkeypatch):
    testset = tmp_path / "ts.jsonl"
    _write_testset(testset, n=2)
    config = tmp_path / "config.yaml"
    config.write_text(
        f"eval:\n  results_dir: {(tmp_path / 'results').as_posix()}\n",
        encoding="utf-8",
    )
    _patch_cli_collaborators(monkeypatch)

    result = CliRunner().invoke(
        _cli_app(),
        [
            "--testset",
            str(testset),
            "--collection",
            "boardgames",
            "--config",
            str(config),
        ],
    )
    assert result.exit_code == 0, result.output

    written = list((tmp_path / "results").glob("*.json"))
    assert len(written) == 2
    labels = {json.loads(p.read_text(encoding="utf-8"))["label"] for p in written}
    assert labels == {"rerank-on", "rerank-off"}


def test_cli_rejects_unknown_judge_provider(tmp_path):
    testset = tmp_path / "ts.jsonl"
    _write_testset(testset, n=1)
    result = CliRunner().invoke(
        _cli_app(),
        ["--testset", str(testset), "--judge-provider", "bogus"],
    )
    assert result.exit_code != 0


def test_cli_judge_provider_override_picks_default_model(tmp_path, monkeypatch):
    testset = tmp_path / "ts.jsonl"
    _write_testset(testset, n=1)
    config = tmp_path / "config.yaml"
    config.write_text(
        f"eval:\n  results_dir: {(tmp_path / 'results').as_posix()}\n",
        encoding="utf-8",
    )
    captured: dict[str, str] = {}

    def capturing_build_judge(settings):
        captured["provider"] = settings.eval.judge_provider
        captured["model"] = settings.eval.judge_model
        return object()

    _patch_cli_collaborators(monkeypatch, build_judge_fn=capturing_build_judge)

    result = CliRunner().invoke(
        _cli_app(),
        [
            "--testset",
            str(testset),
            "--config",
            str(config),
            "--judge-provider",
            "gemini",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["provider"] == "gemini"
    # --judge-model was not given → the provider's default model is selected.
    assert captured["model"] == "gemini-2.5-flash"


def test_cli_pipeline_agent(tmp_path, monkeypatch):
    testset = tmp_path / "ts.jsonl"
    _write_testset(testset, n=2)
    config = tmp_path / "config.yaml"
    config.write_text(
        f"eval:\n  results_dir: {(tmp_path / 'results').as_posix()}\n",
        encoding="utf-8",
    )
    _patch_cli_collaborators(monkeypatch)

    result = CliRunner().invoke(
        _cli_app(),
        [
            "--testset",
            str(testset),
            "--collection",
            "boardgames",
            "--config",
            str(config),
            "--pipeline",
            "agent",
        ],
    )
    assert result.exit_code == 0, result.output

    written = list((tmp_path / "results").glob("*.json"))
    assert len(written) == 2
    labels = {json.loads(p.read_text(encoding="utf-8"))["label"] for p in written}
    assert labels == {"agent", "linear"}


def test_cli_rejects_unknown_pipeline():
    result = CliRunner().invoke(_cli_app(), ["--pipeline", "bogus"])
    assert result.exit_code != 0
