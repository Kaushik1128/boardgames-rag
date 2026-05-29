"""Tests for the answer() orchestrator, OllamaGenerator, and the generate CLI.

All collaborators are faked, so the suite runs with no Ollama server, no
model downloads, and no network.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import typer
from typer.testing import CliRunner

import boardgames_rag.generate as generate_module
import boardgames_rag.retrieve as retrieve_module
from boardgames_rag.config import Settings
from boardgames_rag.generate import NO_ANSWER, OllamaGenerator, RagAnswer, answer, build_generator
from boardgames_rag.retrieve import RetrievedChunk

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _chunks(n: int) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            point_id=f"c{i}",
            score=1.0 - i * 0.1,
            payload={"text": f"text {i}", "source_file": "x.pdf", "heading": "H"},
        )
        for i in range(n)
    ]


class FakeRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.search_calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        self.search_calls.append((query, top_k))
        return list(self.chunks[:top_k])

    def verify_consistency(self) -> None:
        pass


class FakeReranker:
    def __init__(self) -> None:
        self.rerank_calls: list[tuple[str, int, int]] = []

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        self.rerank_calls.append((query, len(chunks), top_k))
        # Reverse order so tests can prove reranking actually ran.
        return list(reversed(chunks))[:top_k]


class FakeGenerator:
    def __init__(self, response: str = "A grounded answer [1].") -> None:
        self.response = response
        self.generate_calls: list[list[dict[str, str]]] = []

    def generate(self, messages: list[dict[str, str]]) -> str:
        self.generate_calls.append(messages)
        return self.response


class FakeOpenAIClient:
    """Stand-in for openai.OpenAI exposing chat.completions.create.

    Also serves streaming requests: when ``stream=True`` is passed, returns
    an iterator of fake chunks built from ``stream_deltas`` (defaults to a
    split of ``response_text`` on word boundaries).
    """

    def __init__(
        self,
        response_text: str | None = "generated answer",
        stream_deltas: list[str] | None = None,
    ) -> None:
        self.response_text = response_text
        self.stream_deltas = stream_deltas
        self.create_calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.create_calls.append(kwargs)
        if kwargs.get("stream"):
            deltas = self.stream_deltas
            if deltas is None:
                deltas = (self.response_text or "").split(" ")
                deltas = [(d + " ") if i < len(deltas) - 1 else d for i, d in enumerate(deltas)]
            return iter(
                SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=d))])
                for d in deltas
            )
        message = SimpleNamespace(content=self.response_text)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


# ---------------------------------------------------------------------------
# answer()
# ---------------------------------------------------------------------------


class TestAnswer:
    def test_returns_rag_answer(self):
        result = answer(
            "How do I trade?",
            retriever=FakeRetriever(_chunks(5)),
            generator=FakeGenerator("the answer"),
        )
        assert isinstance(result, RagAnswer)
        assert result.answer == "the answer"
        assert result.question == "How do I trade?"

    def test_runs_reranker_when_provided(self):
        reranker = FakeReranker()
        result = answer(
            "q",
            retriever=FakeRetriever(_chunks(5)),
            generator=FakeGenerator(),
            reranker=reranker,
            retrieve_k=5,
            rerank_top_k=3,
        )
        assert len(reranker.rerank_calls) == 1
        assert result.debug["reranked"] is True
        assert result.debug["rerank_seconds"] is not None
        # FakeReranker reverses; sources reflect that.
        assert result.sources[0].point_id == "c4"

    def test_skips_reranker_when_none(self):
        result = answer(
            "q",
            retriever=FakeRetriever(_chunks(5)),
            generator=FakeGenerator(),
            reranker=None,
            retrieve_k=5,
            rerank_top_k=3,
        )
        assert result.debug["reranked"] is False
        assert result.debug["rerank_seconds"] is None
        assert len(result.sources) == 3
        # Retrieval order preserved.
        assert result.sources[0].point_id == "c0"

    def test_max_context_chunks_caps_sources(self):
        result = answer(
            "q",
            retriever=FakeRetriever(_chunks(10)),
            generator=FakeGenerator(),
            reranker=None,
            retrieve_k=10,
            rerank_top_k=8,
            max_context_chunks=2,
        )
        assert len(result.sources) == 2

    def test_generator_receives_built_prompt(self):
        gen = FakeGenerator()
        answer(
            "How do I trade?",
            retriever=FakeRetriever(_chunks(3)),
            generator=gen,
            reranker=None,
        )
        messages = gen.generate_calls[0]
        assert messages[0]["role"] == "system"
        assert "How do I trade?" in messages[1]["content"]

    def test_empty_retrieval_skips_reranker_and_still_answers(self):
        reranker = FakeReranker()
        result = answer(
            "q",
            retriever=FakeRetriever([]),
            generator=FakeGenerator(NO_ANSWER),
            reranker=reranker,
        )
        assert result.answer == NO_ANSWER
        assert result.sources == []
        # Reranker is not invoked when there are no candidates.
        assert reranker.rerank_calls == []

    def test_debug_records_counts_and_timings(self):
        result = answer(
            "q",
            retriever=FakeRetriever(_chunks(3)),
            generator=FakeGenerator(),
            reranker=None,
        )
        assert result.debug["n_retrieved"] == 3
        assert "retrieve_seconds" in result.debug
        assert "generate_seconds" in result.debug


# ---------------------------------------------------------------------------
# OllamaGenerator
# ---------------------------------------------------------------------------


class TestOllamaGenerator:
    def test_generate_returns_message_content(self):
        gen = OllamaGenerator(model="test-model")
        gen.client = FakeOpenAIClient("hello answer")  # type: ignore[assignment]
        assert gen.generate([{"role": "user", "content": "q"}]) == "hello answer"

    def test_generate_strips_whitespace(self):
        gen = OllamaGenerator(model="test-model")
        gen.client = FakeOpenAIClient("  padded \n")  # type: ignore[assignment]
        assert gen.generate([{"role": "user", "content": "q"}]) == "padded"

    def test_generate_handles_none_content(self):
        gen = OllamaGenerator(model="test-model")
        gen.client = FakeOpenAIClient(None)  # type: ignore[assignment]
        assert gen.generate([{"role": "user", "content": "q"}]) == ""

    def test_generate_passes_model_and_params(self):
        gen = OllamaGenerator(model="my-model", temperature=0.3, max_tokens=512)
        fake = FakeOpenAIClient()
        gen.client = fake  # type: ignore[assignment]
        gen.generate([{"role": "user", "content": "q"}])
        call = fake.create_calls[0]
        assert call["model"] == "my-model"
        assert call["temperature"] == 0.3
        assert call["max_tokens"] == 512

    def test_connection_error_becomes_friendly_runtime_error(self):
        from openai import APIConnectionError

        gen = OllamaGenerator(model="test-model")

        class FailingClient:
            def __init__(self) -> None:
                self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._fail))

            def _fail(self, **_):
                raise APIConnectionError(request=MagicMock())

        gen.client = FailingClient()  # type: ignore[assignment]
        with pytest.raises(RuntimeError, match="Is Ollama running"):
            gen.generate([{"role": "user", "content": "q"}])

    def test_generate_stream_yields_deltas_in_order(self):
        gen = OllamaGenerator(model="test-model")
        gen.client = FakeOpenAIClient(stream_deltas=["Hello ", "world", "!"])  # type: ignore[assignment]
        chunks = list(gen.generate_stream([{"role": "user", "content": "q"}]))
        assert chunks == ["Hello ", "world", "!"]

    def test_generate_stream_passes_stream_true(self):
        gen = OllamaGenerator(model="test-model")
        fake = FakeOpenAIClient(stream_deltas=["a"])
        gen.client = fake  # type: ignore[assignment]
        list(gen.generate_stream([{"role": "user", "content": "q"}]))
        assert fake.create_calls[0]["stream"] is True

    def test_generate_stream_skips_empty_deltas(self):
        """Some providers emit empty/None deltas as keep-alives."""
        gen = OllamaGenerator(model="test-model")
        gen.client = FakeOpenAIClient(stream_deltas=["A", "", None, "B"])  # type: ignore[assignment]
        chunks = list(gen.generate_stream([{"role": "user", "content": "q"}]))
        assert chunks == ["A", "B"]

    def test_generate_stream_connection_error_becomes_friendly_runtime_error(self):
        from openai import APIConnectionError

        gen = OllamaGenerator(model="test-model")

        class FailingClient:
            def __init__(self) -> None:
                self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._fail))

            def _fail(self, **_):
                raise APIConnectionError(request=MagicMock())

        gen.client = FailingClient()  # type: ignore[assignment]
        with pytest.raises(RuntimeError, match="Is Ollama running"):
            list(gen.generate_stream([{"role": "user", "content": "q"}]))


# ---------------------------------------------------------------------------
# build_generator
# ---------------------------------------------------------------------------


class _FakeOpenAICtor:
    """Captures kwargs the OpenAI client is constructed with."""

    def __init__(self) -> None:
        self.kwargs: dict | None = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return FakeOpenAIClient()


class TestBuildGenerator:
    def test_ollama_backend_uses_local_base_url_and_model(self, monkeypatch):
        fake = _FakeOpenAICtor()
        monkeypatch.setattr("openai.OpenAI", fake)
        settings = Settings()  # defaults: ollama backend
        gen = build_generator(settings)
        assert isinstance(gen, OllamaGenerator)
        assert gen.model == settings.generation.model
        assert fake.kwargs is not None
        assert fake.kwargs["base_url"] == settings.generation.base_url

    def test_groq_backend_uses_groq_endpoint_and_groq_model(self, monkeypatch):
        fake = _FakeOpenAICtor()
        monkeypatch.setattr("openai.OpenAI", fake)
        settings = Settings()
        settings.generation.backend = "groq"
        settings.groq_api_key = "test-key"
        gen = build_generator(settings)
        assert gen.model == settings.generation.groq_model
        assert fake.kwargs is not None
        assert fake.kwargs["base_url"] == "https://api.groq.com/openai/v1"
        assert fake.kwargs["api_key"] == "test-key"

    def test_groq_backend_without_key_raises(self):
        settings = Settings()
        settings.generation.backend = "groq"
        settings.groq_api_key = None
        with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
            build_generator(settings)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_cli_app() -> typer.Typer:
    app = typer.Typer()
    app.command()(generate_module.main)
    return app


def test_cli_smoke_no_rerank(tmp_path, monkeypatch):
    monkeypatch.setattr(
        retrieve_module,
        "build_hybrid_retriever",
        lambda settings, collection: FakeRetriever(_chunks(3)),
    )
    monkeypatch.setattr(
        generate_module,
        "build_generator",
        lambda settings: FakeGenerator("CLI answer [1]"),
    )

    result = CliRunner().invoke(
        _build_cli_app(),
        [
            "--query",
            "How do I trade?",
            "--collection",
            "test",
            "--config",
            str(tmp_path / "nope.yaml"),
            "--no-rerank",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "CLI answer" in result.output


def test_cli_smoke_with_rerank(tmp_path, monkeypatch):
    monkeypatch.setattr(
        retrieve_module,
        "build_hybrid_retriever",
        lambda settings, collection: FakeRetriever(_chunks(3)),
    )
    monkeypatch.setattr(
        retrieve_module,
        "Reranker",
        lambda model_name, device: FakeReranker(),
    )
    monkeypatch.setattr(
        generate_module,
        "build_generator",
        lambda settings: FakeGenerator("reranked CLI answer"),
    )

    result = CliRunner().invoke(
        _build_cli_app(),
        [
            "--query",
            "q",
            "--collection",
            "test",
            "--config",
            str(tmp_path / "nope.yaml"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "reranked CLI answer" in result.output


def test_cli_handles_generator_error(tmp_path, monkeypatch):
    class FailingGenerator:
        def generate(self, messages):
            raise RuntimeError("Could not reach the LLM endpoint")

    monkeypatch.setattr(
        retrieve_module,
        "build_hybrid_retriever",
        lambda settings, collection: FakeRetriever(_chunks(2)),
    )
    monkeypatch.setattr(generate_module, "build_generator", lambda settings: FailingGenerator())

    result = CliRunner().invoke(
        _build_cli_app(),
        [
            "--query",
            "q",
            "--collection",
            "test",
            "--config",
            str(tmp_path / "nope.yaml"),
            "--no-rerank",
            "--skip-consistency-check",
        ],
    )
    assert result.exit_code == 1
    assert "Error" in result.output
