"""Integration-style tests for the orchestrator and CLI.

We swap ``Embedder`` and ``QdrantStore`` for in-memory fakes so the suite
runs without network, without a Qdrant server, and without downloading a
sentence-transformers model. This keeps unit tests fast and fully local.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import typer
from typer.testing import CliRunner

import boardgames_rag.ingest as ingest_module
from boardgames_rag.config import Settings


class FakeEmbedder:
    """Returns deterministic zero-vectors, mimicking the real Embedder API."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.batch_size = kwargs.get("batch_size", 32)
        self.embed_calls: list[list[str]] = []

    @property
    def dim(self) -> int:
        return 8

    def embed(self, texts: list[str]) -> np.ndarray:
        self.embed_calls.append(list(texts))
        return np.zeros((len(texts), 8), dtype=np.float32)


class FakeQdrantStore:
    """Captures upserts in memory; does not touch a Qdrant server."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        qdrant_config = kwargs.get("qdrant_config")
        self.collection = getattr(qdrant_config, "collection", "test")
        self.upserts: list[list[Any]] = []

    def upsert(self, points: list[Any]) -> None:
        self.upserts.append(list(points))


@pytest.fixture
def patched_pipeline(monkeypatch):
    """Replace Embedder and QdrantStore with in-memory fakes."""
    monkeypatch.setattr(ingest_module, "Embedder", FakeEmbedder)
    monkeypatch.setattr(ingest_module, "QdrantStore", FakeQdrantStore)


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Settings whose BM25 index_dir is per-test, isolating disk artifacts."""
    s = Settings()
    s.retrieval.bm25.index_dir = tmp_path / "bm25"
    return s


def _write_minimal_mtg(path) -> None:
    path.write_text(
        "100. General\n"
        "\n"
        "100.1. First rule body for testing.\n"
        "\n"
        "100.2. Second rule body for testing.\n",
        encoding="utf-8",
    )


def test_ingest_directory_runs_end_to_end(tmp_path, patched_pipeline, settings):
    src = tmp_path / "src"
    src.mkdir()
    _write_minimal_mtg(src / "rules.txt")
    total = ingest_module.ingest_directory(src, "test-collection", settings)
    assert total > 0
    # BM25 index was persisted alongside the Qdrant upserts.
    assert (settings.retrieval.bm25.index_dir / "test-collection.pkl").exists()


def test_ingest_directory_missing_dir_raises(tmp_path, patched_pipeline, settings):
    with pytest.raises(FileNotFoundError):
        ingest_module.ingest_directory(tmp_path / "missing", "test", settings)


def test_ingest_directory_empty_dir_returns_zero(tmp_path, patched_pipeline, settings):
    src = tmp_path / "src"
    src.mkdir()
    total = ingest_module.ingest_directory(src, "test", settings)
    assert total == 0


def test_ingest_directory_skips_unsupported_files(tmp_path, patched_pipeline, settings):
    src = tmp_path / "src"
    src.mkdir()
    (src / "ignored.docx").write_text("ignored", encoding="utf-8")
    _write_minimal_mtg(src / "rules.txt")
    total = ingest_module.ingest_directory(src, "test", settings)
    assert total > 0  # only the .txt was ingested


def test_ingest_directory_with_no_chunks_returns_zero(tmp_path, patched_pipeline, settings):
    # A .txt that produces no parseable sections (empty body, no headings).
    src = tmp_path / "src"
    src.mkdir()
    (src / "empty.txt").write_text("", encoding="utf-8")
    total = ingest_module.ingest_directory(src, "test", settings)
    assert total == 0


def test_cli_main_invokes_pipeline(tmp_path, patched_pipeline):
    """Smoke-test the Typer CLI with fakes patched in."""
    app = typer.Typer()
    app.command()(ingest_module.main)

    src = tmp_path / "src"
    src.mkdir()
    _write_minimal_mtg(src / "rules.txt")

    # Materialize a tiny config.yaml that redirects the BM25 index_dir so the
    # test doesn't write into the real ./data/bm25 directory.
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"retrieval:\n  bm25:\n    index_dir: {(tmp_path / 'bm25').as_posix()}\n",
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--source-dir",
            str(src),
            "--collection",
            "test-collection",
            "--config",
            str(config_path),
            "--log-level",
            "WARNING",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Ingested" in result.output


# ---------------------------------------------------------------------------
# build_qdrant_client — picks server vs embedded mode by config
# ---------------------------------------------------------------------------


def test_build_qdrant_client_uses_url_when_path_unset(monkeypatch):
    """Default config: hit the server at qdrant.url, not a local directory."""
    captured: dict[str, Any] = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("qdrant_client.QdrantClient", FakeClient)
    settings = Settings()
    ingest_module.build_qdrant_client(settings.qdrant)
    assert "url" in captured
    assert captured["url"] == settings.qdrant.url
    assert "path" not in captured


def test_build_qdrant_client_uses_path_when_set(monkeypatch, tmp_path):
    """Setting qdrant.path switches to embedded mode and creates the dir."""
    captured: dict[str, Any] = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("qdrant_client.QdrantClient", FakeClient)
    settings = Settings()
    target = tmp_path / "qdrant_data"
    settings.qdrant.path = str(target)
    ingest_module.build_qdrant_client(settings.qdrant)
    assert captured.get("path") == str(target)
    assert "url" not in captured
    assert target.exists()  # mkdir(parents=True, exist_ok=True)


def test_build_qdrant_client_path_wins_over_url(monkeypatch, tmp_path):
    """If both are set, the deploy-style path takes precedence."""
    captured: dict[str, Any] = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("qdrant_client.QdrantClient", FakeClient)
    settings = Settings()
    settings.qdrant.url = "http://example.invalid:6333"
    settings.qdrant.path = str(tmp_path / "data")
    ingest_module.build_qdrant_client(settings.qdrant)
    assert "path" in captured
    assert "url" not in captured
