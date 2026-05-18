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
        self.collection = kwargs.get("collection", "test")
        self.upserts: list[list[Any]] = []

    def upsert(self, points: list[Any]) -> None:
        self.upserts.append(list(points))


@pytest.fixture
def patched_pipeline(monkeypatch):
    """Replace Embedder and QdrantStore with in-memory fakes."""
    monkeypatch.setattr(ingest_module, "Embedder", FakeEmbedder)
    monkeypatch.setattr(ingest_module, "QdrantStore", FakeQdrantStore)


def _write_minimal_mtg(path):
    path.write_text(
        "100. General\n"
        "\n"
        "100.1. First rule body for testing.\n"
        "\n"
        "100.2. Second rule body for testing.\n",
        encoding="utf-8",
    )


def test_ingest_directory_runs_end_to_end(tmp_path, patched_pipeline):
    _write_minimal_mtg(tmp_path / "rules.txt")
    total = ingest_module.ingest_directory(tmp_path, "test-collection", Settings())
    assert total > 0


def test_ingest_directory_missing_dir_raises(tmp_path, patched_pipeline):
    with pytest.raises(FileNotFoundError):
        ingest_module.ingest_directory(tmp_path / "missing", "test", Settings())


def test_ingest_directory_empty_dir_returns_zero(tmp_path, patched_pipeline):
    total = ingest_module.ingest_directory(tmp_path, "test", Settings())
    assert total == 0


def test_ingest_directory_skips_unsupported_files(tmp_path, patched_pipeline):
    (tmp_path / "ignored.docx").write_text("ignored", encoding="utf-8")
    _write_minimal_mtg(tmp_path / "rules.txt")
    total = ingest_module.ingest_directory(tmp_path, "test", Settings())
    assert total > 0  # only the .txt was ingested


def test_ingest_directory_with_no_chunks_returns_zero(tmp_path, patched_pipeline):
    # A .txt that produces no parseable sections (empty body, no headings).
    (tmp_path / "empty.txt").write_text("", encoding="utf-8")
    total = ingest_module.ingest_directory(tmp_path, "test", Settings())
    assert total == 0


def test_cli_main_invokes_pipeline(tmp_path, patched_pipeline):
    """Smoke-test the Typer CLI with fakes patched in."""
    app = typer.Typer()
    app.command()(ingest_module.main)

    src = tmp_path / "src"
    src.mkdir()
    _write_minimal_mtg(src / "rules.txt")
    fake_config = tmp_path / "noconfig.yaml"  # absent → pydantic defaults

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--source-dir",
            str(src),
            "--collection",
            "test-collection",
            "--config",
            str(fake_config),
            "--log-level",
            "WARNING",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Ingested" in result.output
