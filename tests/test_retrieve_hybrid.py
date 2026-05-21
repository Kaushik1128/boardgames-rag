"""HybridRetriever + DenseRetriever tests + CLI smoke test.

Built on fake QdrantClient/Embedder implementations so the suite runs
without a Qdrant server, model downloads, or network.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import typer
from typer.testing import CliRunner

import boardgames_rag.retrieve as retrieve_module
from boardgames_rag.retrieve import (
    BM25Index,
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    IndexOutOfSyncError,
    RetrievedChunk,
    tokenize,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeEmbedder:
    """Returns deterministic zero vectors; matches EmbedderProtocol."""

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim
        self.embed_calls: list[list[str]] = []

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> np.ndarray:
        self.embed_calls.append(list(texts))
        return np.zeros((len(texts), self._dim), dtype=np.float32)


class FakeQdrantClient:
    """In-memory stand-in for QdrantClient.

    Maps point_id → payload. Returns all points (up to ``limit``) on every
    query_points / scroll call. retrieve() filters to requested ids.
    """

    def __init__(self, points: dict[str, dict[str, Any]]) -> None:
        self.points = points
        self.query_calls: list[Any] = []
        self.scroll_calls: list[Any] = []
        self.retrieve_calls: list[Any] = []

    def query_points(self, **kwargs):
        self.query_calls.append(kwargs)
        limit = kwargs.get("limit", 10)
        items = list(self.points.items())[:limit]
        scored = [
            MagicMock(id=pid, score=1.0 - i * 0.1, payload=payload)
            for i, (pid, payload) in enumerate(items)
        ]
        return MagicMock(points=scored)

    def scroll(self, **kwargs):
        self.scroll_calls.append(kwargs)
        records = [MagicMock(id=pid) for pid in self.points]
        return (records, None)

    def retrieve(self, **kwargs):
        self.retrieve_calls.append(kwargs)
        ids = kwargs.get("ids", [])
        return [MagicMock(id=pid, payload=self.points[pid]) for pid in ids if pid in self.points]


def _make_hybrid(
    payloads: dict[str, dict[str, Any]],
    *,
    bm25_ids: list[str] | None = None,
    fusion_method: str = "rrf",
    k_per_retriever: int = 5,
    client_override: FakeQdrantClient | None = None,
) -> tuple[HybridRetriever, FakeQdrantClient]:
    embedder = FakeEmbedder()
    client = client_override or FakeQdrantClient(payloads)
    dense = DenseRetriever(embedder=embedder, client=client, collection="test")

    bm25_ids = bm25_ids if bm25_ids is not None else list(payloads.keys())
    corpus = [tokenize(payloads[pid]["text"]) for pid in bm25_ids]
    bm25 = BM25Retriever(BM25Index.build(bm25_ids, corpus))

    hybrid = HybridRetriever(
        dense=dense,
        bm25=bm25,
        client=client,
        collection="test",
        fusion_method=fusion_method,
        k_per_retriever=k_per_retriever,
    )
    return hybrid, client


# ---------------------------------------------------------------------------
# DenseRetriever
# ---------------------------------------------------------------------------


class TestDenseRetriever:
    def test_returns_id_score_payload_triples(self):
        embedder = FakeEmbedder()
        payloads = {"doc1": {"text": "hello"}, "doc2": {"text": "world"}}
        client = FakeQdrantClient(payloads)
        dense = DenseRetriever(embedder=embedder, client=client, collection="test")
        results = dense.search("query", k=2)
        assert len(results) == 2
        for pid, score, payload in results:
            assert isinstance(pid, str)
            assert isinstance(score, float)
            assert isinstance(payload, dict)

    def test_zero_k_returns_empty(self):
        dense = DenseRetriever(
            embedder=FakeEmbedder(),
            client=FakeQdrantClient({"doc1": {"text": "x"}}),
            collection="test",
        )
        assert dense.search("query", k=0) == []

    def test_calls_embedder_once_with_query(self):
        embedder = FakeEmbedder()
        dense = DenseRetriever(
            embedder=embedder,
            client=FakeQdrantClient({"doc1": {"text": "x"}}),
            collection="test",
        )
        dense.search("How do I attack?", k=1)
        assert embedder.embed_calls == [["How do I attack?"]]


# ---------------------------------------------------------------------------
# Hybrid construction validation
# ---------------------------------------------------------------------------


class TestHybridConstruction:
    def test_invalid_fusion_method_raises(self):
        dense = MagicMock()
        bm25 = MagicMock()
        client = MagicMock()
        with pytest.raises(ValueError, match="unknown fusion_method"):
            HybridRetriever(
                dense=dense,
                bm25=bm25,
                client=client,
                collection="x",
                fusion_method="nope",
            )

    def test_invalid_weighted_alpha_raises(self):
        dense = MagicMock()
        bm25 = MagicMock()
        client = MagicMock()
        with pytest.raises(ValueError, match="weighted_alpha"):
            HybridRetriever(
                dense=dense,
                bm25=bm25,
                client=client,
                collection="x",
                weighted_alpha=2.0,
            )
        with pytest.raises(ValueError, match="weighted_alpha"):
            HybridRetriever(
                dense=dense,
                bm25=bm25,
                client=client,
                collection="x",
                weighted_alpha=-0.1,
            )


# ---------------------------------------------------------------------------
# verify_consistency
# ---------------------------------------------------------------------------


class TestVerifyConsistency:
    def test_matching_ids_passes(self):
        payloads = {"a": {"text": "x"}, "b": {"text": "y"}, "c": {"text": "z"}}
        hybrid, _ = _make_hybrid(payloads)
        hybrid.verify_consistency()  # no raise

    def test_mismatched_count_raises(self):
        payloads = {"a": {"text": "x"}, "b": {"text": "y"}, "c": {"text": "z"}}
        # BM25 only knows about two of the three Qdrant points.
        hybrid, _ = _make_hybrid(payloads, bm25_ids=["a", "b"])
        with pytest.raises(IndexOutOfSyncError, match="out of sync"):
            hybrid.verify_consistency()

    def test_mismatched_ids_raises(self):
        payloads = {"a": {"text": "x"}, "b": {"text": "y"}}
        # Same count, different ids.
        embedder = FakeEmbedder()
        client = FakeQdrantClient(payloads)
        dense = DenseRetriever(embedder=embedder, client=client, collection="test")
        bm25 = BM25Retriever(BM25Index.build(["a", "different"], [["x"], ["y"]]))
        hybrid = HybridRetriever(dense=dense, bm25=bm25, client=client, collection="test")
        with pytest.raises(IndexOutOfSyncError):
            hybrid.verify_consistency()

    def test_paginated_scroll_terminates(self):
        """verify_consistency handles multi-page scroll responses."""

        class PaginatedClient(FakeQdrantClient):
            def __init__(self, points):
                super().__init__(points)
                self._call_count = 0

            def scroll(self, **kwargs):
                self._call_count += 1
                self.scroll_calls.append(kwargs)
                items = list(self.points.keys())
                if self._call_count == 1:
                    return ([MagicMock(id=items[0])], "halfway")
                return ([MagicMock(id=pid) for pid in items[1:]], None)

        payloads = {"a": {"text": "x"}, "b": {"text": "y"}}
        client = PaginatedClient(payloads)
        hybrid, _ = _make_hybrid(payloads, client_override=client)
        hybrid.verify_consistency()
        assert client._call_count == 2


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


class TestHybridSearch:
    def _payloads(self) -> dict[str, dict[str, Any]]:
        return {
            "doc1": {
                "text": "catan settlement build resources",
                "heading": "Building",
                "source_file": "catan.pdf",
            },
            "doc2": {
                "text": "azul tiles pattern matching",
                "heading": "Tiles",
                "source_file": "azul.pdf",
            },
            "doc3": {
                "text": "chess queens checkmate strategy",
                "heading": "Endgame",
                "source_file": "chess.pdf",
            },
        }

    def test_returns_retrieved_chunks(self):
        hybrid, _ = _make_hybrid(self._payloads())
        chunks = hybrid.search("catan", top_k=3)
        assert len(chunks) >= 1
        assert all(isinstance(c, RetrievedChunk) for c in chunks)

    def test_results_sorted_by_score_desc(self):
        hybrid, _ = _make_hybrid(self._payloads())
        chunks = hybrid.search("catan settlement", top_k=3)
        scores = [c.score for c in chunks]
        assert scores == sorted(scores, reverse=True)

    def test_debug_payload_includes_per_retriever_provenance(self):
        hybrid, _ = _make_hybrid(self._payloads())
        chunks = hybrid.search("catan", top_k=3)
        for chunk in chunks:
            assert set(chunk.debug.keys()) >= {
                "dense_rank",
                "dense_score",
                "bm25_rank",
                "bm25_score",
                "fusion_method",
            }
            assert chunk.debug["fusion_method"] == "rrf"

    def test_weighted_fusion_method_recorded(self):
        hybrid, _ = _make_hybrid(self._payloads(), fusion_method="weighted")
        chunks = hybrid.search("catan", top_k=3)
        assert chunks[0].debug["fusion_method"] == "weighted"

    def test_truncates_to_top_k(self):
        hybrid, _ = _make_hybrid(self._payloads())
        chunks = hybrid.search("catan azul chess", top_k=2)
        assert len(chunks) <= 2

    def test_hydrates_bm25_only_hits(self):
        """A point ranked by BM25 but missed by dense gets its payload hydrated."""
        payloads = self._payloads()

        class LimitedDenseClient(FakeQdrantClient):
            """Dense returns only the first 2 points; the 3rd is BM25-only."""

            def query_points(self, **kwargs):
                self.query_calls.append(kwargs)
                items = list(self.points.items())[:2]
                scored = [
                    MagicMock(id=pid, score=1.0 - i * 0.1, payload=payload)
                    for i, (pid, payload) in enumerate(items)
                ]
                return MagicMock(points=scored)

        client = LimitedDenseClient(payloads)
        hybrid, _ = _make_hybrid(payloads, client_override=client, k_per_retriever=3)
        # Query that BM25 finds in doc3 but dense (limited to first two) doesn't.
        chunks = hybrid.search("chess queens", top_k=3)

        ids = {c.point_id for c in chunks}
        assert "doc3" in ids
        # retrieve() must have been called to hydrate doc3.
        assert any("doc3" in (call.get("ids") or []) for call in client.retrieve_calls)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_cli_hybrid(monkeypatch):
    """Patch build_hybrid_retriever to return an in-memory hybrid."""
    payloads = {
        "doc1": {
            "text": "catan settlement build resources",
            "heading": "Building",
            "source_file": "catan.pdf",
        },
        "doc2": {
            "text": "azul tiles pattern matching",
            "heading": "Tiles",
            "source_file": "azul.pdf",
        },
    }
    hybrid, _ = _make_hybrid(payloads)
    monkeypatch.setattr(
        retrieve_module,
        "build_hybrid_retriever",
        lambda settings, collection: hybrid,
    )
    return hybrid


def _build_cli_app() -> typer.Typer:
    app = typer.Typer()
    app.command()(retrieve_module.main)
    return app


def test_cli_table_output(tmp_path, patched_cli_hybrid):
    runner = CliRunner()
    result = runner.invoke(
        _build_cli_app(),
        [
            "--query",
            "catan",
            "--collection",
            "test",
            "--config",
            str(tmp_path / "nope.yaml"),  # doesn't exist → defaults
            "--k",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    # Table title appears in stdout (Rich renders to text in CliRunner).
    assert "results" in result.output.lower()


def test_cli_json_output(tmp_path, patched_cli_hybrid):
    runner = CliRunner()
    result = runner.invoke(
        _build_cli_app(),
        [
            "--query",
            "catan",
            "--collection",
            "test",
            "--config",
            str(tmp_path / "nope.yaml"),
            "--k",
            "2",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    # JSON output should mention at least one point id.
    assert "point_id" in result.output


def test_cli_skip_consistency_check(tmp_path, monkeypatch):
    """With --skip-consistency-check, verify_consistency is never called."""
    payloads = {"a": {"text": "x"}}
    hybrid, _ = _make_hybrid(payloads)

    verify_calls = []
    original_verify = hybrid.verify_consistency

    def tracking_verify():
        verify_calls.append(True)
        original_verify()

    hybrid.verify_consistency = tracking_verify  # type: ignore[method-assign]
    monkeypatch.setattr(
        retrieve_module,
        "build_hybrid_retriever",
        lambda settings, collection: hybrid,
    )

    runner = CliRunner()
    result = runner.invoke(
        _build_cli_app(),
        [
            "--query",
            "x",
            "--collection",
            "test",
            "--config",
            str(tmp_path / "nope.yaml"),
            "--skip-consistency-check",
        ],
    )
    assert result.exit_code == 0, result.output
    assert verify_calls == []  # never called
