"""Tests for cross-encoder reranking — rerank_chunks() and the Reranker class."""

from __future__ import annotations

import sys
import types
from typing import Any

from boardgames_rag.retrieve import Reranker, RetrievedChunk, rerank_chunks


class FakeCrossEncoder:
    """Stand-in for sentence-transformers' CrossEncoder.

    `scores` may be a flat list (positional) or a dict keyed by document text.
    """

    def __init__(self, scores: list[float] | dict[str, float]) -> None:
        self.scores = scores
        self.predict_calls: list[list[tuple[str, str]]] = []

    def predict(self, sentences: list[tuple[str, str]], **_: Any) -> list[float]:
        self.predict_calls.append(list(sentences))
        if isinstance(self.scores, dict):
            return [self.scores[doc] for _, doc in sentences]
        return list(self.scores)


def _chunk(pid: str, text: str, *, fusion_score: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(
        point_id=pid,
        score=fusion_score,
        payload={"text": text, "source_file": "x.pdf"},
        debug={"dense_rank": 1, "fusion_method": "rrf"},
    )


# ---------------------------------------------------------------------------
# rerank_chunks
# ---------------------------------------------------------------------------


class TestRerankChunks:
    def test_sorts_descending_by_cross_encoder_score(self):
        chunks = [_chunk("a", "alpha"), _chunk("b", "bravo"), _chunk("c", "charlie")]
        model = FakeCrossEncoder({"alpha": 0.1, "bravo": 0.9, "charlie": 0.5})
        out = rerank_chunks(model, "query", chunks, top_k=3)
        assert [c.point_id for c in out] == ["b", "c", "a"]

    def test_truncates_to_top_k(self):
        chunks = [_chunk("a", "alpha"), _chunk("b", "bravo"), _chunk("c", "charlie")]
        model = FakeCrossEncoder({"alpha": 0.1, "bravo": 0.9, "charlie": 0.5})
        out = rerank_chunks(model, "query", chunks, top_k=2)
        assert [c.point_id for c in out] == ["b", "c"]

    def test_rerank_score_becomes_primary_score(self):
        chunks = [_chunk("a", "alpha", fusion_score=0.99)]
        model = FakeCrossEncoder({"alpha": 0.42})
        out = rerank_chunks(model, "query", chunks, top_k=1)
        assert out[0].score == 0.42

    def test_debug_preserves_full_provenance(self):
        chunks = [_chunk("a", "alpha", fusion_score=0.77)]
        model = FakeCrossEncoder({"alpha": 0.42})
        out = rerank_chunks(model, "query", chunks, top_k=1)
        debug = out[0].debug
        assert debug["fusion_score"] == 0.77
        assert debug["rerank_score"] == 0.42
        assert debug["rerank_rank"] == 1
        # Pre-existing debug keys survive.
        assert debug["fusion_method"] == "rrf"
        assert debug["dense_rank"] == 1

    def test_rerank_rank_is_sequential_from_one(self):
        chunks = [_chunk("a", "alpha"), _chunk("b", "bravo"), _chunk("c", "charlie")]
        model = FakeCrossEncoder({"alpha": 0.1, "bravo": 0.9, "charlie": 0.5})
        out = rerank_chunks(model, "query", chunks, top_k=3)
        assert [c.debug["rerank_rank"] for c in out] == [1, 2, 3]

    def test_empty_chunks_returns_empty(self):
        assert rerank_chunks(FakeCrossEncoder([]), "query", [], top_k=5) == []

    def test_zero_top_k_returns_empty(self):
        chunks = [_chunk("a", "alpha")]
        assert rerank_chunks(FakeCrossEncoder({"alpha": 0.5}), "q", chunks, top_k=0) == []

    def test_model_receives_query_document_pairs(self):
        chunks = [_chunk("a", "alpha"), _chunk("b", "bravo")]
        model = FakeCrossEncoder({"alpha": 0.1, "bravo": 0.9})
        rerank_chunks(model, "my query", chunks, top_k=2)
        assert model.predict_calls == [[("my query", "alpha"), ("my query", "bravo")]]

    def test_payload_is_preserved(self):
        chunks = [_chunk("a", "alpha")]
        out = rerank_chunks(FakeCrossEncoder({"alpha": 0.5}), "q", chunks, top_k=1)
        assert out[0].payload["source_file"] == "x.pdf"

    def test_does_not_mutate_input_chunk_debug(self):
        chunks = [_chunk("a", "alpha")]
        rerank_chunks(FakeCrossEncoder({"alpha": 0.5}), "q", chunks, top_k=1)
        assert "rerank_score" not in chunks[0].debug


# ---------------------------------------------------------------------------
# Reranker class
# ---------------------------------------------------------------------------


class TestReranker:
    def test_rerank_delegates_to_rerank_chunks(self, monkeypatch):
        # Stub sentence_transformers so no model is downloaded.
        fake_st = types.ModuleType("sentence_transformers")
        fake_st.CrossEncoder = lambda model_name, device="cpu": FakeCrossEncoder(  # type: ignore[attr-defined]
            {"alpha": 0.9, "bravo": 0.1}
        )
        monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st)

        reranker = Reranker(model_name="fake-model", device="cpu")
        chunks = [_chunk("a", "alpha"), _chunk("b", "bravo")]
        out = reranker.rerank("query", chunks, top_k=2)
        assert [c.point_id for c in out] == ["a", "b"]
        assert out[0].debug["rerank_rank"] == 1
