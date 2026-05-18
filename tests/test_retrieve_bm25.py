"""BM25Index build / search / save / load round-trip tests."""

from __future__ import annotations

import pickle

import pytest

from boardgames_rag.retrieve import BM25Index, BM25Retriever, tokenize

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


class TestBM25IndexBuild:
    def test_build_simple_corpus(self):
        ids = ["a", "b", "c"]
        corpus = [
            tokenize("Catan settlement build"),
            tokenize("Azul tiles pattern"),
            tokenize("Chess queens checkmate"),
        ]
        index = BM25Index.build(ids, corpus)
        size, hashval = index.signature
        assert size == 3
        assert isinstance(hashval, str) and len(hashval) == 64  # sha256 hex
        assert index.point_ids == ids

    def test_signature_is_deterministic_for_same_ids(self):
        ix1 = BM25Index.build(["a", "b"], [["x"], ["y"]])
        ix2 = BM25Index.build(["a", "b"], [["DIFFERENT"], ["TOKENS"]])
        # Signature only depends on point_ids — not on tokens.
        assert ix1.signature == ix2.signature

    def test_signature_ignores_id_order(self):
        # Hash is computed over the sorted list, so order shouldn't matter.
        ix1 = BM25Index.build(["a", "b", "c"], [["x"], ["y"], ["z"]])
        ix2 = BM25Index.build(["c", "a", "b"], [["x"], ["y"], ["z"]])
        assert ix1.signature == ix2.signature

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="must have the same length"):
            BM25Index.build(["a", "b"], [["one"]])

    def test_empty_corpus_raises(self):
        with pytest.raises(ValueError, match="empty"):
            BM25Index.build([], [])


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class TestBM25IndexSearch:
    def _make_index(self):
        ids = ["catan", "azul", "chess"]
        corpus = [
            tokenize("Catan is about resources and settlements."),
            tokenize("Azul features beautiful tiles and pattern matching."),
            tokenize("Chess is a strategy game with kings and queens."),
        ]
        return BM25Index.build(ids, corpus)

    def test_finds_most_relevant_doc(self):
        index = self._make_index()
        results = index.search(tokenize("queens chess"), k=3)
        assert results[0][0] == "chess"

    def test_results_are_descending_by_score(self):
        index = self._make_index()
        results = index.search(tokenize("game"), k=3)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_filters_zero_scores(self):
        index = self._make_index()
        # No overlapping tokens → all scores ≤ 0 → empty result.
        results = index.search(tokenize("zzzzz xxxxx qqqqq"), k=5)
        assert results == []

    def test_zero_k_returns_empty(self):
        index = self._make_index()
        assert index.search(tokenize("chess"), k=0) == []

    def test_negative_k_returns_empty(self):
        index = self._make_index()
        assert index.search(tokenize("chess"), k=-1) == []

    def test_empty_query_returns_empty(self):
        index = self._make_index()
        assert index.search([], k=5) == []


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------


class TestBM25IndexPersistence:
    def test_save_load_round_trip(self, tmp_path):
        ids = ["a", "b", "c"]
        corpus = [["alpha"], ["bravo", "charlie"], ["delta"]]
        original = BM25Index.build(ids, corpus)
        path = tmp_path / "index.pkl"
        original.save(path)

        loaded = BM25Index.load(path)
        assert loaded.signature == original.signature
        assert loaded.point_ids == original.point_ids
        # Identical query yields identical results.
        q = ["bravo"]
        assert original.search(q, k=3) == loaded.search(q, k=3)

    def test_save_creates_parent_dirs(self, tmp_path):
        index = BM25Index.build(["a"], [["word"]])
        deep_path = tmp_path / "nested" / "dirs" / "index.pkl"
        index.save(deep_path)
        assert deep_path.exists()

    def test_load_non_bm25_pickle_raises(self, tmp_path):
        path = tmp_path / "bad.pkl"
        with path.open("wb") as f:
            pickle.dump({"not": "a BM25State"}, f)
        with pytest.raises(ValueError, match="Not a BM25Index pickle"):
            BM25Index.load(path)


# ---------------------------------------------------------------------------
# BM25Retriever (the thin search wrapper)
# ---------------------------------------------------------------------------


class TestBM25Retriever:
    # rank-bm25's Okapi IDF collapses to 0 when a term appears in half or
    # more of the corpus, so tests need at least 3-4 distinct docs where
    # the query term is rare.

    def _retriever(self) -> BM25Retriever:
        ids = ["a", "b", "c", "d"]
        corpus = [
            tokenize("catan settlement build"),
            tokenize("azul tile pattern"),
            tokenize("chess queen checkmate"),
            tokenize("wingspan bird habitat"),
        ]
        return BM25Retriever(BM25Index.build(ids, corpus))

    def test_search_tokenizes_query(self):
        retriever = self._retriever()
        # Uppercase query — tokenize lowercases it.
        results = retriever.search("CATAN", k=5)
        assert results
        assert results[0][0] == "a"

    def test_search_returns_id_score_pairs(self):
        retriever = self._retriever()
        results = retriever.search("catan", k=1)
        assert len(results) == 1
        pid, score = results[0]
        assert isinstance(pid, str)
        assert isinstance(score, float)
        assert score > 0
