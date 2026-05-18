"""Pure-function tests for Reciprocal Rank Fusion and weighted-score fusion."""

from __future__ import annotations

import math

import pytest

from boardgames_rag.retrieve import reciprocal_rank_fusion, weighted_score_fusion

# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


class TestReciprocalRankFusion:
    def test_single_retriever_scores_decay_by_rank(self):
        scores = reciprocal_rank_fusion([["a", "b", "c"]], k=60)
        assert math.isclose(scores["a"], 1 / 61)
        assert math.isclose(scores["b"], 1 / 62)
        assert math.isclose(scores["c"], 1 / 63)

    def test_two_retrievers_sum_per_point(self):
        # "b" appears at rank 2 in R1 and rank 1 in R2 — gets both contributions.
        scores = reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=60)
        assert math.isclose(scores["a"], 1 / 61)
        assert math.isclose(scores["b"], 1 / 62 + 1 / 61)
        assert math.isclose(scores["c"], 1 / 62)
        # Sanity: b ranks highest after fusion (in both lists, second list at #1).
        assert scores["b"] > scores["a"]
        assert scores["b"] > scores["c"]

    def test_default_k_is_sixty(self):
        scores = reciprocal_rank_fusion([["a"]])
        assert math.isclose(scores["a"], 1 / 61)

    def test_higher_k_dampens_rank_difference(self):
        scores_low = reciprocal_rank_fusion([["a", "b"]], k=1)
        scores_high = reciprocal_rank_fusion([["a", "b"]], k=1000)
        # With high k, scores[a] and scores[b] are nearly equal.
        diff_low = scores_low["a"] - scores_low["b"]
        diff_high = scores_high["a"] - scores_high["b"]
        assert diff_low > diff_high

    def test_zero_k_raises(self):
        with pytest.raises(ValueError):
            reciprocal_rank_fusion([["a"]], k=0)

    def test_negative_k_raises(self):
        with pytest.raises(ValueError):
            reciprocal_rank_fusion([["a"]], k=-5)

    def test_empty_rankings_list(self):
        assert reciprocal_rank_fusion([], k=60) == {}

    def test_all_empty_rankings(self):
        assert reciprocal_rank_fusion([[], []], k=60) == {}


# ---------------------------------------------------------------------------
# Weighted score fusion
# ---------------------------------------------------------------------------


class TestWeightedScoreFusion:
    def test_weight_one_zero_excludes_second_retriever(self):
        fused = weighted_score_fusion(
            [[("a", 5.0), ("b", 2.0)], [("c", 100.0)]],
            weights=[1.0, 0.0],
        )
        # First retriever spans 5..2; normalize → a=1, b=0.
        assert math.isclose(fused["a"], 1.0)
        assert math.isclose(fused["b"], 0.0)
        # Second retriever's weight is 0 → skipped, "c" never appears.
        assert "c" not in fused

    def test_weight_zero_one_excludes_first_retriever(self):
        fused = weighted_score_fusion(
            [[("a", 5.0)], [("c", 10.0), ("d", 0.0)]],
            weights=[0.0, 1.0],
        )
        assert "a" not in fused
        assert math.isclose(fused["c"], 1.0)
        assert math.isclose(fused["d"], 0.0)

    def test_normalization_balances_disparate_score_scales(self):
        # R1 scores in the thousands, R2 in the tenths — min-max normalization
        # makes the contribution per-retriever comparable.
        fused = weighted_score_fusion(
            [[("a", 1000.0), ("b", 500.0)], [("a", 0.1), ("c", 0.05)]],
            weights=[0.5, 0.5],
        )
        # "a" hits the max of both ranges → 0.5 + 0.5 = 1.0.
        assert math.isclose(fused["a"], 1.0)
        # "b" only in R1, at the min → 0 contribution.
        assert math.isclose(fused["b"], 0.0)
        # "c" only in R2, at the min → 0 contribution.
        assert math.isclose(fused["c"], 0.0)

    def test_single_score_retriever_normalizes_to_one(self):
        fused = weighted_score_fusion([[("a", 5.0)]], weights=[1.0])
        assert math.isclose(fused["a"], 1.0)

    def test_constant_scores_all_normalize_to_one(self):
        # span = 0 → fall back to 1.0 per item.
        fused = weighted_score_fusion([[("a", 5.0), ("b", 5.0)]], weights=[1.0])
        assert math.isclose(fused["a"], 1.0)
        assert math.isclose(fused["b"], 1.0)

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="must have the same length"):
            weighted_score_fusion([[("a", 1.0)]], weights=[0.5, 0.5])

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            weighted_score_fusion([[("a", 1.0)]], weights=[-0.5])

    def test_empty_inputs_return_empty(self):
        assert weighted_score_fusion([], weights=[]) == {}
        assert weighted_score_fusion([[]], weights=[1.0]) == {}
