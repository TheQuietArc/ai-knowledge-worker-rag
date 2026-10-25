"""Retrieval metrics, checked against values worked out by hand."""

import math

import pytest

from eval.metrics import keyword_coverage, mrr, ndcg_at_k

CHUNKS = ["alpha beta", "gamma delta", "epsilon zeta", "alpha gamma"]


class TestMRR:
    def test_keyword_at_rank_one_scores_one(self):
        assert mrr(["alpha"], CHUNKS) == 1.0

    def test_keyword_at_rank_two_scores_half(self):
        assert mrr(["gamma"], CHUNKS) == 0.5

    def test_keyword_never_found_scores_zero(self):
        assert mrr(["omega"], CHUNKS) == 0.0

    def test_averages_across_keywords(self):
        # alpha -> rank 1 (1.0), gamma -> rank 2 (0.5)
        assert mrr(["alpha", "gamma"], CHUNKS) == 0.75

    def test_matching_ignores_case(self):
        assert mrr(["ALPHA"], CHUNKS) == 1.0


class TestKeywordCoverage:
    def test_fraction_of_keywords_found(self):
        assert keyword_coverage(["alpha", "gamma", "omega"], CHUNKS[:2]) == pytest.approx(2 / 3)

    def test_ignores_rank(self):
        assert keyword_coverage(["alpha"], [CHUNKS[2], CHUNKS[3]]) == 1.0

    def test_nothing_found(self):
        assert keyword_coverage(["omega"], CHUNKS) == 0.0


class TestNDCG:
    def test_perfect_ordering_is_one(self):
        ranked = [CHUNKS[0], CHUNKS[3], CHUNKS[1]]
        assert ndcg_at_k(["alpha"], ranked, CHUNKS, 3) == pytest.approx(1.0)

    def test_relevant_chunk_ranked_lower_scores_less(self):
        # Retrieved gains: 1, 0, 1  -> DCG = 1 + 0 + 1/log2(4) = 1.5
        # Ideal gains:     1, 1, 0  -> DCG = 1 + 1/log2(3)
        ranked = [CHUNKS[0], CHUNKS[1], CHUNKS[3]]
        expected = 1.5 / (1 + 1 / math.log2(3))
        assert ndcg_at_k(["alpha"], ranked, CHUNKS, 3) == pytest.approx(expected)

    def test_keyword_missing_from_corpus_is_zero_not_an_error(self):
        assert ndcg_at_k(["omega"], CHUNKS[:3], CHUNKS, 3) == 0.0

    def test_graded_relevance_for_multi_keyword_questions(self):
        corpus = ["alpha beta", "alpha", "beta"]
        # The chunk holding both keywords has relevance 1.0; single-keyword chunks 0.5.
        best_first = ndcg_at_k(["alpha", "beta"], ["alpha beta", "alpha", "beta"], corpus, 3)
        worst_first = ndcg_at_k(["alpha", "beta"], ["beta", "alpha", "alpha beta"], corpus, 3)
        assert best_first == pytest.approx(1.0)
        assert worst_first < best_first
