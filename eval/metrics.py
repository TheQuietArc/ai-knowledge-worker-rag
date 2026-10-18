"""
Retrieval metrics. Pure functions (no API calls, no I/O) so they're easy to test.

A chunk is "relevant" to a question to the degree it contains the question's
keywords. All matching is case-insensitive substring search.

  MRR              Mean Reciprocal Rank. For each keyword, find the rank of the first
                   retrieved chunk that contains it; score = 1/rank (0 if never found).
                   Average over the question's keywords. Rewards putting the right
                   chunk at rank 1 rather than rank 3.

  nDCG@k           Normalized Discounted Cumulative Gain. Each retrieved chunk gets a
                   graded relevance = fraction of the question's keywords it contains.
                   Gains are discounted by log2(rank+1), so a relevant chunk at rank 1
                   counts more than at rank 3, then divided by the best score any
                   ordering of the whole corpus could achieve (the "ideal" DCG), giving
                   a value between 0 and 1.

  Keyword coverage Fraction of the question's keywords found anywhere in the top-k
                   chunks. This is the "could the model even have answered this?"
                   metric — it ignores ranking.
"""

import math


def _contains(text: str, keyword: str) -> bool:
    return keyword.lower() in text.lower()


def mrr(keywords: list[str], ranked_texts: list[str]) -> float:
    """Mean reciprocal rank across a question's keywords."""
    scores = []
    for keyword in keywords:
        score = 0.0
        for rank, text in enumerate(ranked_texts, start=1):
            if _contains(text, keyword):
                score = 1.0 / rank
                break
        scores.append(score)
    return sum(scores) / len(scores)


def keyword_coverage(keywords: list[str], ranked_texts: list[str]) -> float:
    """Fraction of keywords that appear in at least one retrieved chunk."""
    found = sum(
        1 for kw in keywords if any(_contains(text, kw) for text in ranked_texts)
    )
    return found / len(keywords)


def _relevance(text: str, keywords: list[str]) -> float:
    """Graded relevance of one chunk: the fraction of keywords it contains."""
    return sum(1 for kw in keywords if _contains(text, kw)) / len(keywords)


def _dcg(relevances: list[float]) -> float:
    # rank 1 -> log2(2) = 1, so the top result is undiscounted.
    return sum(rel / math.log2(rank + 1) for rank, rel in enumerate(relevances, start=1))


def ndcg_at_k(
    keywords: list[str], ranked_texts: list[str], all_texts: list[str], k: int
) -> float:
    """nDCG@k. `ranked_texts` is what the retriever returned (best first);
    `all_texts` is every chunk in the index, used to work out the ideal ordering."""
    actual = _dcg([_relevance(t, keywords) for t in ranked_texts[:k]])
    ideal_relevances = sorted((_relevance(t, keywords) for t in all_texts), reverse=True)[:k]
    ideal = _dcg(ideal_relevances)
    return actual / ideal if ideal > 0 else 0.0
