"""
Retrieval evaluation: for each chunking configuration, build a fresh in-memory Chroma
index, run every golden question through it, and score what comes back.

This measures ONLY retrieval (embedding + vector search) — no chat-model calls — so it
is fast, cheap, and isolates the part of the pipeline that chunking affects.

Embeddings are cached on disk (eval/.embedding_cache.json), keyed by model + text, so
re-running an experiment — or sharing questions across configs — never re-embeds the
same text twice.
"""

import hashlib
import json
import os
import uuid
from dataclasses import dataclass

import chromadb

from rag.config import client, EMBEDDING_MODEL, TOP_K
from rag.ingestion import load_and_chunk_documents
from rag.vectorstore import call_with_retry
from eval.golden_set import GOLDEN_SET, TestQuestion
from eval.metrics import mrr, ndcg_at_k, keyword_coverage

CACHE_PATH = os.path.join(os.path.dirname(__file__), ".embedding_cache.json")
_cache: dict[str, list[float]] | None = None


@dataclass(frozen=True)
class RetrievalConfig:
    """One experiment: how to chunk, how many chunks to retrieve, which embedder."""
    name: str
    chunk_size: int
    chunk_overlap: int
    k: int = TOP_K
    embedding_model: str = EMBEDDING_MODEL


# The experiment grid. Sizes are written out literally (not read from rag.config) so
# these rows keep meaning the same thing even after you change the production setting.
# Everything varies one thing at a time so any change in the numbers has a single cause.
CONFIGS = [
    RetrievalConfig("baseline (400/80, k=3)", 400, 80, k=3),
    RetrievalConfig("small chunks (200/40, k=3)", 200, 40, k=3),
    RetrievalConfig("large chunks (800/160, k=3)", 800, 160, k=3),
    RetrievalConfig("no overlap (400/0, k=3)", 400, 0, k=3),
    RetrievalConfig("baseline chunks, k=1", 400, 80, k=1),
    RetrievalConfig("baseline chunks, k=5", 400, 80, k=5),
]


# ---------------------------------------------------------------- embedding cache
def _load_cache() -> dict:
    global _cache
    if _cache is None:
        if os.path.exists(CACHE_PATH):
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                _cache = json.load(f)
        else:
            _cache = {}
    return _cache


def _save_cache() -> None:
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(_cache, f)


def cached_embed(text: str, model: str = EMBEDDING_MODEL) -> list[float]:
    """Embed text, hitting the API only if this exact (model, text) is not cached."""
    cache = _load_cache()
    key = f"{model}::{hashlib.sha256(text.encode('utf-8')).hexdigest()}"
    if key not in cache:
        response = call_with_retry(client.embeddings.create, model=model, input=text)
        cache[key] = response.data[0].embedding
        _save_cache()
    return cache[key]


# ---------------------------------------------------------------- corpus validation
def validate_golden_set(questions: list[TestQuestion]) -> list[str]:
    """Check that every keyword exists somewhere in the raw documents. A keyword that
    appears nowhere can never be retrieved, so it would silently drag every score
    down — this catches typos and out-of-date docs before they distort the results."""
    corpus = " ".join(c["text"] for c in load_and_chunk_documents(
        chunk_size=100_000, chunk_overlap=0)).lower()
    problems = []
    for q in questions:
        for kw in q.keywords:
            if kw not in corpus:
                problems.append(f'keyword "{kw}" not found in any document — "{q.question}"')
    return problems


# ---------------------------------------------------------------- evaluation
def build_index(config: RetrievalConfig):
    """Chunk the corpus per the config and load it into a throwaway in-memory Chroma
    collection. Returns (collection, all_chunk_texts)."""
    chunks = load_and_chunk_documents(
        chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap
    )
    chroma = chromadb.EphemeralClient()
    collection = chroma.create_collection(name=f"eval_{uuid.uuid4().hex[:8]}")
    collection.add(
        ids=[c["id"] for c in chunks],
        embeddings=[cached_embed(c["text"], config.embedding_model) for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[{"source": c["source"], "doc_type": c["doc_type"]} for c in chunks],
    )
    return collection, [c["text"] for c in chunks]


def evaluate_config(config: RetrievalConfig, questions: list[TestQuestion]) -> dict:
    """Run every question through one config and aggregate the metrics."""
    collection, all_texts = build_index(config)

    per_question = []
    for q in questions:
        query_vector = cached_embed(q.question, config.embedding_model)
        result = collection.query(query_embeddings=[query_vector], n_results=config.k)
        ranked = result["documents"][0]
        per_question.append({
            "question": q.question,
            "category": q.category,
            "mrr": mrr(q.keywords, ranked),
            "ndcg": ndcg_at_k(q.keywords, ranked, all_texts, config.k),
            "coverage": keyword_coverage(q.keywords, ranked),
        })

    n = len(per_question)
    return {
        "config": config,
        "num_chunks": len(all_texts),
        "mrr": sum(r["mrr"] for r in per_question) / n,
        "ndcg": sum(r["ndcg"] for r in per_question) / n,
        "coverage": sum(r["coverage"] for r in per_question) / n,
        "fully_covered": sum(1 for r in per_question if r["coverage"] == 1.0),
        "total": n,
        "per_question": per_question,
    }


def format_results_table(results: list[dict]) -> str:
    """Markdown table — prints nicely in a terminal and pastes straight into a README."""
    lines = [
        "| Configuration | Chunks | MRR | nDCG@k | Keyword coverage | Fully answerable |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['config'].name} | {r['num_chunks']} | {r['mrr']:.3f} | "
            f"{r['ndcg']:.3f} | {r['coverage']:.3f} | {r['fully_covered']}/{r['total']} |"
        )
    return "\n".join(lines)


def format_failures(result: dict) -> str:
    """List the questions where retrieval missed at least one keyword."""
    failures = [r for r in result["per_question"] if r["coverage"] < 1.0]
    if not failures:
        return "  none — every question had all its keywords retrieved."
    return "\n".join(
        f"  - [{r['category']}] {r['question']} (coverage={r['coverage']:.2f}, mrr={r['mrr']:.2f})"
        for r in failures
    )


def run(questions: list[TestQuestion] | None = None) -> list[dict]:
    questions = questions or GOLDEN_SET

    problems = validate_golden_set(questions)
    if problems:
        print("WARNING: golden set has keywords missing from your documents:")
        for p in problems:
            print(f"  - {p}")
        print("Fix the keywords (or your docs) in eval/golden_set.py — these will score 0.\n")

    results = []
    for i, config in enumerate(CONFIGS, start=1):
        print(f"[{i}/{len(CONFIGS)}] Evaluating: {config.name}")
        results.append(evaluate_config(config, questions))

    print("\n" + format_results_table(results))
    print(f"\nQuestions the baseline missed keywords on:\n{format_failures(results[0])}")
    return results
