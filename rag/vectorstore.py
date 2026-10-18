"""
Embedding (with retry-on-transient-error) and ChromaDB storage/retrieval. Building
the index (embedding every chunk and storing it) is separate from querying it — the
app queries an index that's already been built by build_index.py.
"""

import os
import shutil
import time

import chromadb
import openai

from rag.config import (
    client, EMBEDDING_MODEL, CHROMA_PATH, COLLECTION_NAME, MAX_RETRIES, TOP_K,
)
from rag.ingestion import load_and_chunk_documents

# Errors worth retrying: rate limits (429, too many requests too fast) and
# transient server-side failures (503, the provider is temporarily overloaded).
# Both are "try again shortly," not "something is wrong with our code" — so both
# get the same backoff-and-retry treatment instead of crashing the app.
RETRYABLE_ERRORS = (openai.RateLimitError, openai.InternalServerError)


def is_daily_quota_error(error: Exception) -> bool:
    """True if a 429 is a per-DAY quota (e.g. Gemini's 'GenerateRequestsPerDay...'),
    which resets at a fixed time, as opposed to a per-MINUTE limit worth retrying."""
    message = str(error)
    return "PerDay" in message or "per day" in message.lower()


def call_with_retry(fn, *args, **kwargs):
    """Retry-with-backoff around any API call. Handles both 429 (rate limit) and
    503 (provider temporarily overloaded) — the free tier hits both at times."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except RETRYABLE_ERRORS as e:
            # A per-DAY quota error won't clear in a minute — waiting 15-60s and
            # retrying just burns time. Fail immediately so the caller can save its
            # progress and tell the user to come back after the daily reset.
            if is_daily_quota_error(e):
                raise
            if attempt == MAX_RETRIES:
                raise
            wait_seconds = 15 * attempt
            error_name = type(e).__name__
            print(
                f"  [{error_name}] waiting {wait_seconds}s before retry "
                f"({attempt}/{MAX_RETRIES})..."
            )
            time.sleep(wait_seconds)


def embed(text: str) -> list[float]:
    response = call_with_retry(
        client.embeddings.create, model=EMBEDDING_MODEL, input=text
    )
    return response.data[0].embedding


def build_index() -> None:
    """Chunk every document, embed each chunk, and store it in a fresh Chroma
    collection on disk. Wipes any existing collection first so re-running this never
    produces duplicates."""
    print("Loading and chunking documents...")
    chunks = load_and_chunk_documents()
    print(f"Produced {len(chunks)} chunks.\n")

    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.create_collection(name=COLLECTION_NAME)

    print(f"Embedding {len(chunks)} chunks and storing them in Chroma...")
    for i, chunk in enumerate(chunks, start=1):
        vector = embed(chunk["text"])
        collection.add(
            ids=[chunk["id"]],
            embeddings=[vector],
            documents=[chunk["text"]],
            metadatas=[{"source": chunk["source"], "doc_type": chunk["doc_type"]}],
        )
        print(f"  [{i}/{len(chunks)}] stored {chunk['id']}")

    print(f"\nDone. {collection.count()} chunks stored in ./{CHROMA_PATH}/")


def get_collection():
    """Open the existing Chroma collection for querying. Raises a clear error if
    build_index() hasn't been run yet, instead of a confusing Chroma internal error."""
    if not os.path.exists(CHROMA_PATH):
        raise RuntimeError(
            f"No vector store found at ./{CHROMA_PATH}/. "
            "Run `python build_index.py` first to build it."
        )
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    return chroma_client.get_collection(name=COLLECTION_NAME)


def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    """Embed the query and return the k most similar chunks, each with its text,
    metadata, and distance (lower = more similar, per Chroma's convention)."""
    collection = get_collection()
    query_vector = embed(query)
    results = collection.query(query_embeddings=[query_vector], n_results=k)

    retrieved = []
    for text, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        retrieved.append({"text": text, "doc_type": meta["doc_type"], "source": meta["source"], "distance": dist})
    return retrieved
