"""
Day 1 — Toy RAG: understand the mechanism before using any framework.

This script does RAG "by hand":
  1. Load a handful of documents about Ashutosh's projects.
  2. Embed each document with Gemini's embedding model.
  3. Given a question, embed the question the same way.
  4. Find the most similar document via cosine similarity (plain numpy, no vector DB).
  5. Stuff that document into a prompt and ask Gemini to answer using ONLY that context.

No LangChain, no Chroma yet — that's Day 2. The point of Day 1 is to see exactly
what "retrieval augmented generation" means at the code level before a framework
hides it from you.

This uses Google's Gemini models, called through the OpenAI Python library pointed at
Gemini's OpenAI-compatible endpoint — so the code looks identical to a pure-OpenAI
setup, but it talks to Gemini and uses your Gemini API key.

Setup:
  pip install -r requirements.txt
  export GEMINI_API_KEY=...   (or put it in a .env file, see below)
  Get a free key at https://aistudio.google.com/apikey

Run:
  python day1_toy_rag.py

Note on rate limits: the Gemini free tier allows only a handful of requests per
minute per model. This script retries automatically with a short wait when it hits
that limit, so it may pause for ~10-50 seconds between calls — that's expected, not
a bug.
"""

import os
import glob
import time
import numpy as np
import openai
from openai import OpenAI

# Loads GEMINI_API_KEY from a .env file if you use one. Safe to skip if you export
# the variable in your shell instead.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

client = OpenAI(
    api_key=os.environ["GEMINI_API_KEY"],
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

EMBEDDING_MODEL = "gemini-embedding-001"  # Gemini's current embedding model
CHAT_MODEL = "gemini-3.8-flash"            # current model name (replaces 2.5-flash)
DOCS_FOLDER = "data/raw_docs"
MAX_RETRIES = 5


def call_with_retry(fn, *args, **kwargs):
    """Call an API function, automatically retrying on rate-limit errors (free tier
    allows only a few requests/minute). Waits a bit longer before each retry."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except openai.RateLimitError as e:
            if attempt == MAX_RETRIES:
                raise
            wait_seconds = 15 * attempt  # simple backoff: 15s, 30s, 45s...
            print(
                f"  [rate limit] hit quota, waiting {wait_seconds}s before retry "
                f"({attempt}/{MAX_RETRIES})..."
            )
            time.sleep(wait_seconds)


def load_documents(folder: str) -> list[dict]:
    """Load every .md file from doc_type subfolders (profile/, projects/, skills/)
    as one document each (whole-file chunk for now — Day 2 will split these into
    smaller chunks properly). Each document is tagged with its folder name as a
    doc_type, mirroring how the course's Insurellm example tags chunks by category —
    this sets us up for filtered retrieval later."""
    documents = []
    for path in sorted(glob.glob(os.path.join(folder, "**", "*.md"), recursive=True)):
        doc_type = os.path.basename(os.path.dirname(path))
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        documents.append({
            "source": os.path.basename(path),
            "doc_type": doc_type,
            "text": text,
        })
    return documents


def embed(text: str) -> np.ndarray:
    """Turn a piece of text into a vector. Documents that mean similar things end up
    as vectors that point in similar directions."""
    response = call_with_retry(
        client.embeddings.create, model=EMBEDDING_MODEL, input=text
    )
    return np.array(response.data[0].embedding)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """1.0 = identical direction (very similar meaning), 0.0 = unrelated,
    -1.0 = opposite meaning. This is the core math behind every vector search."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def retrieve_best_match(question_embedding: np.ndarray, documents: list[dict]) -> dict:
    """Compare the question's vector against every document's vector, return the
    single most similar document. A real vector DB (Chroma, Day 2) does this same
    comparison but indexed and at scale — the underlying math doesn't change."""
    scored = [
        (cosine_similarity(question_embedding, doc["embedding"]), doc)
        for doc in documents
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    best_score, best_doc = scored[0]
    print(
        f"  [retrieval] best match: [{best_doc['doc_type']}] {best_doc['source']} "
        f"(similarity={best_score:.3f})"
    )
    return best_doc


def answer_with_context(question: str, context: str) -> str:
    """Ask Gemini to answer using ONLY the retrieved context — this is what makes it
    'grounded' rather than the model guessing from general training knowledge."""
    prompt = (
        "Answer the question using ONLY the context below. "
        "If the context doesn't contain the answer, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )
    response = call_with_retry(
        client.chat.completions.create,
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def main():
    print("Loading documents...")
    documents = load_documents(DOCS_FOLDER)
    print(f"Loaded {len(documents)} documents:")
    for doc in documents:
        print(f"  - [{doc['doc_type']}] {doc['source']}")
    print()

    print("Embedding documents (one API call per document)...")
    for doc in documents:
        doc["embedding"] = embed(doc["text"])
    print("Done.\n")

    questions = [
        "What computer vision project has Ashutosh built?",
        "What is Ashutosh's CGPA and what is he studying?",
        "What data analysis project did Ashutosh build and what is happening to it?",
        "What roadmap has Ashutosh explored for backend AI engineering?",
    ]

    for question in questions:
        print(f"Q: {question}")
        question_embedding = embed(question)
        best_doc = retrieve_best_match(question_embedding, documents)
        answer = answer_with_context(question, best_doc["text"])
        print(f"A: {answer}\n")


if __name__ == "__main__":
    main()
