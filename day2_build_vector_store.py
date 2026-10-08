"""
Day 2 — Real ingestion pipeline: chunking + a real vector database.

Day 1 embedded whole files and compared them with a manual cosine-similarity loop.
That doesn't scale, and whole-file chunks are too coarse — a 40-line document gets
treated as one blob, even if only one sentence actually answers the question.

Day 2 fixes both problems:
  1. Split each document into smaller, overlapping chunks using LangChain's
     RecursiveCharacterTextSplitter, instead of embedding whole files.
  2. Store every chunk's embedding in ChromaDB, a real vector database, instead of
     a Python list we scan by hand.
  3. Each chunk keeps its doc_type metadata (profile / projects / skills), so later
     we can filter retrieval by category.

This script builds the vector store. Run day2_visualize_embeddings.py afterward to
see the chunks plotted in 2D with t-SNE.

Setup:
  pip install -r requirements.txt
  GEMINI_API_KEY must be set in .env (same as Day 1)

Run:
  python day2_build_vector_store.py
"""

import os
import glob
import time
import shutil

import chromadb
import openai
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

client = OpenAI(
    api_key=os.environ["GEMINI_API_KEY"],
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

EMBEDDING_MODEL = "gemini-embedding-001"
DOCS_FOLDER = "data/raw_docs"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "ashutosh_knowledge_base"
MAX_RETRIES = 5

# Chunk size in characters, not tokens — simple and good enough for this project.
# 400 chars (~80-100 words) keeps each chunk focused on one idea; 80 chars of
# overlap means a sentence that straddles a chunk boundary still appears whole in
# at least one chunk, instead of getting cut in half and losing meaning.
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80


def call_with_retry(fn, *args, **kwargs):
    """Same retry-with-backoff pattern as Day 1 — the free tier rate-limits us."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except openai.RateLimitError:
            if attempt == MAX_RETRIES:
                raise
            wait_seconds = 15 * attempt
            print(
                f"  [rate limit] waiting {wait_seconds}s before retry "
                f"({attempt}/{MAX_RETRIES})..."
            )
            time.sleep(wait_seconds)


def embed(text: str) -> list[float]:
    response = call_with_retry(
        client.embeddings.create, model=EMBEDDING_MODEL, input=text
    )
    return response.data[0].embedding


def load_and_chunk_documents(folder: str) -> list[dict]:
    """Load every .md file from doc_type subfolders and split each one into
    overlapping chunks. Returns a flat list of chunk dicts, each tagged with its
    source file and doc_type — this is what actually gets embedded and stored."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Try to split on paragraph breaks first, then sentences, then words —
        # keeps chunks from being cut off mid-thought whenever possible.
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for path in sorted(glob.glob(os.path.join(folder, "**", "*.md"), recursive=True)):
        doc_type = os.path.basename(os.path.dirname(path))
        source = os.path.basename(path)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        pieces = splitter.split_text(text)
        for i, piece in enumerate(pieces):
            chunks.append({
                "id": f"{source}::chunk{i}",
                "text": piece,
                "source": source,
                "doc_type": doc_type,
            })
    return chunks


def main():
    print("Loading and chunking documents...")
    chunks = load_and_chunk_documents(DOCS_FOLDER)
    print(f"Produced {len(chunks)} chunks from the knowledge base.\n")

    # Quick breakdown by doc_type, so you can see the split at a glance.
    counts = {}
    for c in chunks:
        counts[c["doc_type"]] = counts.get(c["doc_type"], 0) + 1
    for doc_type, count in counts.items():
        print(f"  [{doc_type}] {count} chunks")
    print()

    # Show one example chunk so you can see what the splitter actually produced.
    print("Example chunk:")
    print(f"  source: {chunks[0]['source']}")
    print(f"  text:   {chunks[0]['text'][:150]}...\n")

    # Start fresh each run so re-running this script doesn't duplicate chunks.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    print("Connecting to ChromaDB (local, on-disk)...")
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

    print(f"\nDone. Vector store saved to ./{CHROMA_PATH}/")
    print(f"Collection '{COLLECTION_NAME}' now has {collection.count()} chunks.")

    # Prove retrieval actually works end to end before moving on.
    print("\n--- Quick retrieval sanity check ---")
    test_query = "What computer vision project has Ashutosh built?"
    query_vector = embed(test_query)
    results = collection.query(query_embeddings=[query_vector], n_results=2)
    print(f"Query: {test_query}")
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        print(f"  [{meta['doc_type']}/{meta['source']}] (distance={dist:.3f})")
        print(f"    {doc[:120]}...")


if __name__ == "__main__":
    main()
