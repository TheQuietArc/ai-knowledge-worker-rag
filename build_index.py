"""
Build (or rebuild) the vector store. Run this once before starting app.py, and again
any time you change the documents in data/raw_docs/.

Run:
  python build_index.py
"""

from rag.vectorstore import build_index

if __name__ == "__main__":
    build_index()
