"""
Loading and chunking the knowledge base. Chunk size and overlap are parameters (with
the production values from config as defaults) so the Day 4 evaluation can compare
different chunking strategies without touching the app's settings.
"""

import os
import glob
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import DOCS_FOLDER, CHUNK_SIZE, CHUNK_OVERLAP


def load_and_chunk_documents(
    folder: str = DOCS_FOLDER,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """Load every .md file from doc_type subfolders (profile/, projects/, skills/)
    and split each into overlapping chunks. Returns a flat list of chunk dicts, each
    tagged with its source file and doc_type."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
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
