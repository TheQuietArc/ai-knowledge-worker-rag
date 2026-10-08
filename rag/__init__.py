"""
rag — a small, from-scratch Retrieval-Augmented Generation package.

Modules:
  config       — client setup, model names, paths, constants
  ingestion    — loading and chunking the knowledge base documents
  vectorstore  — embedding + ChromaDB storage and retrieval
  chain        — wires retrieval and generation together, with conversation history
"""
