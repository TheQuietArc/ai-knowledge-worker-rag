"""
Central configuration: the Gemini client (via its OpenAI-compatible endpoint), model
names, file paths, and chunking parameters. Every other module imports from here
instead of each redefining its own client/constants — one source of truth.
"""

import os
from openai import OpenAI

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
CHAT_MODEL = "gemini-3.8-flash"

DOCS_FOLDER = "data/raw_docs"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "ashutosh_knowledge_base"

CHUNK_SIZE = 400
CHUNK_OVERLAP = 80

TOP_K = 3           # how many chunks to retrieve per question
MAX_RETRIES = 5      # API retry attempts on rate limits
