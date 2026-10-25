"""
Shared test setup.

rag.config creates the API client at import time, which needs GEMINI_API_KEY to exist.
No test makes a real API call (anything that would is mocked), so a placeholder key is
enough — and it means a test can never accidentally spend your real quota.
"""

import os

os.environ.setdefault("GEMINI_API_KEY", "test-key-not-real")

# Paths like data/raw_docs are relative to the project root, so always run from there.
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
