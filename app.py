"""
Gradio chat UI for the AI Knowledge Worker.

Run build_index.py first if you haven't already (or if you've changed the documents
in data/raw_docs/), then run this file to launch a local chat interface.

Written to work with both older and newer Gradio versions: older Gradio's
ChatInterface passes history as a list of [user, assistant] pairs, while newer
versions (5.0+) can pass a list of {"role": ..., "content": ...} dicts instead. We
normalize either shape into the dict format rag.chain.answer() expects, so this
works regardless of which Gradio version is installed.

Run:
  python app.py
"""

import os
import gradio as gr

from rag.chain import answer
from rag.config import CHROMA_PATH


def normalize_history(history) -> list[dict]:
    """Convert Gradio's history (either old [user, assistant] pairs or new
    {"role", "content"} dicts) into the {"role", "content"} format the chain expects."""
    if not history:
        return []

    normalized = []
    for item in history:
        if isinstance(item, dict):
            # Already in the new format.
            normalized.append({"role": item["role"], "content": item["content"]})
        else:
            # Old format: a [user_message, assistant_message] pair.
            user_msg, assistant_msg = item
            if user_msg:
                normalized.append({"role": "user", "content": user_msg})
            if assistant_msg:
                normalized.append({"role": "assistant", "content": assistant_msg})
    return normalized


def respond(message: str, history) -> str:
    """Gradio calls this with the new message and the conversation so far."""
    return answer(message, history=normalize_history(history))


def main():
    if not os.path.exists(CHROMA_PATH):
        raise RuntimeError(
            f"No vector store found at ./{CHROMA_PATH}/. "
            "Run `python build_index.py` first."
        )

    demo = gr.ChatInterface(
        fn=respond,
        title="Ashutosh's AI Knowledge Worker",
        description=(
            "Ask me about Ashutosh's projects, background, or skills — "
            "I answer using a retrieval-augmented pipeline grounded in his "
            "actual project documents, not general knowledge."
        ),
        examples=[
            "What computer vision project has Ashutosh built?",
            "What's happening to the retail sales analysis project?",
            "What roadmap has he explored for backend AI engineering?",
        ],
        chatbot=gr.Chatbot(height=480, label="Chatbot", avatar_images=(None, "🧠")),

    )
    demo.launch()


if __name__ == "__main__":
    main()