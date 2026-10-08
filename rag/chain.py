"""
The actual RAG chain: given a question and the prior conversation, retrieve relevant
chunks, build a grounded prompt, and generate an answer — while keeping the
conversation's earlier turns so follow-up questions work.

This is "Day 3" in spirit: Day 1 proved retrieval worked with a manual loop, Day 2
built a real vector store, and this module is where retrieval and generation finally
become one callable pipeline instead of two separate scripts.
"""

from rag.config import client, CHAT_MODEL, TOP_K
from rag.vectorstore import retrieve, call_with_retry

SYSTEM_PROMPT = (
    "You are a knowledge assistant that answers questions about Ashutosh — his "
    "projects, background, and skills — using ONLY the context provided with each "
    "question. If the context doesn't contain the answer, say you don't know rather "
    "than guessing. Keep answers concise and specific."
)


def build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a labeled block the model can cite from. Each
    chunk is tagged with its doc_type/source, which keeps the model's answer traceable
    back to where it came from — useful both for debugging and for an interview demo
    ('here's exactly which document backed this answer')."""
    parts = []
    for chunk in chunks:
        parts.append(f"[{chunk['doc_type']}/{chunk['source']}]\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


def answer(message: str, history: list[dict] | None = None, k: int = TOP_K) -> str:
    """Answer one question, grounded in retrieved context, aware of prior turns.

    `history` follows Gradio's `type="messages"` format: a list of
    {"role": "user"|"assistant", "content": str} dicts for every prior turn. Passing
    it to the model (rather than just the current message) is what makes follow-up
    questions like "what stack did it use?" work after an earlier question already
    established which project "it" refers to.
    """
    retrieved_chunks = retrieve(message, k=k)
    context = build_context(retrieved_chunks)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {message}",
    })

    response = call_with_retry(
        client.chat.completions.create,
        model=CHAT_MODEL,
        messages=messages,
    )
    return response.choices[0].message.content
