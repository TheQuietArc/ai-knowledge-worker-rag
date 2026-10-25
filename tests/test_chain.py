"""The RAG chain: how a question, history and retrieved chunks become a model request."""

from types import SimpleNamespace

import pytest

import rag.chain as chain

FAKE_CHUNKS = [
    {"text": "YOLOv8n handles detection.", "doc_type": "projects", "source": "store.md", "distance": 0.1},
    {"text": "Built solo, end to end.", "doc_type": "projects", "source": "store.md", "distance": 0.2},
]


@pytest.fixture
def spy(monkeypatch):
    """Replace retrieval and the chat model, recording what the chain sends them."""
    seen = {"retrieve_calls": [], "request": None}

    def fake_retrieve(query, k=3):
        seen["retrieve_calls"].append((query, k))
        return FAKE_CHUNKS

    def fake_create(**kwargs):
        seen["request"] = kwargs
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="FAKE ANSWER"))])

    monkeypatch.setattr(chain, "retrieve", fake_retrieve)
    monkeypatch.setattr(
        chain, "client",
        SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))),
    )
    return seen


def test_build_context_labels_each_chunk_with_its_source():
    context = chain.build_context(FAKE_CHUNKS)
    assert "[projects/store.md]" in context
    assert "YOLOv8n handles detection." in context
    assert "Built solo, end to end." in context


def test_answer_returns_the_model_reply(spy):
    assert chain.answer("What detects objects?") == "FAKE ANSWER"


def test_request_starts_with_the_grounding_system_prompt(spy):
    chain.answer("What detects objects?")
    first = spy["request"]["messages"][0]
    assert first["role"] == "system"
    assert "ONLY" in first["content"]


def test_last_message_carries_retrieved_context_and_the_question(spy):
    chain.answer("What detects objects?")
    last = spy["request"]["messages"][-1]
    assert last["role"] == "user"
    assert "YOLOv8n handles detection." in last["content"]
    assert "What detects objects?" in last["content"]


def test_history_sits_between_system_prompt_and_new_question(spy):
    history = [
        {"role": "user", "content": "Which project?"},
        {"role": "assistant", "content": "The Store Intelligence System."},
    ]
    chain.answer("What stack did it use?", history=history)
    roles = [m["role"] for m in spy["request"]["messages"]]
    assert roles == ["system", "user", "assistant", "user"]
    assert spy["request"]["messages"][1:3] == history


def test_retrieval_uses_only_the_latest_message(spy):
    # Documents a known limitation: follow-ups are not rewritten into standalone queries.
    history = [{"role": "user", "content": "Which project?"}, {"role": "assistant", "content": "SIS."}]
    chain.answer("What stack did it use?", history=history)
    assert spy["retrieve_calls"] == [("What stack did it use?", chain.TOP_K)]


def test_k_is_passed_through_to_retrieval(spy):
    chain.answer("q", k=5)
    assert spy["retrieve_calls"][0][1] == 5
