"""History normalisation in the Gradio layer (Gradio versions differ in history format)."""

from app import normalize_history


def test_empty_history():
    assert normalize_history(None) == []
    assert normalize_history([]) == []


def test_old_pair_format_is_converted():
    history = [["Which project?", "The Store Intelligence System."]]
    assert normalize_history(history) == [
        {"role": "user", "content": "Which project?"},
        {"role": "assistant", "content": "The Store Intelligence System."},
    ]


def test_new_message_format_passes_through():
    history = [
        {"role": "user", "content": "Which project?"},
        {"role": "assistant", "content": "The Store Intelligence System."},
    ]
    assert normalize_history(history) == history


def test_pair_with_missing_assistant_reply_keeps_only_the_user_turn():
    assert normalize_history([["Hello?", None]]) == [{"role": "user", "content": "Hello?"}]


def test_extra_keys_in_message_dicts_are_dropped():
    history = [{"role": "user", "content": "hi", "metadata": {"x": 1}}]
    assert normalize_history(history) == [{"role": "user", "content": "hi"}]
