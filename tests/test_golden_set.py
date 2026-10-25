"""The evaluation test set must itself be valid, or every metric built on it is suspect."""

import pytest
from pydantic import ValidationError

from eval.golden_set import GOLDEN_SET, TestQuestion
from eval.retrieval_eval import validate_golden_set

ALLOWED_CATEGORIES = {"direct_fact", "multi_fact", "descriptive"}


def test_golden_set_is_not_empty():
    assert len(GOLDEN_SET) >= 10


def test_questions_are_unique():
    questions = [q.question for q in GOLDEN_SET]
    assert len(questions) == len(set(questions))


def test_categories_are_known():
    assert {q.category for q in GOLDEN_SET} <= ALLOWED_CATEGORIES


def test_keywords_are_normalised_to_lowercase():
    for q in GOLDEN_SET:
        assert all(kw == kw.lower().strip() for kw in q.keywords)


def test_every_keyword_exists_in_the_documents():
    # A keyword that appears nowhere can never be retrieved and silently drags scores down.
    assert validate_golden_set(GOLDEN_SET) == []


def test_model_rejects_a_question_with_no_keywords():
    with pytest.raises(ValidationError):
        TestQuestion(question="q", keywords=[], reference_answer="a", category="direct_fact")


def test_model_lowercases_keywords():
    q = TestQuestion(question="q", keywords=["  FastAPI "], reference_answer="a", category="direct_fact")
    assert q.keywords == ["fastapi"]
