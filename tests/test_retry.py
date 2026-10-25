"""Retry behaviour: which API failures are retried, which fail fast."""

import openai
import pytest

import rag.vectorstore as vs


class Transient(Exception):
    """Stand-in for a retryable API error, so these tests don't depend on how the
    installed openai version constructs its exception classes."""


@pytest.fixture
def sleeps(monkeypatch):
    recorded = []
    monkeypatch.setattr(vs.time, "sleep", lambda seconds: recorded.append(seconds))
    monkeypatch.setattr(vs, "RETRYABLE_ERRORS", (Transient,))
    return recorded


def test_daily_quota_errors_are_recognised():
    assert vs.is_daily_quota_error(Exception("quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier"))
    assert vs.is_daily_quota_error(Exception("You exceeded the limit of 20 requests per day"))


def test_per_minute_quota_errors_are_not_daily():
    assert not vs.is_daily_quota_error(Exception("quotaId: GenerateRequestsPerMinutePerProjectPerModel-FreeTier"))


def test_success_needs_no_retry(sleeps):
    assert vs.call_with_retry(lambda: "ok") == "ok"
    assert sleeps == []


def test_transient_errors_are_retried_with_growing_waits(sleeps):
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise Transient("try again")
        return "ok"

    assert vs.call_with_retry(flaky) == "ok"
    assert sleeps == [15, 30]


def test_gives_up_after_max_retries(sleeps, monkeypatch):
    monkeypatch.setattr(vs, "MAX_RETRIES", 3)

    def always_fails():
        raise Transient("still down")

    with pytest.raises(Transient):
        vs.call_with_retry(always_fails)
    assert sleeps == [15, 30]


def test_daily_quota_error_fails_immediately(sleeps):
    calls = []

    def exhausted():
        calls.append(1)
        raise Transient("GenerateRequestsPerDayPerProjectPerModel-FreeTier")

    with pytest.raises(Transient):
        vs.call_with_retry(exhausted)
    assert len(calls) == 1 and sleeps == []


def test_non_retryable_errors_propagate_without_waiting(sleeps):
    def bug():
        raise ValueError("a real bug, retrying won't help")

    with pytest.raises(ValueError):
        vs.call_with_retry(bug)
    assert sleeps == []


def test_real_openai_error_types_are_registered_as_retryable():
    # Guards against someone narrowing the production list by accident.
    real = vs.RETRYABLE_ERRORS
    assert openai.RateLimitError in real
    assert openai.InternalServerError in real
    assert openai.APIConnectionError in real
