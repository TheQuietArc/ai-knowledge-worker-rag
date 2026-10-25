"""Answer evaluation: judge-output parsing, question selection, resumable runs."""

import json
from types import SimpleNamespace

import pytest

import eval.answer_eval as ae


def reply(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def judge_reply(entries):
    return reply("```json\n" + json.dumps(entries) + "\n```")


def score(i, acc=5):
    return {"id": i, "accuracy": acc, "completeness": 4, "relevance": 5, "feedback": f"fb{i}"}


class FakeRateLimit(Exception):
    """Stand-in for openai.RateLimitError (avoids version-specific constructors)."""


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr(ae.time, "sleep", lambda s: None)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Point run storage and the index-exists check at a temp folder."""
    monkeypatch.setattr(ae, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(ae, "CHROMA_PATH", str(tmp_path))
    return tmp_path


def use_judge(monkeypatch, handler):
    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=handler)))
    monkeypatch.setattr(ae, "client", fake)


def batch_judge_everything(**kwargs):
    ids = [int(line.split(": ")[1]) for line in kwargs["messages"][1]["content"].splitlines() if line.startswith("ID: ")]
    return judge_reply([score(i) for i in ids])


# ------------------------------------------------------------------ JSON extraction
def test_extract_json_array_handles_markdown_fences():
    assert ae._extract_json_array('```json\n[{"a": 1}]\n```') == [{"a": 1}]


def test_extract_json_array_handles_surrounding_prose():
    assert ae._extract_json_array('Sure! [{"a": 2}] hope that helps') == [{"a": 2}]


def test_extract_json_array_rejects_replies_without_an_array():
    with pytest.raises(ValueError):
        ae._extract_json_array("no json here")


# ------------------------------------------------------------------ judge_batch
ITEMS = [(1, ae.GOLDEN_SET[0], "a1"), (2, ae.GOLDEN_SET[1], "a2")]


def test_judge_batch_returns_scores_by_id(monkeypatch):
    use_judge(monkeypatch, lambda **kw: judge_reply([score(1), score(2, acc=3)]))
    result = ae.judge_batch(ITEMS)
    assert result[1].accuracy == 5 and result[2].accuracy == 3


def test_judge_batch_skips_invalid_and_unknown_entries(monkeypatch):
    entries = [score(1), score(2, acc=9), score(99)]  # 9 is out of range; 99 was never asked
    use_judge(monkeypatch, lambda **kw: judge_reply(entries))
    assert set(ae.judge_batch(ITEMS)) == {1}


def test_judge_batch_returns_empty_when_output_is_unusable(monkeypatch):
    use_judge(monkeypatch, lambda **kw: reply("sorry, I can't do that"))
    assert ae.judge_batch(ITEMS) == {}


# ------------------------------------------------------------------ question selection
def test_select_only_uses_one_based_positions():
    selected = ae.select_questions([6, 1], None)
    assert [n for n, _ in selected] == [6, 1]
    assert selected[0][1] is ae.GOLDEN_SET[5]


def test_select_limit_truncates():
    assert len(ae.select_questions(None, 3)) == 3


def test_select_rejects_out_of_range_numbers():
    with pytest.raises(ValueError):
        ae.select_questions([999], None)


# ------------------------------------------------------------------ run orchestration
def test_run_costs_one_call_per_answer_plus_one_judge_call(workspace, monkeypatch):
    answers, judge_calls = [], []
    monkeypatch.setattr(ae, "answer", lambda q: answers.append(q) or "A")
    use_judge(monkeypatch, lambda **kw: judge_calls.append(1) or batch_judge_everything(**kw))

    state = ae.run(tag="t", only=[1, 6, 18])

    assert len(answers) == 3 and len(judge_calls) == 1
    assert len(state["scores"]) == 3


def test_rerunning_a_finished_run_makes_no_new_calls(workspace, monkeypatch):
    answers, judge_calls = [], []
    monkeypatch.setattr(ae, "answer", lambda q: answers.append(q) or "A")
    use_judge(monkeypatch, lambda **kw: judge_calls.append(1) or batch_judge_everything(**kw))

    ae.run(tag="t", only=[1, 2])
    answers.clear(); judge_calls.clear()
    ae.run(tag="t", only=[1, 2])

    assert answers == [] and judge_calls == []


def test_daily_quota_stops_cleanly_and_resumes_later(workspace, monkeypatch):
    monkeypatch.setattr(ae.openai, "RateLimitError", FakeRateLimit)
    use_judge(monkeypatch, batch_judge_everything)

    calls = {"n": 0}

    def quota_after_two(question):
        calls["n"] += 1
        if calls["n"] == 3:
            raise FakeRateLimit("GenerateRequestsPerDayPerProjectPerModel-FreeTier")
        return "A"

    monkeypatch.setattr(ae, "answer", quota_after_two)
    state = ae.run(tag="t", only=[1, 2, 3, 4])
    assert len(state["answers"]) == 2 and state["scores"] == {}
    saved = json.loads((workspace / "runs" / "t.json").read_text())
    assert len(saved["answers"]) == 2

    resumed = []                                   # "the next day"
    monkeypatch.setattr(ae, "answer", lambda q: resumed.append(q) or "A")
    state = ae.run(tag="t", only=[1, 2, 3, 4])
    assert len(resumed) == 2                       # only the two missing answers
    assert len(state["scores"]) == 4


def test_unusable_judge_output_keeps_answers_and_does_not_crash(workspace, monkeypatch):
    monkeypatch.setattr(ae, "answer", lambda q: "A")
    use_judge(monkeypatch, lambda **kw: reply("garbage"))
    state = ae.run(tag="t", only=[2])
    assert len(state["answers"]) == 1 and state["scores"] == {}


def test_run_refuses_to_start_without_an_index(tmp_path, monkeypatch):
    monkeypatch.setattr(ae, "CHROMA_PATH", str(tmp_path / "missing"))
    with pytest.raises(RuntimeError):
        ae.run(tag="t")
