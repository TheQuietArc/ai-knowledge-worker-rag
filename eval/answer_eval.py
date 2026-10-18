"""
Answer evaluation with an LLM-as-judge, built to survive a tight free-tier quota.

For each selected golden question we run the REAL production pipeline
(rag.chain.answer, which retrieves from the persisted ./chroma_db index), then ask a
judge model to grade ALL the answers in a single call against the reference answers,
on three 1-5 scales:

  accuracy      — is it factually consistent with the reference answer?
  completeness  — does it include all the key information in the reference?
  relevance     — does it stay on the question without padding or drifting?

Why it is built this way (Gemini's free tier allows ~20 chat requests PER DAY per model):
  * Cost per run is N answer calls + 1 judge call, not 2N.
  * Progress is saved to eval/runs/<tag>.json after every answer, so if the daily quota
    runs out the run stops cleanly and the SAME command resumes where it left off.
  * `--only` runs just the questions you pick, e.g. the ones retrieval struggled with.

Known limitations (worth saying out loud in an interview): the judge is the same model
family that wrote the answers, so it may be lenient toward its own style; and judging
many answers in one call is cheaper but slightly less careful than one call each. LLM
judges are a cheap, useful signal — not ground truth.
"""

import json
import os
import re
import time

import openai
from pydantic import BaseModel, Field, ValidationError

from rag.config import client, CHAT_MODEL, CHROMA_PATH
from rag.chain import answer
from rag.vectorstore import call_with_retry, is_daily_quota_error
from eval.golden_set import GOLDEN_SET, TestQuestion

RUNS_DIR = os.path.join(os.path.dirname(__file__), "runs")

# Free-tier chat is also limited to ~5 requests/minute, so pause between answer calls.
CALL_DELAY_SECONDS = 13


class JudgeScore(BaseModel):
    accuracy: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    relevance: int = Field(ge=1, le=5)
    feedback: str = Field(description="One or two sentences explaining the scores")


class BatchJudgeScore(JudgeScore):
    id: int


JUDGE_SYSTEM_PROMPT = (
    "You are a strict evaluator of question-answering systems. You will receive several "
    "items. Each has an ID, a question, a reference answer that is known to be correct, "
    "and a candidate answer. Grade every candidate against its reference. Reply with "
    "ONLY a JSON array, no markdown, with one object per item, each having exactly these "
    'keys: "id" (the item ID, integer), "accuracy" (1-5), "completeness" (1-5), '
    '"relevance" (1-5), "feedback" (string). '
    "Scoring guide: 5 = fully matches the reference; 3 = partially correct or missing "
    "key details; 1 = wrong or contradicts the reference. If a candidate says it does "
    "not know, its accuracy and completeness must be 1. Grade each item independently."
)


# ---------------------------------------------------------------- run persistence
def _run_path(tag: str) -> str:
    return os.path.join(RUNS_DIR, f"{tag}.json")


def load_run(tag: str) -> dict:
    path = _run_path(tag)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"answers": {}, "scores": {}}


def save_run(tag: str, run: dict) -> None:
    os.makedirs(RUNS_DIR, exist_ok=True)
    with open(_run_path(tag), "w", encoding="utf-8") as f:
        json.dump(run, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------- judging
def _extract_json_array(text: str) -> list:
    """Pull a JSON array out of the model's reply, tolerating markdown fences or stray
    text around it."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE).strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("no JSON array found in judge reply")
    return json.loads(text[start : end + 1])


def judge_batch(items: list[tuple[int, TestQuestion, str]]) -> dict[int, JudgeScore]:
    """Grade many answers in ONE model call. `items` is (id, question, candidate_answer).
    Returns {id: JudgeScore} for every item the judge scored validly; items it skipped
    or mangled are simply absent, so a later rerun can pick them up."""
    blocks = [
        f"ID: {item_id}\nQuestion: {q.question}\n"
        f"Reference answer: {q.reference_answer}\nCandidate answer: {candidate}"
        for item_id, q, candidate in items
    ]
    user_prompt = "\n\n---\n\n".join(blocks)
    wanted_ids = {item_id for item_id, _, _ in items}

    for attempt in range(2):
        response = call_with_retry(
            client.chat.completions.create,
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw = response.choices[0].message.content or ""
        try:
            scores: dict[int, JudgeScore] = {}
            for entry in _extract_json_array(raw):
                try:
                    parsed = BatchJudgeScore(**entry)
                except ValidationError:
                    continue  # skip one bad entry, keep the rest
                if parsed.id in wanted_ids:
                    scores[parsed.id] = JudgeScore(
                        accuracy=parsed.accuracy, completeness=parsed.completeness,
                        relevance=parsed.relevance, feedback=parsed.feedback,
                    )
            if scores:
                return scores
        except (ValueError, json.JSONDecodeError, TypeError):
            pass
        if attempt == 0:
            time.sleep(CALL_DELAY_SECONDS)
    return {}


# ---------------------------------------------------------------- orchestration
def select_questions(
    only: list[int] | None, limit: int | None
) -> list[tuple[int, TestQuestion]]:
    """Choose questions by their 1-based position in the golden set."""
    numbered = list(enumerate(GOLDEN_SET, start=1))
    if only:
        bad = [n for n in only if not 1 <= n <= len(GOLDEN_SET)]
        if bad:
            raise ValueError(f"--only numbers out of range 1-{len(GOLDEN_SET)}: {bad}")
        numbered = [(n, GOLDEN_SET[n - 1]) for n in only]
    if limit:
        numbered = numbered[:limit]
    return numbered


def run(tag: str = "default", only: list[int] | None = None, limit: int | None = None) -> dict:
    if not os.path.exists(CHROMA_PATH):
        raise RuntimeError(
            f"No vector store at ./{CHROMA_PATH}/. Run `python build_index.py` first — "
            "answer evaluation tests the real, persisted pipeline."
        )

    selected = select_questions(only, limit)
    state = load_run(tag)
    already = sum(1 for _, q in selected if q.question in state["answers"])
    to_generate = len(selected) - already
    print(f"Run '{tag}': {len(selected)} questions ({already} already answered, "
          f"{to_generate} to generate; judge adds 1 call if any are unscored).\n")

    # ---- Step 1: generate answers (resumable) ---------------------------------
    for n, q in selected:
        if q.question in state["answers"]:
            print(f"[#{n}] cached: {q.question}")
            continue
        print(f"[#{n}] {q.question}")
        try:
            state["answers"][q.question] = answer(q.question)
        except openai.RateLimitError as e:
            if is_daily_quota_error(e):
                save_run(tag, state)
                print(
                    "\nDaily free-tier quota for the chat model is used up. Progress is "
                    f"saved in eval/runs/{tag}.json. Rerun the SAME command after the "
                    "quota resets (midnight Pacific time) and it will resume."
                )
                return state
            raise
        save_run(tag, state)
        time.sleep(CALL_DELAY_SECONDS)

    # ---- Step 2: judge everything not yet scored, in one call -----------------
    pending = [
        (n, q, state["answers"][q.question])
        for n, q in selected
        if q.question not in state["scores"]
    ]
    if pending:
        print(f"\nJudging {len(pending)} answers in one call...")
        try:
            scored = judge_batch(pending)
        except openai.RateLimitError as e:
            if is_daily_quota_error(e):
                save_run(tag, state)
                print("\nDaily quota used up before judging. All answers are saved — "
                      "rerun the same command after the reset to judge them.")
                return state
            raise
        for n, q, _ in pending:
            if n in scored:
                state["scores"][q.question] = scored[n].model_dump()
        save_run(tag, state)
        missing = [n for n, _, _ in pending if n not in scored]
        if missing:
            print(f"Judge skipped questions {missing}; rerun the same command to retry them.")

    # ---- Step 3: report -------------------------------------------------------
    report(selected, state)
    return state


def report(selected: list[tuple[int, TestQuestion]], state: dict) -> None:
    rows = []
    for n, q in selected:
        score = state["scores"].get(q.question)
        if score:
            rows.append((n, q, score))
    if not rows:
        print("No judged answers to report yet.")
        return

    print("\n" + "=" * 60)
    for n, q, s in rows:
        print(f"[#{n}] acc={s['accuracy']} comp={s['completeness']} rel={s['relevance']}  "
              f"| {q.question}\n       {s['feedback']}")

    count = len(rows)
    avg = lambda key: sum(s[key] for _, _, s in rows) / count
    print("=" * 60)
    print(f"Answer quality over {count}/{len(selected)} judged questions (1-5 scale)")
    print(f"  accuracy:     {avg('accuracy'):.2f}")
    print(f"  completeness: {avg('completeness'):.2f}")
    print(f"  relevance:    {avg('relevance'):.2f}")

    weak = [(n, q, s) for n, q, s in rows if min(s["accuracy"], s["completeness"]) <= 3]
    if weak:
        print("\nWeak answers worth investigating:")
        for n, q, s in weak:
            print(f"  - [#{n}] {q.question}\n      answer: {state['answers'][q.question][:160]}\n"
                  f"      judge:  {s['feedback']}")
