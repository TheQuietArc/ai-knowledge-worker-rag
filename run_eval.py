"""
Run the Day 4 evaluations.

  python run_eval.py retrieval
      Compare chunking/retrieval configs. Embeddings only — fast, cheap.

  python run_eval.py answers --tag baseline --only 1,6,7,10,14,18
      Run the real pipeline on the chosen questions (numbers = position in
      eval/golden_set.py) and score the answers with an LLM judge. Results are saved
      to eval/runs/baseline.json; rerunning the same command resumes after a quota stop.

  python run_eval.py answers --tag large           # all 18 questions
  python run_eval.py answers --tag baseline --limit 5

Gemini's free tier allows ~20 chat requests/day per model. A run costs
(number of questions + 1) calls, so a 6-question run costs 7.

Run `python build_index.py` before `answers` — it evaluates the persisted index.
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Evaluate the RAG system.")
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("retrieval", help="compare chunking/retrieval configurations")

    answers = sub.add_parser("answers", help="score generated answers with an LLM judge")
    answers.add_argument("--tag", default="default",
                         help="name for this run; results saved to eval/runs/<tag>.json")
    answers.add_argument("--only", type=lambda s: [int(x) for x in s.split(",")],
                         default=None, help="comma-separated question numbers, e.g. 1,6,18")
    answers.add_argument("--limit", type=int, default=None, help="only the first N selected")
    args = parser.parse_args()

    if args.mode == "retrieval":
        from eval import retrieval_eval
        results = retrieval_eval.run()
        out_path = os.path.join("eval", "results_retrieval.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(retrieval_eval.format_results_table(results) + "\n")
        print(f"\nSaved table to {out_path}")
    else:
        from eval import answer_eval
        answer_eval.run(tag=args.tag, only=args.only, limit=args.limit)


if __name__ == "__main__":
    main()
