"""Eval runner.

Usage (from the host, with the API stack up on localhost:8058):

    python -m eval.run_eval --strategy vector
    python -m eval.run_eval --strategy hybrid
    python -m eval.run_eval --strategy chat --k 5

Output goes to stdout as a markdown report and is also written to
eval/results/<timestamp>__<strategy>.{md,json} so historical runs are
reproducible side-by-side.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .judge import JudgeCache, score_answer
from .metrics import (
    QuestionScore,
    aggregate,
    keyword_recall_at_k,
    keyword_reciprocal_rank,
    numeric_match,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from .strategies import DEFAULT_API_URL, default_registry


REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO_ROOT / "eval"
RESULTS_DIR = EVAL_DIR / "results"
JUDGE_CACHE_PATH = RESULTS_DIR / ".judge_cache.json"


# ---------------------------------------------------------------------------
# Per-question scoring
# ---------------------------------------------------------------------------


def score_question(
    question: Dict[str, Any],
    chunks: List[Dict[str, Any]],
    answer: Optional[str],
    k: int,
    latency_ms: float,
) -> QuestionScore:
    gold_ids = question.get("gold_chunk_ids") or []
    keywords = question.get("gold_chunk_keywords") or []
    expected_numbers = question.get("expected_numbers") or []
    retrieved_ids = [c["chunk_id"] for c in chunks if c.get("chunk_id")]

    # Negative questions: no gold signal -> retrieval metrics undefined,
    # answer scoring delegated to LLM judge in a later stage.
    is_negative = not gold_ids and not keywords

    if gold_ids:
        recall = recall_at_k(retrieved_ids, gold_ids, k)
        precision = precision_at_k(retrieved_ids, gold_ids, k)
        rr = reciprocal_rank(retrieved_ids, gold_ids)
    elif keywords:
        recall = keyword_recall_at_k(chunks, keywords, k)
        # precision@k is ill-defined for keyword-mode; report None.
        precision = None
        rr = keyword_reciprocal_rank(chunks, keywords)
    else:
        # Negative question: every retrieved chunk is "not gold" by
        # construction, so we leave retrieval scores undefined here and
        # rely on the LLM judge (later) to grade refusal vs hallucination.
        recall = None
        precision = None
        rr = 0.0

    nmatch: Optional[float] = None
    if answer is not None and expected_numbers:
        nmatch = numeric_match(answer, expected_numbers)

    return QuestionScore(
        question_id=question["id"],
        category=question.get("category", "uncategorized"),
        recall_at_k=recall,
        precision_at_k=precision,
        reciprocal_rank=rr,
        numeric_match=nmatch,
        latency_ms=latency_ms,
        answer=answer,
    )


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def _fmt(v: Optional[float], digits: int = 3) -> str:
    if v is None:
        return "—"
    return f"{v:.{digits}f}"


def render_markdown(
    strategy: str,
    k: int,
    per_question: List[QuestionScore],
    summary: Dict[str, Any],
    timestamp: str,
) -> str:
    lines: List[str] = []
    lines.append(f"# Eval run — `{strategy}` (k={k})")
    lines.append("")
    lines.append(f"_Run at {timestamp}._")
    lines.append("")

    overall = summary["overall"]
    lines.append("## Overall")
    lines.append("")
    lines.append(
        "| n | recall@k | precision@k | MRR | numeric-match | faithfulness | relevance | avg latency (ms) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    lines.append(
        "| {n} | {r} | {p} | {m} | {nm} | {f} | {rel} | {lat} |".format(
            n=overall["n"],
            r=_fmt(overall["recall_at_k"]),
            p=_fmt(overall["precision_at_k"]),
            m=_fmt(overall["mrr"]),
            nm=_fmt(overall["numeric_match"]),
            f=_fmt(overall.get("faithfulness")),
            rel=_fmt(overall.get("answer_relevance")),
            lat=_fmt(overall["avg_latency_ms"], digits=0),
        )
    )
    lines.append("")

    lines.append("## Per category")
    lines.append("")
    lines.append(
        "| category | n | recall@k | precision@k | MRR | numeric-match | faithfulness | relevance |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for cat, b in sorted(summary["by_category"].items()):
        lines.append(
            "| {c} | {n} | {r} | {p} | {m} | {nm} | {f} | {rel} |".format(
                c=cat,
                n=b["n"],
                r=_fmt(b["recall_at_k"]),
                p=_fmt(b["precision_at_k"]),
                m=_fmt(b["mrr"]),
                nm=_fmt(b["numeric_match"]),
                f=_fmt(b.get("faithfulness")),
                rel=_fmt(b.get("answer_relevance")),
            )
        )
    lines.append("")

    lines.append("## Per question")
    lines.append("")
    lines.append(
        "| id | category | recall@k | precision@k | RR | numeric-match | faithfulness | relevance | latency ms |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for s in per_question:
        lines.append(
            "| {id} | {c} | {r} | {p} | {rr} | {nm} | {f} | {rel} | {lat} |".format(
                id=s.question_id,
                c=s.category,
                r=_fmt(s.recall_at_k),
                p=_fmt(s.precision_at_k),
                rr=_fmt(s.reciprocal_rank),
                nm=_fmt(s.numeric_match),
                f=_fmt(s.faithfulness),
                rel=_fmt(s.answer_relevance),
                lat=_fmt(s.latency_ms, digits=0),
            )
        )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the RAG eval benchmark.")
    parser.add_argument(
        "--strategy",
        required=True,
        help="Strategy name (e.g. vector, hybrid, chat). See strategies.default_registry.",
    )
    parser.add_argument("--k", type=int, default=5, help="Retrieval top-k (default: 5).")
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=EVAL_DIR / "benchmark.yaml",
        help="Path to the benchmark yaml.",
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help="Base URL of the running API (default: %(default)s).",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Print to stdout but skip writing to eval/results/.",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Score each answer with the LLM-as-judge (faithfulness + relevance). "
        "Requires the strategy to produce an answer (e.g. --strategy chat). "
        "Skipped silently for retrieval-only strategies.",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Override JUDGE_MODEL just for this run.",
    )
    args = parser.parse_args(argv)

    registry = default_registry(base_url=args.api_url)
    if args.strategy not in registry:
        print(
            f"Unknown strategy '{args.strategy}'. Known: {sorted(registry)}",
            file=sys.stderr,
        )
        return 2
    strategy = registry[args.strategy]()

    benchmark = yaml.safe_load(args.benchmark.read_text())
    questions = benchmark.get("questions") or []
    if not questions:
        print(f"Benchmark {args.benchmark} has no questions.", file=sys.stderr)
        return 2

    judge_cache: Optional[JudgeCache] = None
    if args.judge:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        judge_cache = JudgeCache(JUDGE_CACHE_PATH)

    per_question: List[QuestionScore] = []
    raw_per_question: List[Dict[str, Any]] = []
    for q in questions:
        result = strategy.run(q["question"], k=args.k)
        score = score_question(
            question=q,
            chunks=result.chunks,
            answer=result.answer,
            k=args.k,
            latency_ms=result.latency_ms,
        )

        if args.judge and result.answer:
            judged = score_answer(
                question=q["question"],
                answer=result.answer,
                chunks=result.chunks,
                category=q.get("category", "general"),
                model=args.judge_model,
                cache=judge_cache,
            )
            score.faithfulness = judged.faithfulness
            score.answer_relevance = judged.answer_relevance
            score.judge_cached = judged.cached

        per_question.append(score)
        raw_per_question.append(
            {
                "score": asdict(score),
                "retrieved_chunk_ids": [c.get("chunk_id") for c in result.chunks],
            }
        )

    summary = aggregate(per_question)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    md = render_markdown(args.strategy, args.k, per_question, summary, timestamp)
    print(md)

    if not args.no_write:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        stem = f"{timestamp}__{args.strategy}__k{args.k}"
        (RESULTS_DIR / f"{stem}.md").write_text(md)
        (RESULTS_DIR / f"{stem}.json").write_text(
            json.dumps(
                {
                    "strategy": args.strategy,
                    "k": args.k,
                    "timestamp": timestamp,
                    "summary": summary,
                    "per_question": raw_per_question,
                },
                indent=2,
            )
        )
        print(f"\n_Results written to {RESULTS_DIR.relative_to(REPO_ROOT)}/{stem}.{{md,json}}_")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
