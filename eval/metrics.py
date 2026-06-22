"""Deterministic retrieval and answer-quality metrics.

These metrics are intentionally LLM-free so reruns are cheap and reproducible.
LLM-as-judge metrics (faithfulness, answer relevance) live in `judge.py` and
are scored separately, with caching keyed on (question, answer) so repeated
runs don't waste tokens.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional, Sequence


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------


def recall_at_k(retrieved_ids: Sequence[str], gold_ids: Iterable[str], k: int) -> float:
    """Fraction of gold IDs that appear in the top-k retrieved IDs.

    Returns 0.0 when there are no gold IDs (rather than raising), so a
    benchmark question with an empty gold set contributes nothing rather
    than blowing up the run.
    """
    gold_set = set(gold_ids)
    if not gold_set:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return len(gold_set & top_k) / len(gold_set)


def precision_at_k(retrieved_ids: Sequence[str], gold_ids: Iterable[str], k: int) -> float:
    """Fraction of the top-k retrieved IDs that are gold."""
    if k <= 0:
        return 0.0
    gold_set = set(gold_ids)
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    return sum(1 for rid in top_k if rid in gold_set) / len(top_k)


def reciprocal_rank(retrieved_ids: Sequence[str], gold_ids: Iterable[str]) -> float:
    """1 / rank of the first gold hit, or 0.0 if no gold ID was retrieved."""
    gold_set = set(gold_ids)
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in gold_set:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(per_question_rr: Sequence[float]) -> float:
    if not per_question_rr:
        return 0.0
    return sum(per_question_rr) / len(per_question_rr)


# ---------------------------------------------------------------------------
# Keyword-based fallback when ground-truth chunk IDs aren't bootstrapped yet
# ---------------------------------------------------------------------------
#
# Authoring `gold_chunk_ids` for every question requires a one-time human
# pass over the corpus. Until that's done we score a chunk as "gold" if its
# content contains all of a question's `gold_chunk_keywords`. This is a
# noisier signal than ID matching, but it lets us iterate on the framework
# without blocking on labelling work, and tends to be a strict superset of
# the ID-based recall (i.e. keyword-recall >= id-recall in practice).


def keyword_recall_at_k(
    retrieved_chunks: Sequence[Mapping[str, object]],
    gold_keywords: Iterable[str],
    k: int,
    content_field: str = "content",
) -> float:
    """1.0 if any of the top-k retrieved chunks contains every gold keyword.

    Treated as recall of a single "gold concept": the question is answered
    if the keyword cluster appears together in some retrieved chunk.
    """
    keywords = [kw.lower() for kw in gold_keywords if kw]
    if not keywords:
        return 0.0
    for chunk in retrieved_chunks[:k]:
        content = str(chunk.get(content_field, "")).lower()
        if all(kw in content for kw in keywords):
            return 1.0
    return 0.0


def keyword_reciprocal_rank(
    retrieved_chunks: Sequence[Mapping[str, object]],
    gold_keywords: Iterable[str],
    content_field: str = "content",
) -> float:
    keywords = [kw.lower() for kw in gold_keywords if kw]
    if not keywords:
        return 0.0
    for rank, chunk in enumerate(retrieved_chunks, start=1):
        content = str(chunk.get(content_field, "")).lower()
        if all(kw in content for kw in keywords):
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Answer-quality metrics
# ---------------------------------------------------------------------------


_NUMBER_RE = re.compile(r"-?\d+(?:[\.,]\d+)?")


def _normalise_number(token: str) -> str:
    """Strip trailing punctuation, unify decimal/grouping separators."""
    cleaned = token.strip().rstrip(".,;:")
    return cleaned.replace(",", "")


def numeric_match(answer_text: str, expected_numbers: Iterable[str]) -> float:
    """Fraction of expected numeric tokens that appear in the answer.

    Comparison is lexical on the canonicalised number form, so '68%' in the
    expected list matches '68%' or '68 %' in the answer but not '67%'.
    Unitless numbers ('33.4') match 'reduced by 33.4%'.
    """
    expected = [_normalise_number(e).rstrip("%") for e in expected_numbers if e]
    if not expected:
        return 0.0
    found_numbers = {_normalise_number(t).rstrip("%") for t in _NUMBER_RE.findall(answer_text)}
    matched = sum(1 for e in expected if e in found_numbers)
    return matched / len(expected)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


@dataclass
class QuestionScore:
    question_id: str
    category: str
    recall_at_k: Optional[float]
    precision_at_k: Optional[float]
    reciprocal_rank: float
    numeric_match: Optional[float]
    latency_ms: float
    answer: Optional[str] = None


def aggregate(scores: Sequence[QuestionScore]) -> dict:
    """Aggregate per-question scores into mean values + per-category means."""

    def _mean(vals: List[float]) -> Optional[float]:
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    overall = {
        "n": len(scores),
        "recall_at_k": _mean([s.recall_at_k for s in scores if s.recall_at_k is not None]),
        "precision_at_k": _mean([s.precision_at_k for s in scores if s.precision_at_k is not None]),
        "mrr": _mean([s.reciprocal_rank for s in scores]),
        "numeric_match": _mean([s.numeric_match for s in scores if s.numeric_match is not None]),
        "avg_latency_ms": _mean([s.latency_ms for s in scores]),
    }

    by_category: dict = {}
    for s in scores:
        bucket = by_category.setdefault(
            s.category,
            {"n": 0, "recall": [], "precision": [], "rr": [], "numeric": []},
        )
        bucket["n"] += 1
        if s.recall_at_k is not None:
            bucket["recall"].append(s.recall_at_k)
        if s.precision_at_k is not None:
            bucket["precision"].append(s.precision_at_k)
        bucket["rr"].append(s.reciprocal_rank)
        if s.numeric_match is not None:
            bucket["numeric"].append(s.numeric_match)

    summarised_categories = {
        cat: {
            "n": b["n"],
            "recall_at_k": _mean(b["recall"]),
            "precision_at_k": _mean(b["precision"]),
            "mrr": _mean(b["rr"]),
            "numeric_match": _mean(b["numeric"]),
        }
        for cat, b in by_category.items()
    }

    return {"overall": overall, "by_category": summarised_categories}
