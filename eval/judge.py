"""LLM-as-judge metrics: faithfulness and answer relevance.

Two scores, both in [0.0, 1.0]:

* faithfulness    -- how well the answer is grounded in the retrieved
                     passages. 1.0 = every factual claim is supported,
                     0.0 = answer contradicts or fabricates.
* answer_relevance -- how directly the answer addresses the question.
                     1.0 = on-target, 0.0 = unrelated or non-answer.

Calls go through the same OpenAI-compatible endpoint as the rest of the
stack (set via OPENAI_BASE_URL), so by default the judge talks to the
local Ollama. The model is configurable via JUDGE_MODEL so it can (and
should) differ from the chat model used to generate the answer --
self-grading is a known biased setting.

Results are cached on disk by sha256(question + answer + chunks +
model + prompt_version). A rerun with the same inputs costs nothing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from openai import OpenAI


logger = logging.getLogger(__name__)


# Bumping this invalidates every cached score, so do it whenever a prompt
# below is meaningfully edited.
JUDGE_PROMPT_VERSION = "v1"


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


FAITHFULNESS_PROMPT = """You are an impartial judge evaluating whether an AI assistant's answer is faithful to a set of source passages from a corporate sustainability report.

Source passages (separated by `---`):
---
{passages}
---

AI assistant's answer:
---
{answer}
---

Question category hint: {category}

Score the answer's faithfulness on a continuous 0.0-1.0 scale:
- 1.0  Every factual claim in the answer is directly supported by at least one of the source passages.
- 0.7  Most claims are supported; minor unsupported phrasing but no fabricated numbers or facts.
- 0.5  Mix of supported and unsupported claims, or the answer overgeneralises beyond the passages.
- 0.3  Several claims have no basis in the passages.
- 0.0  The answer contradicts the passages or fabricates entire facts.

Special cases:
- For category="negative", a refusal ("the report does not state...", "I cannot find...") is correct and scores 1.0.
- An empty or off-topic answer scores 0.0.

Respond with ONLY a single JSON object on one line, no surrounding prose:
{{"score": <float between 0.0 and 1.0>, "reason": "<one short sentence>"}}"""


RELEVANCE_PROMPT = """You are an impartial judge evaluating whether an AI assistant's answer addresses the user's question.

User question: {question}

AI assistant's answer:
---
{answer}
---

Question category hint: {category}

Score the answer's relevance on a continuous 0.0-1.0 scale:
- 1.0  The answer directly addresses the question with specific information.
- 0.7  The answer is on-topic and largely addresses the question, but is partial or buries the key fact.
- 0.5  The answer is on-topic but vague or tangential.
- 0.3  The answer barely addresses the question; mostly off-topic.
- 0.0  The answer is unrelated to the question.

Special cases:
- For category="negative" (the asked-about information genuinely isn't in the source corpus), an honest refusal IS the correct answer and scores 1.0. Confidently fabricating an answer for a negative question scores 0.0 on relevance even if it sounds plausible.

Respond with ONLY a single JSON object on one line, no surrounding prose:
{{"score": <float between 0.0 and 1.0>, "reason": "<one short sentence>"}}"""


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class JudgeCache:
    """Thin disk-backed JSON cache for judge calls.

    Keyed on a content hash so identical (question, answer, passages,
    model, prompt_version) tuples reuse prior scores.
    """

    def __init__(self, path: Path):
        self.path = path
        self.data: Dict[str, Dict[str, Any]] = {}
        if path.exists():
            try:
                self.data = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to load judge cache at %s: %s", path, exc)
                self.data = {}

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        return self.data.get(key)

    def set(self, key: str, value: Dict[str, Any]) -> None:
        self.data[key] = value
        self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, sort_keys=True))
        tmp.replace(self.path)


def _make_cache_key(
    question: str,
    answer: str,
    chunks: Sequence[Dict[str, Any]],
    model: str,
    version: str,
) -> str:
    canon = json.dumps(
        {
            "q": question,
            "a": answer,
            # Truncate chunk content so trivial whitespace differences don't
            # invalidate the cache, but enough text to fingerprint each chunk.
            "c": [str(c.get("content", ""))[:1000] for c in chunks],
            "model": model,
            "version": version,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Score parsing
# ---------------------------------------------------------------------------


_SCORE_RE = re.compile(r'"?score"?\s*[:=]\s*([0-9]*\.?[0-9]+)', re.IGNORECASE)


def _extract_score(raw: Optional[str]) -> Optional[float]:
    """Pull a numeric score out of the judge's reply.

    Tries strict JSON first, then a permissive regex, so we don't fail an
    entire eval run because the judge added stray prose around a JSON
    object.
    """
    if not raw:
        return None
    text = raw.strip()
    # Strip ```json fences if the model wraps its answer.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "score" in obj:
            return _clamp_unit(float(obj["score"]))
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    m = _SCORE_RE.search(text)
    if m:
        try:
            return _clamp_unit(float(m.group(1)))
        except ValueError:
            return None
    return None


def _clamp_unit(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


@dataclass
class JudgeScores:
    faithfulness: Optional[float]
    answer_relevance: Optional[float]
    raw_faithfulness: Optional[str] = None
    raw_relevance: Optional[str] = None
    cached: bool = False


def _make_client(base_url: Optional[str], api_key: Optional[str]) -> OpenAI:
    return OpenAI(
        base_url=base_url or os.getenv("OPENAI_BASE_URL"),
        api_key=api_key or os.getenv("OPENAI_API_KEY") or "no-key",
    )


def _resolve_model(explicit: Optional[str]) -> str:
    """Pick the judge model, preferring the dedicated env var.

    Falls back to LLM_CHOICE so the framework runs out of the box, but
    logs a warning because using the same model for chat and judging is
    a known bias in self-grading.
    """
    if explicit:
        return explicit
    judge = os.getenv("JUDGE_MODEL")
    if judge:
        return judge
    chat = os.getenv("LLM_CHOICE", "qwen2.5:14b")
    logger.warning(
        "JUDGE_MODEL not set; falling back to LLM_CHOICE=%s. "
        "For unbiased grading, set JUDGE_MODEL to a different model.",
        chat,
    )
    return chat


def score_answer(
    question: str,
    answer: str,
    chunks: Sequence[Dict[str, Any]],
    *,
    category: str = "general",
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    cache: Optional[JudgeCache] = None,
    max_passages: int = 5,
    timeout_s: float = 120.0,
) -> JudgeScores:
    """Score a single (question, answer, retrieved-chunks) triple.

    Returns NaN-free `JudgeScores`; either field is `None` if parsing the
    judge's reply failed (logged with the raw output for debugging).
    Successful scores are cached on disk.
    """
    if not answer:
        return JudgeScores(faithfulness=0.0, answer_relevance=0.0)

    resolved_model = _resolve_model(model)
    passages_for_cache = list(chunks[:max_passages])
    cache_key = _make_cache_key(
        question=question,
        answer=answer,
        chunks=passages_for_cache,
        model=resolved_model,
        version=JUDGE_PROMPT_VERSION,
    )

    if cache is not None:
        hit = cache.get(cache_key)
        if hit is not None:
            return JudgeScores(
                faithfulness=hit.get("faithfulness"),
                answer_relevance=hit.get("answer_relevance"),
                raw_faithfulness=hit.get("raw_faithfulness"),
                raw_relevance=hit.get("raw_relevance"),
                cached=True,
            )

    client = _make_client(base_url, api_key)
    passages_text = "\n\n---\n\n".join(
        str(c.get("content", "")) for c in passages_for_cache
    ) or "(no retrieved passages)"

    f_prompt = FAITHFULNESS_PROMPT.format(
        passages=passages_text, answer=answer, category=category
    )
    r_prompt = RELEVANCE_PROMPT.format(
        question=question, answer=answer, category=category
    )

    f_raw = _ask_judge(client, resolved_model, f_prompt, timeout_s=timeout_s)
    r_raw = _ask_judge(client, resolved_model, r_prompt, timeout_s=timeout_s)
    f_score = _extract_score(f_raw)
    r_score = _extract_score(r_raw)

    if f_score is None:
        logger.warning(
            "Could not parse faithfulness score from judge output: %r", f_raw
        )
    if r_score is None:
        logger.warning(
            "Could not parse relevance score from judge output: %r", r_raw
        )

    result = JudgeScores(
        faithfulness=f_score,
        answer_relevance=r_score,
        raw_faithfulness=f_raw,
        raw_relevance=r_raw,
    )

    if cache is not None and (f_score is not None or r_score is not None):
        cache.set(
            cache_key,
            {
                "faithfulness": f_score,
                "answer_relevance": r_score,
                "raw_faithfulness": f_raw,
                "raw_relevance": r_raw,
                "model": resolved_model,
                "prompt_version": JUDGE_PROMPT_VERSION,
            },
        )

    return result


def _ask_judge(
    client: OpenAI, model: str, prompt: str, *, timeout_s: float
) -> Optional[str]:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a careful, terse evaluation judge. "
                    "Always reply with a single JSON object on one line.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            timeout=timeout_s,
        )
        return resp.choices[0].message.content
    except Exception as exc:  # noqa: BLE001 -- judge errors must not kill the run
        logger.warning("Judge call failed (model=%s): %s", model, exc)
        return None


__all__ = [
    "JUDGE_PROMPT_VERSION",
    "JudgeCache",
    "JudgeScores",
    "score_answer",
]
