"""Strategy adapters: a uniform interface over the running API's search and
chat endpoints, so the eval runner doesn't care whether it's exercising
vector retrieval, hybrid retrieval, or (later) the SQL-as-a-tool feature.

A `Strategy` returns a `StrategyResult` containing:
- the ordered list of retrieved chunk dicts (with `chunk_id` and `content`)
- the optional final answer string (set only for chat-based strategies)
- elapsed latency in milliseconds

Strategies must be deterministic for retrieval-only modes (limit and ordering
fully specified by the caller). Chat-based strategies can be non-deterministic
because the LLM samples; eval runs that include them should be repeated and
averaged where possible.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol

import httpx


DEFAULT_API_URL = "http://localhost:8058"


@dataclass
class StrategyResult:
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    answer: Optional[str] = None
    latency_ms: float = 0.0
    raw: Optional[Dict[str, Any]] = None


class Strategy(Protocol):
    name: str

    def run(self, question: str, *, k: int) -> StrategyResult: ...


# ---------------------------------------------------------------------------
# Concrete strategies
# ---------------------------------------------------------------------------


@dataclass
class HttpSearchStrategy:
    """Adapter for /search/vector and /search/hybrid.

    Retrieval-only: leaves `answer` as None.
    """

    name: str
    endpoint: str  # e.g. "/search/vector"
    base_url: str = DEFAULT_API_URL
    timeout_s: float = 60.0

    def run(self, question: str, *, k: int) -> StrategyResult:
        payload = {"query": question, "limit": k}
        t0 = time.perf_counter()
        with httpx.Client(timeout=self.timeout_s) as client:
            resp = client.post(f"{self.base_url}{self.endpoint}", json=payload)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        resp.raise_for_status()
        body = resp.json()
        chunks = [
            {
                "chunk_id": r.get("chunk_id"),
                "content": r.get("content", ""),
                "score": r.get("score"),
                "document_title": r.get("document_title"),
            }
            for r in body.get("results", [])
        ]
        return StrategyResult(chunks=chunks, latency_ms=latency_ms, raw=body)


@dataclass
class HttpChatStrategy:
    """Adapter for /chat. Captures the agent's final answer plus any tool
    calls reported in the response, so we can still score retrieval recall
    against the chunks the agent actually pulled in.

    Note that the upstream /chat response currently returns sources=[] even
    when tools were used; until that's fixed, retrieval metrics for chat
    runs are best-effort (we fall back to scoring the answer text against
    the gold keywords / numeric expectations).
    """

    name: str
    base_url: str = DEFAULT_API_URL
    timeout_s: float = 180.0
    chat_payload: Dict[str, Any] = field(default_factory=dict)

    def run(self, question: str, *, k: int) -> StrategyResult:
        payload = {"message": question, **self.chat_payload}
        t0 = time.perf_counter()
        with httpx.Client(timeout=self.timeout_s) as client:
            resp = client.post(f"{self.base_url}/chat", json=payload)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        resp.raise_for_status()
        body = resp.json()
        # `retrieved_chunks` is the surface populated from the agent's
        # ToolReturnPart messages (the chunks it actually pulled in via
        # vector_search / hybrid_search). `sources` was the original
        # upstream field but it was always empty -- kept here as a
        # fallback for compatibility with older API builds.
        retrieved = body.get("retrieved_chunks") or body.get("sources") or []
        chunks = [
            {
                "chunk_id": s.get("chunk_id"),
                "content": s.get("content", ""),
                "score": s.get("score"),
                "document_title": s.get("document_title"),
            }
            for s in retrieved[:k]
        ]
        return StrategyResult(
            chunks=chunks,
            answer=body.get("message"),
            latency_ms=latency_ms,
            raw=body,
        )


# ---------------------------------------------------------------------------
# Registry: names used in benchmark.yaml / run_eval.py
# ---------------------------------------------------------------------------


def default_registry(base_url: str = DEFAULT_API_URL) -> Dict[str, Callable[[], Strategy]]:
    """Map of strategy name -> factory.

    Lazy factories let the runner accept a CLI arg like `--strategy hybrid`
    and instantiate only the requested strategy.
    """

    return {
        "vector": lambda: HttpSearchStrategy(
            name="vector", endpoint="/search/vector", base_url=base_url
        ),
        "hybrid": lambda: HttpSearchStrategy(
            name="hybrid", endpoint="/search/hybrid", base_url=base_url
        ),
        "chat": lambda: HttpChatStrategy(name="chat", base_url=base_url),
    }
