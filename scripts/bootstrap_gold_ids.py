"""Bootstrap gold_chunk_ids for benchmark questions that have none yet.

For each question without strict gold IDs, this script:
  1. Hits /search/hybrid at k=8
  2. For every returned chunk, checks whether every keyword in
     gold_chunk_keywords appears in the chunk content (case-insensitive)
  3. Writes a side-car YAML report (eval/_bootstrap_candidates.yaml)
     listing the candidate chunks per question, with the keyword-match
     verdict and the first 240 chars of content for quick visual review.

The script does NOT mutate benchmark.yaml. After review, the human
edits benchmark.yaml manually based on the candidates.

Usage (from host, with the API stack up on localhost:8058):

    python scripts/bootstrap_gold_ids.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import httpx
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK = REPO_ROOT / "eval" / "benchmark.yaml"
OUTPUT = REPO_ROOT / "eval" / "_bootstrap_candidates.yaml"
API = "http://localhost:8058"


def search_hybrid(query: str, k: int = 8) -> list[dict[str, Any]]:
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(f"{API}/search/hybrid", json={"query": query, "limit": k})
    resp.raise_for_status()
    return resp.json().get("results", [])


def keyword_match(content: str, keywords: list[str]) -> bool:
    if not keywords:
        return False
    lc = content.lower()
    return all(k.lower() in lc for k in keywords)


def main() -> int:
    benchmark = yaml.safe_load(BENCHMARK.read_text())
    questions = benchmark.get("questions") or []

    out: dict[str, Any] = {"questions": []}

    for q in questions:
        qid = q.get("id")
        category = q.get("category")
        gold_ids = q.get("gold_chunk_ids") or []
        keywords = q.get("gold_chunk_keywords") or []

        # Skip questions that already have strict gold IDs.
        if gold_ids:
            print(f"[skip] {qid}: already labelled ({len(gold_ids)} ids)")
            continue
        # Negative questions intentionally have no gold; nothing to bootstrap.
        if category == "negative":
            print(f"[skip] {qid}: negative category")
            continue

        print(f"[run]  {qid} -> /search/hybrid")
        results = search_hybrid(q["question"], k=8)

        candidates = []
        for r in results:
            content = r.get("content", "") or ""
            candidates.append(
                {
                    "chunk_id": r.get("chunk_id"),
                    "score": r.get("score"),
                    "matches_all_keywords": keyword_match(content, keywords),
                    "preview": content[:240].replace("\n", " ").strip(),
                }
            )

        out["questions"].append(
            {
                "id": qid,
                "category": category,
                "question": q.get("question"),
                "gold_chunk_keywords": keywords,
                "expected_numbers": q.get("expected_numbers") or [],
                "candidates": candidates,
            }
        )

    OUTPUT.write_text(yaml.safe_dump(out, sort_keys=False, allow_unicode=True))
    print(f"\nWrote {OUTPUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
