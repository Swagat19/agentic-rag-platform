"""Auto-label gold_chunk_ids for benchmark questions by ILIKE-matching chunk content.

For each unlabeled (non-negative) question, this script issues a SQL
query against `chunks` that requires every keyword in
`gold_chunk_keywords` AND every value in `expected_numbers` to appear
in the chunk content (case-insensitive substring). Any chunk that
matches is a strong candidate for the gold label.

This labelling is independent of the SQL-as-tool feature: it never
consults `kpi_facts` and never uses the agent or LLM. The eval thus
remains fair — the gold IDs are derived from the raw chunk text, the
same surface that vector and hybrid retrieval can see.

Usage:

    python scripts/label_gold_ids.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import asyncio
import asyncpg
import yaml
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK = REPO_ROOT / "eval" / "benchmark.yaml"
OUTPUT = REPO_ROOT / "eval" / "_gold_id_proposals.yaml"


def _build_dsn() -> str:
    load_dotenv()
    user = os.getenv("DB_USER", "postgres")
    password = os.getenv("DB_PASSWORD", "postgres")
    host = os.getenv("DB_HOST", "postgres")
    port = os.getenv("DB_PORT", "5432")
    name = os.getenv("DB_NAME", "postgres")
    return f"postgresql://{user}:{password}@{host}:{port}/{name}"


async def find_matches(
    conn: asyncpg.Connection,
    keywords: list[str],
    numbers: list[str],
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Return chunks containing every keyword AND every expected number."""
    needles = list(keywords) + list(numbers)
    if not needles:
        return []
    where_clauses = []
    params: list[Any] = []
    for n in needles:
        params.append(f"%{n}%")
        where_clauses.append(f"content ILIKE ${len(params)}")
    where_sql = " AND ".join(where_clauses)
    params.append(limit)
    sql = (
        "SELECT id::text AS chunk_id, "
        "       substring(content, 1, 280) AS preview, "
        "       length(content) AS len "
        f"FROM chunks WHERE {where_sql} "
        f"ORDER BY length(content) DESC LIMIT ${len(params)}"
    )
    rows = await conn.fetch(sql, *params)
    return [
        {
            "chunk_id": r["chunk_id"],
            "len": r["len"],
            "preview": (r["preview"] or "").replace("\n", " ").strip(),
        }
        for r in rows
    ]


async def main() -> int:
    benchmark = yaml.safe_load(BENCHMARK.read_text())
    questions = benchmark.get("questions") or []

    out: dict[str, Any] = {"questions": []}
    conn = await asyncpg.connect(_build_dsn())
    try:
        for q in questions:
            qid = q.get("id")
            category = q.get("category")
            if (q.get("gold_chunk_ids") or []):
                print(f"[skip] {qid}: already labelled")
                continue
            if category == "negative":
                print(f"[skip] {qid}: negative")
                continue
            keywords = q.get("gold_chunk_keywords") or []
            numbers = [str(n) for n in (q.get("expected_numbers") or [])]
            matches = await find_matches(conn, keywords, numbers)
            print(f"[probe] {qid}: {len(matches)} matches")
            out["questions"].append(
                {
                    "id": qid,
                    "category": category,
                    "question": q.get("question"),
                    "gold_chunk_keywords": keywords,
                    "expected_numbers": q.get("expected_numbers") or [],
                    "proposed_gold_chunk_ids": [m["chunk_id"] for m in matches],
                    "candidates": matches,
                }
            )
    finally:
        await conn.close()

    OUTPUT.write_text(yaml.safe_dump(out, sort_keys=False, allow_unicode=True))
    print(f"\nWrote {OUTPUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
