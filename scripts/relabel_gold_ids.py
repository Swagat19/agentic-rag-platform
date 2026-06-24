"""Refresh ``gold_chunk_ids`` in ``eval/benchmark.yaml`` in place.

This is the script you run when chunk UUIDs have changed (for example
after re-running ingestion with a different chunker). It uses the same
deterministic ILIKE probe as ``scripts/label_gold_ids.py`` — every
non-negative question's ``gold_chunk_keywords`` + ``expected_numbers``
are joined with AND against the live ``chunks`` table — and writes the
top-N matching chunks back into the YAML, preserving comments and key
order via ``ruamel.yaml`` where available (falls back to PyYAML).

The labelling never reads ``kpi_facts`` and never invokes the agent, so
the gold labels stay independent of the SQL feature being measured.

Usage::

    docker compose exec -T api python -m scripts.relabel_gold_ids \\
        --top-n 2
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any, List

import asyncpg
import yaml
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK = REPO_ROOT / "eval" / "benchmark.yaml"


def _build_dsn() -> str:
    load_dotenv()
    user = os.getenv("DB_USER", "postgres")
    password = os.getenv("DB_PASSWORD", "postgres")
    host = os.getenv("DB_HOST", "postgres")
    port = os.getenv("DB_PORT", "5432")
    name = os.getenv("DB_NAME", "vector_db")
    return f"postgresql://{user}:{password}@{host}:{port}/{name}"


async def find_matches(
    conn: asyncpg.Connection,
    keywords: List[str],
    numbers: List[str],
    limit: int,
) -> List[str]:
    needles = list(keywords) + list(numbers)
    if not needles:
        return []
    where_clauses: List[str] = []
    params: List[Any] = []
    for n in needles:
        params.append(f"%{n}%")
        where_clauses.append(f"content ILIKE ${len(params)}")
    where_sql = " AND ".join(where_clauses)
    params.append(limit)
    # Ordering rationale: pick the SHORTEST chunks that match all the
    # answer tokens. A short chunk containing every required token is
    # by construction focused on the answer, which is exactly what a
    # well-tuned dense or hybrid retriever should be able to find. With
    # the older single-pass labelling (ORDER BY length DESC LIMIT 2),
    # gold tended to be the single longest container chunk for a
    # question, which made retrieval scores look optimistic on the
    # original 316-chunk corpus because that chunk was also the only
    # plausible match for many other keywords.
    sql = (
        "SELECT id::text AS chunk_id "
        f"FROM chunks WHERE {where_sql} "
        f"ORDER BY length(content) ASC LIMIT ${len(params)}"
    )
    rows = await conn.fetch(sql, *params)
    return [r["chunk_id"] for r in rows]


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--top-n",
        type=int,
        default=2,
        help="How many top ILIKE matches to write back as gold per question.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the proposed updates without writing the YAML.",
    )
    args = parser.parse_args()

    text = BENCHMARK.read_text()
    benchmark = yaml.safe_load(text)
    questions = benchmark.get("questions") or []

    conn = await asyncpg.connect(_build_dsn())
    proposals: dict[str, List[str]] = {}
    try:
        for q in questions:
            qid = q.get("id")
            category = q.get("category")
            if category == "negative":
                continue
            keywords = q.get("gold_chunk_keywords") or []
            numbers = [str(n) for n in (q.get("expected_numbers") or [])]
            matches = await find_matches(conn, keywords, numbers, limit=args.top_n)
            print(f"{qid:<35} -> {len(matches)} matches: {matches}")
            proposals[qid] = matches
    finally:
        await conn.close()

    if args.dry_run:
        print("\n--dry-run: not writing benchmark.yaml")
        return 0

    # Update gold_chunk_ids in place. We use a simple targeted text
    # replace so that surrounding comments and formatting survive.
    new_text = text
    for q in questions:
        qid = q.get("id")
        if qid not in proposals:
            continue
        old_ids = q.get("gold_chunk_ids") or []
        new_ids = proposals[qid]
        if old_ids == new_ids:
            continue
        # Render the new YAML block for this question's gold_chunk_ids
        # then splice it into the file text.
        new_block_lines = ["    gold_chunk_ids:"]
        if not new_ids:
            new_block_lines = ["    gold_chunk_ids: []"]
        else:
            for cid in new_ids:
                new_block_lines.append(f'      - "{cid}"')
        new_block = "\n".join(new_block_lines)

        # Build the old block as it currently appears (so we can replace).
        anchor = f"  - id: {qid}\n"
        anchor_idx = new_text.find(anchor)
        if anchor_idx < 0:
            print(f"!! could not locate question anchor for {qid}")
            continue
        gold_idx = new_text.find("gold_chunk_ids", anchor_idx)
        if gold_idx < 0:
            print(f"!! could not find gold_chunk_ids for {qid}")
            continue
        # Walk back to start of the line for clean replacement.
        line_start = new_text.rfind("\n", 0, gold_idx) + 1
        # Find the end of the block: next line that isn't a list item.
        cursor = new_text.find("\n", gold_idx) + 1
        while cursor < len(new_text):
            line_end = new_text.find("\n", cursor)
            line = new_text[cursor : line_end if line_end >= 0 else len(new_text)]
            stripped = line.lstrip()
            if (
                stripped.startswith("-")
                and stripped.lstrip("-").startswith(' "')
                and line.startswith("      ")
            ):
                cursor = line_end + 1 if line_end >= 0 else len(new_text)
                continue
            break
        new_text = new_text[:line_start] + new_block + "\n" + new_text[cursor:]
        print(f"updated {qid}")

    BENCHMARK.write_text(new_text)
    print(f"\nWrote {BENCHMARK.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
