"""Populate kpi_facts by LLM-extracting structured KPI rows from chunks.

This is the "agentic data extraction" pass that builds the structured
retrieval surface for the SQL-as-tool feature. For each chunk that
plausibly contains quantitative facts, the script asks the configured
LLM to emit an array of `{metric_name, value, unit, year, scope,
baseline_year, category}` objects. Each emitted row is validated and
inserted into kpi_facts with a back-reference to the source chunk so
the agent can return both the structured fact and the surrounding text.

Honest about its limits:
    - The LLM is the bottleneck. Extraction is noisy: it can miss
      facts, invent values, or choose ambiguous metric names. The
      eval framework measures whether the agent's downstream answers
      improve, not whether this table is a perfect ground truth.
    - We pre-filter chunks with a regex for numeric content so we
      don't waste calls on prose-only sections.
    - Rows whose `value` does not contain at least one digit are
      rejected so we don't pollute the table with hallucinated text.

Usage (run from repo root, after the kpi_facts migration is applied):

    docker compose exec api python -m scripts.extract_kpi_facts

Env vars consulted:
    OPENAI_BASE_URL, OPENAI_API_KEY, LLM_CHOICE  (matches the rest of the stack)
    DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME
    KPI_EXTRACT_LIMIT       optional int, cap chunks processed (debug)
    KPI_EXTRACT_RESET       set to 1 to truncate kpi_facts first
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Iterable

import asyncpg
from openai import AsyncOpenAI
from dotenv import load_dotenv


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("extract_kpi_facts")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_NUMERIC_HINT = re.compile(r"(\d+\s*%)|(\$\s*\d)|(FY\s*\d{2,4})|(20\d{2})|(\d+(\.\d+)?\s*(million|billion|tons?|tonnes?|MWh|kWh|GWh|tCO2e?|kg|m\^?2|hours?))", re.IGNORECASE)
_HAS_DIGIT = re.compile(r"\d")

VALID_CATEGORIES = {
    "emissions",
    "energy",
    "water",
    "waste",
    "diversity",
    "governance",
    "supply_chain",
    "social",
    "financial",
    "other",
}

EXTRACTION_PROMPT = """You extract structured KPI rows from sustainability-report text.

Return ONLY a JSON array. Each element is an object with these keys:
  metric_name (string, required) - short noun phrase, e.g. "Scope 1+2 GHG emissions reduction target"
  value (string, required) - the literal value as it appears, e.g. "68%", "30,000 tonnes", "FY2030"
  unit (string|null) - "%", "tCO2e", "tonnes", "MWh", "USD millions", etc.; null if value already encodes it
  year (integer|null) - the year the value refers to (target or reporting year)
  scope (string|null) - "Scope 1", "Scope 2", "Scope 1+2", "Scope 3", "all", or a business unit; null if absent
  baseline_year (integer|null) - reference year for relative targets; null if absent
  category (string, required) - one of: emissions, energy, water, waste, diversity, governance, supply_chain, social, financial, other

Rules:
  - Only extract rows where the chunk explicitly states a quantitative value with at least one digit.
  - Do not invent metric names, values, or years. If unsure, omit the row.
  - Preserve the exact value string from the chunk; do not re-format numbers.
  - One metric per row even if the chunk lists several values for the same metric across years.
  - If the chunk has no quantitative KPIs, return [].
  - Output ONLY the JSON array. No prose, no markdown fences, no commentary.

Chunk text:
\"\"\"
{chunk_text}
\"\"\"
"""


@dataclass
class ChunkRow:
    chunk_id: str
    document_id: str
    content: str


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _build_dsn() -> str:
    user = os.getenv("DB_USER", "postgres")
    password = os.getenv("DB_PASSWORD", "postgres")
    host = os.getenv("DB_HOST", "postgres")
    port = os.getenv("DB_PORT", "5432")
    name = os.getenv("DB_NAME", "postgres")
    return f"postgresql://{user}:{password}@{host}:{port}/{name}"


async def fetch_candidate_chunks(
    conn: asyncpg.Connection,
    limit: int | None,
    resume: bool = False,
) -> list[ChunkRow]:
    """Pull chunks that look quantitative.

    When resume=True, skip any chunk that already has at least one row
    in kpi_facts so partial runs can be continued without re-extraction.
    """
    if resume:
        rows = await conn.fetch(
            """
            SELECT c.id::text AS chunk_id,
                   c.document_id::text AS document_id,
                   c.content
            FROM chunks c
            WHERE NOT EXISTS (
                SELECT 1 FROM kpi_facts kf WHERE kf.source_chunk_id = c.id
            )
            ORDER BY c.chunk_index
            """
        )
    else:
        rows = await conn.fetch(
            """
            SELECT id::text AS chunk_id, document_id::text AS document_id, content
            FROM chunks
            ORDER BY chunk_index
            """
        )
    candidates = [
        ChunkRow(chunk_id=r["chunk_id"], document_id=r["document_id"], content=r["content"])
        for r in rows
        if _NUMERIC_HINT.search(r["content"] or "")
    ]
    if limit is not None:
        candidates = candidates[:limit]
    return candidates


async def truncate_facts(conn: asyncpg.Connection) -> None:
    await conn.execute("TRUNCATE TABLE kpi_facts;")
    logger.info("kpi_facts truncated")


async def insert_fact(conn: asyncpg.Connection, fact: dict[str, Any], chunk: ChunkRow) -> bool:
    try:
        await conn.execute(
            """
            INSERT INTO kpi_facts (
                metric_name, value, unit, year, scope, baseline_year,
                category, source_chunk_id, source_document_id,
                extracted_text, confidence
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8::uuid, $9::uuid, $10, $11)
            """,
            fact["metric_name"],
            fact["value"],
            fact.get("unit"),
            fact.get("year"),
            fact.get("scope"),
            fact.get("baseline_year"),
            fact["category"],
            chunk.chunk_id,
            chunk.document_id,
            chunk.content[:2000],
            fact.get("confidence", 0.7),
        )
        return True
    except Exception as exc:
        logger.warning("Insert failed for %r: %s", fact.get("metric_name"), exc)
        return False


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def _make_llm_client() -> tuple[AsyncOpenAI, str]:
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY", "ollama")
    model = os.getenv("LLM_CHOICE", "qwen2.5:14b")
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    return client, model


_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def _parse_extraction(text: str) -> list[dict[str, Any]]:
    """Parse the LLM's response into a list of dicts. Tolerates fences and stray prose."""
    if not text:
        return []
    text = text.strip()
    # Strip markdown fences if present.
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_ARRAY_RE.search(text)
        if not match:
            return []
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            logger.debug("JSON re-parse failed: %s; raw=%r", exc, text[:300])
            return []
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


def _validate_fact(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Tighten an LLM-emitted dict into a row we'd be willing to insert."""
    metric = (raw.get("metric_name") or "").strip()
    value = str(raw.get("value") or "").strip()
    if not metric or not value:
        return None
    if not _HAS_DIGIT.search(value):
        return None
    if len(metric) > 240:
        metric = metric[:240]
    if len(value) > 240:
        value = value[:240]
    category = (raw.get("category") or "other").strip().lower()
    if category not in VALID_CATEGORIES:
        category = "other"

    def _coerce_int(field: str) -> int | None:
        v = raw.get(field)
        if v is None or v == "":
            return None
        try:
            iv = int(v)
        except (TypeError, ValueError):
            return None
        # Sanity bound: reject obvious junk like 99999 or 1.
        if iv < 1990 or iv > 2100:
            return None
        return iv

    def _coerce_str(field: str) -> str | None:
        v = raw.get(field)
        if v is None:
            return None
        s = str(v).strip()
        if not s:
            return None
        return s[:80]

    return {
        "metric_name": metric,
        "value": value,
        "unit": _coerce_str("unit"),
        "year": _coerce_int("year"),
        "scope": _coerce_str("scope"),
        "baseline_year": _coerce_int("baseline_year"),
        "category": category,
        "confidence": 0.7,
    }


async def extract_from_chunk(
    client: AsyncOpenAI,
    model: str,
    chunk: ChunkRow,
) -> list[dict[str, Any]]:
    prompt = EXTRACTION_PROMPT.format(chunk_text=chunk.content)
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You output only valid JSON arrays as instructed."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=1200,
        )
    except Exception as exc:
        logger.warning("LLM call failed for chunk %s: %s", chunk.chunk_id, exc)
        return []
    raw_text = (resp.choices[0].message.content or "") if resp.choices else ""
    rows = _parse_extraction(raw_text)
    cleaned: list[dict[str, Any]] = []
    for row in rows:
        validated = _validate_fact(row)
        if validated:
            cleaned.append(validated)
    return cleaned


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> int:
    limit_env = os.getenv("KPI_EXTRACT_LIMIT")
    limit = int(limit_env) if limit_env else None
    reset = os.getenv("KPI_EXTRACT_RESET", "0") == "1"
    resume = os.getenv("KPI_EXTRACT_RESUME", "0") == "1"
    if reset and resume:
        logger.warning("Both KPI_EXTRACT_RESET and KPI_EXTRACT_RESUME set; reset wins.")
        resume = False

    dsn = _build_dsn()
    client, model = _make_llm_client()
    logger.info("Using model %s via base_url=%s", model, os.getenv("OPENAI_BASE_URL", "<default>"))

    conn = await asyncpg.connect(dsn)
    try:
        if reset:
            await truncate_facts(conn)

        chunks = await fetch_candidate_chunks(conn, limit=limit, resume=resume)
        logger.info("Candidate chunks with numeric hints: %d", len(chunks))

        sleep_ms = int(os.getenv("KPI_EXTRACT_SLEEP_MS", "0"))

        total_inserted = 0
        for idx, chunk in enumerate(chunks, start=1):
            facts = await extract_from_chunk(client, model, chunk)
            inserted = 0
            for fact in facts:
                ok = await insert_fact(conn, fact, chunk)
                if ok:
                    inserted += 1
            total_inserted += inserted
            logger.info(
                "[%d/%d] chunk=%s extracted=%d inserted=%d",
                idx,
                len(chunks),
                chunk.chunk_id[:8],
                len(facts),
                inserted,
            )
            # Optional cool-down between calls; keeps a long single-thread
            # Ollama job from pinning the CPU at 100% the whole time.
            if sleep_ms > 0 and idx < len(chunks):
                await asyncio.sleep(sleep_ms / 1000)

        logger.info("Done. Inserted %d facts total.", total_inserted)
    finally:
        await conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
