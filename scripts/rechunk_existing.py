"""Re-chunk an already-ingested document with the current chunker, in place.

Reads the document's pre-parsed markdown straight out of ``documents.content``
(populated by Docling on the original ingest), runs it through the current
``PDFSemanticChunker`` (which now includes table-aware splitting), re-embeds
the new chunks, and atomically swaps the ``chunks`` rows.

The point is to compare retrieval *before* and *after* a chunking change
without re-running Docling, which is the most CPU-heavy step of full
ingestion. Embedding the new chunks is the only sustained CPU work and is
deliberately throttled (small batch size + short sleeps) so a laptop fan
doesn't spin up.

Side effect: any rows in ``kpi_facts`` tied to the old chunks are removed by
``ON DELETE CASCADE``. Re-run ``scripts/extract_kpi_facts.py`` to rebuild
the SQL surface; until then the ``sql_kpi_search`` tool will return nothing.

Usage (from the host, with the stack up):

    docker compose exec -T api python -m scripts.rechunk_existing \
        --document sr2024 \
        --batch-size 4 \
        --sleep-ms 1000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, List

import asyncpg
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

# Make the in-repo packages importable regardless of cwd.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ingestion.chunker import ChunkingConfig, PDFSemanticChunker, DocumentChunk  # noqa: E402


logger = logging.getLogger("rechunk")


def _build_dsn() -> str:
    load_dotenv()
    user = os.getenv("DB_USER", "postgres")
    password = os.getenv("DB_PASSWORD", "postgres")
    host = os.getenv("DB_HOST", "postgres")
    port = os.getenv("DB_PORT", "5432")
    name = os.getenv("DB_NAME", "vector_db")
    return f"postgresql://{user}:{password}@{host}:{port}/{name}"


async def _fetch_document(conn: asyncpg.Connection, title: str) -> tuple[str, str, str] | None:
    row = await conn.fetchrow(
        "SELECT id::text AS id, title, content FROM documents WHERE title = $1",
        title,
    )
    if not row:
        return None
    return row["id"], row["title"], row["content"]


def _build_chunks(
    content: str,
    title: str,
    source: str,
    rows_per_chunk: int,
    use_table_aware: bool,
) -> List[DocumentChunk]:
    cfg = ChunkingConfig(
        chunk_size=int(os.getenv("CHUNK_SIZE", "850")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
        # Keep semantic splitting on so prose segments behave the same as
        # the original ingest; only the table-aware path is new.
        use_semantic_splitting=os.getenv("USE_SEMANTIC_SPLITTING", "1") == "1",
        use_table_aware_chunking=use_table_aware,
        table_rows_per_chunk=rows_per_chunk,
    )
    chunker = PDFSemanticChunker(cfg)
    return chunker.chunk_content(
        content=content,
        title=title,
        source=source,
        metadata={"chunk_method_override": "rechunk_existing"},
    )


async def _embed_chunks_cool(
    chunks: List[DocumentChunk], batch_size: int, sleep_ms: int
) -> List[DocumentChunk]:
    """Embed every chunk, but in small batches with a short sleep between
    batches so sustained CPU stays low. Returns chunks with ``.embedding``
    attached.
    """
    model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    embeddings = OpenAIEmbeddings(
        model=model,
        tiktoken_enabled=False,
        check_embedding_ctx_length=False,
        # NB: this chunk_size is the HTTP batch size, not text chunk size.
        chunk_size=batch_size,
    )

    out: List[DocumentChunk] = []
    total = len(chunks)
    t0 = time.time()
    for batch_start in range(0, total, batch_size):
        batch = chunks[batch_start : batch_start + batch_size]
        texts = [c.content for c in batch]
        vectors = await embeddings.aembed_documents(texts)
        for c, v in zip(batch, vectors):
            embedded = DocumentChunk(
                content=c.content,
                index=c.index,
                start_char=c.start_char,
                end_char=c.end_char,
                metadata={
                    **c.metadata,
                    "embedding_model": model,
                    "embedding_generated_at": datetime.now().isoformat(),
                },
            )
            embedded.embedding = v
            out.append(embedded)
        done = batch_start + len(batch)
        elapsed = time.time() - t0
        logger.info(
            "embedded %d/%d (%.0f%%, %.1fs elapsed)",
            done,
            total,
            100 * done / total,
            elapsed,
        )
        if done < total and sleep_ms > 0:
            await asyncio.sleep(sleep_ms / 1000)
    return out


async def _swap_chunks(
    conn: asyncpg.Connection,
    document_id: str,
    new_chunks: List[DocumentChunk],
) -> tuple[int, int]:
    """Atomically delete old chunks for ``document_id`` and insert the new
    set in a single transaction. Returns (deleted_count, inserted_count).
    """
    async with conn.transaction():
        deleted = await conn.fetchval(
            "WITH d AS (DELETE FROM chunks WHERE document_id = $1::uuid RETURNING 1) "
            "SELECT count(*) FROM d",
            document_id,
        )
        for chunk in new_chunks:
            embedding_data = None
            if getattr(chunk, "embedding", None):
                embedding_data = "[" + ",".join(map(str, chunk.embedding)) + "]"
            meta = {
                **chunk.metadata,
                "chunk_type": chunk.metadata.get("content_type", "text"),
            }
            await conn.execute(
                """
                INSERT INTO chunks (document_id, content, embedding, chunk_index, metadata, token_count)
                VALUES ($1::uuid, $2, $3::vector, $4, $5, $6)
                """,
                document_id,
                chunk.content,
                embedding_data,
                chunk.index,
                json.dumps(meta),
                chunk.token_count if hasattr(chunk, "token_count") else len(chunk.content.split()),
            )
        return int(deleted), len(new_chunks)


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-chunk an already-ingested document in place (skips Docling).",
    )
    parser.add_argument(
        "--document",
        required=True,
        help="Document title (the 'title' column in documents). E.g. 'sr2024'.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--sleep-ms",
        type=int,
        default=1000,
        help="Milliseconds to pause between embedding batches (CPU throttle).",
    )
    parser.add_argument(
        "--rows-per-chunk",
        type=int,
        default=4,
        help="How many table rows to put in each chunk.",
    )
    parser.add_argument(
        "--no-table-aware",
        action="store_true",
        help="Disable table-aware chunking (semantic chunker only).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and report the new chunks without touching the database.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    conn = await asyncpg.connect(_build_dsn())
    try:
        doc = await _fetch_document(conn, args.document)
        if not doc:
            logger.error("No document found with title %r", args.document)
            return 2
        doc_id, title, content = doc
        logger.info(
            "Loaded document %s (id=%s, %d chars)", title, doc_id, len(content)
        )

        # Cheap: chunk in memory, log shape, abort early if dry-run.
        old_count = await conn.fetchval(
            "SELECT count(*) FROM chunks WHERE document_id = $1::uuid", doc_id
        )
        chunks = _build_chunks(
            content,
            title,
            title,
            args.rows_per_chunk,
            use_table_aware=not args.no_table_aware,
        )
        n = len(chunks)
        sizes = sorted(len(c.content) for c in chunks)
        table_n = sum(1 for c in chunks if c.metadata.get("segment_type") == "table")
        prose_n = sum(1 for c in chunks if c.metadata.get("segment_type") == "prose")
        logger.info(
            "Will replace %d existing chunks with %d new chunks "
            "(%d prose / %d table). Sizes min/median/max = %d / %d / %d.",
            old_count,
            n,
            prose_n,
            table_n,
            sizes[0],
            sizes[len(sizes) // 2],
            sizes[-1],
        )
        if args.dry_run:
            logger.info("--dry-run: stopping before embedding/insert.")
            return 0

        # Expensive: embedding. Throttled to keep the laptop cool.
        embedded = await _embed_chunks_cool(
            chunks, batch_size=args.batch_size, sleep_ms=args.sleep_ms
        )

        deleted, inserted = await _swap_chunks(conn, doc_id, embedded)
        logger.info("Swap complete: deleted %d, inserted %d.", deleted, inserted)
        logger.info(
            "Reminder: kpi_facts rows tied to the old chunks were cascaded "
            "away. Run scripts/extract_kpi_facts.py to rebuild the SQL surface."
        )
        return 0
    finally:
        await conn.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
