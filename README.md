# Agentic RAG Platform with Structured + Unstructured Retrieval

A chat-with-your-PDFs system. You drop in PDFs (sustainability reports,
annual reports, anything with tables of numbers), an LLM agent answers
questions about them, and a small evaluation framework actually measures
how well it does.

> Built on top of [serkanyasr/agentic_rag_project](https://github.com/serkanyasr/agentic_rag_project)
> (the agent, the FastAPI server, pgvector storage, hybrid text + vector
> search, Docling-based ingestion, Streamlit UI). My additions are listed
> in the "What I added" section below.

---

## The problem I set out to fix

The base system handles PDF questions by embedding the question into a
vector and looking for similar-vector chunks of the PDF. That works fine
for conceptual questions like *"how does NTT DATA approach materiality?"*
but is bad at number questions like *"what's the Scope 1+2 emissions
reduction target?"*. The answer to a number question often lives in one
row of a giant KPI table, and that row's vector is drowned in the rest
of the table.

Concretely, on a 25-question benchmark I built (NTT DATA sustainability
reports 2023 + 2024, 686 chunks):

- Only **12.7 % of the chunks the base vector retriever returned were
  actually relevant** (precision@5 = 0.127).
- Hybrid search (vector blended with full-text trigram rank) did not
  improve recall at all — same numbers, twice the latency.
- Every query took **36-157 ms**, because each one has to embed the
  question and run a vector similarity search.

## What I changed

Three things, in increasing order of difficulty:

1. **SQL-as-a-tool retrieval.** During ingestion, an offline LLM pass
   pulls every number out of the PDF chunks and writes them to a Postgres
   table (`kpi_facts`) along with metric name, year, scope, etc. At query
   time, the agent has a new tool — `sql_kpi_search` — that does a fuzzy
   SQL match on the metric name. For specific-number questions it goes
   there first; for conceptual questions it falls back to the existing
   vector / hybrid search.
2. **An evaluation framework.** Before this, you had no way to say
   whether a change to the agent helped or hurt. I built a 25-question
   benchmark with strict gold-chunk labels, a CLI runner, and an
   LLM-as-judge for the open-ended answer-quality metrics.
3. **Table-aware chunking.** The biggest single table in the PDF was a
   37 000-character KPI block; the next biggest was a narrative TCFD
   risk-disclosure table where individual rows are themselves 5-10 kB
   of prose. Both kinds of "fat" chunks have useless embeddings. The
   new ingestion path detects markdown tables, splits them into
   row-groups capped by **both** a row count and a character budget,
   and repeats the column headers on every piece. Max table chunk
   dropped from 43 kB to ~24 kB; the corpus grew from 316 chunks to
   328 for the same document. Implemented in `ingestion/chunker.py`
   and covered by 22 unit tests in `tests/ingestion/test_chunker.py`.

## The headline result

Same 25 questions, base system (vector / hybrid) vs the new SQL tool,
k = 5:

| What we measure                          | Base system | My SQL tool | Change            |
| ---------------------------------------- | :---------: | :---------: | :---------------: |
| **Precision** (% of returned chunks that are relevant) | 12.7 %      | **26.4 %**  | **+108 %**        |
| **Content-recall@5** (top 5 contains the answer text)  | 50.0 %      | **72.7 %**  | **+45 %**         |
| **Multi-hop content-recall@5** (compare two KPIs)      | 66.7 %      | **100.0 %** | **perfect**       |
| **Latency** (time per query)             | 36-157 ms   | **5 ms**    | **7-30× faster**  |

The agent keeps all three retrievers and picks per-question. SQL is the
right answer for KPI lookups but only knows about numbers the extractor
caught; conceptual questions ("which framework certifies the targets?")
still fall back to vector / hybrid. The eval framework reports the
per-category breakdown so the trade-off is explicit, not hidden.

## What "recall@5", "precision@5", "MRR" mean in plain English

Retrieval quality is reported with five numbers throughout this README
and the eval reports:

| Term                    | Plain-English question it answers                                                                                                |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **recall@5** (strict)   | Of the *specific gold chunk IDs* labelled for this question, what fraction did the system put in its top 5?                      |
| **content-recall@5**    | Did *any* of the top 5 chunks' text contain the answer keywords? — chunking-independent, robust to chunk-ID drift.               |
| **precision@5**         | Of the 5 chunks the system returned, what fraction were the labelled gold IDs?                                                   |
| **MRR**                 | How high up the list did the first correct chunk appear? 1.0 = always rank 1, 0.5 ≈ usually rank 2, etc.                         |
| **latency**             | Wall-clock milliseconds per query, end-to-end through the API.                                                                   |

`@5` just means "looking at the top 5 results"; you can run with a
different `k`. Recall and precision pull in opposite directions, which
is why a tool can be much more precise (less noise) and only slightly
better on recall (less coverage) at the same time. The two recall
numbers exist because chunk-ID-based metrics break when you change the
chunker: a new chunking might find the same answer in *different*
chunks, which looks like a regression under strict ID matching but is
caught by content-recall.

---

## What I added

### 1. SQL-as-a-tool retrieval

A new way for the agent to answer KPI questions: a Postgres table called
`kpi_facts` that holds extracted numbers, and a `sql_kpi_search` tool the
agent can call.

- An offline script (`scripts/extract_kpi_facts.py`) runs the LLM over every
  chunk that contains numbers and asks it to pull out
  `{metric_name, value, unit, year, scope, category}` rows.
- Each row keeps a foreign key back to its source chunk, so answers can
  still cite the original prose.
- The agent picks this tool first for specific-number questions, and falls
  back to hybrid search for everything else.

### 2. Retrieval evaluation framework

A small benchmark + CLI runner that puts the agent's retrieval on a
scorecard, so claims of improvement can be checked rather than trusted.

- 25 questions (factoid / multi-hop / table-lookup / negative).
- Metrics: `recall@k`, `precision@k`, `MRR`, `numeric_match`, plus an
  optional LLM judge for `faithfulness` and `answer_relevance`.
- Gold chunk IDs were bootstrapped by direct ILIKE-matching against the
  raw chunk content — the bootstrap never reads `kpi_facts` and never
  calls the agent, so the labels don't favour the SQL feature.
- Reports are written to `eval/results/` as both markdown and JSON.

### 3. Table-aware chunking

The biggest single-chunk failure on the original benchmark was the 37 000-
character "FY2024 KPI table". Its embedding was too unfocused for vector
search to rank it. The chunker now handles markdown tables specially.

- Detects pipe-syntax tables before any semantic splitting.
- Splits each table into row groups capped by **both** a row count
  (default 4) **and** a character budget (default 10 000) — the
  character cap is what handles "narrative tables" like the TCFD
  risk-disclosure blocks where one row is multiple paragraphs of prose.
- Repeats the first three header rows on every chunk so the column
  headers stay attached to the data.
- Implemented in `ingestion/chunker.py`; 22 tests in
  `tests/ingestion/test_chunker.py`.

Operationally, `scripts/rechunk_existing.py` lets you re-chunk an
already-ingested document in place: it reads the parsed markdown back
out of `documents.content`, runs the new chunker, re-embeds in
throttled batches, and atomically swaps the chunks rows. This avoids
re-running the most CPU-heavy step of ingestion (Docling parsing) when
all you want to evaluate is a chunker change.

### 4. Provider-pluggable LLM stack

`OPENAI_BASE_URL` plumbing so the whole stack (chat agent, embeddings, KPI
extraction, judge) can run against Ollama, OpenAI, Groq, vLLM, or any other
OpenAI-compatible endpoint without code changes. Embedding dimension is
configurable to match (768 for Ollama `nomic-embed-text`, 1536 for OpenAI
`text-embedding-3-small`).

### 5. Dev tooling

- Makefile wrapper that ignores polluting shell env vars (`DB_USER`,
  `DB_PASSWORD` etc. set by user shells were silently overriding `.env`).
- Dockerfile pinned to the Python version `pyproject.toml` actually
  requires; system graphics libraries added so Docling can render PDFs
  inside `python:3.12-slim`.

---

## How it works

### Architecture

```
                              ┌──────────────────────────┐
                              │  Streamlit UI  (:8501)   │
                              └────────────┬─────────────┘
                                           │
                              ┌────────────▼─────────────┐
                              │  FastAPI agent (:8058)   │
                              │  - Pydantic AI runtime   │
                              │  - tools: vector_search, │
                              │           hybrid_search, │
                              │           sql_kpi_search │
                              └────────────┬─────────────┘
                                           │
                              ┌────────────▼─────────────┐
                              │  Postgres (:6543)        │
                              │  - documents, chunks     │
                              │  - kpi_facts             │
                              │  - pgvector + pg_trgm    │
                              └──────────────────────────┘
```

### Where each piece lives

| Piece                  | File / function                                  | One-line role                                                  |
| ---------------------- | ------------------------------------------------ | -------------------------------------------------------------- |
| PDF parsing            | `ingestion/extract_files.py`                     | Docling -> markdown (tables become pipe-syntax tables).        |
| Chunking               | `ingestion/chunker.py`                           | Splits markdown into chunks; table-aware splitter for tables.  |
| Embeddings             | `ingestion/ingest.py` `aembed_chunks()`          | Sends chunk text to the configured embedding model.            |
| KPI extraction         | `scripts/extract_kpi_facts.py`                   | LLM pulls `{metric, value, year, ...}` rows from chunks.       |
| Agent + tools          | `agent/agent.py`, `agent/tools.py`               | Pydantic AI agent registering all tools.                       |
| Vector search          | `agent/db_utils.py` `vector_search()`            | pgvector cosine similarity over `chunks.embedding`.            |
| Hybrid search          | `agent/db_utils.py` `hybrid_search()`            | Vector + pg_trgm text rank, weighted.                          |
| SQL KPI search         | `agent/db_utils.py` `sql_kpi_search()`           | Fuzzy text match on `kpi_facts.metric_name`.                   |
| API                    | `agent/api.py`                                   | FastAPI routes; `/chat`, `/search/{vector,hybrid,sql}`.        |
| Eval CLI               | `eval/run_eval.py`                               | Runs a strategy against `benchmark.yaml`, writes reports.      |

---

## Results in detail

The summary table at the top of this README compares the base system's
*best* retriever (vector and hybrid tied) against the new SQL tool.
This section shows all three side by side, plus a per-category
breakdown so you can see where each strategy is actually being used.

All numbers come from the same 25 questions, k = 5, against a corpus
of two NTT DATA Sustainability Reports (2023 + 2024 — 686 chunks total,
328 of them produced by the new table-aware chunker on the 2024
report). Reproduce with:

```bash
docker compose exec api python -m eval.run_eval --strategy vector --k 5
docker compose exec api python -m eval.run_eval --strategy hybrid --k 5
docker compose exec api python -m eval.run_eval --strategy sql    --k 5
```

### Overall scoreboard

| Strategy            | recall@5 | content-recall@5 | precision@5 |  MRR  | latency |
| ------------------- | :------: | :--------------: | :---------: | :---: | :-----: |
| `vector` (upstream) |  0.185   |     0.500        |    0.127    | 0.273 |  42 ms  |
| `hybrid` (upstream) |  0.185   |     0.500        |    0.127    | 0.273 | 157 ms  |
| **`sql`** (new)     | **0.324**|   **0.727**      |  **0.264**  |**0.400**|**5 ms**|

**Reading the row:** vector retrieves the exact labelled gold chunk
18.5 % of the time, but a top-5 chunk *containing the answer text* 50 %
of the time — so the answer is more findable than strict ID matching
suggests. SQL gets both right far more often: 32 % strict, 73 % content,
and twice the precision per returned chunk, in ~5 ms.

### Per-category content-recall@5 (where each tool actually wins)

Questions are split into categories on purpose, because the aggregate
hides trade-offs.

| Category       | # questions | vector | hybrid |    sql    | Best at                                                |
| -------------- | :---------: | :----: | :----: | :-------: | :----------------------------------------------------- |
| factoid        |  10         | 0.400  | 0.400  | **0.700** | **sql**: any single KPI lookup                         |
| multi-hop      |   6         | 0.667  | 0.667  | **1.000** | **sql**: comparing two numeric KPIs (perfect recall)   |
| table-lookup   |   6         | 0.500  | 0.500  |   0.500   | three-way tie — every retriever still misses half      |

### Why SQL wins where it wins

- **Precision** (0.264 vs 0.127, ~2×): everything in `kpi_facts` is a
  number with a metric name. There's no off-topic prose, so what comes
  back is usually relevant.
- **Speed** (~5 ms vs 42-157 ms): a fuzzy SQL match on a small table is
  much cheaper than embedding a question and scanning a vector index.
- **Multi-hop perfect recall**: questions like *"is the Scope 1+2 cut
  bigger than the Scope 3 cut by 2030?"* need two related numbers. A
  structured table makes that one query; vector search has to hope
  both numbers land in the top 5 of the same query.

### Where SQL is weaker (and why)

- **Conceptual factoids**: questions like *"which framework certifies
  the targets?"* don't have numeric answers and aren't in `kpi_facts`.
  SQL coverage is bounded by what the extractor pulled, so the agent
  falls back to vector / hybrid for these.
- **Table-lookup**: every retriever ties at 0.5 content-recall here.
  These questions ("on the table, what's the value for X in row Y?")
  need both keyword grounding (which the vector retriever provides) and
  numeric grounding (which SQL provides), and ranking is the hard part
  — the answer is in the top-N retrieved but not always at rank 1.

### An honest negative result

Hybrid retrieval — vector blended with full-text trigram rank — was
supposed to beat plain vector retrieval. On this 25-question benchmark
**it doesn't**: same recall, same MRR, twice the latency. An earlier
5-question micro-eval suggested hybrid had an edge; the larger
benchmark didn't back that up. Worth knowing.

### How the gold labels were chosen

To keep the eval honest, gold chunk IDs were chosen **without using the
SQL feature**. `scripts/label_gold_ids.py` (and the in-place updater
`scripts/relabel_gold_ids.py`) ILIKE-match each question's keywords and
expected numbers directly against the `chunks` table. They never read
`kpi_facts` and never invoke the agent, so the labels can't be biased
toward what SQL happens to find.

22 of 25 questions have strict gold IDs; the other 3 are intentional
"unanswerable" negatives where the LLM judge grades whether the agent
refuses cleanly. The `content-recall@5` column above is computed
straight from chunk text and so doesn't rely on the gold IDs at all —
it's the metric to trust when comparing chunkings, because chunk UUIDs
change but answer text does not.

### Chat-mode reliability caveat

The `/chat` end-to-end strategy depends on the underlying LLM actually
calling the tools it's offered. Small open-source models like
`qwen2.5:14b` sometimes return a free-text reply instead of a tool call,
even at temperature 0. Larger / commercial models are much more reliable.
The deterministic `/search/sql` strategy above is not affected by this
and gives the cleanest before/after picture of the SQL feature.

---

## Quick start

### Requirements

- Docker + Docker Compose
- An LLM provider — Ollama (free, local) is the simplest; OpenAI or any
  OpenAI-compatible endpoint also works.
- ~10 GB free disk if you run models locally with Ollama.

### 1. Boot the stack

```bash
make up       # docker compose up -d, with shell env vars neutralised
make status   # confirm /health is green
```

| What     | URL                                                |
| -------- | -------------------------------------------------- |
| API      | http://localhost:8058 (Swagger at `/docs`)         |
| UI       | http://localhost:8501                              |
| Postgres | `localhost:6543` (postgres / postgres / vector_db) |

### 2. Configure the LLM

Copy `.env.example` to `.env` and fill in values. The default file is
Ollama-friendly:

```env
OPENAI_BASE_URL=http://host.docker.internal:11434/v1
OPENAI_API_KEY=ollama   # any non-empty string
LLM_CHOICE=qwen2.5:14b
EMBEDDING_MODEL=nomic-embed-text
```

For OpenAI: set a real `OPENAI_API_KEY` and leave `OPENAI_BASE_URL` unset.

### 3. Ingest PDFs and build the KPI table

```bash
# Drop PDFs into ./documents/, then:
make ingest

# Then extract KPIs from those chunks into kpi_facts:
docker compose exec api python -m scripts.extract_kpi_facts
```

Useful flags on the extractor:

- `KPI_EXTRACT_RESET=1` — truncate `kpi_facts` first.
- `KPI_EXTRACT_RESUME=1` — skip chunks that already have facts (so you can
  continue a partial run).

### 4. Run the eval

```bash
# Retrieval-only (deterministic and fast):
docker compose exec api python -m eval.run_eval --strategy vector --k 5
docker compose exec api python -m eval.run_eval --strategy hybrid --k 5
docker compose exec api python -m eval.run_eval --strategy sql    --k 5

# End-to-end agent with LLM-as-judge faithfulness + relevance:
docker compose exec api python -m eval.run_eval --strategy chat --k 5 --judge
```

Reports land in `eval/results/<timestamp>__<strategy>__k<k>.{md,json}`.

### 5. Tear down

```bash
make stop    # stop containers, keep the data volume
make down    # remove containers + data volume
make clean   # also remove built images (full reset)
```

---

## API endpoints

```
GET  /health                  - liveness + DB + LLM client status
POST /chat                    - single-turn chat (agent picks tools)
POST /chat/stream             - SSE-streamed chat
POST /search/vector           - direct vector search (no agent)
POST /search/hybrid           - direct hybrid search (no agent)
POST /search/sql              - direct KPI lookup     (no agent)   [new]
GET  /documents               - list ingested documents
GET  /sessions/{session_id}   - conversation history
```

Full schema at `http://localhost:8058/docs`.

---

## Repository layout

```
agent/                  FastAPI app, Pydantic AI agent, tools, DB helpers
ingestion/              Docling PDF parsing + chunking + embedding
  chunker.py              table-aware splitter (new)
scripts/                One-off scripts
  extract_kpi_facts.py    LLM-driven KPI extraction (new)
  label_gold_ids.py       deterministic gold-label bootstrap (new)
  relabel_gold_ids.py     in-place gold-ID refresh after re-chunking (new)
  rechunk_existing.py     re-chunk + re-embed without re-running Docling (new)
  bootstrap_gold_ids.py   hybrid-search candidate dump (new)
sql/
  schema.sql              full DB schema (rebuild from scratch)
  migrations/             idempotent in-place migrations
eval/                   Retrieval evaluation framework (new)
  benchmark.yaml          25 questions + gold chunk IDs
  metrics.py              recall@k, MRR, precision@k, numeric_match
  strategies.py           vector / hybrid / sql / chat adapters
  judge.py                LLM-as-judge with on-disk cache
  run_eval.py             CLI runner
  results/                timestamped per-run reports (gitignored)
documents/              source PDFs
tests/                  pytest suite
```

---

## License

Same as the upstream project. See
[serkanyasr/agentic_rag_project](https://github.com/serkanyasr/agentic_rag_project)
for original terms.
