# Agentic RAG Platform with Hybrid Retrieval and Evaluation

An agentic Retrieval-Augmented Generation system built on Pydantic AI and FastAPI,
extending the base architecture with hybrid structured + unstructured retrieval
(SQL-as-a-tool) and a reproducible evaluation suite that measures retrieval and
answer-quality across heterogeneous data sources.

> **Attribution.** This project is built on top of
> [serkanyasr/agentic_rag_project](https://github.com/serkanyasr/agentic_rag_project),
> which provides the base architecture (Pydantic AI agent, FastAPI server, pgvector
> storage, hybrid text + vector search, Docling-based ingestion, Streamlit UI). My
> contributions are listed below; the rest is upstream code I am studying,
> extending, and operating.

---

## My Contributions

- **Hybrid structured + unstructured retrieval.** Added a second searchable
  surface alongside the chunk index: a Postgres `kpi_facts` table populated by
  an LLM-driven extraction pass over ingested chunks, and a new
  `sql_kpi_search` agent tool that queries it via trigram-similar metric-name
  matching with optional year and category filters. Each row is foreign-keyed
  to its source chunk, so the agent can cite both the structured value and the
  surrounding text.
- **Retrieval evaluation framework.** A pre-registered micro-benchmark with a
  CLI runner that measures `recall@k`, `precision@k`, `MRR`, and `numeric_match`
  across vector, hybrid, SQL, and chat strategies, plus an opt-in LLM-as-judge
  for `faithfulness` and `answer_relevance`. Reports are timestamped markdown
  + JSON, with anti-bias guarantees documented in `eval/README.md` (external
  question sourcing, per-category reporting, distinct judge model, content-hashed
  judge cache so reruns are free).
- **Provider-pluggable agent runtime.** `OPENAI_BASE_URL` plumbing so the
  whole stack (chat agent, embeddings, KPI extraction, judge) runs against
  Ollama, OpenAI, Groq, vLLM, or any OpenAI-compatible endpoint with zero code
  changes. Embedding dimension is configurable to match local providers
  (e.g. 768 for Ollama `nomic-embed-text`, 1536 for OpenAI `text-embedding-3-small`).
- **Dev tooling.** Makefile wrapper that neutralises shell-env pollution
  (`DB_USER`, `DB_PASSWORD`, etc. set by user shells override `.env`); Dockerfile
  pinned to the Python version `pyproject.toml` actually requires; system-graphics
  libraries added so Docling renders PDFs in `python:3.12-slim`.

---

## Results: SQL-as-tool vs vector / hybrid

Reproduced via `python -m eval.run_eval --strategy {vector,hybrid,sql} --k 5`
against the v0 micro-benchmark (5 questions across factoid, multi-hop,
table-lookup, and negative categories) on the NTT DATA Sustainability Report 2024 corpus.

| Strategy           | recall@5 | precision@5 |  MRR  | avg latency |
| ------------------ | :------: | :---------: | :---: | :---------: |
| `vector` (upstream)|  0.750   |    0.250    | 0.500 |   234 ms    |
| `hybrid` (upstream)|  0.750   |    0.250    | 0.500 |    91 ms    |
| **`sql`** (new)    | **0.875**|  **0.500**  |**0.600**|  **7 ms** |

**Headline:** the table-lookup question (`waste-recycling-target`, asking for an
explicit numeric target buried in a dense KPI table) scored **0.000 recall@5**
on both vector and hybrid retrieval and **0.500** on SQL. Precision doubled
across the whole benchmark, and SQL latency is an order of magnitude lower
because it skips the embedding round-trip.

Run artifacts in `eval/results/2026-06-23T10-19-*__*__k5.{md,json}`.

### How the structured layer is built

- `scripts/extract_kpi_facts.py` walks every chunk that contains numeric
  hints (regex-pre-filtered to skip prose-only sections), prompts the LLM
  for a strict JSON array of `{metric_name, value, unit, year, scope,
  baseline_year, category}` tuples, validates each row (must contain a
  digit, year ∈ [1990, 2100], category ∈ a small enum), and inserts into
  `kpi_facts` with a foreign-key back to the source chunk.
- The extraction pass on this corpus produced **96 facts across 16 chunks
  in 10 categories** (emissions, waste, diversity, governance, supply_chain,
  social, water, energy, financial, other) in roughly 12 minutes against a
  local Ollama `qwen2.5:14b`.
- The pass is intentionally noisy. The contribution is not that the table
  is a perfect ground-truth (it isn't — the LLM can miss rows or pick
  ambiguous metric names) but that *querying a noisy structured layer
  beats fuzzy vector retrieval on lookup-style questions*. The eval
  framework is what makes this claim auditable rather than vibes-based.

### How the agent uses it

A new `sql_kpi_search` tool is registered with the Pydantic AI agent:

```
sql_kpi_search(query: str, year: int|None, category: str|None, limit: int)
  → [{metric_name, value, unit, year, scope, baseline_year, category,
      similarity, chunk_id, document_title, supporting_text}, ...]
```

The system prompt instructs the agent to call `sql_kpi_search` *first* for
specific quantitative KPI questions, falling back to `hybrid_search` for
conceptual or multi-hop questions. The eval framework projects each SQL hit's
`source_chunk_id` onto the standard chunk-level retrieval metrics, so SQL and
vector hits are scored uniformly.

### Note on chat-mode reliability

`/chat` results are sensitive to the underlying LLM's tool-calling discipline.
Smaller open-source models (e.g. `qwen2.5:14b` on Ollama) occasionally emit a
free-text reply instead of a tool call, which manifests as silent retrieval drops
even at temperature 0. Larger / commercial models (GPT-4-class, Llama 3.1 70B+)
are dramatically more consistent. The deterministic `/search/sql` strategy
above is unaffected by this and is the cleanest before/after measurement of the
new feature.

---

## Architecture

Three services orchestrated via Docker Compose:

| Service             | Tech                                                    | Role                                                              |
| ------------------- | ------------------------------------------------------- | ----------------------------------------------------------------- |
| `agent_api`         | Python 3.12 · FastAPI · Pydantic AI (`:8058`)           | Agent runtime; exposes retrieval, chat, and streaming endpoints   |
| `agent_ui`          | Streamlit (`:8501`)                                     | Interactive chat UI                                               |
| `postgres_pgvector` | Postgres 17 · pgvector · pg_trgm (host `:6543`)         | Document store, vector index, full-text trigram index, KPI facts  |

**Agent tools:**

| Tool              | What it does                                                                            |
| ----------------- | --------------------------------------------------------------------------------------- |
| `sql_kpi_search`  | **(new)** Trigram-matched metric-name lookup over the structured `kpi_facts` table      |
| `vector_search`   | Semantic similarity search via pgvector cosine distance                                 |
| `hybrid_search`   | Weighted blend of vector similarity and trigram text rank                               |
| `get_document`    | Fetch a single document by id                                                           |
| `list_documents`  | Paginated corpus listing                                                                |

**Postgres surfaces:**

| Surface                  | Purpose                                                                              |
| ------------------------ | ------------------------------------------------------------------------------------ |
| `documents`, `chunks`    | Source documents and their embedded chunks (pgvector + pg_trgm indexed)              |
| `kpi_facts`              | LLM-extracted structured KPIs, FK-linked to the chunk they were derived from         |
| `search_kpi_facts(...)`  | SQL function backing `sql_kpi_search`: trigram match + year / category filters       |
| `match_chunks(...)`      | SQL function backing `vector_search`                                                 |
| `hybrid_search(...)`     | SQL function backing `hybrid_search`                                                 |

---

## Quick Start

### Prerequisites

- Docker + Docker Compose
- An LLM provider — **Ollama** (free, local) is the recommended path; OpenAI
  and any OpenAI-compatible endpoint are also supported.
- ~10 GB free disk if running models locally with Ollama.

### Boot the stack

```bash
make up        # wraps `docker compose up -d` and unsets polluting shell env
make status    # confirm containers are healthy and API /health is green
```

| What           | URL                                                  |
| -------------- | ---------------------------------------------------- |
| API            | http://localhost:8058 (Swagger at `/docs`)           |
| UI             | http://localhost:8501                                |
| Postgres       | `localhost:6543`  (postgres / postgres / vector_db)  |

### Configure the LLM provider

Copy `.env.example` to `.env` and fill in values. Default `.env.example` is
Ollama-friendly (set `OPENAI_BASE_URL=http://host.docker.internal:11434/v1`
and use any string as the API key). For OpenAI, set a real `OPENAI_API_KEY`
and leave `OPENAI_BASE_URL` unset.

### Ingest documents and build the KPI table

```bash
make ingest                                                # PDFs → chunks + embeddings
docker compose exec api python -m scripts.extract_kpi_facts # chunks → kpi_facts
```

`KPI_EXTRACT_RESET=1` truncates `kpi_facts` first; `KPI_EXTRACT_RESUME=1` skips
chunks that already have facts (useful to continue a partial run).

### Run the eval

```bash
# Retrieval-only strategies (deterministic, fast):
docker compose exec api python -m eval.run_eval --strategy vector --k 5
docker compose exec api python -m eval.run_eval --strategy hybrid --k 5
docker compose exec api python -m eval.run_eval --strategy sql    --k 5

# End-to-end agent with LLM-as-judge faithfulness + relevance:
docker compose exec api python -m eval.run_eval --strategy chat --k 5 --judge
```

Reports land in `eval/results/<timestamp>__<strategy>__k<k>.{md,json}`.

### Tear down

```bash
make stop      # stop containers, keep the volume
make down      # remove containers + volume
make clean     # also remove built images (full reset)
```

---

## Endpoints

```
GET  /health                   - health + DB + LLM client status
POST /chat                     - single-turn chat (agent picks tools)
POST /chat/stream              - SSE-streamed chat
POST /search/vector            - direct vector search (bypasses the agent)
POST /search/hybrid            - direct hybrid search (bypasses the agent)
POST /search/sql               - direct structured KPI lookup (bypasses the agent)
GET  /documents                - list ingested documents
GET  /sessions/{session_id}    - conversation history for a session
```

Full schema at `http://localhost:8058/docs`.

---

## Repository layout

```
agent/                  FastAPI app, Pydantic AI agent, tools, DB helpers
ingestion/              Docling-based PDF ingestion + chunking + embedding
scripts/                One-off scripts (e.g. extract_kpi_facts.py)
sql/
  schema.sql            Canonical schema (rebuilds full DB)
  migrations/           Idempotent in-place migrations (e.g. 001_kpi_facts.sql)
eval/
  benchmark.yaml        Pre-registered micro-benchmark (questions + gold IDs)
  metrics.py            Deterministic retrieval + answer-quality metrics
  strategies.py         Adapters: vector / hybrid / sql / chat
  judge.py              LLM-as-judge with disk-backed cache
  run_eval.py           CLI runner
  results/              Timestamped per-run reports (gitignored except .gitkeep)
documents/              Source PDFs (DVC-friendly; see .gitignore)
```

---

## License

Same as the upstream project. See
[serkanyasr/agentic_rag_project](https://github.com/serkanyasr/agentic_rag_project)
for original license terms.
