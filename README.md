# Agentic RAG Platform with Hybrid Retrieval and Evaluation

An agentic Retrieval-Augmented Generation system built on Pydantic AI and FastAPI,
extending the base architecture with hybrid structured + unstructured retrieval
(SQL-as-a-tool) and a reproducible evaluation suite that measures retrieval and
generation quality across heterogeneous data sources.

> **Attribution.** This project is built on top of
> [serkanyasr/agentic_rag_project](https://github.com/serkanyasr/agentic_rag_project),
> which provides the base architecture (Pydantic AI agent, FastAPI server, pgvector
> storage, hybrid text + vector search, Docling-based ingestion, Streamlit UI). My
> contributions are listed below; the rest is upstream code I am studying,
> extending, and operating.

---

## My Contributions

> _(Filled in as the project evolves. Each item maps to one or more commits.)_

- [ ] **Retrieval evaluation framework** — benchmark dataset and metrics
      (recall@k, MRR, hit rate, faithfulness, answer relevance) with a CLI
      runner and markdown reports for head-to-head retrieval-strategy
      comparison.
- [ ] **Hybrid structured + unstructured retrieval** — add a SQL-as-a-tool
      capability over a separate domain database, with an LLM-driven
      query-type router that selects between vector search, SQL, or fused
      multi-source retrieval.
- [ ] **Provider-pluggable agent runtime** — `OPENAI_BASE_URL` support so the
      stack runs against OpenAI, Ollama, Groq, or any OpenAI-compatible
      endpoint with zero code changes.
- [x] **Dev tooling** — Makefile wrapper that neutralises shell-env
      pollution; Dockerfile pinned to the Python version `pyproject.toml`
      actually requires.

---

## Architecture

Three services orchestrated via Docker Compose:

| Service             | Tech                                                    | Role                                                              |
| ------------------- | ------------------------------------------------------- | ----------------------------------------------------------------- |
| `agent_api`         | Python 3.12 · FastAPI · Pydantic AI (`:8058`)           | Agent runtime; exposes retrieval, chat, and streaming endpoints   |
| `agent_ui`          | Streamlit (`:8501`)                                     | Interactive chat UI                                               |
| `postgres_pgvector` | Postgres 17 · pgvector · pg_trgm (host `:6543`)         | Document store, vector index, and full-text trigram index         |

**Current agent tools (upstream):**

| Tool             | What it does                                                      |
| ---------------- | ----------------------------------------------------------------- |
| `vector_search`  | Semantic similarity search via pgvector cosine distance           |
| `hybrid_search`  | Weighted blend of vector similarity and BM25 trigram text rank    |
| `get_document`   | Fetch a single document by id                                     |
| `list_documents` | Paginated corpus listing                                          |

**Planned tools (this project):**

| Tool           | What it will do                                                                |
| -------------- | ------------------------------------------------------------------------------ |
| `sql_search`   | Agent-issued NL → SQL against a separate structured domain database            |
| `query_router` | Upstream classifier that selects which tool(s) to invoke based on the question |

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

| What           | URL                                       |
| -------------- | ----------------------------------------- |
| API            | http://localhost:8058 (Swagger at `/docs`) |
| UI             | http://localhost:8501                     |
| Postgres       | `localhost:6543`  (postgres / postgres / vector_db) |

### Configure the LLM provider

Copy `.env.example` to `.env` and fill in values. Default `.env.example` is
Ollama-friendly (set `OPENAI_BASE_URL=http://host.docker.internal:11434/v1`
and use any string as the API key). For OpenAI, set a real `OPENAI_API_KEY`
and leave `OPENAI_BASE_URL` unset.

### Ingest documents

```bash
make ingest    # runs the Docling-based ingestion pipeline over documents/
```

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
POST /chat                     - single-turn chat
POST /chat/stream              - SSE-streamed chat
POST /search/vector            - direct vector search (bypasses the agent)
POST /search/hybrid            - direct hybrid search (bypasses the agent)
GET  /documents                - list ingested documents
GET  /sessions/{session_id}    - conversation history for a session
```

Full schema at `http://localhost:8058/docs`.

---

## Roadmap

- [x] Initial import + attribution
- [x] Dev tooling: Makefile wrapper, Dockerfile python pin
- [ ] Provider-pluggable runtime (`OPENAI_BASE_URL`, configurable embedding
      dimension to match local providers like Ollama)
- [ ] Retrieval evaluation framework (benchmark dataset + metrics + runner)
- [ ] Baseline evaluation run on the upstream tools (checked-in results)
- [ ] Hybrid structured + unstructured retrieval: domain DB + `sql_search`
      tool + safety guards (read-only, allowlisted tables, parameterised
      queries)
- [ ] Query-type router (LLM-classified) and multi-source result fusion
- [ ] Post-improvement evaluation run + head-to-head comparison table
- [ ] Final README polish: design choices, evaluation methodology, demo
      numbers

---

## License

Same as the upstream project. See
[serkanyasr/agentic_rag_project](https://github.com/serkanyasr/agentic_rag_project)
for original license terms.
