# Retrieval Evaluation Framework

This directory holds a small, reproducible framework for measuring the
retrieval and answer-generation quality of the agent. It is deliberately
LLM-light: deterministic metrics (recall@k, MRR, precision@k,
numeric-match) run with no model calls, and the only LLM-in-the-loop
metrics (faithfulness, answer relevance) are isolated in `judge.py` and
cached so reruns are cheap.

## Why this exists

The base repository ships a hybrid agentic RAG stack but no way to say
how good it is. Without a measurement harness:

* You can't tell whether changes (e.g. a new retrieval strategy, a tuned
  chunk size, a different embedding model) help or hurt.
* You can't claim a feature improved retrieval quality on a resume or
  in a code review.
* Regressions go unnoticed until they affect a real user query.

This framework solves all three by giving each retrieval strategy a
shared scorecard.

## Layout

```
eval/
  benchmark.yaml      # the question set, with ground-truth signals
  metrics.py          # pure functions: recall@k, MRR, numeric-match, ...
  strategies.py       # adapters for vector / hybrid / chat / (later) sql
  run_eval.py         # CLI runner; outputs markdown + JSON to results/
  results/            # gitignored timestamped run outputs
  README.md
```

## Running it

The runner targets the live API on `localhost:8058`, so the docker-compose
stack must be up:

```bash
make up
docker exec agent_api python -m eval.run_eval --strategy vector --k 5
docker exec agent_api python -m eval.run_eval --strategy hybrid --k 5
docker exec agent_api python -m eval.run_eval --strategy chat   --k 5
```

Each invocation prints a markdown report to stdout and (unless
`--no-write` is passed) writes a timestamped pair of `<run>.md` and
`<run>.json` files into `eval/results/` so historical runs are
side-by-side comparable.

## Metrics

| Metric | Type | Definition |
|---|---|---|
| recall@k | retrieval | Fraction of gold chunk IDs (or gold keyword clusters) that appear in the top-k retrieved chunks. |
| precision@k | retrieval | Fraction of the top-k retrieved chunks that are gold. Reported only when strict gold IDs are available. |
| MRR | retrieval | Mean of 1/rank of the first gold hit per question. Punishes putting the right chunk at rank 5 instead of rank 1. |
| numeric-match | answer | For chat-style runs, the fraction of `expected_numbers` from the benchmark that appear in the agent's final answer. Lexical comparison, percent-aware. |
| faithfulness *(planned)* | answer | LLM-judged: does every claim in the answer have support in the retrieved chunks? Penalises hallucinations even when they happen to be correct. |
| answer relevance *(planned)* | answer | LLM-judged: does the answer actually address the question that was asked? |

### Ground-truth signals

Each question in `benchmark.yaml` carries up to three signals:

* `gold_chunk_ids` — the strict signal: UUIDs of chunks that contain the
  answer. Populated by a one-time human pass over the corpus. Empty
  until that pass is done.
* `gold_chunk_keywords` — the loose signal: a list of substrings that
  must all appear (case-insensitive) in a gold chunk. Easy to author,
  noisier than IDs, but sufficient to validate the framework end-to-end
  before the labelling pass.
* `expected_numbers` — used only for `numeric-match`. Kept separate
  from keywords so the lexical answer scoring isn't pulled toward
  retrieval signals.

The runner prefers `gold_chunk_ids` when present and falls back to
`gold_chunk_keywords` otherwise. This means we can grow the benchmark
incrementally: new questions land in keyword-mode and get upgraded to
ID-mode as time permits.

## Anti-bias design

The single biggest credibility risk in any "I improved X by Y%" claim
is a benchmark that flatters the change being measured. The framework
is structured to make that hard:

1. **Pre-registration.** The benchmark is committed to git *before* any
   feature work that might be measured against it. The git timestamp on
   `benchmark.yaml` is the proof. Edits after that point are tracked in
   git history; a reviewer can see exactly which questions existed at
   each measurement.
2. **External question sourcing.** Questions are written from a generic
   ESG/sustainability analyst perspective, drawn from the report's own
   table of contents and from public frameworks (TCFD, SBTi, GRI), not
   from "what we expect a particular retrieval strategy to be good at".
3. **Per-category reporting, not just an aggregate.** Aggregate numbers
   hide trade-offs. The runner emits per-category breakdowns
   (`factoid`, `multi-hop`, `definitional`, `table-lookup`, `negative`)
   so a feature that wins on numerics but loses on definitions can't
   hide behind a flattering mean.
4. **Negative questions.** Some questions have answers that are *not*
   in the corpus. The expected behaviour is for retrieval to surface
   weak/off-topic chunks and for `/chat` to either refuse or admit the
   gap. This catches the failure mode where a retrieval improvement
   makes the system more confident at fabricating answers.
5. **Mix that excludes the change being measured.** No single category
   should dominate. If the SQL-as-tool feature wins uniformly across
   *every* category that's the suspicious outcome, not the reassuring
   one — pure-prose definitional questions should be roughly neutral
   to it.
6. **Different LLM as judge.** Once `judge.py` lands, the LLM grading
   the answers must not be the same model that generated them.
   Self-grading is a known biased setting.
7. **Fixed retrieval params across runs.** Only the strategy changes
   between runs. `k`, prompt templates, chunk sizes, and seeds stay
   constant, so any delta in metrics is attributable to the strategy
   under test rather than incidental tuning.

The result is a measurement harness whose conclusions a reviewer can
trust: not because the framework is exhaustive, but because the design
choices are visible and auditable.

## Roadmap (in this repo)

* `judge.py` — LLM-as-judge for faithfulness and answer relevance, with
  per-(question, answer) caching keyed on a content hash.
* Strict ID-level gold labelling for the existing 5 questions, then
  expansion to ~25 questions across the full document set.
* `sql` strategy adapter, once the structured-retrieval feature lands.
* Side-by-side report (`run_eval.py --compare vector hybrid sql`) that
  emits a single markdown table contrasting all strategies on the same
  benchmark, suitable for dropping straight into the project README.
