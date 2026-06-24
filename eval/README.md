# Retrieval Evaluation Framework

A small scorecard for the agent's retrieval and answer quality. The point is
to make claims like "this change helped" something you can check rather
than something you have to take on trust.

## Why this exists

The base repo ships an agentic RAG stack but no way to measure it. Without
measurement:

- You can't tell if a change (new retrieval strategy, different chunk size,
  different embedding model) helped or hurt.
- You can't say a feature improved things in a review or on a resume.
- Regressions stay invisible until a user notices.

This framework gives every retrieval strategy the same scorecard so
deltas are real and reproducible.

## Layout

```
eval/
  benchmark.yaml    The questions plus ground truth (gold chunks + numbers).
  metrics.py        Pure functions: recall@k, MRR, precision@k, numeric_match.
  strategies.py     Adapters for vector / hybrid / sql / chat.
  judge.py          LLM-as-judge with on-disk content-hash cache.
  run_eval.py       CLI runner; writes markdown + JSON to results/.
  results/          gitignored timestamped run outputs.
  README.md
```

## Running it

The runner hits the live API on `localhost:8058`, so the stack must be up:

```bash
make up
docker exec agent_api python -m eval.run_eval --strategy vector --k 5
docker exec agent_api python -m eval.run_eval --strategy hybrid --k 5
docker exec agent_api python -m eval.run_eval --strategy sql    --k 5
docker exec agent_api python -m eval.run_eval --strategy chat   --k 5 --judge
```

Each run prints a markdown report and (unless `--no-write` is passed)
writes a `<run>.md` + `<run>.json` pair to `eval/results/` so historical
runs can be compared side by side.

## Metrics

| Metric             | Type      | What it measures                                                                                                                                                |
| ------------------ | --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| recall@k           | retrieval | Of the gold chunk **IDs** for a question, what fraction appeared in the top-k retrieved IDs.                                                                    |
| content-recall@k   | retrieval | Of the top-k retrieved chunks, did any one of them *contain all the answer keywords*. Chunking-independent — the right metric when comparing two chunkings.     |
| precision@k        | retrieval | Of the top-k retrieved IDs, what fraction were gold. Only reported when strict gold IDs are available.                                                          |
| MRR                | retrieval | Mean of `1 / rank` of the first gold-ID hit. Punishes putting the right chunk at rank 5 instead of rank 1.                                                      |
| content-MRR        | retrieval | Same as MRR but using keyword content match, so it's also chunking-independent.                                                                                 |
| numeric-match      | answer    | For chat runs: fraction of `expected_numbers` from the benchmark that appear in the agent's answer.                                                             |
| faithfulness       | answer    | LLM-judged (`judge.py`, `--judge` flag): does every claim in the answer have support in the retrieved chunks?                                                   |
| answer relevance   | answer    | LLM-judged: does the answer actually address the question that was asked?                                                                                       |

> **When to use which recall.** Use `recall@k` for the cleanest
> apples-to-apples comparison of two retrievers *against the same
> chunking* (it's the strictest signal). Use `content-recall@k` when
> the chunking itself is the thing you're comparing — chunk UUIDs
> shift, but the underlying answer text doesn't. Reporting both makes
> regressions vs. genuine improvements distinguishable.

### Ground-truth signals

Each question in `benchmark.yaml` carries up to three signals:

- **`gold_chunk_ids`** — the strict signal. The UUIDs of the chunks that
  contain the answer. These were bootstrapped semi-automatically by
  `scripts/label_gold_ids.py`, which probes the `chunks` table directly
  with `ILIKE` over the question's keywords + expected numbers. The
  labelling never reads `kpi_facts` and never invokes the agent, so the
  gold IDs are independent of the SQL feature being measured.
- **`gold_chunk_keywords`** — the loose signal. A list of substrings that
  must all appear (case-insensitive) in a gold chunk. Used as a fallback
  for negative questions, and as a sanity check on the ILIKE bootstrap.
- **`expected_numbers`** — only used for `numeric_match`. Kept separate
  from keywords so answer scoring isn't pulled toward retrieval signals.

22 of 25 questions are ID-labelled; the other 3 are intentional
"unanswerable" negatives where the judge grades refusal instead.

## Anti-bias design

The biggest credibility risk in any "I improved X by Y%" claim is a
benchmark that flatters the change. The framework is structured to make
that hard:

1. **Pre-registration.** The benchmark is committed to git *before* the
   feature being measured against it. Git history is the audit trail.
2. **External question sourcing.** Questions are written from a generic
   ESG / sustainability analyst's perspective — drawn from the report's
   own table of contents and public frameworks (TCFD, SBTi, GRI) — not
   from "what we expect this retrieval strategy to be good at".
3. **Per-category reporting.** Aggregates hide trade-offs. The runner
   emits per-category breakdowns (`factoid`, `multi-hop`, `table-lookup`,
   `negative`) so a feature that wins on numerics but loses on definitions
   can't hide behind a flattering mean.
4. **Negative questions.** Some questions intentionally don't have
   answers in the corpus. The expected behaviour is for retrieval to
   surface weak/off-topic chunks and for `/chat` to refuse or admit the
   gap. This catches the case where a retrieval improvement makes the
   system more confident at fabricating.
5. **Independent gold labelling.** The script that picked gold chunk IDs
   never reads `kpi_facts` and never calls the agent. It's pure ILIKE
   matching on chunk content. So the gold labels can't favour the SQL
   feature. Two scripts cover this surface — `label_gold_ids.py` for
   the initial pass and `relabel_gold_ids.py` for in-place refresh when
   the chunker changes — and both use the same content-only probe.
6. **Chunking-independent metric.** `content_recall_at_k` is reported
   alongside `recall_at_k` precisely so a chunker change isn't
   penalised for finding the same answer in a different chunk UUID.
   This is the metric to trust for cross-chunker comparisons.
7. **Different LLM for judging.** When using `--judge`, the model
   grading the answers should not be the same one that generated them
   (`JUDGE_MODEL` env var). Self-grading is a known biased setting.
8. **Fixed retrieval params across runs.** Only the strategy changes
   between runs. `k`, prompt templates, chunk sizes, and seeds stay
   constant, so any delta is attributable to the strategy.

The point isn't that the framework is exhaustive — it's that the design
choices are visible and you can argue with them.
