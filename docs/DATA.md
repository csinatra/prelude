# Research data handling

What is published, what stays local, and why. The rule this all follows: **the
results dataset contains exactly what supports the reported claims.** Extra data
is not extra transparency — it creates ambiguity about what was actually
analyzed.

## Stages

Runs are organized by *experiment stage*, and stages are isolated at the
filesystem level rather than by a field or a convention.

| stage | registry | artifacts | published |
|---|---|---|---|
| `dev` (default) | `results/runs_dev.jsonl` | `results/dev/{run_key}/` | no |
| `eval_v1` | `results/runs_eval_v1.jsonl` | `results/eval_v1/{run_key}/` | yes |

The active stage comes from `PRELUDE_REGISTRY_STAGE`, defaulting to `dev`, so
writing into an eval registry has to be asked for explicitly:

```bash
PRELUDE_REGISTRY_STAGE=eval_v1 python -m harness.batch ...
```

**Why separate files rather than a `stage` column.** `run_key` is
`{competition}_{condition}_{seed}` and carries no corpus identifier, so a run
rebuilt against a new corpus produces the *same* key as its predecessor — and
both the registry merge (`load_runs()` merges per `run_key`) and the artifact
path would silently blend the two. Separation makes that impossible rather than
merely detectable. It also makes the publish boundary enforceable: git tracks
files, not rows, so a single registry would force dev rows to be published
alongside eval results or the eval results to stay unpublished.

Exploratory data collected before the analysis plan was frozen cannot support
confirmatory claims. Runs made during development are archived under
`results/_archive_testing/` — kept, because they answer "what did you try?", but
never published and never poolable with eval results.

Append-only is a property *within* a stage. Retiring a stage means starting a new
file, never rewriting an existing one.

Cross-stage comparison stays available — `load_runs(stage=...)` reads a named
registry, so `eval_v1` and a later `eval_v1_5` can be read side by side. Report
that as **replication across corpus generations**, with the corpus difference
stated. Do not pool them: the paired within-competition statistics in
`analysis/stats.py` assume a single corpus and apply within a stage.

## What is published

| Published | Why it is needed |
|---|---|
| `results/runs_eval_*.jsonl` | the results table |
| `results/eval_*/{run_key}/spec.md` | the treatment itself; results are not reproducible without it |
| `.../manifest.json` | provenance: git commit, model, stage, corpus fingerprint |
| `.../retrievals.json`, `pipeline_output.json`, `llm_usage.json` | re-derive retrieval and spec-side cost |
| `.../journal.json`, `best_solution.py`, `prelude_token_usage.jsonl` | judge inputs (H2) and convergence measures (H3) |
| `results/corpus_export/manifest.json` | the corpus fingerprint results cite |
| `retrieval_audit.md`, `retrieval_diversity.json`, `summary_prompt_validation.json` | evidence `DECISIONS.md` cites for design choices |

Not published: dev and testing runs; `submission.csv` and `trajectory.log` (large,
and their information is already in the registry metrics and judged artifacts);
the ChromaDB store; raw Code4ML downloads.

## Corpus provenance

The corpus is **not reproducible from code and sources alone.** Summarization is
an LLM step, so re-running `ingest_summaries` against the same Code4ML CSVs with
the same pinned model yields different text. A pointer to Code4ML plus our ingest
code therefore does not reconstruct what the results were computed over — which
is precisely the case where preserving derived data is expected practice.

`python -m ingest.export_corpus` writes:

- `notebook_summaries.jsonl.gz` — one record per notebook (~18 MB). Deterministic:
  records sorted by id, gzip written with `mtime=0`, so an unchanged corpus
  exports byte-identically and its SHA-256 is stable.
- `manifest.json` — counts, model identifiers, and that SHA-256. Small and
  version-controlled.

Every run's `manifest.json` carries `corpus_sha256`, `corpus_documents`, and
`corpus_competitions`, so published results reference a hash. That lets the
deposit venue for the bytes (Git LFS or a DOI-issuing archive such as Zenodo) be
decided at writeup time without weakening the provenance chain.

The vector store is never an artifact: it is a derived binary index, ~964 MB, and
rebuilds from the exported text under the named embedding model.

**Fallback if the export is ever lost.** Every retrieved document appears
verbatim in the injected `spec.md` context block, so every document that
influenced a published result is recoverable from the published specs. The export
matters for re-running retrieval, not for auditing reported results.

## Before eval runs

1. `python -m ingest.export_corpus` — so runs carry a non-null corpus hash. A run
   whose provenance records a null hash cannot be tied to a specific corpus
   afterwards.
2. Confirm `PRELUDE_REGISTRY_STAGE` is set to the eval stage on both machines.
3. Confirm the eval registry does not already exist — it is append-only from
   first write.
