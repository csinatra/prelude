# Cost / Scope Tracking

Update this file whenever conditions, competition count, or seed count
change. Last updated: 2026-08-03 (voyage-4-large embedding swap; summary-unit
retrieval — corpus embed cost is now summaries, not code chunks). Prior: no
eval runs yet).

## Run matrix (POC scope — decided 2026-07-22)

| Condition | Competitions | Seeds | Agent runs |
|---|---|---|---|
| A | contingent matched anchor (see RESEARCH_DESIGN) | — | 0 unless triggered |
| B1 | 3–5 | 3 | 9–15 |
| B2 | 3–5 | 3 | 9–15 |
| C1 (pilot) | 3 | 1 | 3 |
| C2 | 3–5 | 3 | 9–15 |
| **Total** | | | **~30–48** |

Full Lite-22 (~94 runs, AWS-g5-class GPU) is the v1.5 plan if an initial
result is pursued.

## Cost estimate

- **GPU compute (dominant):** ~30–48 runs × Lambda A10 (~\$1.29/hr) at
  MLE-bench Lite runtimes, with a reduced per-run step/time cap (set from
  smoke timings — TBD) → **low hundreds of dollars, well under the v1 estimate**
- **API (spec pipeline + judge):** B2/C1/C2 synthesis, C2's four structured
  stages, judge passes — Sonnet for eval, Haiku elsewhere; spec-builds scale
  with competition×condition, not seeds → **~\$15–50** at POC scope
- **Embeddings:** `voyage-4-large` (switched 2026-08-03 — general-purpose fits
  the NL↔NL summary retrieval, \$0.12/M) has its own **200M free** grant
  (voyage-code-3's remaining ~182M does not carry over). Re-embedding the dev
  corpus (metadata + summaries) is a few M tokens; eval runs then cost only
  query embeddings → **~\$0**. The voyage-4 family shares one embedding space,
  so query-side can drop to a cheaper tier without re-embedding if ever needed.
- Contingent matched-A arm (triggers per RESEARCH_DESIGN.md Condition A
  note): **+GPU for one run per eval competition if triggered**; the small
  matched-A anchor comes free from registered smoke runs either way.

## One-time / infrastructure

- Notebook-summary ingestion (`ingest.ingest_summaries`): one Haiku call per
  unique notebook in the corpus slice — thousands of short calls, est.
  \$5–15, resumable. **Actual (2026-07-15): ~\$20 for 5,937 notebooks**
  (~\$0.003–0.005/notebook; estimate predated the notebook count).
- Full-corpus expansion (pre-production): the retrieval unit is the notebook
  summary, so the embedding cost is the full-Code4ML **summaries** (~55–110M
  tokens), not code chunks — those are no longer embedded (unqueried). That fits
  inside `voyage-4-large`'s **200M free** grant → ~\$0. Summary *generation* (the
  Haiku Batch job) is the real spend — see below. MLEModernizer
  ingestion is cloud-box disk/bandwidth, not API cost. Note: naively
  re-running Haiku
  summaries over full Code4ML (~20× notebooks) would be ~\$400 at the
  observed per-notebook rate — budget deliberately before that step.
  **Decision (2026-07-15): use the Anthropic Batch API (50% discount, same
  pinned SUMMARY_MODEL — no homogeneity impact) for the Code4ML expansion:
  ~\$200 for ~110–115k remaining notebooks.** MLEModernizer summary cost is
  NOT included and is unsized until the tarball is opened on the cloud box:
  two-level retrieval requires a level-one abstract per retrieval unit, so
  either its units get the same batched-Haiku treatment (cost scales with
  unit count, unknown) or native docs (READMEs/abstracts) serve as level-one
  text (embedding-only cost). Count units and check for native abstracts
  before estimating.
