# Cost / Scope Tracking

Update this file whenever conditions, competition count, or seed count
change. Last updated: 2026-07-14 (design-review refactor; no eval runs yet).

## Run matrix (v1 plan)

| Condition | Competitions | Seeds | Agent runs |
|---|---|---|---|
| A | — (published baseline, cited) | — | 0 |
| B1 | 10 | 3 | 30 |
| B2 | 10 | 3 | 30 |
| C1 (pilot) | 3–4 | 1 | 3–4 |
| C2 | 10 | 3 | 30 |
| **Total** | | | **~94** |

## Cost estimate

- **GPU compute (dominant):** ~94 runs × AWS g5.xlarge (~$1.50/hr) at
  MLE-bench Lite runtimes → **~$500–600**
- **API (spec pipeline + judge):** B2/C1/C2 synthesis calls, C2's four
  structured stages, judge passes — Sonnet for eval runs, Haiku everywhere
  else → **~$50–100**
- **Embeddings:** within Voyage's 200M free-token allowance (dev corpus
  ~13M tokens; notebook summaries small) → **~$0**
- **Total: ~$600–700**, dominated by GPU compute, not API cost.

## One-time / infrastructure

- Notebook-summary ingestion (`ingest.ingest_summaries`): one Haiku call per
  unique notebook in the corpus slice — thousands of short calls, est.
  $5–15, resumable. **Actual (2026-07-15): ~$20 for 5,937 notebooks**
  (~$0.003–0.005/notebook; estimate predated the notebook count).
- Full-corpus expansion (pre-production): re-embedding at ~10× dev-slice
  volume stays within the free-token allowance; MLEModernizer ingestion is
  cloud-box disk/bandwidth, not API cost. Note: naively re-running Haiku
  summaries over full Code4ML (~20× notebooks) would be ~$400 at the
  observed per-notebook rate — budget deliberately before that step.
  **Decision (2026-07-15): use the Anthropic Batch API (50% discount, same
  pinned SUMMARY_MODEL — no homogeneity impact) for the Code4ML expansion:
  ~$200 for ~110–115k remaining notebooks.** MLEModernizer summary cost is
  NOT included and is unsized until the tarball is opened on the cloud box:
  two-level retrieval requires a level-one abstract per retrieval unit, so
  either its units get the same batched-Haiku treatment (cost scales with
  unit count, unknown) or native docs (READMEs/abstracts) serve as level-one
  text (embedding-only cost). Count units and check for native abstracts
  before estimating.
