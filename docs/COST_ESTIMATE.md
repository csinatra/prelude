# Cost / Scope Tracking

Update this file whenever conditions, competition count, or seed count change.
Last updated: 2026-08-11 (batch-summary re-cost after the richer prompt;
superseded code-chunk-era notes collapsed into History). No eval runs yet.

## Run matrix (POC scope — decided 2026-07-22)

| Condition | Competitions | Seeds | Agent runs |
|---|---|---|---|
| A | contingent matched anchor (see RESEARCH_DESIGN) | — | 0 unless triggered |
| B1 | 3–5 | 3 | 9–15 |
| B2 | 3–5 | 3 | 9–15 |
| C1 (pilot) | 3 | 1 | 3 |
| C2 | 3–5 | 3 | 9–15 |
| **Total** | | | **~30–48** |

Scope is bounded by budget, not by a judgment that this n is scientifically
correct (RESEARCH_DESIGN.md, Roadmap). Full Lite-22 (~94 runs) is the v1.5 plan
if an initial result is pursued.

## Cost estimate

- **GPU compute (dominant):** ~30–48 runs × Lambda A10 (~\$1.29/hr) at MLE-bench
  Lite runtimes, with a reduced per-run step/time cap set from smoke timings
  (TBD) → **low hundreds of dollars**
- **API (spec pipeline + judge):** B2/C1/C2 synthesis, C2's four structured
  stages, and judge passes. Sonnet for eval, Haiku elsewhere. Spec builds scale
  with competition × condition, not with seeds → **~\$15–50** at POC scope
- **Embeddings:** `voyage-4-large` at \$0.12/M with its own **200M free** grant.
  Re-embedding the dev corpus is a few M tokens and eval runs cost only query
  embeddings → **~\$0**. The voyage-4 family shares one embedding space, so the
  query side can drop to a cheaper tier without re-embedding if volume ever
  demands it.
- **Judge validation:** human labeling of a 20–30 item sample → **\$0**, review
  time only (docs/JUDGE_VALIDATION.md)
- **Contingent matched-A arm:** triggers per the RESEARCH_DESIGN Condition A
  note → **+GPU for one run per eval competition if triggered**. The small
  matched-A anchor comes free from registered smoke runs either way.

## One-time / infrastructure

- **Dev-corpus summaries (Lite-22, 5,937 notebooks):** one pinned-Haiku call per
  notebook, resumable. Actual on the original prompt (2026-07-15): **~\$20**.
  The 2026-08-03 prompt is richer (60k-char input, up to 1024 output tokens), so
  a rebuild runs somewhat higher.
- **Full-Code4ML expansion (~110–115k remaining notebooks):** summary
  *generation* is the real spend, not embedding. Via the Anthropic Batch API
  (50% discount, same pinned `SUMMARY_MODEL`, no homogeneity impact) →
  **~\$250–300** under the current prompt, revised up from the ~\$200 estimated
  on 2026-07-15 against the shorter prompt. Embedding the resulting summaries
  (~55–110M tokens) fits inside the free grant → **~\$0**.
- **MLEModernizer:** ingestion is cloud-box disk and bandwidth, not API cost.
  Summary cost is **not** included above and stays unsized until the tarball is
  opened on the box. Each retrieval unit needs an abstract, so either its units
  get the same batched-Haiku treatment (cost scales with unit count, unknown) or
  native README/abstract text serves directly (embedding-only cost). Count units
  and check for native abstracts before estimating.

## History

Superseded cost models, kept so past decisions stay legible:

- **Code-chunk retrieval era (through 2026-08-03).** When the retrieval unit was
  the code chunk, the dominant embedding line was practitioner chunks at an
  estimated ~130–260M tokens. Those chunks are no longer embedded or queried
  (DECISIONS.md 2026-08-03), so corpus embedding cost is now summaries only.
- **`voyage-code-3` grant.** The earlier one-time ~200M grant (~182M remaining)
  does not carry over to `voyage-4-large`, which has its own 200M.
