# Prelude — Research Design

*v1 scope. Drafted July 2026, before any evaluation run. Amendments after
runs begin must be dated and listed at the bottom.*

## Research question and hypothesis

**Question:** Does structured reasoning over retrieved prior work improve ML
agent performance on benchmark tasks beyond what unstructured knowledge
provision achieves?

**Falsifiable hypothesis (H1):** On MLE-bench Lite competitions, an AIDE
agent given Condition C2's structured specification will achieve a higher
Any-Medal rate than the same agent given Condition B's unstructured context,
which in turn will not significantly outperform the published no-assistance
baseline (A). H1 is falsified if C2 fails to separate from B beyond seed
noise, or if unstructured provision alone matches structured specification.

**Secondary hypothesis (H2, mechanistic):** C2's advantage, if any, is
mediated by specification flags being acted on — flag categories with higher
action rates (per the frozen judge rubric) contribute disproportionately to
outcome differences.

## Related work positioning

- **MLE-bench** (OpenAI, ICLR 2025): evaluation infrastructure and the
  Condition A baseline. Measures agent capability; does not study what
  knowledge or specification the agent starts with.
- **AssistedDS** (EMNLP 2025): showed LLMs adopt provided unstructured
  knowledge uncritically. Defines Condition B's design; does not test
  structured alternatives.
- **CatDB** (VLDB 2025): closest analog — catalog-grounded data-science
  automation. Assumes a populated catalog; does not address reasoning about
  unknown/missing signals, and provides no staged specification process.
- **Yang et al. 2023, "LLMs as Optimizers":** conceptual grounding for
  prompt-level structured reasoning; not applied to ML problem
  specification.

## Experimental design

Five conditions, structured as a 2 (retrieval: flat vs staged) × 3
(synthesis: none vs freeform vs structured-critical) grid with three cells
built plus one grid-external anchor:

| Condition | Retrieval | Synthesis | Isolates (vs) |
|---|---|---|---|
| A | none | none | published MLE-bench baseline (cited, not run) |
| B1 | flat, single query | none — raw context block | knowledge provision per se (vs A) |
| B2 | flat, single query | freeform, stance-free | LLM preprocessing (vs B1) |
| C1 | staged, 4 directed queries | freeform, stance-free (B2's path) | staged retrieval (vs B2) |
| C2 | staged, 4 directed queries | structured schemas + critical-integration stance | synthesis structure (vs C1) |

Design notes:

- **Retrieval unit held constant.** All conditions receive practitioner
  knowledge as "notebook cards" (top-M chunks from each of N notebooks
  surfaced via a summaries index) plus flat competition-metadata chunks.
  B selects notebooks with one flat query; C1/C2 with four stage-directed
  queries. This confines the flat-vs-staged contrast to query structure
  rather than retrieval granularity.
- **Document budgets parity-matched** via `pipeline/config.py` (B's flat
  budget = staged conditions' total). Exact values are calibration
  parameters, not design constants.
- **C1 runs parse's structured extraction** (task type, metric, goal) solely
  to build the directed queries — its synthesis input is description +
  context block only, identical in form to B2's. This is inherent to
  "staged": directed queries must be directed by something.
- **C1 is a pilot condition:** 3–4 competitions, single seed, not the full
  matrix. Built to run at full scale; the harness defaults to the pilot
  subset.
- AIDE scaffold, agent model, and MLE-bench grading are held constant across
  all run conditions.

## Corpus construction

- **Current dev corpus:** Code4ML filtered to MLE-bench Lite-22 competitions
  — 62,379 code chunks (deduped from 132,796 cell-level blocks), 1,005
  competition-metadata chunks (1,156 Code4ML descriptions + 22 mle-bench
  `description.md`), plus a `notebook_summaries` collection (one Haiku
  abstract per unique notebook) as level one of two-level retrieval.
- **Planned before production runs:** expansion to full Code4ML (~2.74M
  blocks, ~1,150 competitions; the Lite slice is ~5%) and MLEModernizer
  (107 GB replication package — cloud-box ingestion, separate swap).
- **Leave-one-out:** every retrieval carries
  `competition_id != current_competition`, enforced inside the retrieval
  seam (`pipeline/retriever.py`); for two-level retrieval the filter applies
  at the notebook level, which subsumes the chunk level. A code path
  bypassing this filter is solution leakage (CLAUDE.md core constraint 5).
- **Two-level rationale:** flat chunk retrieval returns decontextualized
  fragments from arbitrary notebooks; notebook-then-chunk retrieval returns
  coherent excerpt groups from notebooks selected as wholes, and lets
  retrieval reason at the level practitioners work at (a notebook is an
  approach; a chunk is a step).

## Outcome metrics (defined in advance)

**Primary:** Any-Medal rate (MLE-bench grading), mean ± one standard error
across 3 seeds per competition. Acknowledged as low-powered at n≈10
competitions; treated as directional, not confirmatory.

**Secondary (higher resolution):**
1. Leaderboard percentile of the final submission
2. Valid-submission rate (fraction of runs producing a gradeable submission)
3. Time-to-first-valid-submission (wall-clock within the AIDE run)

## Mechanistic evaluation

Per-flag flag→action→outcome judging against the **frozen rubric**
(`docs/JUDGE_RUBRIC.md`; frozen before any run, amendment-controlled):
each `SpecificationFlag` from a C2 run is classified `not_acted_on` /
`acted_on_unclear` / `acted_on_positive` by an LLM judge
(`analysis/judge.py`) that sees the solution artifacts but never the score.
Aggregation per category: detection rate, action rate, outcome
contribution, and retrieval-grounded fraction (non-empty
`evidence_doc_ids`). Requires the artifact preservation layout in
`analysis/artifacts.py`.

## Threats to validity

- **Construct validity (acknowledged, central):** Kaggle competitions are
  *pre-specified by construction* — task, metric, and data are given, which
  limits the available specification-failure signal that Prelude is designed
  to catch in production settings. Mitigation: competition selection favors
  Lite competitions with known data quirks (leakage paths, temporal
  structure, measurement gaps). A synthetic-corruption arm (deliberately
  under-specified task descriptions) is named as future work, not current
  scope.
- **Budget/attention confounds:** document budgets are parity-matched; token
  counts of injected artifacts are logged per run and reported.
- **Judge circularity:** rubric frozen pre-run; judge blinded to outcomes;
  evidence quotes required.
- **Corpus coverage asymmetry:** 18/22 Lite competitions have practitioner
  notebooks; retrieved-doc counts are reported per competition per
  condition as descriptive statistics.
- **Low n:** primary metric is directional; secondary metrics and
  mechanistic analysis carry the evidential weight.

## Roadmap

**v1 (this design):** everything above — B1/B2/C1-pilot/C2 on ~10 Lite
competitions, 3 seeds (1 for C1 pilot), mechanistic judging, writeup.

**v2 (explicitly out of scope):** MLE-Dojo / interactive specification
reasoning; iteration depth as an experimental variable; real
organizational-corpus case study; synthetic-corruption arm.

## Amendments

(none)
