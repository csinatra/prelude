"""Central tunables for retrieval and condition budgets.

One place to calibrate; nothing here is hardcoded design. Parity invariant to
preserve when editing: Condition B and the staged conditions must reason over
the same number of DISTINCT notebook-summary documents (plus equal metadata).
B draws BASELINE_N_NOTEBOOKS summaries in one flat pass; the staged conditions
draw STAGE_N_NOTEBOOKS per directed stage across 3 stages, with cross-stage
top-up (retriever.retrieve_with_topup) so each stage contributes
STAGE_N_NOTEBOOKS NEW distinct docs. Hence BASELINE_N_NOTEBOOKS ==
3 x STAGE_N_NOTEBOOKS, and METADATA_K == RETRIEVAL_K. Parity is on distinct
documents, not tokens (duplicates re-surfaced across stages are retained as an
importance signal and add logged tokens). See docs/DECISIONS.md (2026-08-03).
"""

import os

# ── Models ──────────────────────────────────────────────────────────
# Eval-run spec-pipeline model, pinned pre-run (recorded 2026-07-19 in
# docs/RESEARCH_DESIGN.md). Set MODEL to this value for runs whose results
# go in the writeup; dev iteration stays on Haiku. Distinct from the AIDE
# agent model (cloudbox/agents/aide-prelude/config.yaml), which has its
# own pin.
EVAL_MODEL = "claude-sonnet-5"

# ── Collections ─────────────────────────────────────────────────────
COMPETITION_METADATA = "competition_metadata"
# Collection DROPPED at the voyage-4-large re-ingest (2026-08-03): unqueried since
# the retrieval unit became the notebook summary, and its voyage-code-3 vectors are
# an incompatible-space liability. Constant + ingest_notebooks module kept as a
# dormant rebuild path if curated-cell grounding is ever revived (re-embed from
# scratch). Not referenced by any spec-time code.
PRACTITIONER_KNOWLEDGE = "practitioner_knowledge"
NOTEBOOK_SUMMARIES = "notebook_summaries"

# ── Staged conditions (C1/C2) ───────────────────────────────────────
RETRIEVAL_K = 5  # parse stage, competition_metadata (flat chunk retrieval)
STAGE_N_NOTEBOOKS = 8  # distinct notebook-summary docs contributed per directed stage

# ── Condition B (flat) ──────────────────────────────────────────────
METADATA_K = 5  # parity with the parse stage
BASELINE_N_NOTEBOOKS = 24  # parity: == 3 staged stages x STAGE_N_NOTEBOOKS distinct summaries

# ── Retrieval quality ───────────────────────────────────────────────
# Deliberately unset by default — calibrate against the real corpus before
# eval runs (observed good-match similarities: ~0.48-0.66; sweep ~0.45-0.70).
SIMILARITY_THRESHOLD: float | None = (
    float(os.environ["SIMILARITY_THRESHOLD"]) if "SIMILARITY_THRESHOLD" in os.environ else None
)
CHROMA_PATH = os.environ.get("CHROMA_PATH", "data/chroma")
