"""Central tunables for retrieval and condition budgets.

One place to calibrate; nothing here is hardcoded design. Parity invariant to
preserve when editing: Condition B's flat budget should match the staged
conditions' total (METADATA_K ~ parse; BASELINE_N_NOTEBOOKS x
BASELINE_CHUNKS_PER_NOTEBOOK ~ 3 practitioner stages x STAGE_N_NOTEBOOKS x
STAGE_CHUNKS_PER_NOTEBOOK) so flat-vs-staged comparisons are not confounded
by document budget.
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
PRACTITIONER_KNOWLEDGE = "practitioner_knowledge"
NOTEBOOK_SUMMARIES = "notebook_summaries"

# ── Staged conditions (C1/C2) ───────────────────────────────────────
RETRIEVAL_K = 5  # parse stage, competition_metadata (flat chunk retrieval)
# Per-stage notebook-card budget (review-doc starting range; calibrate
# before eval runs): 3 practitioner stages x 8 x 3 = 72 chunks.
STAGE_N_NOTEBOOKS = 8
STAGE_CHUNKS_PER_NOTEBOOK = 3

# ── Condition B (flat) ──────────────────────────────────────────────
METADATA_K = 5  # parity with the parse stage
BASELINE_N_NOTEBOOKS = 24  # parity with 3 stages x STAGE_N_NOTEBOOKS
BASELINE_CHUNKS_PER_NOTEBOOK = 3

# ── Retrieval quality ───────────────────────────────────────────────
# Deliberately unset by default — calibrate against the real corpus before
# eval runs (observed good-match similarities: ~0.48-0.66; sweep ~0.45-0.70).
SIMILARITY_THRESHOLD: float | None = (
    float(os.environ["SIMILARITY_THRESHOLD"]) if "SIMILARITY_THRESHOLD" in os.environ else None
)
CHROMA_PATH = os.environ.get("CHROMA_PATH", "data/chroma")
