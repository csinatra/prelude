"""Condition B baselines — unstructured knowledge provision (AssistedDS-style).

Two variants sharing one flat retrieval pass (single query = the raw
problem description; no stage-directed queries):

- B1 (`run_b1`): retrieved material formatted into a context block and
  injected into AIDE as-is. No LLM pass.
- B2 (`run_b2`): the same context block plus one freeform, schema-less LLM
  call producing prose advice. Controls for "any LLM preprocessing at all" so
  the staged conditions' advantage is attributable to structure specifically.

Retrieval unit is held constant with the staged conditions: practitioner
knowledge arrives as notebook summaries via a flat retrieve() here (stage-
directed queries in C1/C2), plus flat chunk retrieval over competition_metadata.
Budgets are parity-matched to C's staged totals on DISTINCT documents
(metadata k=5 ~ parse; BASELINE_N_NOTEBOOKS summaries == 3 stages x
STAGE_N_NOTEBOOKS distinct, achieved in C via cross-stage top-up) — calibration
knobs, not hardcoded design.

B2's prompt deliberately omits Condition C2's RETRIEVAL_STANCE: uncritical
adoption of provided knowledge is the AssistedDS failure mode the baseline
must be free to exhibit.
"""

from pipeline.config import (
    BASELINE_N_NOTEBOOKS,
    COMPETITION_METADATA,
    METADATA_K,
    NOTEBOOK_SUMMARIES,
)
from pipeline.llm_client import SYNTHESIS_MAX_TOKENS, call_llm_text
from pipeline.nodes import _format_docs
from pipeline.retriever import RetrievedDoc, retrieve

# Shared by B2 and C1: freeform synthesis over provided context, deliberately
# without Condition C2's RETRIEVAL_STANCE (uncritical adoption is the
# AssistedDS failure mode these conditions must be free to exhibit).
FREEFORM_SYSTEM = (
    "You are an experienced ML engineer advising a colleague on a machine learning "
    "problem. You are given the problem description and reference material from "
    "similar past problems. Provide advice on how to approach the problem."
)


def _flat_retrieve(*, raw_problem: str, competition_id: str) -> list[RetrievedDoc]:
    metadata_docs = retrieve(
        query=raw_problem,
        collection=COMPETITION_METADATA,
        exclude_competition=competition_id,
        k=METADATA_K,
    )
    # One flat pass over notebook summaries — a single top-k has no internal
    # repeats, so B naturally sees BASELINE_N_NOTEBOOKS distinct summaries (no
    # top-up needed; that machinery exists only to restore distinct parity for
    # the staged conditions).
    notebook_docs = retrieve(
        query=raw_problem,
        collection=NOTEBOOK_SUMMARIES,
        exclude_competition=competition_id,
        k=BASELINE_N_NOTEBOOKS,
    )
    return metadata_docs + notebook_docs


def run_b1(*, raw_problem: str, competition_id: str) -> dict:
    """Raw retrieval block — no LLM pass."""
    docs = _flat_retrieve(raw_problem=raw_problem, competition_id=competition_id)
    return {
        "condition": "B1",
        "competition_id": competition_id,
        "retrieved": [doc.model_dump() for doc in docs],
        "context_block": _format_docs(docs),
    }


def run_b2(*, raw_problem: str, competition_id: str) -> dict:
    """Flat retrieval plus one freeform LLM pass."""
    docs = _flat_retrieve(raw_problem=raw_problem, competition_id=competition_id)
    context_block = _format_docs(docs)
    advice = call_llm_text(
        system=FREEFORM_SYSTEM,
        user=(
            f"Problem description:\n{raw_problem}\n\n"
            f"Reference material from similar problems:\n{context_block}"
        ),
        max_tokens=SYNTHESIS_MAX_TOKENS,
    )
    return {
        "condition": "B2",
        "competition_id": competition_id,
        "retrieved": [doc.model_dump() for doc in docs],
        "context_block": context_block,
        "advice": advice,
    }
