"""Condition B baselines — unstructured knowledge provision (AssistedDS-style).

Two variants sharing one flat retrieval pass (single query = the raw
competition description, both collections, merged to top-k by similarity —
same corpus access as Condition C, minus the stage-directed structure):

- B1 (`run_b1`): retrieved excerpts formatted into a context block and
  injected into AIDE as-is. No LLM pass — the literal AssistedDS replication.
- B2 (`run_b2`): the same context block plus one freeform, schema-less LLM
  call producing prose advice. Controls for "any LLM preprocessing at all" so
  Condition C's advantage is attributable to structure specifically.

B2's prompt deliberately omits Condition C's RETRIEVAL_STANCE: uncritical
adoption of provided knowledge is the AssistedDS failure mode the baseline
must be free to exhibit.
"""

from pipeline.llm_client import call_llm_text
from pipeline.nodes import COMPETITION_METADATA, PRACTITIONER_KNOWLEDGE, _format_docs
from pipeline.retriever import RetrievedDoc, retrieve

BASELINE_K = 20  # 4x Condition C's per-stage k — same maximum document budget


def _flat_retrieve(*, raw_problem: str, competition_id: str) -> list[RetrievedDoc]:
    docs: list[RetrievedDoc] = []
    for collection in [COMPETITION_METADATA, PRACTITIONER_KNOWLEDGE]:
        docs.extend(
            retrieve(
                query=raw_problem,
                collection=collection,
                exclude_competition=competition_id,
                k=BASELINE_K,
            )
        )
    docs.sort(key=lambda doc: doc.similarity, reverse=True)
    return docs[:BASELINE_K]


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
        system=(
            "You are an experienced ML engineer advising a colleague on a machine learning "
            "competition. You are given the competition description and reference material from "
            "similar past competitions. Provide advice on how to approach the competition."
        ),
        user=(
            f"Competition description:\n{raw_problem}\n\n"
            f"Reference material from similar competitions:\n{context_block}"
        ),
        max_tokens=4096,
    )
    return {
        "condition": "B2",
        "competition_id": competition_id,
        "retrieved": [doc.model_dump() for doc in docs],
        "context_block": context_block,
        "advice": advice,
    }
