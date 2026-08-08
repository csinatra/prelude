"""Corpus retrieval — the single retrieval seam for pipeline nodes.

Mirrors the call_llm pattern: every node calls retrieve(), tests mock it,
LangSmith traces it. Leave-one-out is enforced here: every query excludes the
competition currently being specified, so the pipeline can never see the
held-out competition's own artifacts.

SIMILARITY_THRESHOLD is unset by default — to be calibrated against the real
corpus before eval runs (see implementation brief).
"""

import logging

import chromadb
from chromadb.api.models.Collection import Collection
from langsmith import traceable
from pydantic import BaseModel

from pipeline.config import CHROMA_PATH, SIMILARITY_THRESHOLD
from pipeline.embeddings import embed

logger = logging.getLogger("pipeline.retriever")

_client: chromadb.ClientAPI | None = None

# Process-local ledger of distinct-document shortfalls, drained per run by
# harness.runner into the run artifact (same pattern as llm_client's usage log).
# A shortfall means a staged retrieval could not contribute its full k NEW
# distinct docs, so the parity invariant is not met for that run and the
# analysis must know. Kept out of stdout-only logging for exactly that reason.
_shortfall_log: list[dict] = []


def reset_shortfalls() -> None:
    _shortfall_log.clear()


def shortfall_log() -> list[dict]:
    """Recorded parity shortfalls since the last reset, in call order."""
    return list(_shortfall_log)


def _record_shortfall(*, shortfall: dict) -> None:
    _shortfall_log.append(shortfall)
    logger.warning(
        "retrieval parity shortfall: %s contributed %d/%d new distinct docs "
        "(pool=%d, repeats=%d)",
        shortfall["collection"],
        shortfall["actual_distinct"],
        shortfall["requested_distinct"],
        shortfall["pool_size"],
        shortfall["repeats_in_primary"],
    )


class RetrievedDoc(BaseModel):
    doc_id: str
    competition_id: str
    source_type: str
    text: str
    similarity: float
    kaggle_id: int | None = None  # source notebook, when the doc is a code chunk


def _get_collection(*, name: str) -> Collection:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _client.get_collection(name=name)


@traceable(run_type="retriever")
def retrieve(
    *,
    query: str,
    collection: str,
    exclude_competition: str,
    k: int = 5,
    score_threshold: float | None = SIMILARITY_THRESHOLD,
) -> list[RetrievedDoc]:
    """Top-k cosine retrieval with the leave-one-out competition filter applied."""
    query_vector = embed(texts=[query], input_type="query")[0]
    result = _get_collection(name=collection).query(
        query_embeddings=[query_vector],
        n_results=k,
        where={"competition_id": {"$ne": exclude_competition}},
        include=["documents", "metadatas", "distances"],
    )
    docs = []
    for doc_id, text, metadata, distance in zip(
        result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        similarity = 1.0 - distance  # cosine distance -> similarity
        if score_threshold is not None and similarity < score_threshold:
            continue
        docs.append(
            RetrievedDoc(
                doc_id=doc_id,
                competition_id=str(metadata["competition_id"]),
                source_type=str(metadata.get("source_type", collection)),
                text=text,
                similarity=similarity,
                kaggle_id=metadata.get("kaggle_id"),
            )
        )
    return docs


@traceable(run_type="retriever")
def retrieve_with_topup(
    *,
    query: str,
    collection: str,
    exclude_competition: str,
    k: int,
    seen: set[str],
    score_threshold: float | None = SIMILARITY_THRESHOLD,
) -> list[RetrievedDoc]:
    """Directed retrieval with cross-stage top-up for distinct-document parity.

    Returns the stage's context: its top-k docs by `query`, where a doc already
    in `seen` from an earlier stage is RETAINED (re-selection across stages is
    an importance signal, not a duplicate to drop) AND, for each such repeat,
    the next-best doc not yet in `seen` is appended. Every call therefore
    contributes k documents new to `seen`, so the staged conditions surface the
    same distinct-document budget as Condition B's flat pass (parity is on
    distinct documents, not tokens). `seen` is mutated with the new doc_ids.
    See docs/DECISIONS.md (2026-08-03 retrieval-unit entry).
    """
    pool = retrieve(
        query=query,
        collection=collection,
        exclude_competition=exclude_competition,
        k=k * 6,  # headroom for top-up: covers up to k repeats past the primary k
        score_threshold=score_threshold,
    )
    primary = pool[:k]
    primary_ids = {doc.doc_id for doc in primary}
    repeats = sum(1 for doc in primary if doc.doc_id in seen)
    topups: list[RetrievedDoc] = []
    for doc in pool[k:]:
        if len(topups) >= repeats:
            break
        if doc.doc_id not in seen and doc.doc_id not in primary_ids:
            topups.append(doc)
    context = primary + topups
    new_distinct = len({doc.doc_id for doc in context} - seen)
    for doc in context:
        seen.add(doc.doc_id)
    if new_distinct < k:
        # Parity depends on every stage contributing k NEW distinct docs. A small
        # or heavily-overlapping corpus slice can exhaust the unseen pool, and a
        # silent shortfall would degrade parity without leaving a trace in the
        # run record. Record it rather than papering over it: the run stays
        # valid, but the analysis needs to know the budget was not met.
        _record_shortfall(
            shortfall={
                "collection": collection,
                "requested_distinct": k,
                "actual_distinct": new_distinct,
                "pool_size": len(pool),
                "repeats_in_primary": repeats,
                "query": query[:200],
            }
        )
    return context
