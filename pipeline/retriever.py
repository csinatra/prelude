"""Corpus retrieval — the single retrieval seam for pipeline nodes.

Mirrors the call_llm pattern: every node calls retrieve(), tests mock it,
LangSmith traces it. Leave-one-out is enforced here: every query excludes the
competition currently being specified, so the pipeline can never see the
held-out competition's own artifacts.

SIMILARITY_THRESHOLD is unset by default — to be calibrated against the real
corpus before eval runs (see implementation brief).
"""

import chromadb
from chromadb.api.models.Collection import Collection
from langsmith import traceable
from pydantic import BaseModel

from pipeline.config import (
    CHROMA_PATH,
    NOTEBOOK_SUMMARIES,
    PRACTITIONER_KNOWLEDGE,
    SIMILARITY_THRESHOLD,
)
from pipeline.embeddings import embed

_client: chromadb.ClientAPI | None = None


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
def retrieve_two_level(
    *,
    query: str,
    exclude_competition: str,
    n_notebooks: int = 8,
    chunks_per_notebook: int = 3,
    score_threshold: float | None = SIMILARITY_THRESHOLD,
) -> list[RetrievedDoc]:
    """Notebook-then-chunk retrieval over practitioner knowledge.

    (a) query notebook_summaries (leave-one-out filter) for top-N notebooks,
    (b) query practitioner_knowledge restricted to those kaggle_ids for top-M
    chunks per notebook. No leave-one-out re-check needed at (b): step (a)
    already excluded the current competition's notebooks.

    This is the retrieval unit for ALL conditions' practitioner-knowledge
    access ("notebook cards" — coherent per-notebook excerpt groups): B calls
    it once with a flat query; C1/C2 call it per stage with directed queries.
    Holding the unit constant means the flat-vs-staged comparison isolates
    query structure, not retrieval granularity. The chunk-level `retrieve()`
    above remains for competition_metadata (no notebook structure).
    """
    query_vector = embed(texts=[query], input_type="query")[0]
    summaries = _get_collection(name=NOTEBOOK_SUMMARIES).query(
        query_embeddings=[query_vector],
        n_results=n_notebooks,
        where={"competition_id": {"$ne": exclude_competition}},
        include=["metadatas"],
    )
    kaggle_ids = [int(metadata["kaggle_id"]) for metadata in summaries["metadatas"][0]]

    chunks_collection = _get_collection(name=PRACTITIONER_KNOWLEDGE)
    docs: list[RetrievedDoc] = []
    for kaggle_id in kaggle_ids:
        result = chunks_collection.query(
            query_embeddings=[query_vector],
            n_results=chunks_per_notebook,
            where={"kaggle_id": kaggle_id},
            include=["documents", "metadatas", "distances"],
        )
        for doc_id, text, metadata, distance in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            similarity = 1.0 - distance
            if score_threshold is not None and similarity < score_threshold:
                continue
            docs.append(
                RetrievedDoc(
                    doc_id=doc_id,
                    competition_id=str(metadata["competition_id"]),
                    source_type=str(metadata["source_type"]),
                    text=text,
                    similarity=similarity,
                    kaggle_id=kaggle_id,
                )
            )
    return docs
