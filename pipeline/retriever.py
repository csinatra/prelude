"""Corpus retrieval — the single retrieval seam for pipeline nodes.

Mirrors the call_llm pattern: every node calls retrieve(), tests mock it,
LangSmith traces it. Leave-one-out is enforced here: every query excludes the
competition currently being specified, so the pipeline can never see the
held-out competition's own artifacts.

SIMILARITY_THRESHOLD is unset by default — to be calibrated against the real
corpus before eval runs (see implementation brief).
"""

import os

import chromadb
from chromadb.api.models.Collection import Collection
from langsmith import traceable
from pydantic import BaseModel

from pipeline.embeddings import embed

CHROMA_PATH = os.environ.get("CHROMA_PATH", "data/chroma")
SIMILARITY_THRESHOLD = (
    float(os.environ["SIMILARITY_THRESHOLD"]) if "SIMILARITY_THRESHOLD" in os.environ else None
)

_client: chromadb.ClientAPI | None = None


class RetrievedDoc(BaseModel):
    doc_id: str
    competition_id: str
    source_type: str
    text: str
    similarity: float


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
                source_type=str(metadata["source_type"]),
                text=text,
                similarity=similarity,
            )
        )
    return docs
