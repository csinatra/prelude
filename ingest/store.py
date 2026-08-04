"""ChromaDB write helper shared by the two ingestion scripts.

Collections are created with cosine distance and populated with explicit
Voyage embeddings — Chroma is a pure vector store here, no default embedder.
"""

import chromadb
from chromadb.api.models.Collection import Collection

from ingest.config import CHROMA_PATH
from pipeline.embeddings import embed


def get_collection(*, name: str) -> Collection:
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def drop_collection(*, name: str) -> None:
    """Delete a collection so the next ingest rebuilds it from scratch (--rebuild).

    Used after a change that makes existing vectors stale — an embedding-model
    swap, or a summary-prompt change (where the skip-existing resumability would
    otherwise never regenerate). No-op if the collection is absent.
    """
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    existing = {getattr(c, "name", c) for c in client.list_collections()}
    if name in existing:
        client.delete_collection(name=name)


def add_documents(
    *,
    collection: Collection,
    ids: list[str],
    texts: list[str],
    metadatas: list[dict],
) -> None:
    """Embed texts as documents and upsert. Chroma caps ~5k records per call."""
    vectors = embed(texts=texts, input_type="document")
    for start in range(0, len(ids), 1000):
        end = start + 1000
        collection.upsert(
            ids=ids[start:end],
            embeddings=vectors[start:end],
            documents=texts[start:end],
            metadatas=metadatas[start:end],
        )
