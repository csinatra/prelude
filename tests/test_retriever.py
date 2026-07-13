"""Retriever tests against a temp ChromaDB — embeddings monkeypatched, no API calls."""

import chromadb
import pytest

from pipeline import retriever


@pytest.fixture()
def fixture_collection(tmp_path, monkeypatch):
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.create_collection(
        name="practitioner_knowledge", metadata={"hnsw:space": "cosine"}
    )
    collection.add(
        ids=["doc_held_out", "doc_near", "doc_far"],
        embeddings=[[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
        documents=["held-out competition code", "similar competition code", "unrelated code"],
        metadatas=[
            {"competition_id": "current-comp", "source_type": "code_block"},
            {"competition_id": "other-comp", "source_type": "code_block"},
            {"competition_id": "far-comp", "source_type": "code_block"},
        ],
    )
    monkeypatch.setattr(retriever, "_client", client)
    # Query vector points at doc_held_out / doc_near.
    monkeypatch.setattr(
        retriever, "embed", lambda *, texts, input_type: [[1.0, 0.0] for _ in texts]
    )


def test_leave_one_out_excludes_current_competition(fixture_collection):
    docs = retriever.retrieve(
        query="anything",
        collection="practitioner_knowledge",
        exclude_competition="current-comp",
        k=3,
    )
    ids = [doc.doc_id for doc in docs]
    assert "doc_held_out" not in ids
    assert ids[0] == "doc_near"


def test_score_threshold_filters_weak_matches(fixture_collection):
    docs = retriever.retrieve(
        query="anything",
        collection="practitioner_knowledge",
        exclude_competition="current-comp",
        k=3,
        score_threshold=0.5,
    )
    assert [doc.doc_id for doc in docs] == ["doc_near"]


def test_k_caps_results(fixture_collection):
    docs = retriever.retrieve(
        query="anything",
        collection="practitioner_knowledge",
        exclude_competition="none",
        k=1,
    )
    assert len(docs) == 1
    assert docs[0].similarity == pytest.approx(1.0, abs=1e-6)
