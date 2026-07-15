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


@pytest.fixture()
def two_level_fixture(tmp_path, monkeypatch):
    client = chromadb.PersistentClient(path=str(tmp_path))
    summaries = client.create_collection(
        name="notebook_summaries", metadata={"hnsw:space": "cosine"}
    )
    summaries.add(
        ids=["nb_1", "nb_2", "nb_3"],
        embeddings=[[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
        documents=["current-comp notebook", "relevant notebook", "irrelevant notebook"],
        metadatas=[
            {"competition_id": "current-comp", "kaggle_id": 1},
            {"competition_id": "other-comp", "kaggle_id": 2},
            {"competition_id": "far-comp", "kaggle_id": 3},
        ],
    )
    chunks = client.create_collection(
        name="practitioner_knowledge", metadata={"hnsw:space": "cosine"}
    )
    chunks.add(
        ids=["c1a", "c2a", "c2b", "c2c", "c3a"],
        embeddings=[[1.0, 0.0], [0.95, 0.05], [0.9, 0.1], [0.5, 0.5], [0.1, 0.9]],
        documents=[
            "chunk of held-out notebook",
            "chunk 2a",
            "chunk 2b",
            "chunk 2c",
            "only chunk of notebook 3",
        ],
        metadatas=[
            {"competition_id": "current-comp", "source_type": "code_block", "kaggle_id": 1},
            {"competition_id": "other-comp", "source_type": "code_block", "kaggle_id": 2},
            {"competition_id": "other-comp", "source_type": "code_block", "kaggle_id": 2},
            {"competition_id": "other-comp", "source_type": "code_block", "kaggle_id": 2},
            {"competition_id": "far-comp", "source_type": "code_block", "kaggle_id": 3},
        ],
    )
    monkeypatch.setattr(retriever, "_client", client)
    monkeypatch.setattr(
        retriever, "embed", lambda *, texts, input_type: [[1.0, 0.0] for _ in texts]
    )


def test_two_level_leave_one_out_at_notebook_level(two_level_fixture):
    docs = retriever.retrieve_two_level(
        query="anything", exclude_competition="current-comp", n_notebooks=3, chunks_per_notebook=2
    )
    assert all(doc.competition_id != "current-comp" for doc in docs)
    assert "c1a" not in [doc.doc_id for doc in docs]


def test_two_level_chunks_restricted_to_surfaced_notebooks(two_level_fixture):
    docs = retriever.retrieve_two_level(
        query="anything", exclude_competition="current-comp", n_notebooks=1, chunks_per_notebook=3
    )
    # only notebook 2 is surfaced (closest non-excluded summary)
    assert {doc.kaggle_id for doc in docs} == {2}
    assert len(docs) == 3


def test_two_level_notebook_with_fewer_chunks_than_m(two_level_fixture):
    docs = retriever.retrieve_two_level(
        query="anything", exclude_competition="other-comp", n_notebooks=3, chunks_per_notebook=4
    )
    # far-comp's notebook 3 has a single chunk; requesting M=4 returns just it
    assert {doc.kaggle_id for doc in docs} == {1, 3}
    assert len([doc for doc in docs if doc.kaggle_id == 3]) == 1
