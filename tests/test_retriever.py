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


def test_threshold_is_read_from_config_at_call_time(fixture_collection, monkeypatch):
    """A threshold set after import must take effect, not the import-time value.

    Guards the config/retriever split: config resolves SIMILARITY_THRESHOLD from
    the environment on import, so binding it as a default argument froze it.
    """
    from pipeline import config

    monkeypatch.setattr(config, "SIMILARITY_THRESHOLD", 0.5)
    docs = retriever.retrieve(
        query="anything",
        collection="practitioner_knowledge",
        exclude_competition="current-comp",
        k=3,
    )
    assert [doc.doc_id for doc in docs] == ["doc_near"]  # doc_far filtered out


def test_explicit_none_threshold_still_disables_filtering(fixture_collection, monkeypatch):
    """None passed by a caller means 'no filtering', and must not be read as 'use config'."""
    from pipeline import config

    monkeypatch.setattr(config, "SIMILARITY_THRESHOLD", 0.99)
    docs = retriever.retrieve(
        query="anything",
        collection="practitioner_knowledge",
        exclude_competition="current-comp",
        k=3,
        score_threshold=None,
    )
    assert len(docs) == 2  # nothing filtered despite the strict config threshold


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
def summaries_fixture(tmp_path, monkeypatch):
    """Six notebook summaries ranked by decreasing similarity to query [1, 0]."""
    client = chromadb.PersistentClient(path=str(tmp_path))
    summaries = client.create_collection(
        name="notebook_summaries", metadata={"hnsw:space": "cosine"}
    )
    summaries.add(
        ids=[f"nb_{i}" for i in range(1, 7)],
        embeddings=[[1.0, 0.0], [0.99, 0.14], [0.95, 0.31], [0.9, 0.44], [0.8, 0.6], [0.6, 0.8]],
        documents=[f"summary {i}" for i in range(1, 7)],
        metadatas=[{"competition_id": "other-comp", "kaggle_id": i} for i in range(1, 7)],
    )
    monkeypatch.setattr(retriever, "_client", client)
    monkeypatch.setattr(
        retriever, "embed", lambda *, texts, input_type: [[1.0, 0.0] for _ in texts]
    )


def test_topup_no_repeats_returns_plain_top_k(summaries_fixture):
    seen: set[str] = set()
    docs = retriever.retrieve_with_topup(
        query="anything", collection="notebook_summaries",
        exclude_competition="none", k=2, seen=seen,
    )
    assert [doc.doc_id for doc in docs] == ["nb_1", "nb_2"]
    assert seen == {"nb_1", "nb_2"}  # contributed k=2 distinct


def test_topup_retains_repeat_and_adds_distinct_doc(summaries_fixture):
    seen = {"nb_1"}  # a prior stage already surfaced nb_1
    docs = retriever.retrieve_with_topup(
        query="anything", collection="notebook_summaries",
        exclude_competition="none", k=2, seen=seen,
    )
    # nb_1 (repeat) retained as importance signal; nb_3 topped up for the repeat
    assert [doc.doc_id for doc in docs] == ["nb_1", "nb_2", "nb_3"]
    # still contributes k=2 NEW distinct docs (nb_2, nb_3)
    assert seen == {"nb_1", "nb_2", "nb_3"}


def test_topup_leave_one_out_applies(summaries_fixture):
    seen: set[str] = set()
    docs = retriever.retrieve_with_topup(
        query="anything", collection="notebook_summaries",
        exclude_competition="other-comp", k=2, seen=seen,
    )
    assert docs == []  # every notebook is other-comp; leave-one-out excludes all
