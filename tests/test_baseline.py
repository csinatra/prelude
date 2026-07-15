"""Condition B baseline tests — retrieval seams and call_llm_text mocked."""

import pytest

from pipeline import baseline
from pipeline.retriever import RetrievedDoc


def _doc(*, doc_id: str, similarity: float, source_type: str = "code_block", kaggle_id=None):
    return RetrievedDoc(
        doc_id=doc_id,
        competition_id="other-comp",
        source_type=source_type,
        text=f"content of {doc_id}",
        similarity=similarity,
        kaggle_id=kaggle_id,
    )


@pytest.fixture()
def mock_retrieval(monkeypatch):
    calls = {"flat": [], "two_level": []}

    def fake_retrieve(*, query, collection, exclude_competition, k=5, score_threshold=None):
        calls["flat"].append({"query": query, "collection": collection, "exclude": exclude_competition, "k": k})
        return [_doc(doc_id="meta_0", similarity=0.9, source_type="competition_description")]

    def fake_two_level(
        *, query, exclude_competition, n_notebooks=8, chunks_per_notebook=3, score_threshold=None
    ):
        calls["two_level"].append(
            {"query": query, "exclude": exclude_competition, "n": n_notebooks, "m": chunks_per_notebook}
        )
        return [
            _doc(doc_id="chunk_a", similarity=0.8, kaggle_id=111),
            _doc(doc_id="chunk_b", similarity=0.7, kaggle_id=222),
        ]

    monkeypatch.setattr(baseline, "retrieve", fake_retrieve)
    monkeypatch.setattr(baseline, "retrieve_two_level", fake_two_level)
    return calls


def test_flat_retrieve_combines_metadata_and_notebook_cards(mock_retrieval):
    docs = baseline._flat_retrieve(raw_problem="desc", competition_id="current-comp")
    assert [doc.doc_id for doc in docs] == ["meta_0", "chunk_a", "chunk_b"]
    # single flat query on each path, leave-one-out applied, budget from config
    assert len(mock_retrieval["flat"]) == 1
    assert len(mock_retrieval["two_level"]) == 1
    assert mock_retrieval["flat"][0]["k"] == baseline.METADATA_K
    assert mock_retrieval["two_level"][0]["n"] == baseline.BASELINE_N_NOTEBOOKS
    assert mock_retrieval["two_level"][0]["query"] == "desc"
    assert mock_retrieval["two_level"][0]["exclude"] == "current-comp"


def test_b1_returns_context_block_without_llm(mock_retrieval, monkeypatch):
    monkeypatch.setattr(
        baseline, "call_llm_text", lambda **_: pytest.fail("B1 must not call the LLM")
    )
    result = baseline.run_b1(raw_problem="desc", competition_id="current-comp")
    assert result["condition"] == "B1"
    assert "content of meta_0" in result["context_block"]
    assert len(result["retrieved"]) == 3
    assert "advice" not in result


def test_b2_adds_freeform_advice(mock_retrieval, monkeypatch):
    seen = {}

    def fake_llm_text(*, system, user, max_tokens=4096):
        seen["system"] = system
        seen["user"] = user
        return "freeform advice prose"

    monkeypatch.setattr(baseline, "call_llm_text", fake_llm_text)
    result = baseline.run_b2(raw_problem="desc", competition_id="current-comp")
    assert result["condition"] == "B2"
    assert result["advice"] == "freeform advice prose"
    assert "content of chunk_a" in seen["user"]
    # B2 must not carry Condition C2's critical-integration stance
    assert "not a boundary" not in seen["system"]
