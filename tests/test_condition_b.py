"""Condition B baseline tests — retrieval seams and call_llm_text mocked."""

import pytest

from pipeline import condition_b as baseline
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
    calls = []

    def fake_retrieve(*, query, collection, exclude_competition, k=5, score_threshold=None):
        calls.append({"query": query, "collection": collection, "exclude": exclude_competition, "k": k})
        if collection == "competition_metadata":
            return [_doc(doc_id="meta_0", similarity=0.9, source_type="competition_description")]
        return [
            _doc(doc_id="nb_a", similarity=0.8, source_type="notebook_summaries", kaggle_id=111),
            _doc(doc_id="nb_b", similarity=0.7, source_type="notebook_summaries", kaggle_id=222),
        ]

    monkeypatch.setattr(baseline, "retrieve", fake_retrieve)
    return calls


def test_flat_retrieve_combines_metadata_and_notebook_summaries(mock_retrieval):
    docs = baseline._flat_retrieve(raw_problem="desc", competition_id="current-comp")
    assert [doc.doc_id for doc in docs] == ["meta_0", "nb_a", "nb_b"]
    # one flat query per collection, leave-one-out applied, budgets from config
    assert len(mock_retrieval) == 2
    meta_call = next(c for c in mock_retrieval if c["collection"] == "competition_metadata")
    nb_call = next(c for c in mock_retrieval if c["collection"] == "notebook_summaries")
    assert meta_call["k"] == baseline.METADATA_K
    assert nb_call["k"] == baseline.BASELINE_N_NOTEBOOKS
    assert nb_call["query"] == "desc"
    assert nb_call["exclude"] == "current-comp"


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
    assert "content of nb_a" in seen["user"]
    # B2 must not carry Condition C2's critical-integration stance
    assert "not a boundary" not in seen["system"]
