"""Condition B baseline tests — retrieve and call_llm_text mocked, no API calls."""

import pytest

from pipeline import baseline
from pipeline.retriever import RetrievedDoc


def _doc(*, doc_id: str, similarity: float, competition_id: str = "other-comp") -> RetrievedDoc:
    return RetrievedDoc(
        doc_id=doc_id,
        competition_id=competition_id,
        source_type="code_block",
        text=f"content of {doc_id}",
        similarity=similarity,
    )


@pytest.fixture()
def mock_retrieval(monkeypatch):
    calls = []

    def fake_retrieve(*, query, collection, exclude_competition, k=5, score_threshold=None):
        calls.append({"query": query, "collection": collection, "exclude": exclude_competition, "k": k})
        if collection == "competition_metadata":
            return [_doc(doc_id="meta_hi", similarity=0.9), _doc(doc_id="meta_lo", similarity=0.1)]
        return [_doc(doc_id="code_hi", similarity=0.8), _doc(doc_id="code_lo", similarity=0.2)]

    monkeypatch.setattr(baseline, "retrieve", fake_retrieve)
    return calls


def test_flat_retrieve_merges_collections_by_similarity(mock_retrieval):
    docs = baseline._flat_retrieve(raw_problem="desc", competition_id="current-comp")
    assert [doc.doc_id for doc in docs] == ["meta_hi", "code_hi", "code_lo", "meta_lo"]
    # one flat query per collection, same query string, leave-one-out applied
    assert len(mock_retrieval) == 2
    assert all(call["query"] == "desc" for call in mock_retrieval)
    assert all(call["exclude"] == "current-comp" for call in mock_retrieval)
    assert all(call["k"] == baseline.BASELINE_K for call in mock_retrieval)


def test_b1_returns_context_block_without_llm(mock_retrieval, monkeypatch):
    monkeypatch.setattr(
        baseline, "call_llm_text", lambda **_: pytest.fail("B1 must not call the LLM")
    )
    result = baseline.run_b1(raw_problem="desc", competition_id="current-comp")
    assert result["condition"] == "B1"
    assert "content of meta_hi" in result["context_block"]
    assert len(result["retrieved"]) == 4
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
    assert "content of code_hi" in seen["user"]
    # B2 must not carry Condition C's critical-integration stance
    assert "not a boundary" not in seen["system"]
