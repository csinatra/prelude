"""Condition C1 tests — parse, retrieval seams, and call_llm_text mocked."""

import pytest

from pipeline import condition_c1 as c1
from pipeline.retriever import RetrievedDoc


def _doc(*, doc_id: str, kaggle_id=None, source_type: str = "code_block"):
    return RetrievedDoc(
        doc_id=doc_id,
        competition_id="other-comp",
        source_type=source_type,
        text=f"content of {doc_id}",
        similarity=0.8,
        kaggle_id=kaggle_id,
    )


@pytest.fixture()
def mock_seams(monkeypatch):
    def fake_parse(state):
        return {
            "task_type": "text classification",
            "evaluation_metric": "log loss",
            "parsed_goal": "classify authors",
            "retrieved_parse": [_doc(doc_id="meta_0", source_type="competition_description").model_dump()],
            "stage_trace": ["parse_problem"],
        }

    queries = []

    def fake_two_level(
        *, query, exclude_competition, n_notebooks=8, chunks_per_notebook=3, score_threshold=None
    ):
        queries.append(query)
        # same doc returned for every stage — exercises cross-stage dedupe
        return [_doc(doc_id="chunk_shared", kaggle_id=111), _doc(doc_id=f"chunk_{len(queries)}", kaggle_id=222)]

    captured = {}

    def fake_llm_text(*, system, user, max_tokens=4096):
        captured["system"] = system
        captured["user"] = user
        return "freeform staged advice"

    monkeypatch.setattr(c1, "parse_problem", fake_parse)
    monkeypatch.setattr(c1, "retrieve_two_level", fake_two_level)
    monkeypatch.setattr(c1, "call_llm_text", fake_llm_text)
    return {"queries": queries, "captured": captured}


def test_c1_returns_both_artifacts(mock_seams):
    result = c1.run_c1(raw_problem="desc", competition_id="current-comp")
    assert result["condition"] == "C1"
    assert "context_block" in result and "advice" in result
    assert result["advice"] == "freeform staged advice"
    assert set(result["retrieved"]) == {"parse", "surface", "flag", "advise"}


def test_c1_dedupes_across_stages(mock_seams):
    result = c1.run_c1(raw_problem="desc", competition_id="current-comp")
    assert result["context_block"].count("content of chunk_shared") == 1


def test_c1_uses_directed_queries_and_freeform_synthesis(mock_seams):
    c1.run_c1(raw_problem="desc", competition_id="current-comp")
    assert len(mock_seams["queries"]) == 3  # surface, flag, advise
    assert any("pitfalls" in query for query in mock_seams["queries"])
    # synthesis is B2's stance-free freeform prompt, without parse's fields
    assert "not a boundary" not in mock_seams["captured"]["system"]
    assert "text classification" not in mock_seams["captured"]["user"]
