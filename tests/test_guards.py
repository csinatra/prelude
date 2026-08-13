"""Research-validity guards: parity assertion, retrieval shortfall, batch-state resume."""

import json

import pytest

from ingest import ingest_summaries as ing
from pipeline import config, retriever


# ── document-budget parity ──────────────────────────────────────────

def test_validate_parity_passes_on_current_config():
    config.validate_parity()  # must not raise as shipped


def test_validate_parity_rejects_broken_notebook_budget(monkeypatch):
    monkeypatch.setattr(config, "BASELINE_N_NOTEBOOKS", 25)
    with pytest.raises(ValueError, match="document-budget parity broken"):
        config.validate_parity()


def test_validate_parity_rejects_broken_metadata_budget(monkeypatch):
    monkeypatch.setattr(config, "METADATA_K", 7)
    with pytest.raises(ValueError, match="metadata parity broken"):
        config.validate_parity()


# ── retrieval distinct-doc shortfall ────────────────────────────────

def _doc(doc_id: str) -> retriever.RetrievedDoc:
    return retriever.RetrievedDoc(
        doc_id=doc_id,
        competition_id="other-comp",
        source_type="notebook_summaries",
        text=f"summary {doc_id}",
        similarity=0.5,
    )


def test_no_shortfall_when_pool_is_deep(monkeypatch):
    retriever.reset_shortfalls()
    monkeypatch.setattr(retriever, "retrieve", lambda **_: [_doc(f"nb_{i}") for i in range(12)])
    docs = retriever.retrieve_with_topup(
        query="q", collection="notebook_summaries", exclude_competition="x", k=2, seen=set()
    )
    assert len(docs) == 2
    assert retriever.shortfall_log() == []


def test_shortfall_recorded_when_unseen_pool_exhausted(monkeypatch):
    retriever.reset_shortfalls()
    # only 3 docs exist and 2 were already seen: cannot contribute k=2 new distinct
    monkeypatch.setattr(retriever, "retrieve", lambda **_: [_doc("a"), _doc("b"), _doc("c")])
    seen = {"a", "b"}
    retriever.retrieve_with_topup(
        query="q", collection="notebook_summaries", exclude_competition="x", k=2, seen=seen
    )
    log = retriever.shortfall_log()
    assert len(log) == 1
    assert log[0]["requested_distinct"] == 2
    assert log[0]["actual_distinct"] == 1  # only 'c' was new
    assert log[0]["collection"] == "notebook_summaries"


def test_shortfall_log_resets(monkeypatch):
    retriever.reset_shortfalls()
    monkeypatch.setattr(retriever, "retrieve", lambda **_: [_doc("a")])
    retriever.retrieve_with_topup(
        query="q", collection="notebook_summaries", exclude_competition="x", k=3, seen={"a"}
    )
    assert retriever.shortfall_log()
    retriever.reset_shortfalls()
    assert retriever.shortfall_log() == []


# ── batch-state resume safety ───────────────────────────────────────

def test_resume_accepts_matching_corpus():
    state = {"batch_ids": ["b1"], "custom_ids": ["nb_1", "nb_2"]}
    ing._check_resume_matches(state=state, notebooks={1: {}, 2: {}, 3: {}})  # no raise


def test_resume_rejects_corpus_changed_under_batch():
    state = {"batch_ids": ["b1"], "custom_ids": ["nb_1", "nb_9"]}
    with pytest.raises(SystemExit, match="batch state mismatch"):
        ing._check_resume_matches(state=state, notebooks={1: {}})


def test_resume_skips_check_for_legacy_state_without_custom_ids():
    ing._check_resume_matches(state={"batch_ids": ["b1"]}, notebooks={1: {}})  # no raise


def test_submit_writes_custom_ids_into_state(monkeypatch, tmp_path):
    """State must carry the ids so a later resume can verify them."""
    monkeypatch.setattr(ing, "BATCH_STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(ing, "_submit", lambda **_: ["batch_1"])
    monkeypatch.setattr(ing, "_collect", lambda **_: None)
    monkeypatch.setattr(ing, "_pending", lambda **_: {7: {"blocks": ["x"], "competition_id": "c", "kaggle_score": None}})
    captured = {}
    original_write = ing.BATCH_STATE_PATH.write_text

    def capture(text):
        captured.update(json.loads(text))
        return original_write(text)

    monkeypatch.setattr(type(ing.BATCH_STATE_PATH), "write_text", lambda self, text: capture(text))
    monkeypatch.setattr(ing.anthropic, "Anthropic", lambda: object())
    ing._run_batch(notebooks={7: {}}, collection=object(), limit=None)
    assert captured["custom_ids"] == ["nb_7"]
