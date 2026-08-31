"""Judge scaffold tests against synthetic fixtures — call_llm mocked."""

import pytest

from analysis import judge
from analysis.judge import FlagJudgment, aggregate_by_category, judge_flags

FLAGS = [
    {"category": "iid_violation", "explanation": "group leakage", "evidence_doc_ids": ["d1"], "confidence": "high"},
    {"category": "iid_violation", "explanation": "temporal leakage", "evidence_doc_ids": [], "confidence": "medium"},
    {"category": "resource_constraint_violation", "explanation": "GPU budget", "evidence_doc_ids": [], "confidence": "low"},
]

JUDGMENTS = [
    FlagJudgment(classification="acted_on", evidence_quote="GroupKFold(...)", reasoning="used"),
    FlagJudgment(classification="acted_on", evidence_quote="TimeSeriesSplit(...)", reasoning="used, effect unclear"),
    FlagJudgment(classification="not_acted_on", evidence_quote="", reasoning="no budgeting found"),
]


def test_judge_flags_calls_llm_per_flag(monkeypatch):
    calls = []

    def fake_call_llm(*, system, user, response_model, max_tokens=512):
        calls.append(user)
        assert "FROZEN" in system  # rubric text is embedded
        return FlagJudgment(
            classification="acted_on", evidence_quote="some_code()", reasoning="r"
        )

    monkeypatch.setattr(judge, "call_llm", fake_call_llm)
    judgments = judge_flags(flags=FLAGS, solution="code", logs="logs")
    assert len(judgments) == 3
    assert "group leakage" in calls[0]


def test_acted_on_without_quote_is_invalidated(monkeypatch):
    monkeypatch.setattr(
        judge,
        "call_llm",
        lambda **_: FlagJudgment(classification="acted_on", evidence_quote="  ", reasoning="r"),
    )
    judgments = judge_flags(flags=FLAGS[:1], solution="code")
    assert judgments[0].classification == "not_acted_on"
    assert "invalidated" in judgments[0].reasoning


def test_aggregate_by_category():
    table = aggregate_by_category(flags=FLAGS, judgments=JUDGMENTS)
    iid = table["iid_violation"]
    assert iid["detected"] == 2
    assert iid["action_rate"] == 1.0
    assert iid["retrieval_grounded_fraction"] == 0.5
    resource = table["resource_constraint_violation"]
    assert resource["detected"] == 1
    assert resource["action_rate"] == 0.0
