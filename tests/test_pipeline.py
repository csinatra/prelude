"""Unit tests for the four-stage spec pipeline. No real API or corpus calls —
pipeline.nodes.call_llm is monkeypatched with canned JSON responses keyed
off the system prompt for each stage, and pipeline.nodes.retrieve with a
canned document list.
"""

import pytest

from pipeline import nodes
from pipeline.graph import build_graph
from pipeline.retriever import RetrievedDoc
from pipeline.runner import run
from pipeline.state import PipelineState

FAKE_DOC = RetrievedDoc(
    doc_id="block_abc_0",
    competition_id="some-other-competition",
    source_type="code_block",
    text="model = XGBClassifier()",
    similarity=0.83,
)

FAKE_PAYLOADS = {
    "translates machine learning competition": {
        "goal": (
            "estimate, per customer, the causal effect of each candidate "
            "intervention (no intervention, small discount, medium discount, "
            "large discount) on churn, to select the cheapest intervention "
            "that actually prevents churn"
        ),
        "task_type": "tabular classification",
        "evaluation_metric": "AUC",
        "target_variable": "churned",
        "framing_type": "causal",
        "constraints": [
            "four treatment arms: none, small discount, medium discount, large discount",
            "past discount assignment was not randomized",
        ],
    },
    "ML data scientist": {
        "available_signals": [
            "historical discount offers and assigned tier per customer",
            "subscription, billing, and support-ticket history",
        ],
        "desired_signals": ["randomized holdout of discount tier assignment"],
        "prior_work": ["uplift modeling / CATE estimation for multi-armed treatment personalization"],
    },
    "assumptions auditor": {
        "assumption_flags": [
            "confounding: past discount assignment likely targeted customers already flagged as at-risk",
            "positivity violation risk: some customer segments may never have received certain discount tiers",
        ],
    },
    "modeling advisor": {
        "recommended_approaches": [
            "per-arm CATE estimation (causal forest / T-learner) over the four treatment arms",
            "policy learning to select the argmin-cost discount tier subject to a retention constraint",
        ],
        "tradeoffs": ["personalization granularity vs. sample size per treatment arm"],
        "failure_modes": ["overfit CATE estimates in low-overlap customer segments"],
    },
}


def _fake_call_llm(*, system, user, response_model, max_tokens=512):
    for key, payload in FAKE_PAYLOADS.items():
        if key in system:
            return response_model(**payload)
    raise AssertionError(f"no fake payload registered for system prompt: {system}")


def _fake_retrieve(*, query, collection, exclude_competition, k=5, score_threshold=None):
    assert collection in {"competition_metadata", "practitioner_knowledge"}
    return [FAKE_DOC]


@pytest.fixture(autouse=True)
def mock_llm(monkeypatch):
    monkeypatch.setattr(nodes, "call_llm", _fake_call_llm)
    monkeypatch.setattr(nodes, "retrieve", _fake_retrieve)


def test_graph_compiles():
    app = build_graph()
    assert app is not None


def test_state_keys():
    keys = set(PipelineState.__optional_keys__) | set(PipelineState.__required_keys__)
    assert {
        "raw_problem",
        "competition_id",
        "parsed_goal",
        "task_type",
        "evaluation_metric",
        "target_variable",
        "framing_type",
        "constraints",
        "available_signals",
        "desired_signals",
        "prior_work",
        "assumption_flags",
        "recommended_approaches",
        "tradeoffs",
        "failure_modes",
        "retrieved_parse",
        "retrieved_surface",
        "retrieved_flag",
        "retrieved_advise",
        "stage_trace",
    } <= keys


def test_parse_problem_updates_state():
    state = {
        "raw_problem": "which customers should get a discount to prevent churn?",
        "stage_trace": [],
    }
    result = nodes.parse_problem(state)
    assert result["framing_type"] == "causal"
    assert result["task_type"] == "tabular classification"
    assert result["evaluation_metric"] == "AUC"
    assert result["target_variable"] == "churned"
    assert "small discount" in " ".join(result["constraints"])
    assert result["retrieved_parse"] == [FAKE_DOC.model_dump()]
    assert result["stage_trace"] == ["parse_problem"]


def test_surface_signals_updates_state():
    state = {
        "raw_problem": "which customers should get a discount to prevent churn?",
        "parsed_goal": FAKE_PAYLOADS["translates machine learning competition"]["goal"],
        "framing_type": "causal",
        "stage_trace": ["parse_problem"],
    }
    result = nodes.surface_signals(state)
    assert result["available_signals"] == FAKE_PAYLOADS["ML data scientist"]["available_signals"]
    assert result["desired_signals"] == FAKE_PAYLOADS["ML data scientist"]["desired_signals"]
    assert result["prior_work"] == FAKE_PAYLOADS["ML data scientist"]["prior_work"]
    assert result["stage_trace"] == ["parse_problem", "surface_signals"]


def test_flag_assumptions_updates_state():
    state = {
        "raw_problem": "which customers should get a discount to prevent churn?",
        "parsed_goal": FAKE_PAYLOADS["translates machine learning competition"]["goal"],
        "framing_type": "causal",
        "available_signals": FAKE_PAYLOADS["ML data scientist"]["available_signals"],
        "desired_signals": FAKE_PAYLOADS["ML data scientist"]["desired_signals"],
        "stage_trace": ["parse_problem", "surface_signals"],
    }
    result = nodes.flag_assumptions(state)
    assert result["assumption_flags"] == FAKE_PAYLOADS["assumptions auditor"]["assumption_flags"]
    assert result["stage_trace"] == ["parse_problem", "surface_signals", "flag_assumptions"]


def test_advise_approach_updates_state():
    state = {
        "raw_problem": "which customers should get a discount to prevent churn?",
        "parsed_goal": FAKE_PAYLOADS["translates machine learning competition"]["goal"],
        "framing_type": "causal",
        "constraints": FAKE_PAYLOADS["translates machine learning competition"]["constraints"],
        "available_signals": FAKE_PAYLOADS["ML data scientist"]["available_signals"],
        "desired_signals": FAKE_PAYLOADS["ML data scientist"]["desired_signals"],
        "prior_work": FAKE_PAYLOADS["ML data scientist"]["prior_work"],
        "assumption_flags": FAKE_PAYLOADS["assumptions auditor"]["assumption_flags"],
        "stage_trace": ["parse_problem", "surface_signals", "flag_assumptions"],
    }
    result = nodes.advise_approach(state)
    assert result["recommended_approaches"] == FAKE_PAYLOADS["modeling advisor"]["recommended_approaches"]
    assert result["tradeoffs"] == FAKE_PAYLOADS["modeling advisor"]["tradeoffs"]
    assert result["failure_modes"] == FAKE_PAYLOADS["modeling advisor"]["failure_modes"]
    assert result["stage_trace"] == [
        "parse_problem",
        "surface_signals",
        "flag_assumptions",
        "advise_approach",
    ]


def test_full_run_populates_all_stages_in_order():
    sample_problem = (
        "We want to identify which customers are at risk of churning and, for "
        "each one, decide whether an intervention would actually prevent them "
        "from churning — not just correlate with lower churn. We have several "
        "possible interventions to choose from per customer: no intervention, "
        "a small discount, a medium discount, and a large discount. We have 18 "
        "months of subscription, billing, and support-ticket data, plus records "
        "of past discount offers and their outcomes."
    )
    result = run(raw_problem=sample_problem)
    assert result["stage_trace"] == [
        "parse_problem",
        "surface_signals",
        "flag_assumptions",
        "advise_approach",
    ]
    assert result["framing_type"] == "causal"
    assert result["assumption_flags"] == FAKE_PAYLOADS["assumptions auditor"]["assumption_flags"]
    assert result["recommended_approaches"] == FAKE_PAYLOADS["modeling advisor"]["recommended_approaches"]
