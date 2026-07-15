"""Unit tests for the C2 four-stage spec pipeline. No real API or corpus
calls — pipeline.nodes.call_llm is monkeypatched with canned responses keyed
off the system prompt for each stage, and both retrieval seams
(pipeline.nodes.retrieve / retrieve_two_level) with canned document lists.
"""

import pytest

from pipeline import nodes
from pipeline.graph import build_graph
from pipeline.retriever import RetrievedDoc
from pipeline.condition_c2 import run_c2
from pipeline.state import PipelineState

FAKE_META_DOC = RetrievedDoc(
    doc_id="code4ml_other-comp_0",
    competition_id="some-other-competition",
    source_type="competition_description",
    text="a similar competition description",
    similarity=0.71,
)

FAKE_CHUNK_DOC = RetrievedDoc(
    doc_id="block_abc_0",
    competition_id="some-other-competition",
    source_type="code_block",
    text="model = XGBClassifier()",
    similarity=0.83,
    kaggle_id=12345,
)

FAKE_FLAG = {
    "category": "iid_violation",
    "explanation": (
        "customer-level records repeat across train rows; a random split leaks "
        "customer identity into validation"
    ),
    "evidence_doc_ids": ["block_abc_0"],
    "confidence": "high",
}

FAKE_RECOMMENDATION = {
    "approach": "grouped cross-validation by customer_id with LightGBM",
    "tradeoff": "fewer effective folds vs. leakage-free validation estimates",
    "failure_mode": "optimistic validation if grouping key is wrong",
    "addresses_flags": [0],
}

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
    "assumptions auditor": {"flags": [FAKE_FLAG]},
    "modeling advisor": {"recommendations": [FAKE_RECOMMENDATION]},
}


def _fake_call_llm(*, system, user, response_model, max_tokens=512):
    for key, payload in FAKE_PAYLOADS.items():
        if key in system:
            return response_model(**payload)
    raise AssertionError(f"no fake payload registered for system prompt: {system}")


def _fake_retrieve(*, query, collection, exclude_competition, k=5, score_threshold=None):
    assert collection == "competition_metadata"
    return [FAKE_META_DOC]


def _fake_retrieve_two_level(
    *, query, exclude_competition, n_notebooks=8, chunks_per_notebook=3, score_threshold=None
):
    return [FAKE_CHUNK_DOC]


@pytest.fixture(autouse=True)
def mock_seams(monkeypatch):
    monkeypatch.setattr(nodes, "call_llm", _fake_call_llm)
    monkeypatch.setattr(nodes, "retrieve", _fake_retrieve)
    monkeypatch.setattr(nodes, "retrieve_two_level", _fake_retrieve_two_level)


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
        "recommendations",
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
    assert result["retrieved_parse"] == [FAKE_META_DOC.model_dump()]
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
    assert result["retrieved_surface"] == [FAKE_CHUNK_DOC.model_dump()]
    assert result["stage_trace"] == ["parse_problem", "surface_signals"]


def test_flag_assumptions_returns_structured_flags():
    state = {
        "raw_problem": "which customers should get a discount to prevent churn?",
        "parsed_goal": FAKE_PAYLOADS["translates machine learning competition"]["goal"],
        "framing_type": "causal",
        "available_signals": FAKE_PAYLOADS["ML data scientist"]["available_signals"],
        "desired_signals": FAKE_PAYLOADS["ML data scientist"]["desired_signals"],
        "stage_trace": ["parse_problem", "surface_signals"],
    }
    result = nodes.flag_assumptions(state)
    assert result["assumption_flags"] == [FAKE_FLAG]
    assert result["assumption_flags"][0]["category"] == "iid_violation"
    assert result["assumption_flags"][0]["evidence_doc_ids"] == ["block_abc_0"]
    assert result["stage_trace"] == ["parse_problem", "surface_signals", "flag_assumptions"]


def test_advise_approach_returns_linked_recommendations():
    state = {
        "raw_problem": "which customers should get a discount to prevent churn?",
        "parsed_goal": FAKE_PAYLOADS["translates machine learning competition"]["goal"],
        "framing_type": "causal",
        "constraints": FAKE_PAYLOADS["translates machine learning competition"]["constraints"],
        "available_signals": FAKE_PAYLOADS["ML data scientist"]["available_signals"],
        "desired_signals": FAKE_PAYLOADS["ML data scientist"]["desired_signals"],
        "prior_work": FAKE_PAYLOADS["ML data scientist"]["prior_work"],
        "assumption_flags": [FAKE_FLAG],
        "stage_trace": ["parse_problem", "surface_signals", "flag_assumptions"],
    }
    result = nodes.advise_approach(state)
    assert result["recommendations"] == [FAKE_RECOMMENDATION]
    assert result["recommendations"][0]["addresses_flags"] == [0]
    assert result["stage_trace"] == [
        "parse_problem",
        "surface_signals",
        "flag_assumptions",
        "advise_approach",
    ]


def test_flags_render_into_advise_prompt(monkeypatch):
    captured = {}

    def capturing_call_llm(*, system, user, response_model, max_tokens=512):
        if "modeling advisor" in system:
            captured["user"] = user
        return _fake_call_llm(
            system=system, user=user, response_model=response_model, max_tokens=max_tokens
        )

    monkeypatch.setattr(nodes, "call_llm", capturing_call_llm)
    nodes.advise_approach(
        {
            "raw_problem": "problem",
            "assumption_flags": [FAKE_FLAG],
            "stage_trace": [],
        }
    )
    assert "[0] iid_violation (high confidence)" in captured["user"]


def test_full_run_populates_all_stages_in_order():
    sample_problem = (
        "We want to identify which customers are at risk of churning and, for "
        "each one, decide whether an intervention would actually prevent them "
        "from churning — not just correlate with lower churn."
    )
    result = run_c2(raw_problem=sample_problem, competition_id="some-held-out-comp")
    assert result["stage_trace"] == [
        "parse_problem",
        "surface_signals",
        "flag_assumptions",
        "advise_approach",
    ]
    assert result["framing_type"] == "causal"
    assert result["assumption_flags"] == [FAKE_FLAG]
    assert result["recommendations"] == [FAKE_RECOMMENDATION]
