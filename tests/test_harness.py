"""Harness: spec rendering per condition, registry, and the runner end-to-end
(condition runners and token counting mocked — no API calls)."""

import json

import pytest

from analysis import artifacts
from harness import registry, renderer, runner


def _doc(doc_id: str, competition_id: str = "other-comp") -> dict:
    return {
        "doc_id": doc_id,
        "competition_id": competition_id,
        "source_type": "code_chunk",
        "text": f"text of {doc_id}",
        "similarity": 0.6,
        "kaggle_id": 123,
    }


B1_OUTPUT = {
    "condition": "B1",
    "competition_id": "spooky-author-identification",
    "retrieved": [_doc("d1")],
    "context_block": "[d1] block text",
}

B2_OUTPUT = {**B1_OUTPUT, "condition": "B2", "advice": "Use logistic regression."}

C1_OUTPUT = {
    "condition": "C1",
    "competition_id": "spooky-author-identification",
    "retrieved": {"parse": [_doc("d1")], "surface": [_doc("d2")]},
    "context_block": "[d1][d2] staged block",
    "advice": "Try TF-IDF features.",
}

C2_OUTPUT = {
    "raw_problem": "desc",
    "competition_id": "spooky-author-identification",
    "parsed_goal": "classify authors",
    "task_type": "multiclass classification",
    "evaluation_metric": "log loss",
    "target_variable": "author",
    "framing_type": "predictive",
    "constraints": ["kernel-only"],
    "available_signals": ["text"],
    "desired_signals": ["author metadata"],
    "prior_work": ["TF-IDF + NB"],
    "assumption_flags": [
        {
            "flag_id": "F0",
            "category": "iid_violation",
            "confidence": "high",
            "explanation": "sentences from same document",
            "evidence_doc_ids": ["d1"],
        }
    ],
    "recommendations": [
        {
            "approach": "TF-IDF + linear model",
            "tradeoff": "less expressive",
            "failure_mode": "overfits rare words",
            "addresses_flags": ["F0"],
        }
    ],
    "retrieved_parse": [_doc("d1")],
    "retrieved_surface": [_doc("d2")],
    "retrieved_flag": [_doc("d1")],  # duplicate across stages — must dedupe
    "retrieved_advise": [_doc("d3")],
}


def test_render_b1_is_context_only():
    spec = renderer.render_spec(condition="B1", output=B1_OUTPUT)
    assert renderer.CONTEXT_HEADER in spec
    assert "[d1] block text" in spec
    assert renderer.ADVICE_HEADER not in spec
    assert renderer.SPEC_HEADER not in spec


@pytest.mark.parametrize("output", [B2_OUTPUT, C1_OUTPUT])
def test_render_freeform_conditions_add_advice(output):
    spec = renderer.render_spec(condition=output["condition"], output=output)
    assert output["context_block"] in spec
    assert renderer.ADVICE_HEADER in spec
    assert output["advice"] in spec


def test_render_c2_merges_stage_retrievals_and_renders_spec():
    spec = renderer.render_spec(condition="C2", output=C2_OUTPUT)
    assert renderer.CONTEXT_HEADER in spec
    assert spec.count("text of d1") == 1  # deduped across stages
    assert "text of d2" in spec and "text of d3" in spec
    assert renderer.SPEC_HEADER in spec
    assert "[F0] iid_violation (high confidence)" in spec
    assert "TF-IDF + linear model" in spec
    assert "Addresses flags: F0" in spec


def test_render_unknown_condition_raises():
    with pytest.raises(ValueError):
        renderer.render_spec(condition="Z9", output={})


def test_registry_latest_entry_wins(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "RUNS_PATH", tmp_path / "runs.jsonl")
    registry.append_run(entry={"run_key": "k1", "status": "spec_built"})
    registry.append_run(entry={"run_key": "k1", "status": "agent_run"})
    registry.append_run(entry={"run_key": "k2", "status": "spec_built"})
    runs = registry.load_runs()
    assert runs["k1"]["status"] == "agent_run"
    assert runs["k2"]["status"] == "spec_built"


def test_run_condition_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(registry, "RUNS_PATH", tmp_path / "results" / "runs.jsonl")
    monkeypatch.setattr(runner, "DESCRIPTIONS_DIR", tmp_path / "descriptions")
    monkeypatch.setattr(runner, "_count_tokens", lambda *, text: 42)
    monkeypatch.setitem(
        runner.CONDITION_RUNNERS, "B2", lambda *, raw_problem, competition_id: B2_OUTPUT
    )
    (tmp_path / "descriptions").mkdir()
    (tmp_path / "descriptions" / "spooky-author-identification.md").write_text("the description")

    run_dir = runner.run_condition(
        competition_id="spooky-author-identification", condition="B2", seed=1
    )

    assert (run_dir / "spec.md").read_text() == renderer.render_spec(
        condition="B2", output=B2_OUTPUT
    )
    assert json.loads((run_dir / "retrievals.json").read_text()) == B2_OUTPUT["retrieved"]
    runs = registry.load_runs()
    entry = runs["spooky-author-identification_B2_1"]
    assert entry["status"] == "spec_built"
    assert entry["spec_tokens"] == 42
    assert entry["block_tokens"] == 42
    assert entry["synthesis_tokens"] == 42
    assert entry["git_commit"]
    # Efficiency ledger: mocked condition runner makes no LLM calls.
    assert entry["spec_llm_calls"] == 0
    assert entry["spec_llm_input_tokens"] == 0
    assert entry["spec_build_secs"] >= 0
    assert json.loads((run_dir / "llm_usage.json").read_text()) == []


def test_spec_sections_split_composes_to_render():
    for condition, output in [("B1", B1_OUTPUT), ("B2", B2_OUTPUT), ("C1", C1_OUTPUT), ("C2", C2_OUTPUT)]:
        sections = renderer.spec_sections(condition=condition, output=output)
        parts = [sections["context"]] + ([sections["synthesis"]] if sections["synthesis"] else [])
        assert "\n\n".join(parts) + "\n" == renderer.render_spec(condition=condition, output=output)
    assert renderer.spec_sections(condition="B1", output=B1_OUTPUT)["synthesis"] == ""
