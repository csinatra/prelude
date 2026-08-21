"""Registry advancement: merge semantics and the agent_run/graded transitions."""

import pytest

from harness import advance, registry


@pytest.fixture
def seeded_registry(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "RESULTS_DIR", tmp_path)
    registry.append_run(
        entry={
            "run_key": "comp_B2_0",
            "competition_id": "comp",
            "condition": "B2",
            "seed": 0,
            "status": "spec_built",
            "spec_tokens": 111,
        }
    )


def test_agent_run_transition_preserves_spec_fields(seeded_registry):
    advance.record_agent_run(
        run_key="comp_B2_0",
        submission_path="/x/submission.csv",
        wallclock_secs=3600.0,
        steps=42,
        time_to_first_valid_secs=900.0,
    )
    run = registry.load_runs()["comp_B2_0"]
    assert run["status"] == "agent_run"
    assert run["spec_tokens"] == 111  # merged, not erased
    assert run["agent_id"] == "aide-prelude"
    assert run["agent_wallclock_secs"] == 3600.0
    assert run["agent_steps"] == 42
    assert run["agent_time_to_first_valid_secs"] == 900.0


def test_graded_transition_records_metric_subset(seeded_registry):
    report = {
        "competition_id": "comp",
        "score": 0.42,
        "any_medal": True,
        "gold_medal": False,
        "silver_medal": False,
        "bronze_medal": True,
        "above_median": True,
        "submission_exists": True,
        "valid_submission": True,
        "is_lower_better": True,
        "gold_threshold": 0.30,
        "silver_threshold": 0.35,
        "bronze_threshold": 0.45,
        "median_threshold": 0.60,
        "submission_path": "/x/submission.csv",  # not a registry field
    }
    advance.record_graded(run_key="comp_B2_0", report=report)
    run = registry.load_runs()["comp_B2_0"]
    assert run["status"] == "graded"
    assert run["any_medal"] is True and run["bronze_medal"] is True
    assert run["score"] == 0.42
    assert run["spec_tokens"] == 111
    assert "submission_path" not in run


def test_advance_unknown_run_key_fails(seeded_registry):
    with pytest.raises(SystemExit):
        advance.record_agent_run(run_key="nope_C2_9")


def test_register_creates_specless_run_then_advances(seeded_registry):
    advance.register_run(competition_id="comp", condition="A", seed=0)
    run = registry.load_runs()["comp_A_0"]
    assert run["status"] == "registered"
    assert run["condition"] == "A"
    advance.record_agent_run(run_key="comp_A_0", wallclock_secs=100.0)
    assert registry.load_runs()["comp_A_0"]["status"] == "agent_run"


def test_register_duplicate_run_key_fails(seeded_registry):
    with pytest.raises(SystemExit):
        advance.register_run(competition_id="comp", condition="B2", seed=0)


def test_report_for_extracts_from_aggregated_report(tmp_path):
    import json

    aggregated = {
        "competition_reports": [
            {"competition_id": "other", "score": 0.1},
            {"competition_id": "comp", "score": 0.42},
        ]
    }
    path = tmp_path / "grading_report.json"
    path.write_text(json.dumps(aggregated))
    report = advance._report_for(report_path=path, run_key="comp_B2_0", competition_id="comp")
    assert report["score"] == 0.42
