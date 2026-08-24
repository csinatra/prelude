"""Batch driver orchestration: queue selection, sequencing, failure isolation,
registry advancement, and terminate gating — box seams (agent/grade) mocked."""

from pathlib import Path

import pytest

from harness import advance, batch, registry


@pytest.fixture
def seeded_registry(tmp_path, monkeypatch):
    """Runs across two competitions, appended out of order: one graded (done),
    the rest pending, so ordering (by competition) is observable."""
    monkeypatch.setattr(registry, "RESULTS_DIR", tmp_path)
    registry.append_run(entry={
        "run_key": "comp_C2_0", "competition_id": "comp", "condition": "C2",
        "seed": 0, "status": "graded", "score": 0.9,
    })
    registry.append_run(entry={
        "run_key": "comp_B2_0", "competition_id": "comp", "condition": "B2",
        "seed": 0, "status": "spec_built", "spec_path": "results/comp_B2_0/spec.md",
    })
    registry.append_run(entry={
        "run_key": "alpha_B1_0", "competition_id": "alpha", "condition": "B1",
        "seed": 0, "status": "spec_built", "spec_path": "results/alpha_B1_0/spec.md",
    })
    registry.append_run(entry={
        "run_key": "comp_A_0", "competition_id": "comp", "condition": "A",
        "seed": 0, "status": "registered",
    })


def test_pending_excludes_graded_and_groups_by_competition(seeded_registry):
    keys = [run["run_key"] for run in batch.pending_runs()]
    # graded comp_C2_0 dropped; alpha's runs before comp's; within comp, A before B2
    assert keys == ["alpha_B1_0", "comp_A_0", "comp_B2_0"]


def test_run_batch_executes_each_pending_in_order(seeded_registry):
    seen = []
    batch.run_batch(
        data_dir=Path("/data"),
        execute=lambda *, run, data_dir: seen.append(run["run_key"]),
    )
    assert seen == ["alpha_B1_0", "comp_A_0", "comp_B2_0"]


def test_one_failure_does_not_stall_the_batch(seeded_registry):
    def execute(*, run, data_dir):
        if run["run_key"] == "comp_A_0":
            raise RuntimeError("agent crashed")

    summary = batch.run_batch(data_dir=Path("/data"), execute=execute)
    assert summary["abandoned"] == ["comp_A_0"]
    assert summary["succeeded"] == ["alpha_B1_0", "comp_B2_0"]  # runs before and after still ran


def test_failure_parks_run_and_excludes_it_from_next_batch(seeded_registry):
    def execute(*, run, data_dir):
        if run["run_key"] == "comp_A_0":
            raise RuntimeError("boom")

    batch.run_batch(data_dir=Path("/data"), execute=execute)

    parked = registry.load_runs()["comp_A_0"]
    assert parked["abandoned"] is True
    assert "RuntimeError: boom" in parked["last_error"]
    assert parked["status"] == "registered"  # phase preserved for a later retry
    # excluded from the queue → a re-run does NOT re-attempt it (unbounded-cost guard)
    assert "comp_A_0" not in [run["run_key"] for run in batch.pending_runs()]


def test_retry_abandoned_requeues_parked_runs(seeded_registry):
    attempts = []

    def execute(*, run, data_dir):
        attempts.append(run["run_key"])
        if run["run_key"] == "comp_A_0" and attempts.count("comp_A_0") == 1:
            raise RuntimeError("transient")

    batch.run_batch(data_dir=Path("/data"), execute=execute)  # comp_A_0 parked
    assert "comp_A_0" not in [run["run_key"] for run in batch.pending_runs()]

    summary = batch.run_batch(data_dir=Path("/data"), execute=execute, retry_abandoned=True)
    assert "comp_A_0" in summary["succeeded"]  # re-queued and succeeded on 2nd attempt
    assert registry.load_runs()["comp_A_0"]["abandoned"] is False


def test_terminate_on_done_fires_once_when_flagged(seeded_registry, monkeypatch):
    calls = []
    monkeypatch.setattr(
        batch.lambda_ctl, "terminate_instance",
        lambda *, instance_id: calls.append(instance_id),
    )
    batch.run_batch(
        data_dir=Path("/data"), execute=lambda *, run, data_dir: None,
        terminate_on_done=True, instance_id="i-123",
    )
    assert calls == ["i-123"]


def test_no_terminate_without_flag(seeded_registry, monkeypatch):
    calls = []
    monkeypatch.setattr(
        batch.lambda_ctl, "terminate_instance",
        lambda *, instance_id: calls.append(instance_id),
    )
    batch.run_batch(data_dir=Path("/data"), execute=lambda *, run, data_dir: None)
    assert calls == []


def test_terminate_requires_instance_id(seeded_registry):
    with pytest.raises(SystemExit, match="instance-id"):
        batch.run_batch(
            data_dir=Path("/data"), execute=lambda *, run, data_dir: None,
            terminate_on_done=True, instance_id=None,
        )


# ── execute_run wiring: box seams (_run_agent/_grade) mocked ─────────────

def test_execute_run_advances_spec_built_through_graded(seeded_registry, monkeypatch, tmp_path):
    submission = tmp_path / "submission.csv"
    submission.write_text("id,pred\n")
    (tmp_path / "grading_report.json").write_text('{"competition_id": "comp", "score": 0.5}')

    report = tmp_path / "grading_report.json"
    monkeypatch.setattr(batch.artifacts, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(batch, "_run_agent", lambda *, run, data_dir: batch.AgentOutputs(
        submission_path=str(submission), journal_path=str(tmp_path / "journal.json"),
        metrics={"steps": 12, "wallclock_secs": 300.0, "time_to_first_valid_secs": 60.0},
    ))
    monkeypatch.setattr(batch, "_grade", lambda **kw: report)

    batch.execute_run(run=batch.load_runs()["comp_B2_0"], data_dir=Path("/data"))

    run = registry.load_runs()["comp_B2_0"]
    assert run["status"] == "graded"
    assert run["agent_steps"] == 12
    assert run["spec_path"] == "results/comp_B2_0/spec.md"  # spec-time field survived
    assert run["score"] == 0.5


def test_execute_run_condition_a_runs_agent_without_spec(seeded_registry, monkeypatch, tmp_path):
    submission = tmp_path / "submission.csv"
    submission.write_text("id,pred\n")
    report = tmp_path / "grading_report.json"
    report.write_text('{"competition_id": "comp", "score": 0.3}')
    seen_spec = {}

    def fake_agent(*, run, data_dir):
        seen_spec["spec_path"] = run.get("spec_path")
        return batch.AgentOutputs(submission_path=str(submission), journal_path=None, metrics={})

    monkeypatch.setattr(batch.artifacts, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(batch, "_run_agent", fake_agent)
    monkeypatch.setattr(batch, "_grade", lambda **kw: report)

    batch.execute_run(run=batch.load_runs()["comp_A_0"], data_dir=Path("/data"))

    assert seen_spec["spec_path"] is None  # Condition A: no spec mounted
    assert registry.load_runs()["comp_A_0"]["status"] == "graded"


# ── _run_agent spec-mount wiring (subprocess/output seams mocked) ─────────

def test_run_agent_sets_spec_env_for_bc(monkeypatch, tmp_path):
    spec = tmp_path / "spec.md"
    spec.write_text("ADVISOR CONTEXT\n")
    captured = {}
    monkeypatch.setattr(batch, "MLEBENCH_DIR", tmp_path)
    monkeypatch.setattr(batch.subprocess, "run",
                        lambda argv, **kw: captured.update(argv=argv, env=kw.get("env")))
    monkeypatch.setattr(batch, "_locate_outputs",
                        lambda *, run_output_dir: ("/x/submission.csv", None, None, None))

    batch._run_agent(
        run={"run_key": "comp_C2_0", "competition_id": "comp", "spec_path": str(spec)},
        data_dir=Path("/data"),
    )
    assert captured["env"]["PRELUDE_SPEC_PATH"] == str(spec.resolve())  # B/C: spec mounted
    assert "--competition-set" in captured["argv"]  # file, not --competition


def test_run_agent_uses_mlebenchs_own_interpreter(monkeypatch, tmp_path):
    """Bare `python` resolves to prelude's venv, which lacks mle-bench's deps.

    Regression: the driver invoked `python run_agent.py`, which under
    `.venv/bin/python -m harness.batch` picked up prelude's interpreter and died
    on `ModuleNotFoundError: py7zr` before any container started.
    """
    captured = {}
    monkeypatch.setattr(batch, "MLEBENCH_DIR", tmp_path)
    monkeypatch.setattr(batch, "MLEBENCH_PYTHON", tmp_path / ".venv" / "bin" / "python")
    monkeypatch.setattr(batch.subprocess, "run", lambda argv, **kw: captured.update(argv=argv))
    monkeypatch.setattr(batch, "_locate_outputs",
                        lambda *, run_output_dir: (None, None, None, None))

    batch._run_agent(
        run={"run_key": "comp_A_0", "competition_id": "comp"},
        data_dir=Path("/data"),
    )
    assert captured["argv"][0] == str(tmp_path / ".venv" / "bin" / "python")
    assert "--container-config" in captured["argv"]  # benchmark resources, not Docker defaults


def test_execute_run_preserves_agent_artifacts_to_volume(seeded_registry, monkeypatch, tmp_path):
    # mle-bench writes these under an ephemeral run dir; execute_run must copy
    # them into results/{run_key}/ (the persistent volume) so --terminate-on-done
    # can't destroy the mechanistic-judge inputs.
    src = tmp_path / "rundir"
    src.mkdir()
    for name, body in [
        ("submission.csv", "id,pred\n"), ("journal.json", "{}"),
        ("best_solution.py", "print(1)\n"), ("prelude_token_usage.jsonl", "{}\n"),
    ]:
        (src / name).write_text(body)
    report = tmp_path / "grading_report.json"
    report.write_text('{"competition_id": "comp", "score": 0.5}')

    monkeypatch.setattr(batch.artifacts, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(batch, "_run_agent", lambda *, run, data_dir: batch.AgentOutputs(
        submission_path=str(src / "submission.csv"), journal_path=str(src / "journal.json"),
        metrics={"steps": 8}, solution_path=str(src / "best_solution.py"),
        token_usage_path=str(src / "prelude_token_usage.jsonl"),
    ))
    monkeypatch.setattr(batch, "_grade", lambda **kw: report)

    batch.execute_run(run=batch.load_runs()["comp_B2_0"], data_dir=Path("/data"))

    # staged: agent outputs must land in the SAME stage tree the registry row
    # points at, or spec_path and the preserved journal would disagree
    preserved = batch.artifacts.run_root() / "comp_B2_0"
    assert (preserved / "submission.csv").exists()
    assert (preserved / "journal.json").exists()
    assert (preserved / "best_solution.py").exists()      # judge input
    assert (preserved / "prelude_token_usage.jsonl").exists()  # agent cost ledger


def test_run_agent_no_spec_env_for_a(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(batch, "MLEBENCH_DIR", tmp_path)
    monkeypatch.setattr(batch.subprocess, "run",
                        lambda argv, **kw: captured.update(env=kw.get("env")))
    monkeypatch.setattr(batch, "_locate_outputs", lambda *, run_output_dir: (None, None, None, None))

    batch._run_agent(run={"run_key": "comp_A_0", "competition_id": "comp"}, data_dir=Path("/data"))
    assert "PRELUDE_SPEC_PATH" not in captured["env"]  # Condition A: unset -> stock aide


def test_run_agent_missing_spec_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(batch, "MLEBENCH_DIR", tmp_path)
    with pytest.raises(RuntimeError, match="spec not found"):
        batch._run_agent(
            run={"run_key": "comp_B2_0", "competition_id": "comp",
                 "spec_path": str(tmp_path / "missing.md")},
            data_dir=Path("/data"),
        )
