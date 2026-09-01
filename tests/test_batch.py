"""Batch driver orchestration: queue selection, sequencing, failure isolation,
registry advancement, and terminate gating — box seams (agent/grade) mocked."""

import json
import os
from pathlib import Path

import pytest

from harness import advance, batch, registry


@pytest.fixture
def seeded_registry(tmp_path, monkeypatch):
    """Runs across two competitions, appended out of order: one graded (done),
    the rest pending, so ordering (by competition) is observable.

    Returns a data dir with both competitions prepared, since run_batch now
    refuses to drain without that (see unprepared_competitions).
    """
    monkeypatch.setattr(registry, "RESULTS_DIR", tmp_path)
    data_dir = tmp_path / "data"
    for competition in ("comp", "alpha"):
        (data_dir / competition / "prepared").mkdir(parents=True)
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
    return data_dir


def test_drain_refuses_to_start_when_a_competition_is_not_prepared(seeded_registry):
    """Kaggle rules acceptance is browser-only, so an unjoined competition can
    never be prepared. Both causes look the same here, and without this check
    both surface hours into a drain as a parked run."""
    (seeded_registry / "alpha" / "prepared").rmdir()
    ran = []
    with pytest.raises(SystemExit) as excinfo:
        batch.run_batch(
            data_dir=seeded_registry,
            execute=lambda *, run, data_dir: ran.append(run["run_key"]),
        )
    assert "alpha" in str(excinfo.value)
    assert "comp" not in str(excinfo.value).split("prepared data for:")[1].split("\n")[0]
    assert ran == []  # refuses before spending any GPU time


def test_pending_excludes_graded_and_groups_by_competition(seeded_registry):
    keys = [run["run_key"] for run in batch.pending_runs()]
    # graded comp_C2_0 dropped; alpha's runs before comp's; within comp, A before B2
    assert keys == ["alpha_B1_0", "comp_A_0", "comp_B2_0"]


def test_run_batch_executes_each_pending_in_order(seeded_registry):
    seen = []
    batch.run_batch(
        data_dir=seeded_registry,
        execute=lambda *, run, data_dir: seen.append(run["run_key"]),
    )
    assert seen == ["alpha_B1_0", "comp_A_0", "comp_B2_0"]


def test_one_failure_does_not_stall_the_batch(seeded_registry):
    def execute(*, run, data_dir):
        if run["run_key"] == "comp_A_0":
            raise RuntimeError("agent crashed")

    summary = batch.run_batch(data_dir=seeded_registry, execute=execute)
    assert summary["abandoned"] == ["comp_A_0"]
    assert summary["succeeded"] == ["alpha_B1_0", "comp_B2_0"]  # runs before and after still ran


def test_failure_parks_run_and_excludes_it_from_next_batch(seeded_registry):
    def execute(*, run, data_dir):
        if run["run_key"] == "comp_A_0":
            raise RuntimeError("boom")

    batch.run_batch(data_dir=seeded_registry, execute=execute)

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

    batch.run_batch(data_dir=seeded_registry, execute=execute)  # comp_A_0 parked
    assert "comp_A_0" not in [run["run_key"] for run in batch.pending_runs()]

    summary = batch.run_batch(data_dir=seeded_registry, execute=execute, retry_abandoned=True)
    assert "comp_A_0" in summary["succeeded"]  # re-queued and succeeded on 2nd attempt
    assert registry.load_runs()["comp_A_0"]["abandoned"] is False


def test_terminate_on_done_fires_once_when_flagged(seeded_registry, monkeypatch):
    calls = []
    monkeypatch.setattr(batch, "_results_survive_termination", lambda: True)
    monkeypatch.setattr(
        batch.lambda_ctl, "terminate_instance",
        lambda *, instance_id: calls.append(instance_id),
    )
    batch.run_batch(
        data_dir=seeded_registry, execute=lambda *, run, data_dir: None,
        terminate_on_done=True, instance_id="i-123",
    )
    assert calls == ["i-123"]


def test_no_terminate_without_flag(seeded_registry, monkeypatch):
    calls = []
    monkeypatch.setattr(
        batch.lambda_ctl, "terminate_instance",
        lambda *, instance_id: calls.append(instance_id),
    )
    batch.run_batch(data_dir=seeded_registry, execute=lambda *, run, data_dir: None)
    assert calls == []


def test_terminate_requires_instance_id(seeded_registry):
    with pytest.raises(SystemExit, match="instance-id"):
        batch.run_batch(
            data_dir=seeded_registry, execute=lambda *, run, data_dir: None,
            terminate_on_done=True, instance_id=None,
        )


# ── execute_run wiring: box seams (_run_agent/_grade) mocked ─────────────

def test_execute_run_advances_spec_built_through_graded(seeded_registry, monkeypatch, tmp_path):
    submission = tmp_path / "submission.csv"
    submission.write_text("id,pred\n")
    (tmp_path / "grading_report.json").write_text('{"competition_id": "comp", "score": 0.5}')

    report = tmp_path / "grading_report.json"
    monkeypatch.setattr(batch, "_run_agent", lambda *, run, data_dir: batch.AgentOutputs(
        submission_path=str(submission), journal_path=str(tmp_path / "journal.json"),
        metrics={"steps": 12, "wallclock_secs": 300.0, "time_to_first_valid_secs": 60.0},
    ))
    monkeypatch.setattr(batch, "_grade", lambda **kw: report)

    batch.execute_run(run=batch.load_runs()["comp_B2_0"], data_dir=seeded_registry)

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

    monkeypatch.setattr(batch, "_run_agent", fake_agent)
    monkeypatch.setattr(batch, "_grade", lambda **kw: report)

    batch.execute_run(run=batch.load_runs()["comp_A_0"], data_dir=seeded_registry)

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
    monkeypatch.setattr(batch, "_locate_run_group", lambda *, started_at: tmp_path)
    monkeypatch.setattr(batch, "_locate_outputs",
                        lambda *, run_output_dir: ("/x/submission.csv", None, None, None))

    batch._run_agent(
        run={"run_key": "comp_C2_0", "competition_id": "comp", "spec_path": str(spec)},
        data_dir=seeded_registry,
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
    monkeypatch.setattr(batch, "_locate_run_group", lambda *, started_at: tmp_path)
    monkeypatch.setattr(batch, "_locate_outputs",
                        lambda *, run_output_dir: (None, None, None, None))

    batch._run_agent(
        run={"run_key": "comp_A_0", "competition_id": "comp"},
        data_dir=seeded_registry,
    )
    assert captured["argv"][0] == str(tmp_path / ".venv" / "bin" / "python")
    assert "--container-config" in captured["argv"]  # benchmark resources, not Docker defaults


def test_locate_run_group_picks_the_one_just_created(monkeypatch, tmp_path):
    """mle-bench names its own output dir and ignores --run-dir.

    Regression: the driver looked in runs/batch_{run_key}/, which never exists,
    so every run failed at grading with "no submission to grade" while the real
    outputs sat in runs/<timestamp>_run-group_<agent>/.
    """
    import time as time_mod

    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(batch, "MLEBENCH_DIR", tmp_path)

    stale = runs / "2024-09-06T18-25-55-UTC_run-group_aide"
    stale.mkdir()
    os.utime(stale, (1_000_000, 1_000_000))  # one of ~150 pre-existing groups

    started = time_mod.time()
    fresh = runs / "2026-08-24T17-35-13-GMT_run-group_aide-prelude"
    fresh.mkdir()

    assert batch._locate_run_group(started_at=started) == fresh


def test_locate_run_group_returns_none_when_nothing_was_created(monkeypatch, tmp_path):
    import time as time_mod

    (tmp_path / "runs").mkdir()
    monkeypatch.setattr(batch, "MLEBENCH_DIR", tmp_path)
    assert batch._locate_run_group(started_at=time_mod.time()) is None


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

    monkeypatch.setattr(batch, "_run_agent", lambda *, run, data_dir: batch.AgentOutputs(
        submission_path=str(src / "submission.csv"), journal_path=str(src / "journal.json"),
        metrics={"steps": 8}, solution_path=str(src / "best_solution.py"),
        token_usage_path=str(src / "prelude_token_usage.jsonl"),
    ))
    monkeypatch.setattr(batch, "_grade", lambda **kw: report)

    batch.execute_run(run=batch.load_runs()["comp_B2_0"], data_dir=seeded_registry)

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
    monkeypatch.setattr(batch, "_locate_run_group", lambda *, started_at: tmp_path)
    monkeypatch.setattr(batch, "_locate_outputs", lambda *, run_output_dir: (None, None, None, None))

    batch._run_agent(run={"run_key": "comp_A_0", "competition_id": "comp"}, data_dir=seeded_registry)
    assert "PRELUDE_SPEC_PATH" not in captured["env"]  # Condition A: unset -> stock aide


def test_run_agent_missing_spec_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(batch, "MLEBENCH_DIR", tmp_path)
    with pytest.raises(RuntimeError, match="spec not found"):
        batch._run_agent(
            run={"run_key": "comp_B2_0", "competition_id": "comp",
                 "spec_path": str(tmp_path / "missing.md")},
            data_dir=seeded_registry,
        )


def test_submissionless_run_stays_retryable(seeded_registry, monkeypatch, tmp_path):
    """A run that produced nothing must not advance to agent_run.

    Regression: record_agent_run fired before the submission check, so the row
    became agent_run with a null submission — and the next --retry-abandoned
    skipped the agent, went straight to grading, and failed in 0s forever.
    """
    monkeypatch.setattr(batch, "_run_agent", lambda *, run, data_dir: batch.AgentOutputs(
        submission_path=None, journal_path=None, metrics={}))

    with pytest.raises(RuntimeError, match="no submission"):
        batch.execute_run(run={"run_key": "comp_B2_0", "competition_id": "comp",
                               "status": "spec_built"}, data_dir=seeded_registry)

    assert registry.load_runs()["comp_B2_0"].get("status") != "agent_run"


def test_grading_report_is_preserved_to_the_volume(seeded_registry, monkeypatch, tmp_path):
    """The report is written to the ephemeral run dir; the registry keeps only
    extracted fields, so --terminate-on-done would destroy the primary evidence."""
    submission = tmp_path / "run" / "submission" / "submission.csv"
    submission.parent.mkdir(parents=True)
    submission.write_text("id,pred\n")
    report = tmp_path / "grading_report.json"
    report.write_text('{"competition_id": "comp", "score": 0.5}')

    monkeypatch.setattr(batch, "_run_agent", lambda *, run, data_dir: batch.AgentOutputs(
        submission_path=str(submission), journal_path=None, metrics={}))
    monkeypatch.setattr(batch, "_grade", lambda **kw: report)

    batch.execute_run(run=batch.load_runs()["comp_B2_0"], data_dir=seeded_registry)

    preserved = batch.artifacts.run_root() / "comp_B2_0" / "grading_report.json"
    assert preserved.is_file()


def test_terminate_refuses_when_results_are_on_the_boot_disk(seeded_registry, monkeypatch):
    """Terminating with results on ephemeral storage destroys the whole run set.

    Every earlier mechanism for this failed silently — a skipped symlink, an
    unset env var — so the check sits immediately before the irreversible step
    and leaves the box running instead.
    """
    calls = []
    monkeypatch.setattr(batch, "_results_survive_termination", lambda: False)
    monkeypatch.setattr(
        batch.lambda_ctl, "terminate_instance",
        lambda *, instance_id: calls.append(instance_id),
    )
    batch.run_batch(
        data_dir=seeded_registry, execute=lambda *, run, data_dir: None,
        terminate_on_done=True, instance_id="i-123",
    )
    assert calls == []  # box left running; results recoverable


def test_results_root_follows_the_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv(registry.RESULTS_ENV, str(tmp_path / "volume"))
    assert registry.results_root() == tmp_path / "volume"
    assert registry.registry_path().parent == tmp_path / "volume"
    monkeypatch.delenv(registry.RESULTS_ENV)
    assert registry.results_root() == registry.RESULTS_DIR


def test_spec_resolves_against_this_machines_results_root(monkeypatch, tmp_path):
    """spec_path is written relative to the DEV machine's root.

    Regression: with the box writing to a persistent volume, the stored
    `results/dev/.../spec.md` resolved against the repo and every run failed
    with "spec not found" before starting a container.
    """
    volume = tmp_path / "volume"
    spec = volume / "dev" / "comp_C2_0" / "spec.md"
    spec.parent.mkdir(parents=True)
    spec.write_text("ADVISOR CONTEXT\n")
    monkeypatch.setenv(registry.RESULTS_ENV, str(volume))

    captured = {}
    monkeypatch.setattr(batch, "MLEBENCH_DIR", tmp_path)
    monkeypatch.setattr(batch.subprocess, "run",
                        lambda argv, **kw: captured.update(env=kw.get("env")))
    monkeypatch.setattr(batch, "_locate_run_group", lambda *, started_at: tmp_path)
    monkeypatch.setattr(batch, "_locate_outputs",
                        lambda *, run_output_dir: ("/x/submission.csv", None, None, None))

    batch._run_agent(
        run={"run_key": "comp_C2_0", "competition_id": "comp",
             "spec_path": "results/dev/comp_C2_0/spec.md"},  # the dev machine's path
        data_dir=seeded_registry,
    )
    assert captured["env"]["PRELUDE_SPEC_PATH"] == str(spec.resolve())


# ── convergence + cost measures (H3) ────────────────────────────────────
#
# Node shapes copied from a real aideml journal (2026-08-24 box run): `ctime` is
# stamped when the drafting call RETURNS, so node N+1's ctime is node N's ctime
# plus its exec_time plus the next draft. `metric` is a dict carrying its own
# direction, and a buggy node's value is None inside that dict.

def _journal(*, tmp_path, nodes: list[dict]) -> str:
    path = tmp_path / "journal.json"
    path.write_text(json.dumps({"nodes": nodes}))
    return str(path)


def _node(*, step, ctime, exec_time=1.0, value=None, buggy=False, maximize=True) -> dict:
    return {
        "step": step,
        "ctime": ctime,
        "exec_time": exec_time,
        "is_buggy": buggy,
        "metric": {"value": value, "maximize": maximize},
    }


def _token_log(*, tmp_path, starts: list[float]) -> str:
    path = tmp_path / "prelude_token_usage.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"t_start": start, "t_end": start + 1, "input_tokens": 10,
                        "output_tokens": 2, "cache_read_input_tokens": 3,
                        "cache_creation_input_tokens": 0})
            for start in starts
        )
    )
    return str(path)


def test_first_valid_at_node_zero_is_not_time_zero(tmp_path):
    """Regression: measuring from min(ctime) floored this metric at exactly 0.0.

    A spec good enough to make the agent's first draft work is the case H3 most
    needs to resolve, and creation-to-creation timing reported it as no time at
    all.
    """
    nodes = [_node(step=0, ctime=1000.0, exec_time=118.8, value=0.63)]
    metrics = batch._read_journal_metrics(
        journal_path=_journal(tmp_path=tmp_path, nodes=nodes),
        token_usage_path=_token_log(tmp_path=tmp_path, starts=[986.0]),
    )
    assert metrics["steps_to_first_valid"] == 1
    assert metrics["time_to_first_valid_secs"] == pytest.approx(132.8)
    assert metrics["timing_origin"] == "first_llm_call"


def test_milestones_skip_buggy_nodes_and_track_the_metric_direction(tmp_path):
    nodes = [
        _node(step=0, ctime=1000.0, exec_time=5.0, buggy=True),
        _node(step=1, ctime=1020.0, exec_time=10.0, value=0.50),
        _node(step=2, ctime=1050.0, exec_time=10.0, value=0.70),
        _node(step=3, ctime=1080.0, exec_time=10.0, value=0.60),
    ]
    metrics = batch._read_journal_metrics(
        journal_path=_journal(tmp_path=tmp_path, nodes=nodes),
        token_usage_path=_token_log(tmp_path=tmp_path, starts=[990.0]),
    )
    assert metrics["steps_to_first_valid"] == 2       # 1-based, buggy node 0 skipped
    assert metrics["steps_to_best"] == 3              # the 0.70 node, not the last
    assert metrics["best_validation_score"] == 0.70
    assert metrics["time_to_best_secs"] == pytest.approx(70.0)


def test_best_node_honours_lower_is_better(tmp_path):
    nodes = [
        _node(step=0, ctime=1000.0, value=0.30, maximize=False),
        _node(step=1, ctime=1010.0, value=0.90, maximize=False),
    ]
    metrics = batch._read_journal_metrics(
        journal_path=_journal(tmp_path=tmp_path, nodes=nodes),
        token_usage_path=None,
    )
    assert metrics["best_validation_score"] == 0.30
    assert metrics["timing_origin"] == "first_node"   # no token log to anchor to


def test_all_buggy_run_reports_no_milestones(tmp_path):
    nodes = [_node(step=0, ctime=1000.0, buggy=True), _node(step=1, ctime=1010.0, buggy=True)]
    metrics = batch._read_journal_metrics(
        journal_path=_journal(tmp_path=tmp_path, nodes=nodes), token_usage_path=None
    )
    assert metrics["steps"] == 2
    assert metrics["steps_to_first_valid"] is None
    assert metrics["time_to_first_valid_secs"] is None
    assert metrics["best_validation_score"] is None


def test_agent_token_totals_complete_the_cost_ledger(tmp_path):
    """The spec side is already in the registry; without this the agent side
    lived only in a preserved artifact and the two could not be compared."""
    usage = batch._read_token_usage(
        token_usage_path=_token_log(tmp_path=tmp_path, starts=[1.0, 2.0, 3.0])
    )
    assert usage == {
        "llm_calls": 3,
        "llm_input_tokens": 30,
        "llm_output_tokens": 6,
        "llm_cache_read_tokens": 9,
        "llm_cache_creation_tokens": 0,
    }


def test_leaderboard_percentile_is_direction_aware(tmp_path, monkeypatch):
    board = tmp_path / "mlebench" / "competitions" / "comp"
    board.mkdir(parents=True)
    (board / "leaderboard.csv").write_text("teamId,score\n1,0.1\n2,0.2\n3,0.3\n4,0.4\n")
    monkeypatch.setattr(batch, "MLEBENCH_DIR", tmp_path)

    higher = {"score": 0.35, "valid_submission": True, "is_lower_better": False}
    assert batch._leaderboard_percentile(competition_id="comp", report=higher) == 0.75

    lower = {"score": 0.15, "valid_submission": True, "is_lower_better": True}
    assert batch._leaderboard_percentile(competition_id="comp", report=lower) == 0.75


def test_leaderboard_percentile_none_without_a_valid_submission(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "MLEBENCH_DIR", tmp_path)
    report = {"score": 0.5, "valid_submission": False, "is_lower_better": False}
    assert batch._leaderboard_percentile(competition_id="comp", report=report) is None


def test_leaderboard_is_preserved_onto_the_results_root(tmp_path, monkeypatch):
    """The leaderboard must outlive the instance that held it.

    It is a git-lfs file inside the mle-bench checkout, which exists only on the
    cloud box. Without this copy the percentile is computable exactly once, at
    grade time, and can never be backfilled or audited afterwards — two smoke
    runs lost the field that way when their box was terminated.
    """
    board = tmp_path / "mlebench" / "competitions" / "comp"
    board.mkdir(parents=True)
    (board / "leaderboard.csv").write_text("teamId,score\n1,0.1\n2,0.3\n")
    monkeypatch.setattr(batch, "MLEBENCH_DIR", tmp_path)
    monkeypatch.setenv(registry.RESULTS_ENV, str(tmp_path / "volume"))

    report = {"score": 0.2, "valid_submission": True, "is_lower_better": False}
    assert batch._leaderboard_percentile(competition_id="comp", report=report) == 0.5

    preserved = tmp_path / "volume" / "leaderboards" / "comp.csv"
    assert preserved.is_file()  # copied without an operator step


def test_percentile_works_from_the_preserved_copy_alone(tmp_path, monkeypatch):
    """The dev machine has no mle-bench checkout, so this is the analysis path."""
    preserved = tmp_path / "volume" / "leaderboards" / "comp.csv"
    preserved.parent.mkdir(parents=True)
    preserved.write_text("teamId,score\n1,0.1\n2,0.3\n")
    monkeypatch.setattr(batch, "MLEBENCH_DIR", tmp_path / "absent")
    monkeypatch.setenv(registry.RESULTS_ENV, str(tmp_path / "volume"))

    report = {"score": 0.2, "valid_submission": True, "is_lower_better": False}
    assert batch._leaderboard_percentile(competition_id="comp", report=report) == 0.5


def test_preserved_leaderboard_is_not_overwritten_by_the_checkout(tmp_path, monkeypatch):
    """The percentile must be re-derivable from the bytes it was computed from."""
    board = tmp_path / "mlebench" / "competitions" / "comp"
    board.mkdir(parents=True)
    (board / "leaderboard.csv").write_text("teamId,score\n1,9.9\n")
    preserved = tmp_path / "volume" / "leaderboards" / "comp.csv"
    preserved.parent.mkdir(parents=True)
    preserved.write_text("teamId,score\n1,0.1\n2,0.3\n")
    monkeypatch.setattr(batch, "MLEBENCH_DIR", tmp_path)
    monkeypatch.setenv(registry.RESULTS_ENV, str(tmp_path / "volume"))

    assert batch.leaderboard_path(competition_id="comp") == preserved
    assert "9.9" not in preserved.read_text()
