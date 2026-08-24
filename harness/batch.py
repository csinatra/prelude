"""Sequential batch driver for the cloud box — drain the run queue back-to-back.

Manual per-run execution leaves the GPU idle twice: between one run finishing
and a human starting the next, and after the last run until someone tears the
box down. This driver removes both gaps. It reads the registry, runs every
unfinished run to completion in sequence (agent -> grade -> advance), and with
--terminate-on-done destroys the Lambda instance once the queue is drained.

Granularity: one queue item = one run_key = one (competition x condition x
seed). Each condition is its own independent AIDE run with its own spec (or no
spec, for Condition A) — they never share a run, because the experiment
compares what AIDE does under each condition on the same competition. The
registry already keys at this granularity, so pending_runs() yields one item
per condition-variant, ordered so a competition's variants run contiguously.

Completion signal: a full AIDE run is one run_agent.py process. AIDE searches
until it exhausts the agent config's step/time cap (config.yaml — fixed and
uniform across all competitions and conditions by benchmark-fairness design;
the driver does not set it). Actual runtime varies by problem: a light
competition exhausts the step budget and exits early, a heavy one runs to the
time wall. Whenever the process exits, that is the signal the run is done and
the next may start; blocking on it serializes the queue (one GPU = one run at a
time, so the loop is deliberately serial — no scheduler, no parallelism).

A run that raises is parked — marked `abandoned` in the registry with its
error — and skipped, so a single failure never stalls the batch and, crucially,
a deterministic failure is not re-attempted on the next invocation (which would
burn GPU unbounded and keep the box from ever draining to termination). There
is no automatic retry: fix the root cause, then re-queue parked runs explicitly
with --retry-abandoned.

Runs on the cloud box only: it executes Docker/AIDE and MLE-bench grading. The
spec pipeline never runs here (CLAUDE.md core constraint 1) — specs are built
on the dev machine and shipped in.

The box-specific invocations — the mle-bench run_agent + grade commands and
the AIDE journal schema — are isolated in the functions marked [confirm on
box]. The 2026-07-24 smoke confirmed the output layout and run_agent
invocation these target; the batch's own grade (via JSONL) and journal-metric
parsing are first exercised on the initial automated drain, and should be
re-verified if the pinned mle-bench commit changes. The orchestration (queue
selection, sequencing, failure isolation, registry advancement, terminate
gating) is covered by tests with those seams mocked.
"""

import argparse
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from analysis import artifacts
from harness import advance, lambda_ctl, registry
from harness.registry import load_runs

logger = logging.getLogger("harness.batch")

# A run needs work unless it has been graded. spec_built/registered runs need
# the agent then grading; an interrupted agent_run resumes at grading.
DONE_STATUS = "graded"
NEEDS_AGENT_STATUSES = {"spec_built", "registered"}

MLEBENCH_DIR = Path(os.environ.get("MLEBENCH_DIR", str(Path.home() / "work" / "mle-bench")))
# mle-bench lives in its own 3.11 venv (setup_cloudbox.sh); prelude has a
# separate one. Bare `python` picks up whichever is on PATH — which is prelude's
# under `.venv/bin/python -m harness.batch`, and lacks mle-bench's dependencies.
MLEBENCH_PYTHON = MLEBENCH_DIR / ".venv" / "bin" / "python"
MLEBENCH_CLI = MLEBENCH_DIR / ".venv" / "bin" / "mlebench"
CONTAINER_CONFIG = Path(__file__).resolve().parent.parent / "cloudbox" / "container_config.json"


@dataclass
class AgentOutputs:
    submission_path: str | None
    journal_path: str | None
    metrics: dict
    solution_path: str | None = None
    token_usage_path: str | None = None
    viz_paths: tuple[str, ...] = ()


def pending_runs() -> list[dict]:
    """Unfinished, non-abandoned runs, grouped by competition then condition then seed.

    Grouping by competition keeps all of a problem's condition-variants
    contiguous, so a competition's full cross-condition set completes together
    and can be analyzed without waiting for the rest of the grid. Abandoned
    (parked-failure) runs are excluded so they never re-run automatically;
    --retry-abandoned clears the flag to re-queue them.
    """
    unfinished = [
        run
        for run in load_runs().values()
        if run.get("status") != DONE_STATUS and not run.get("abandoned")
    ]
    return sorted(
        unfinished, key=lambda run: (run["competition_id"], run["condition"], run["seed"])
    )


# ── box seams (interface confirmed on the smoke run 2026-07-23) ──────────

def _run_agent(*, run: dict, data_dir: Path) -> AgentOutputs:
    """Launch one full AIDE run via mle-bench; block until it exits.

    subprocess.run returns only when AIDE has run its search to the config cap
    (or the time wall) — that exit is the completion signal serializing the
    queue. The step/time budget is the agent config's, not set here.

    mle-bench takes a `--competition-set` FILE (one competition id per line),
    not a `--competition` flag; we give it a per-run split file and a dedicated
    `--run-dir`. B/C spec injection: run_agent has no `--extra-mount`, so our
    setup_cloudbox.sh patches mle-bench's agents/run.py with a hook that mounts
    the file named by the PRELUDE_SPEC_PATH env var read-only at /home/spec/spec.md
    (which aide-prelude/start.sh appends as ADVISOR CONTEXT). Condition A leaves
    the var unset -> stock aide. The B/C spec mount was confirmed end-to-end on
    the 2026-07-24 smoke run (ADVISOR CONTEXT appended, valid submission — see
    docs/DECISIONS.md).
    """
    run_output_dir = MLEBENCH_DIR / "runs" / f"batch_{run['run_key']}"
    comp_set = MLEBENCH_DIR / "experiments" / "splits" / f"{run['run_key']}.txt"
    comp_set.parent.mkdir(parents=True, exist_ok=True)
    comp_set.write_text(run["competition_id"] + "\n")

    argv = [
        str(MLEBENCH_PYTHON), "run_agent.py",
        "--agent-id", run.get("agent_id", "aide-prelude"),
        "--competition-set", str(comp_set),
        "--data-dir", str(data_dir),
        "--run-dir", str(run_output_dir),
        # mle-bench's default container config gives the agent 4 vCPUs and no
        # GPU, which is a Docker default rather than the benchmark's stated
        # baseline (36 vCPUs + one A10). See cloudbox/README.md.
        "--container-config", str(CONTAINER_CONFIG),
    ]
    env = os.environ.copy()
    spec_path = run.get("spec_path")  # relative to the prelude repo = the batch driver's cwd
    if spec_path:
        spec_abs = Path(spec_path).resolve()
        if not spec_abs.is_file():
            raise RuntimeError(f"{run['run_key']}: spec not found at {spec_abs}")
        env["PRELUDE_SPEC_PATH"] = str(spec_abs)
    logger.info("agent argv: %s (spec=%s)", " ".join(argv), env.get("PRELUDE_SPEC_PATH", "-"))
    subprocess.run(argv, cwd=MLEBENCH_DIR, check=True, env=env)
    submission_path, journal_path, solution_path, token_usage_path = _locate_outputs(
        run_output_dir=run_output_dir
    )
    metrics = _read_journal_metrics(journal_path=journal_path) if journal_path else {}
    return AgentOutputs(
        submission_path=submission_path,
        journal_path=journal_path,
        metrics=metrics,
        solution_path=solution_path,
        token_usage_path=token_usage_path,
        viz_paths=_locate_viz(run_output_dir=run_output_dir),
    )


def _locate_outputs(
    *, run_output_dir: Path
) -> tuple[str | None, str | None, str | None, str | None]:
    """(submission.csv, journal.json, best_solution.py, prelude_token_usage.jsonl).

    mle-bench writes <run-dir>/<group>/<competition>_<uuid>/ with submission/,
    logs/, and code/. Glob so we don't depend on the exact group/uuid nesting;
    None for whichever is absent (e.g. a buggy run with no submission). The
    solution falls back to code/solution.py if the best_solution snapshot is
    missing; the token log is the agent-side usage side-channel."""
    def find(pattern: str) -> str | None:
        hit = next(run_output_dir.glob(pattern), None)
        return str(hit) if hit else None

    submission = find("**/submission/submission.csv")
    journal = find("**/logs/journal.json")
    solution = find("**/logs/best_solution.py") or find("**/code/solution.py")
    token_usage = find("**/logs/prelude_token_usage.jsonl")
    return (submission, journal, solution, token_usage)


def _locate_viz(*, run_output_dir: Path) -> tuple[str, ...]:
    """AIDE's own search-tree visualization, if it wrote one.

    The journal is the machine-readable record; this is the navigable one, and
    it is what makes a 500-step trajectory reviewable by a human at all
    (docs/JUDGE_VALIDATION.md). Globbed rather than named because the filename
    is aideml's, not ours, and unverified against a real run — anything the agent
    left in logs/ as HTML is worth keeping. Absent is fine: nothing downstream
    requires it."""
    return tuple(str(path) for path in sorted(run_output_dir.glob("**/logs/*.html")))


def _read_journal_metrics(*, journal_path: str) -> dict:
    """steps / wallclock / time-to-first-valid from the AIDE journal.

    Journal is {"nodes": [{"step", "ctime", "exec_time", "metric", "is_buggy",
    ...}]}. Defensive — any schema surprise returns {}, never raises, so a graded
    run keeps its score even if timing can't be parsed."""
    try:
        nodes = json.loads(Path(journal_path).read_text()).get("nodes", [])
        ctimes = [n["ctime"] for n in nodes if "ctime" in n]
        if not ctimes:
            return {"steps": len(nodes)}
        start = min(ctimes)
        first_valid = next((n["ctime"] for n in nodes if not n.get("is_buggy")), None)
        return {
            "steps": len(nodes),
            "wallclock_secs": round(max(ctimes) - start, 3),
            "time_to_first_valid_secs": round(first_valid - start, 3)
            if first_valid is not None
            else None,
        }
    except Exception:
        logger.exception("journal parse failed: %s", journal_path)
        return {}


def _grade(*, run: dict, submission_path: str, data_dir: Path, report_dir: Path) -> Path:
    """Grade one submission with mle-bench; return the grading_report.json path.

    `mlebench grade` takes a JSONL (one {competition_id, submission_path} per
    line) via --submission and writes a timestamped grading_report.json into
    --output-dir. Requires the mle-bench leaderboards pulled from git-lfs
    (setup_cloudbox.sh) or medal-ranking asserts on a missing `score` column."""
    report_dir.mkdir(parents=True, exist_ok=True)
    jsonl = report_dir / "submission.jsonl"
    jsonl.write_text(
        json.dumps({"competition_id": run["competition_id"], "submission_path": submission_path})
        + "\n"
    )
    argv = [
        str(MLEBENCH_CLI), "grade",
        "--submission", str(jsonl),
        "--output-dir", str(report_dir),
        "--data-dir", str(data_dir),
    ]
    logger.info("grade argv: %s", " ".join(argv))
    subprocess.run(argv, cwd=MLEBENCH_DIR, check=True)
    report = next(report_dir.glob("*grading_report.json"), None)
    if report is None:
        raise RuntimeError(f"no grading report written to {report_dir}")
    return report


# ── orchestration ───────────────────────────────────────────────────────

def execute_run(*, run: dict, data_dir: Path) -> dict:
    """Take one run from its current status through to graded; advance registry."""
    run_key = run["run_key"]
    submission_path = run.get("agent_submission_path")

    if run.get("status") in NEEDS_AGENT_STATUSES:
        outputs = _run_agent(run=run, data_dir=data_dir)
        submission_path = outputs.submission_path
        # Preserve agent outputs onto the persistent volume BEFORE grading: the
        # mle-bench run dir is on the ephemeral boot disk, so without this copy
        # --terminate-on-done would destroy the journal + solution the
        # mechanistic judge needs (the registry keeps only outcome fields).
        artifacts.preserve_agent_outputs(
            run_key=run_key,
            submission_path=outputs.submission_path,
            journal_path=outputs.journal_path,
            solution_path=outputs.solution_path,
            extra_paths=(outputs.token_usage_path, *outputs.viz_paths),
        )
        advance.record_agent_run(
            run_key=run_key,
            submission_path=outputs.submission_path,
            trajectory_path=outputs.journal_path,
            wallclock_secs=outputs.metrics.get("wallclock_secs"),
            steps=outputs.metrics.get("steps"),
            time_to_first_valid_secs=outputs.metrics.get("time_to_first_valid_secs"),
            agent_id=run.get("agent_id", "aide-prelude"),
        )

    if not submission_path:
        raise RuntimeError(f"{run_key}: no submission to grade")

    report_dir = Path(submission_path).parent.parent / "grade"
    report_path = _grade(
        run=run, submission_path=submission_path, data_dir=data_dir, report_dir=report_dir
    )
    report = advance._report_for(
        report_path=report_path, run_key=run_key, competition_id=run["competition_id"]
    )
    return advance.record_graded(run_key=run_key, report=report)


def _abandon(*, run: dict, error: str) -> None:
    """Park a failed run: mark it abandoned + record the error, keeping the phase
    status so a later --retry-abandoned resumes at the right point."""
    registry.append_run(
        entry={
            "run_key": run["run_key"],
            "abandoned": True,
            "last_error": error,
            "abandoned_at": datetime.now(tz=timezone.utc).isoformat(),
        }
    )


def _clear_abandoned() -> list[str]:
    """Un-park every abandoned run so pending_runs re-queues it. Operator-driven."""
    parked = [run["run_key"] for run in load_runs().values() if run.get("abandoned")]
    for run_key in parked:
        registry.append_run(entry={"run_key": run_key, "abandoned": False})
    return parked


def run_batch(
    *,
    data_dir: Path,
    terminate_on_done: bool = False,
    instance_id: str | None = None,
    retry_abandoned: bool = False,
    execute=execute_run,
) -> dict:
    """Drain pending runs in sequence; optionally terminate the box afterward.

    Returns a summary dict (attempted / succeeded / abandoned run_keys). A failed
    run is parked (abandoned) with its error and skipped — never auto-retried.
    Pass retry_abandoned to un-park previously-failed runs first (after fixing
    the root cause). Terminate only fires on normal loop completion — if the loop
    itself raises, the box is left up for inspection.
    """
    if retry_abandoned:
        cleared = _clear_abandoned()
        logger.info("retry-abandoned: re-queued %d parked run(s)", len(cleared))

    queue = pending_runs()
    logger.info("batch: %d pending run(s)", len(queue))
    succeeded: list[str] = []
    abandoned: list[str] = []

    for run in queue:
        run_key = run["run_key"]
        started = time.monotonic()
        logger.info("run %s: starting (status=%s)", run_key, run.get("status"))
        try:
            execute(run=run, data_dir=data_dir)
            succeeded.append(run_key)
            logger.info("run %s: graded (%.0fs)", run_key, time.monotonic() - started)
        except Exception as exc:
            abandoned.append(run_key)
            _abandon(run=run, error=f"{type(exc).__name__}: {exc}")
            logger.exception(
                "run %s: failed after %.0fs — parked (abandoned), skipping",
                run_key,
                time.monotonic() - started,
            )

    summary = {"attempted": len(queue), "succeeded": succeeded, "abandoned": abandoned}
    logger.info("batch done: %d ok, %d abandoned", len(succeeded), len(abandoned))

    if terminate_on_done:
        if not instance_id:
            raise SystemExit(
                "--terminate-on-done requires --instance-id (from the launch response)"
            )
        logger.info("terminating instance %s", instance_id)
        lambda_ctl.terminate_instance(instance_id=instance_id)

    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("MLEBENCH_DATA_DIR"),
        help="prepared-competition data dir (default: $MLEBENCH_DATA_DIR)",
    )
    parser.add_argument(
        "--terminate-on-done",
        action="store_true",
        help="terminate the Lambda instance once the queue is drained",
    )
    parser.add_argument(
        "--instance-id",
        default=os.environ.get("LAMBDA_INSTANCE_ID"),
        help="Lambda instance id to terminate (from the launch response)",
    )
    parser.add_argument(
        "--retry-abandoned",
        action="store_true",
        help="re-queue previously-parked (failed) runs before draining",
    )
    args = parser.parse_args()
    if not args.data_dir:
        raise SystemExit("--data-dir or $MLEBENCH_DATA_DIR is required")
    run_batch(
        data_dir=Path(args.data_dir),
        terminate_on_done=args.terminate_on_done,
        instance_id=args.instance_id,
        retry_abandoned=args.retry_abandoned,
    )
