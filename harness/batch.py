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

A run that raises is logged and skipped so a single failure never stalls the
batch (the whole point is to keep the GPU busy).

Runs on the cloud box only: it executes Docker/AIDE and MLE-bench grading. The
spec pipeline never runs here (CLAUDE.md core constraint 1) — specs are built
on the dev machine and shipped in.

The box-specific invocations — the mle-bench run_agent + grade commands and
the AIDE journal schema — are isolated in the functions marked [confirm on
box] and verified on the first smoke run. The orchestration (queue selection,
sequencing, failure isolation, registry advancement, terminate gating) is
covered by tests with those seams mocked.
"""

import argparse
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from harness import advance, lambda_ctl
from harness.registry import load_runs

logger = logging.getLogger("harness.batch")

# A run needs work unless it has been graded. spec_built/registered runs need
# the agent then grading; an interrupted agent_run resumes at grading.
DONE_STATUS = "graded"
NEEDS_AGENT_STATUSES = {"spec_built", "registered"}

MLEBENCH_DIR = Path(os.environ.get("MLEBENCH_DIR", str(Path.home() / "work" / "mle-bench")))


@dataclass
class AgentOutputs:
    submission_path: str | None
    journal_path: str | None
    metrics: dict


def pending_runs() -> list[dict]:
    """Unfinished runs, grouped by competition then condition then seed.

    Grouping by competition keeps all of a problem's condition-variants
    contiguous, so a competition's full cross-condition set completes together
    and can be analyzed without waiting for the rest of the grid.
    """
    unfinished = [run for run in load_runs().values() if run.get("status") != DONE_STATUS]
    return sorted(
        unfinished, key=lambda run: (run["competition_id"], run["condition"], run["seed"])
    )


# ── box seams: verify on the first smoke run ([confirm on box]) ──────────

def _run_agent(*, run: dict, data_dir: Path) -> AgentOutputs:
    """Launch one full AIDE run via mle-bench; block until it exits.

    subprocess.run returns only when AIDE has run its search to the config cap
    (or the time wall) — that exit is the completion signal serializing the
    queue. The step/time budget is the agent config's, not set here.

    [confirm on box]: exact run_agent.py flags, the spec-mount mechanism, and
    the output directory layout. Condition A (no spec_path) runs with no spec
    mounted (stock aide); B/C mount results/{run_key}/spec.md at
    /home/spec/spec.md, which aide-prelude/start.sh appends as ADVISOR CONTEXT.
    """
    spec_path = run.get("spec_path")
    argv = [
        "python", "run_agent.py",
        "--agent-id", run.get("agent_id", "aide-prelude"),
        "--competition", run["competition_id"],
        "--data-dir", str(data_dir),
    ]
    if spec_path:
        argv += ["--extra-mount", f"{spec_path}:/home/spec/spec.md"]
    logger.info("agent argv: %s", " ".join(argv))
    subprocess.run(argv, cwd=MLEBENCH_DIR, check=True)
    submission_path, journal_path = _locate_outputs(run_key=run["run_key"])
    metrics = _read_journal_metrics(journal_path=journal_path) if journal_path else {}
    return AgentOutputs(
        submission_path=submission_path, journal_path=journal_path, metrics=metrics
    )


def _locate_outputs(*, run_key: str) -> tuple[str | None, str | None]:
    """(submission.csv, journal) paths from the agent run's output dir.

    [confirm on box]: mle-bench writes a per-run output dir (best_submission +
    AIDE journal). Resolve the real layout on the smoke run and return the two
    paths; None for whichever is absent (e.g. a run that made no submission).
    """
    raise NotImplementedError("resolve mle-bench output layout on the smoke run")


def _read_journal_metrics(*, journal_path: str) -> dict:
    """steps / wallclock / time-to-first-valid from the AIDE journal.

    [confirm on box]: parse the journal schema for the efficiency ledger
    (RESEARCH_DESIGN.md). Defensive — missing keys return None, never raise, so
    a schema surprise degrades to a graded run with blank timing, not a lost
    batch.
    """
    raise NotImplementedError("parse the AIDE journal schema on the smoke run")


def _grade(*, run: dict, submission_path: str, data_dir: Path, report_path: Path) -> Path:
    """Grade one submission with mle-bench; return the grading_report.json path.

    [confirm on box]: exact `mlebench grade` flags and report output path.
    """
    argv = [
        ".venv/bin/mlebench", "grade",
        "--submission", submission_path,
        "--competition", run["competition_id"],
        "--data-dir", str(data_dir),
        "--output", str(report_path),
    ]
    logger.info("grade argv: %s", " ".join(argv))
    subprocess.run(argv, cwd=MLEBENCH_DIR, check=True)
    return report_path


# ── orchestration ───────────────────────────────────────────────────────

def execute_run(*, run: dict, data_dir: Path) -> dict:
    """Take one run from its current status through to graded; advance registry."""
    run_key = run["run_key"]
    submission_path = run.get("agent_submission_path")

    if run.get("status") in NEEDS_AGENT_STATUSES:
        outputs = _run_agent(run=run, data_dir=data_dir)
        submission_path = outputs.submission_path
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

    report_path = Path(submission_path).parent / "grading_report.json"
    _grade(run=run, submission_path=submission_path, data_dir=data_dir, report_path=report_path)
    report = advance._report_for(
        report_path=report_path, run_key=run_key, competition_id=run["competition_id"]
    )
    return advance.record_graded(run_key=run_key, report=report)


def run_batch(
    *,
    data_dir: Path,
    terminate_on_done: bool = False,
    instance_id: str | None = None,
    execute=execute_run,
) -> dict:
    """Drain pending runs in sequence; optionally terminate the box afterward.

    Returns a summary dict (attempted / succeeded / failed run_keys). Terminate
    only fires on normal loop completion — if the loop itself raises, the box is
    left up for inspection.
    """
    queue = pending_runs()
    logger.info("batch: %d pending run(s)", len(queue))
    succeeded: list[str] = []
    failed: list[str] = []

    for run in queue:
        run_key = run["run_key"]
        started = time.monotonic()
        logger.info("run %s: starting (status=%s)", run_key, run.get("status"))
        try:
            execute(run=run, data_dir=data_dir)
            succeeded.append(run_key)
            logger.info("run %s: graded (%.0fs)", run_key, time.monotonic() - started)
        except Exception:
            failed.append(run_key)
            logger.exception(
                "run %s: failed after %.0fs — skipping", run_key, time.monotonic() - started
            )

    summary = {"attempted": len(queue), "succeeded": succeeded, "failed": failed}
    logger.info("batch done: %d ok, %d failed", len(succeeded), len(failed))

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
    args = parser.parse_args()
    if not args.data_dir:
        raise SystemExit("--data-dir or $MLEBENCH_DATA_DIR is required")
    run_batch(
        data_dir=Path(args.data_dir),
        terminate_on_done=args.terminate_on_done,
        instance_id=args.instance_id,
    )
