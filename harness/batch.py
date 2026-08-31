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
import csv
import json
import logging
import os
import shutil
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
    not a `--competition` flag; we give it a per-run split file. It does NOT
    accept an output location — `--run-dir` is parsed and never referenced, and
    outputs land in `runs/<timestamp>_run-group_<agent>/` under mle-bench's own
    tree — so the group directory is located after the fact.
    B/C spec injection: run_agent has no `--extra-mount`, so our
    setup_cloudbox.sh patches mle-bench's agents/run.py with a hook that mounts
    the file named by the PRELUDE_SPEC_PATH env var read-only at /home/spec/spec.md
    (which aide-prelude/start.sh appends as ADVISOR CONTEXT). Condition A leaves
    the var unset -> stock aide. The B/C spec mount was confirmed end-to-end on
    the 2026-07-24 smoke run (ADVISOR CONTEXT appended, valid submission — see
    docs/DECISIONS.md).
    """
    comp_set = MLEBENCH_DIR / "experiments" / "splits" / f"{run['run_key']}.txt"
    comp_set.parent.mkdir(parents=True, exist_ok=True)
    comp_set.write_text(run["competition_id"] + "\n")

    argv = [
        str(MLEBENCH_PYTHON), "run_agent.py",
        "--agent-id", run.get("agent_id", "aide-prelude"),
        "--competition-set", str(comp_set),
        "--data-dir", str(data_dir),
        # mle-bench's default container config gives the agent 4 vCPUs and no
        # GPU, which is a Docker default rather than the benchmark's stated
        # baseline (36 vCPUs + one A10). See cloudbox/README.md.
        "--container-config", str(CONTAINER_CONFIG),
    ]
    env = os.environ.copy()
    if run.get("spec_path"):  # presence = this condition has a spec; A does not
        env["PRELUDE_SPEC_PATH"] = str(_resolve_spec(run=run))
    logger.info("agent argv: %s (spec=%s)", " ".join(argv), env.get("PRELUDE_SPEC_PATH", "-"))
    started_at = time.time()
    subprocess.run(argv, cwd=MLEBENCH_DIR, check=True, env=env)
    run_output_dir = _locate_run_group(started_at=started_at)
    if run_output_dir is None:
        raise RuntimeError(f"{run['run_key']}: no run group created under {MLEBENCH_DIR / 'runs'}")
    logger.info("run group: %s", run_output_dir.name)
    submission_path, journal_path, solution_path, token_usage_path = _locate_outputs(
        run_output_dir=run_output_dir
    )
    metrics = (
        _read_journal_metrics(journal_path=journal_path, token_usage_path=token_usage_path)
        if journal_path
        else {}
    )
    metrics.update(_read_token_usage(token_usage_path=token_usage_path))
    return AgentOutputs(
        submission_path=submission_path,
        journal_path=journal_path,
        metrics=metrics,
        solution_path=solution_path,
        token_usage_path=token_usage_path,
        viz_paths=_locate_viz(run_output_dir=run_output_dir)
        + _locate_run_log(run_output_dir=run_output_dir),
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


def _locate_run_log(*, run_output_dir: Path) -> tuple[str, ...]:
    """The container's own log — evidence, not just debugging.

    It records that the spec was mounted (the ADVISOR CONTEXT banner and the
    `cat /home/spec/spec.md` that follows) and what hardware the agent saw from
    its startup probe. Both are claims the writeup makes about how runs were
    configured, so the log is preserved with the outputs rather than left on the
    ephemeral disk."""
    hit = next(run_output_dir.glob("**/run.log"), None)
    return (str(hit),) if hit else ()


def _resolve_spec(*, run: dict) -> Path:
    """Where this run's spec lives on THIS machine.

    The registry's `spec_path` is written by the dev machine relative to its own
    results root, so it does not resolve on a box whose root is the persistent
    volume. The canonical location is derivable — artifacts always live at
    run_root()/{run_key}/ — so the registry field is treated as a flag for
    whether the condition has a spec at all, and the literal path only as a
    fallback for rows written before the roots could differ.
    """
    run_key = run["run_key"]
    canonical = artifacts.run_root() / run_key / "spec.md"
    if canonical.is_file():
        return canonical.resolve()
    literal = Path(run["spec_path"])
    if literal.is_file():
        return literal.resolve()
    raise RuntimeError(f"{run_key}: spec not found at {canonical} or {literal}")


def _locate_run_group(*, started_at: float) -> Path | None:
    """The run-group directory mle-bench just created.

    It names its own output dir `runs/<timestamp>_run-group_<agent>/` and gives
    no way to choose one, so the only handle is what appeared during this call.
    Safe because the driver is deliberately serial — one agent run at a time —
    and the mtime floor keeps the ~150 pre-existing group dirs out.
    """
    groups = [
        path
        for path in (MLEBENCH_DIR / "runs").glob("*_run-group_*")
        if path.is_dir() and path.stat().st_mtime >= started_at - 5
    ]
    return max(groups, key=lambda path: path.stat().st_mtime, default=None)


def _locate_viz(*, run_output_dir: Path) -> tuple[str, ...]:
    """AIDE's own search-tree visualization, if it wrote one.

    The journal is the machine-readable record; this is the navigable one, and
    it is what makes a 500-step trajectory reviewable by a human at all
    (docs/JUDGE_VALIDATION.md). Globbed rather than named because the filename
    is aideml's, not ours, and unverified against a real run — anything the agent
    left in logs/ as HTML is worth keeping. Absent is fine: nothing downstream
    requires it."""
    return tuple(str(path) for path in sorted(run_output_dir.glob("**/logs/*.html")))


def _node_end(*, node: dict) -> float | None:
    """When a node finished executing.

    AIDE stamps `ctime` when the *drafting call returns*, not when the node
    finishes — verified against a real journal, where node N+1's ctime equals
    node N's ctime plus node N's exec_time plus the next draft. So a node's end
    is ctime + exec_time, and no journal field marks the run's true beginning.
    """
    if "ctime" not in node:
        return None
    return node["ctime"] + (node.get("exec_time") or 0)


def _node_metric(*, node: dict) -> tuple[float | None, bool]:
    """(value, maximize) from a node, tolerating both serialization shapes.

    A buggy node carries `{"value": None}` — a dict, not None — so testing the
    field itself would count failed nodes as scored.
    """
    metric = node.get("metric")
    if isinstance(metric, dict):
        return metric.get("value"), bool(metric.get("maximize", True))
    return metric, True


def _read_journal_metrics(*, journal_path: str, token_usage_path: str | None = None) -> dict:
    """Convergence measures for H3, from the AIDE journal.

    Journal is {"nodes": [{"step", "ctime", "exec_time", "metric", "is_buggy",
    ...}]}. Defensive — any schema surprise returns {}, never raises, so a graded
    run keeps its score even if timing can't be parsed.

    Two milestones per run, recorded in both steps and seconds:

    - **first valid** — the first node that ran without error. `steps_to_first_valid`
      leads the timing pair in H3's supporting analysis (RESEARCH_DESIGN.md),
      since wall-clock varies with data size, with whichever model the agent
      tries, and with GPU contention.
    - **best** — the node holding the run's best *validation* score. Note this is
      best-so-far under a fixed step budget, so it is censored and biased toward
      runs that happened to peak early; it summarizes the pre-registered per-step
      score curve and is descriptive only, carrying no separation criterion.

    Timing is measured from the first LLM call (`token_usage_path`), not from
    `min(ctime)`. Measuring creation-to-creation returns exactly 0.0 whenever
    node 0 is already valid — which is what a good spec is most likely to
    produce, and precisely the case H3 needs to resolve. The floor would silence
    the effect rather than measure it. Without the token log the origin falls
    back to the first node's ctime, and both measures are then understated by
    the first draft; the `timing_origin` field records which was used.
    """
    try:
        nodes = json.loads(Path(journal_path).read_text()).get("nodes", [])
        ends = [end for node in nodes for end in [_node_end(node=node)] if end is not None]
        if not ends:
            return {"steps": len(nodes)}
        start, origin = _agent_start(token_usage_path=token_usage_path), "first_llm_call"
        if start is None:
            start, origin = min(node["ctime"] for node in nodes if "ctime" in node), "first_node"

        def milestone(*, index: int | None, prefix: str) -> dict:
            if index is None:
                return {f"steps_to_{prefix}": None, f"time_to_{prefix}_secs": None}
            end = _node_end(node=nodes[index])
            return {
                # 1-based: "the agent's first attempt worked" is 1 step, not 0.
                f"steps_to_{prefix}": index + 1,
                f"time_to_{prefix}_secs": round(end - start, 3) if end is not None else None,
            }

        scored = [
            (value, maximize, index)
            for index, node in enumerate(nodes)
            if not node.get("is_buggy")
            for value, maximize in [_node_metric(node=node)]
            if value is not None
        ]
        best = None
        if scored:
            pick = max if scored[0][1] else min
            best = pick(scored, key=lambda item: item[0])[2]
        return {
            "steps": len(nodes),
            "wallclock_secs": round(max(ends) - start, 3),
            "timing_origin": origin,
            **milestone(
                index=next(
                    (i for i, node in enumerate(nodes) if not node.get("is_buggy")), None
                ),
                prefix="first_valid",
            ),
            **milestone(index=best, prefix="best"),
            "best_validation_score": _node_metric(node=nodes[best])[0]
            if best is not None
            else None,
        }
    except Exception:
        logger.exception("journal parse failed: %s", journal_path)
        return {}


def _agent_start(*, token_usage_path: str | None) -> float | None:
    """When the agent began work — the first LLM call's start.

    The token side-channel (`prelude_token_usage.jsonl`, appended by the
    aide-prelude backend) is the only record of the run's true origin: the
    journal's earliest ctime is stamped when the first draft *returns*. Verified
    against a real run, where the first node's ctime equals the first call's
    t_end exactly."""
    if not token_usage_path or not Path(token_usage_path).is_file():
        return None
    try:
        starts = [
            json.loads(line)["t_start"]
            for line in Path(token_usage_path).read_text().splitlines()
            if line.strip()
        ]
        return min(starts) if starts else None
    except Exception:
        logger.exception("token usage parse failed: %s", token_usage_path)
        return None


def _read_token_usage(*, token_usage_path: str | None) -> dict:
    """Agent-side token totals — the other half of H3's two-sided cost ledger.

    The spec side already lands in the registry (calls, in/out tokens); without
    this the agent side lived only in a preserved artifact, so the cost
    comparison the design calls for could not be made from the registry alone.
    Per-call detail stays in the artifact for per-step attribution."""
    if not token_usage_path or not Path(token_usage_path).is_file():
        return {}
    try:
        calls = [
            json.loads(line)
            for line in Path(token_usage_path).read_text().splitlines()
            if line.strip()
        ]
    except Exception:
        logger.exception("token usage parse failed: %s", token_usage_path)
        return {}
    def total(*, field: str) -> int:
        return sum(call.get(field) or 0 for call in calls)

    return {
        "llm_calls": len(calls),
        "llm_input_tokens": total(field="input_tokens"),
        "llm_output_tokens": total(field="output_tokens"),
        "llm_cache_read_tokens": total(field="cache_read_input_tokens"),
        "llm_cache_creation_tokens": total(field="cache_creation_input_tokens"),
    }


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


def leaderboard_path(*, competition_id: str) -> Path | None:
    """The competition's leaderboard, preserved onto the results root on first use.

    The leaderboards are git-lfs files inside the mle-bench checkout, which lives
    only on the cloud box and dies with the instance. That made
    `leaderboard_percentile` computable exactly once, at grade time, and
    unrecoverable afterwards: a run graded without it could never have it
    backfilled, and nothing about the analysis could be re-derived or audited on
    the dev machine. Two smoke runs lost the field that way.

    So the first computation copies the file under the results root, which is on
    the persistent volume and syncs back with every other artifact. No operator
    step, and nothing to remember before terminating an instance. The copy is
    git-tracked on the dev machine (see .gitignore), joining the corpus manifest
    and retrieval characterization as versioned reference data.

    Nothing is ever written back to a leaderboard: it is a static snapshot of the
    historical Kaggle standings, our runs are graded locally and never submitted,
    and `rank_score` only reads it. That is what makes a placement reconstructible
    later from a preserved score alone.

    The preserved copy wins over the checkout when both exist. mle-bench is
    pinned, so they agree; preferring the copy means the analysis reads the same
    bytes the percentile was computed from even if the checkout is later moved
    to a different commit.
    """
    preserved = registry.results_root() / "leaderboards" / f"{competition_id}.csv"
    if preserved.is_file():
        return preserved
    source = MLEBENCH_DIR / "mlebench" / "competitions" / competition_id / "leaderboard.csv"
    if not source.is_file():
        return None
    preserved.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, preserved)
    logger.info("preserved leaderboard: %s", preserved)
    return preserved


def _leaderboard_percentile(*, competition_id: str, report: dict) -> float | None:
    """Where the final submission lands in the competition's real leaderboard.

    H1's "higher resolution" measure (RESEARCH_DESIGN.md): it carries more
    information per run than a binary medal and guards against a medal
    difference that is really threshold luck. mle-bench's CompetitionReport
    gives medal booleans and thresholds but no percentile, so it is computed
    here.

    Reported as the fraction of leaderboard teams the submission beats, so higher
    is better under both metric directions.
    """
    if report.get("score") is None or not report.get("valid_submission"):
        return None
    path = leaderboard_path(competition_id=competition_id)
    if path is None:
        logger.warning("no leaderboard available for %s", competition_id)
        return None
    try:
        with path.open() as handle:
            scores = [
                float(row["score"])
                for row in csv.DictReader(handle)
                if row.get("score") not in (None, "")
            ]
        if not scores:
            return None
        score = report["score"]
        if report.get("is_lower_better"):
            beaten = sum(1 for other in scores if score < other)
        else:
            beaten = sum(1 for other in scores if score > other)
        return round(beaten / len(scores), 5)
    except Exception:
        logger.exception("leaderboard percentile failed: %s", path)
        return None


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
        # Check BEFORE advancing status. Recording agent_run for a run that
        # produced nothing leaves it unretryable: the next --retry-abandoned
        # sees agent_run, skips the agent entirely, and fails at grading in 0s.
        # Artifacts are preserved above either way, so a failed run is still
        # inspectable.
        if not outputs.submission_path:
            raise RuntimeError(f"{run_key}: agent produced no submission")
        advance.record_agent_run(
            run_key=run_key,
            submission_path=outputs.submission_path,
            trajectory_path=outputs.journal_path,
            steps=outputs.metrics.get("steps"),
            metrics=outputs.metrics,
            agent_id=run.get("agent_id", "aide-prelude"),
        )

    if not submission_path:
        raise RuntimeError(f"{run_key}: no submission to grade")

    report_dir = Path(submission_path).parent.parent / "grade"
    report_path = _grade(
        run=run, submission_path=submission_path, data_dir=data_dir, report_dir=report_dir
    )
    # The report lives on the ephemeral run dir like the agent outputs did, and
    # the registry keeps only the extracted fields — so without this copy
    # --terminate-on-done destroys the primary grading evidence.
    artifacts.preserve_agent_outputs(run_key=run_key, extra_paths=(str(report_path),))
    report = advance._report_for(
        report_path=report_path, run_key=run_key, competition_id=run["competition_id"]
    )
    report["leaderboard_percentile"] = _leaderboard_percentile(
        competition_id=run["competition_id"], report=report
    )
    return advance.record_graded(run_key=run_key, report=report)


def _results_survive_termination() -> bool:
    """Is the results root on a different filesystem than the repo?

    The guard exists because every previous mechanism for this failed silently:
    setup_cloudbox.sh's symlink was skipped with a warning buried in a long
    provisioning log, and an unset env var looks identical to a correct one until
    the instance is gone. Comparing st_dev is the only check that reflects
    physical reality rather than intent — whatever the configuration claims, the
    files are either on the volume or they are not.

    Terminating with results on the boot disk destroys every artifact the run
    produced, so the check runs immediately before the irreversible step.
    """
    results = registry.results_root()
    if not results.exists():
        return False
    repo = Path(__file__).resolve().parent.parent
    try:
        return results.resolve().stat().st_dev != repo.stat().st_dev
    except OSError:
        return False


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
        if not _results_survive_termination():
            logger.error(
                "REFUSING to terminate %s: results root %s is on the boot disk. "
                "Set %s to a path on the persistent volume, or rsync the results "
                "off the box first. The instance is left running.",
                instance_id,
                registry.results_root(),
                registry.RESULTS_ENV,
            )
            return summary
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
