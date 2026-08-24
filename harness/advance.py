"""Cloud-box side of the registry: advance runs through agent_run → graded.

Runs on the machine that executes AIDE and MLE-bench grading. Appends
transition entries to runs.jsonl — each entry carries only its new fields;
load_runs() merges per run_key, so spec-time fields survive. The resulting
registry file is what gets merged back to the dev machine.

Usage:
    python -m harness.advance register --competition <ID> --condition A --seed 0
    python -m harness.advance agent-run --run-key <K> \
        [--submission PATH] [--trajectory PATH] [--wallclock-secs N]
    python -m harness.advance graded --run-key <K> --report grading_report.json

`register` exists for runs with no spec-build phase — Condition A (stock
AIDE, no spec mounted; see the Condition A note in RESEARCH_DESIGN.md).
B/C runs enter the registry via harness.runner instead.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from analysis.artifacts import _git_provenance, run_key as make_run_key
from harness.registry import append_run, load_runs

# Subset of mle-bench's CompetitionReport (mlebench/grade_helpers.py, pinned
# commit 507f92e) recorded into the registry. Thresholds kept so medal
# distance is computable without re-opening report files.
GRADED_FIELDS = [
    "score",
    # Not from mle-bench: computed at grade time by harness.batch, since the
    # leaderboards it needs live only in the mle-bench checkout on the box.
    "leaderboard_percentile",
    "any_medal",
    "gold_medal",
    "silver_medal",
    "bronze_medal",
    "above_median",
    "submission_exists",
    "valid_submission",
    "is_lower_better",
    "gold_threshold",
    "silver_threshold",
    "bronze_threshold",
    "median_threshold",
]


def _advance(*, run_key: str, status: str, fields: dict) -> dict:
    known = load_runs()
    if run_key not in known:
        raise SystemExit(f"unknown run_key: {run_key} (build the spec first; check merge state)")
    entry = {
        "run_key": run_key,
        "status": status,
        **fields,
        "updated_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    append_run(entry=entry)
    return entry


def register_run(*, competition_id: str, condition: str, seed: int) -> dict:
    """Create the initial registry entry for a run with no spec-build phase."""
    key = make_run_key(competition_id=competition_id, condition=condition, seed=seed)
    if key in load_runs():
        raise SystemExit(f"run_key already registered: {key}")
    entry = {
        "run_key": key,
        "competition_id": competition_id,
        "condition": condition,
        "seed": seed,
        "status": "registered",
        **_git_provenance(),
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    append_run(entry=entry)
    return entry


# Journal- and token-log-derived fields (harness.batch._read_journal_metrics /
# _read_token_usage), stored under an `agent_` prefix to keep the agent side of
# the two-sided cost ledger distinct from the spec side's `spec_llm_*`.
AGENT_METRIC_FIELDS = [
    "wallclock_secs",
    "timing_origin",
    "steps_to_first_valid",
    "time_to_first_valid_secs",
    "steps_to_best",
    "time_to_best_secs",
    "best_validation_score",
    "llm_calls",
    "llm_input_tokens",
    "llm_output_tokens",
    "llm_cache_read_tokens",
    "llm_cache_creation_tokens",
]


def record_agent_run(
    *,
    run_key: str,
    submission_path: str | None = None,
    trajectory_path: str | None = None,
    steps: int | None = None,
    metrics: dict | None = None,
    agent_id: str = "aide-prelude",
) -> dict:
    """Register that the AIDE run for this run_key completed.

    `metrics` carries the journal- and token-log-derived measures; the full
    journal is preserved via trajectory_path for the per-step score/time curves
    (H3 — see RESEARCH_DESIGN.md). Only AGENT_METRIC_FIELDS are recorded, so a
    new key in the metrics dict has to be declared here before it reaches the
    registry and the writeup.
    """
    metrics = metrics or {}
    return _advance(
        run_key=run_key,
        status="agent_run",
        fields={
            "agent_id": agent_id,
            "agent_submission_path": submission_path,
            "agent_trajectory_path": trajectory_path,
            "agent_steps": steps,
            **{f"agent_{field}": metrics.get(field) for field in AGENT_METRIC_FIELDS},
        },
    )


def record_graded(*, run_key: str, report: dict) -> dict:
    """Register MLE-bench grading output (one CompetitionReport dict)."""
    return _advance(
        run_key=run_key,
        status="graded",
        fields={field: report.get(field) for field in GRADED_FIELDS},
    )


def _report_for(*, report_path: Path, run_key: str, competition_id: str) -> dict:
    """Accept either a single CompetitionReport or an aggregated grading report."""
    data = json.loads(report_path.read_text())
    if "competition_reports" in data:
        matches = [
            report
            for report in data["competition_reports"]
            if report["competition_id"] == competition_id
        ]
        if len(matches) != 1:
            raise SystemExit(
                f"{report_path} has {len(matches)} reports for {competition_id}; "
                f"pass a single-report file for {run_key}"
            )
        return matches[0]
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    register_parser = subparsers.add_parser("register")
    register_parser.add_argument("--competition", required=True)
    register_parser.add_argument("--condition", required=True)
    register_parser.add_argument("--seed", type=int, default=0)

    agent_parser = subparsers.add_parser("agent-run")
    agent_parser.add_argument("--run-key", required=True)
    agent_parser.add_argument("--submission")
    agent_parser.add_argument("--trajectory")
    agent_parser.add_argument("--steps", type=int)
    agent_parser.add_argument("--wallclock-secs", type=float)
    agent_parser.add_argument("--agent-id", default="aide-prelude")

    graded_parser = subparsers.add_parser("graded")
    graded_parser.add_argument("--run-key", required=True)
    graded_parser.add_argument("--report", required=True)

    args = parser.parse_args()
    if args.command == "register":
        entry = register_run(
            competition_id=args.competition, condition=args.condition, seed=args.seed
        )
    elif args.command == "agent-run":
        # Manual fallback for a run recorded by hand; the batch driver passes the
        # full metrics dict rather than going through these flags.
        entry = record_agent_run(
            run_key=args.run_key,
            submission_path=args.submission,
            trajectory_path=args.trajectory,
            steps=args.steps,
            metrics={"wallclock_secs": args.wallclock_secs},
            agent_id=args.agent_id,
        )
    else:
        competition_id = load_runs()[args.run_key]["competition_id"]
        report = _report_for(
            report_path=Path(args.report), run_key=args.run_key, competition_id=competition_id
        )
        entry = record_graded(run_key=args.run_key, report=report)
    print(json.dumps(entry, indent=2))
