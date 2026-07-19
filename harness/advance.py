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


def record_agent_run(
    *,
    run_key: str,
    submission_path: str | None = None,
    trajectory_path: str | None = None,
    wallclock_secs: float | None = None,
    steps: int | None = None,
    time_to_first_valid_secs: float | None = None,
    agent_id: str = "aide-prelude",
) -> dict:
    """Register that the AIDE run for this run_key completed.

    steps and time_to_first_valid_secs come from the AIDE journal; the full
    journal is preserved via trajectory_path for score-vs-time trajectory
    analysis (efficiency accounting — see RESEARCH_DESIGN.md).
    """
    return _advance(
        run_key=run_key,
        status="agent_run",
        fields={
            "agent_id": agent_id,
            "agent_submission_path": submission_path,
            "agent_trajectory_path": trajectory_path,
            "agent_wallclock_secs": wallclock_secs,
            "agent_steps": steps,
            "agent_time_to_first_valid_secs": time_to_first_valid_secs,
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
    agent_parser.add_argument("--wallclock-secs", type=float)
    agent_parser.add_argument("--steps", type=int)
    agent_parser.add_argument("--time-to-first-valid-secs", type=float)
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
        entry = record_agent_run(
            run_key=args.run_key,
            submission_path=args.submission,
            trajectory_path=args.trajectory,
            wallclock_secs=args.wallclock_secs,
            steps=args.steps,
            time_to_first_valid_secs=args.time_to_first_valid_secs,
            agent_id=args.agent_id,
        )
    else:
        competition_id = load_runs()[args.run_key]["competition_id"]
        report = _report_for(
            report_path=Path(args.report), run_key=args.run_key, competition_id=competition_id
        )
        entry = record_graded(run_key=args.run_key, report=report)
    print(json.dumps(entry, indent=2))
