"""Record a graded result for a run whose `execute_run` never returned.

Runs normally reach the registry through `harness.advance.record_graded`, called
at the end of `execute_run`. When the agent process dies mid-run that code is
never reached, so nothing is preserved and nothing is graded — the run stays at
`registered`/`spec_built` and is invisible to `load_runs()` even though a usable
submission exists inside the container.

That is not hypothetical: an Anthropic credit exhaustion on 2026-09-08 left
aideml deadlocked (parent in `multiprocessing`'s atexit `join`, child blocked on
`queue.get()`), so `run_agent` never returned and the harness waited
indefinitely. The container still held 116 nodes and a valid submission.

Recovery is: copy `/home/logs`, `/home/submission`, `/home/code` out of the
container into the run's artifact dir, grade the submission with
`mlebench grade-sample` into `manual_grading_report.json`, then run this to put
the result in the registry. Rows are marked `salvaged` with a note, so a
truncated run is never silently read as a clean one.
"""

import argparse
import datetime
import json
from pathlib import Path


def _metric(*, node: dict) -> float | None:
    return (node.get("metric") or {}).get("value")


def salvage_row(*, run_dir: Path, run_key: str, note: str) -> dict:
    """Build a graded registry row from preserved artifacts."""
    report = json.loads((run_dir / "manual_grading_report.json").read_text())
    nodes = json.loads((run_dir / "logs" / "journal.json").read_text())
    nodes = nodes["nodes"] if isinstance(nodes, dict) else nodes
    good = [n for n in nodes if not n.get("is_buggy")]
    scored = [m for n in good if (m := _metric(node=n)) is not None]

    return {
        "run_key": run_key,
        "status": "graded",
        **{
            k: report[k]
            for k in (
                "score", "any_medal", "gold_medal", "silver_medal", "bronze_medal",
                "above_median", "submission_exists", "valid_submission",
                "is_lower_better", "gold_threshold", "silver_threshold",
                "bronze_threshold", "median_threshold",
            )
        },
        "steps": len(nodes),
        "good_nodes": len(good),
        "first_good_step": next((n["step"] for n in nodes if not n.get("is_buggy")), None),
        "best_step": max(good, key=lambda n: _metric(node=n) or -1)["step"] if good else None,
        "best_validation_metric": max(scored) if scored else None,
        "salvaged": True,
        "salvage_note": note,
        "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--run-key", required=True)
    parser.add_argument("--note", required=True, help="why this run needed salvaging")
    args = parser.parse_args()

    row = salvage_row(run_dir=args.run_dir, run_key=args.run_key, note=args.note)
    with args.registry.open("a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(json.dumps(row, indent=1))


if __name__ == "__main__":
    main()
