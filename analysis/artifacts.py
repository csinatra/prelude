"""Run-artifact preservation — required input for post-run mechanistic analysis.

Layout (results/ is gitignored during dev; flipped to committed for final
eval outputs):

    results/{competition_id}_{condition}_{seed}/
    ├── spec.md            # injected specification/context artifact (all conditions)
    ├── retrievals.json    # every retrieval call: doc IDs + similarities
    ├── pipeline_output.json  # full condition output (state / advice / flags)
    ├── submission.csv     # copied final submission (when available)
    └── trajectory.log     # copied agent trajectory/logs (when available)
"""

import json
import shutil
from pathlib import Path

RESULTS_DIR = Path("results")


def run_key(*, competition_id: str, condition: str, seed: int) -> str:
    return f"{competition_id}_{condition}_{seed}"


def save_artifacts(
    *,
    competition_id: str,
    condition: str,
    seed: int,
    spec_document: str,
    retrievals: dict | list,
    pipeline_output: dict,
    submission_path: Path | None = None,
    trajectory_path: Path | None = None,
) -> Path:
    """Persist one run's artifacts; returns the run directory."""
    run_dir = RESULTS_DIR / run_key(competition_id=competition_id, condition=condition, seed=seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "spec.md").write_text(spec_document)
    (run_dir / "retrievals.json").write_text(json.dumps(retrievals, indent=2))
    (run_dir / "pipeline_output.json").write_text(json.dumps(pipeline_output, indent=2, default=str))
    if submission_path is not None:
        shutil.copy(src=submission_path, dst=run_dir / "submission.csv")
    if trajectory_path is not None:
        shutil.copy(src=trajectory_path, dst=run_dir / "trajectory.log")
    return run_dir
