"""Run-artifact preservation — required input for post-run mechanistic analysis.

Layout (results/ is gitignored during dev; flipped to committed for final
eval outputs):

    results/{competition_id}_{condition}_{seed}/
    ├── spec.md            # injected specification/context artifact (all conditions)
    ├── retrievals.json    # every retrieval call: doc IDs + similarities
    ├── pipeline_output.json  # full condition output (state / advice / flags)
    ├── manifest.json      # provenance: git commit (pins prompts/config), model, timestamp
    ├── submission.csv     # copied final submission (when available)
    └── trajectory.log     # copied agent trajectory/logs (when available)

LangSmith traces are ephemeral (14-day retention on the base tier); this
directory is the durable research record. spec.md holds the exact injected
prompt; the manifest's commit hash pins the stage prompts and retrieval
config that produced it without duplicating them per run.
"""

import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

RESULTS_DIR = Path("results")


def _git_provenance() -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
            ).stdout.strip()
        )
        return {"git_commit": commit, "git_dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"git_commit": None, "git_dirty": None}


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
    manifest = {
        "competition_id": competition_id,
        "condition": condition,
        "seed": seed,
        **_git_provenance(),
        "llm_provider": os.environ.get("LLM_PROVIDER", "anthropic"),
        "model": os.environ.get("MODEL"),
        "saved_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (run_dir / "spec.md").write_text(spec_document)
    (run_dir / "retrievals.json").write_text(json.dumps(retrievals, indent=2))
    (run_dir / "pipeline_output.json").write_text(json.dumps(pipeline_output, indent=2, default=str))
    if submission_path is not None:
        shutil.copy(src=submission_path, dst=run_dir / "submission.csv")
    if trajectory_path is not None:
        shutil.copy(src=trajectory_path, dst=run_dir / "trajectory.log")
    return run_dir
