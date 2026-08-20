"""Run-artifact preservation — required input for post-run mechanistic analysis.

Layout — one tree per experiment stage, matching the per-stage registry in
harness/registry.py. Testing stages stay local and unpublished; the eval stage
is the results dataset and is version-controlled:

    results/{stage}/{competition_id}_{condition}_{seed}/
    ├── spec.md            # injected specification/context artifact (all conditions)
    ├── retrievals.json    # every retrieval call: doc IDs + similarities
    ├── pipeline_output.json  # full condition output (state / advice / flags)
    ├── llm_usage.json     # per-call spec-build usage (written by harness.runner;
    │                      #   call order == stage order, sequential nodes)
    ├── manifest.json      # provenance: git commit (pins prompts/config), model, timestamp
    ├── submission.csv     # copied final submission (agent side)
    ├── journal.json       # copied AIDE journal — per-step trajectory (agent side)
    ├── best_solution.py   # copied final/best solution code — judge + evidence mining
    └── prelude_token_usage.jsonl  # per-call agent token usage (agent side, when logged)

Two write paths feed this dir. The spec side (save_artifacts, on the dev
machine) writes the top block. The agent side (preserve_agent_outputs, on the
cloud box after an AIDE run) copies the bottom block off the ephemeral mle-bench
run dir onto the persistent volume that results/ is symlinked to — without it,
--terminate-on-done destroys the journal + solution the mechanistic judge needs,
keeping only the outcome fields already in the registry.

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

from harness.registry import active_stage
from ingest.export_corpus import fingerprint as corpus_fingerprint

RESULTS_DIR = Path("results")


def run_root() -> Path:
    """Artifacts live under the active stage, beside that stage's registry.

    run_key is not unique across experiment stages — a run rebuilt against a new
    corpus reuses its predecessor's key — so an unstaged path would let an eval
    run silently overwrite the testing artifacts already there. Same reasoning as
    the per-stage registry in harness/registry.py, and it must resolve the stage
    the same way or a registry row's spec_path would not point at its spec.
    """
    return RESULTS_DIR / active_stage()


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
    run_dir = run_root() / run_key(competition_id=competition_id, condition=condition, seed=seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "competition_id": competition_id,
        "condition": condition,
        "seed": seed,
        **_git_provenance(),
        "llm_provider": os.environ.get("LLM_PROVIDER", "anthropic"),
        "model": os.environ.get("MODEL"),
        "stage": active_stage(),
        **corpus_fingerprint(),
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


def preserve_agent_outputs(
    *,
    run_key: str,
    submission_path: str | Path | None = None,
    journal_path: str | Path | None = None,
    solution_path: str | Path | None = None,
    extra_paths: tuple[str | Path | None, ...] = (),
) -> Path:
    """Copy an AIDE run's outputs into its artifact dir on durable storage.

    Runs on the cloud box after the agent, where mle-bench writes these under an
    ephemeral run dir; results/ is symlinked to the persistent volume, so this
    copy is what survives instance termination (the mechanistic judge reads the
    solution + journal here, offline). Creates the dir — Condition A has no
    spec-side artifacts to seed it. Best-effort per file: a missing optional
    output (e.g. a buggy run with no submission) still preserves the rest.
    """
    run_dir = run_root() / run_key
    run_dir.mkdir(parents=True, exist_ok=True)
    named = {
        "submission.csv": submission_path,
        "journal.json": journal_path,
        "best_solution.py": solution_path,
    }
    for dst_name, src in named.items():
        if src is not None and Path(src).is_file():
            shutil.copy(src=src, dst=run_dir / dst_name)
    for src in extra_paths:
        if src is not None and Path(src).is_file():
            shutil.copy(src=src, dst=run_dir / Path(src).name)
    return run_dir
