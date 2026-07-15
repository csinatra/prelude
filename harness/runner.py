"""Spec-side harness runner: execute one condition for one competition/seed.

Produces everything AIDE needs (the injectable spec.md), preserves the run
artifact, and registers the run as status=spec_built in runs.jsonl. Agent
execution inside the MLE-bench container and grading happen on the cloud box
and advance the same run_key through later statuses.

Runs entirely OUTSIDE the MLE-bench execution environment (CLAUDE.md core
constraint 1): all LLM calls happen here, at spec-build time.

Usage: python -m harness.runner --competition spooky-author-identification \
           --condition C2 --seed 0
"""

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

from analysis.artifacts import _git_provenance, run_key, save_artifacts
from harness.registry import append_run
from harness.renderer import render_spec
from pipeline.condition_b import run_b1, run_b2
from pipeline.condition_c1 import run_c1
from pipeline.condition_c2 import run_c2

DESCRIPTIONS_DIR = Path("data/raw/mlebench_descriptions")


def load_description(*, competition_id: str) -> str:
    return (DESCRIPTIONS_DIR / f"{competition_id}.md").read_text()


def _run_c2_output(*, raw_problem: str, competition_id: str) -> dict:
    return dict(run_c2(raw_problem=raw_problem, competition_id=competition_id))


CONDITION_RUNNERS = {
    "B1": run_b1,
    "B2": run_b2,
    "C1": run_c1,
    "C2": _run_c2_output,
}


def _extract_retrievals(*, condition: str, output: dict) -> dict | list:
    if condition == "C2":
        return {
            key.removeprefix("retrieved_"): output.get(key, [])
            for key in ["retrieved_parse", "retrieved_surface", "retrieved_flag", "retrieved_advise"]
        }
    return output["retrieved"]


def _count_tokens(*, text: str) -> int | None:
    """Injected-artifact token count (a reported design metric). None if unavailable."""
    try:
        import anthropic

        response = anthropic.Anthropic().messages.count_tokens(
            model=os.environ["MODEL"],
            messages=[{"role": "user", "content": text}],
        )
        return response.input_tokens
    except Exception:
        return None


def run_condition(*, competition_id: str, condition: str, seed: int) -> Path:
    """Build one spec, save artifacts, register the run. Returns the run directory."""
    raw_problem = load_description(competition_id=competition_id)
    output = CONDITION_RUNNERS[condition](
        raw_problem=raw_problem, competition_id=competition_id
    )
    spec_document = render_spec(condition=condition, output=output)
    run_dir = save_artifacts(
        competition_id=competition_id,
        condition=condition,
        seed=seed,
        spec_document=spec_document,
        retrievals=_extract_retrievals(condition=condition, output=output),
        pipeline_output=output,
    )
    append_run(
        entry={
            "run_key": run_key(competition_id=competition_id, condition=condition, seed=seed),
            "competition_id": competition_id,
            "condition": condition,
            "seed": seed,
            "status": "spec_built",
            "spec_path": str(run_dir / "spec.md"),
            "spec_chars": len(spec_document),
            "spec_tokens": _count_tokens(text=spec_document),
            **_git_provenance(),
            "llm_provider": os.environ.get("LLM_PROVIDER", "anthropic"),
            "model": os.environ.get("MODEL"),
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
        }
    )
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition", required=True)
    parser.add_argument("--condition", required=True, choices=sorted(CONDITION_RUNNERS))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_dir = run_condition(
        competition_id=args.competition, condition=args.condition, seed=args.seed
    )
    print(f"spec built: {run_dir}")
