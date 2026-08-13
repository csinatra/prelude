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
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from analysis.artifacts import _git_provenance, run_key, save_artifacts
from harness.registry import append_run
from harness.renderer import render_spec, spec_sections
from pipeline.condition_b import run_b1, run_b2
from pipeline.condition_c1 import run_c1
from pipeline.condition_c2 import run_c2
from pipeline.llm_client import reset_usage, usage_log, usage_snapshot
from pipeline.retriever import reset_shortfalls, shortfall_log

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


def _spec_metrics(*, condition: str, output: dict) -> dict:
    """Spec-side sanity metrics, consolidated so condition-level tables fall out
    of the registry without re-parsing artifacts.

    Flag fields are C2-only (B/C1 produce no structured flags), and are reported
    as None rather than 0 there so "no flags produced" stays distinguishable
    from "condition has no flag stage."
    """
    if condition != "C2":
        return {
            "flag_count": None,
            "flags_by_category": None,
            "flag_grounded_fraction": None,
            "recommendation_count": None,
        }
    flags = output.get("assumption_flags", [])
    by_category: dict[str, int] = {}
    for flag in flags:
        by_category[flag["category"]] = by_category.get(flag["category"], 0) + 1
    grounded = sum(1 for flag in flags if flag.get("evidence_doc_ids"))
    return {
        "flag_count": len(flags),
        "flags_by_category": by_category,
        "flag_grounded_fraction": round(grounded / len(flags), 3) if flags else None,
        "recommendation_count": len(output.get("recommendations", [])),
    }


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
    reset_usage()
    reset_shortfalls()
    build_started = time.monotonic()
    output = CONDITION_RUNNERS[condition](
        raw_problem=raw_problem, competition_id=competition_id
    )
    spec_build_secs = round(time.monotonic() - build_started, 3)
    usage = usage_snapshot()
    sections = spec_sections(condition=condition, output=output)
    spec_document = render_spec(condition=condition, output=output)
    run_dir = save_artifacts(
        competition_id=competition_id,
        condition=condition,
        seed=seed,
        spec_document=spec_document,
        retrievals=_extract_retrievals(condition=condition, output=output),
        pipeline_output=output,
    )
    # Per-call spec-build usage, call order == stage order (sequential nodes).
    (run_dir / "llm_usage.json").write_text(json.dumps(usage_log(), indent=2))
    # Distinct-document parity shortfalls, if any. Written only when non-empty so
    # the file's presence is itself the signal that a run needs a closer look.
    shortfalls = shortfall_log()
    if shortfalls:
        (run_dir / "retrieval_shortfalls.json").write_text(json.dumps(shortfalls, indent=2))
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
            "block_tokens": _count_tokens(text=sections["context"]),
            "synthesis_tokens": _count_tokens(text=sections["synthesis"])
            if sections["synthesis"]
            else 0,
            "spec_build_secs": spec_build_secs,
            "spec_llm_calls": usage["llm_calls"],
            "spec_llm_input_tokens": usage["input_tokens"],
            "spec_llm_output_tokens": usage["output_tokens"],
            "retrieval_shortfall_count": len(shortfalls),
            **_spec_metrics(condition=condition, output=output),
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
