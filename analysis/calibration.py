"""Retrieval-threshold calibration sweep — pre-run tooling, re-runnable.

Collects raw similarity distributions (no threshold applied) for every query
kind the pipeline issues, across a spread of dev competitions, then prints
retained-fraction tables over the sweep grid. Thresholds are corpus-relative:
re-run this after any corpus expansion (full Code4ML, MLEModernizer) before
trusting a previously calibrated SIMILARITY_THRESHOLD.

Cost per run: one Haiku parse call per competition (to build the directed
stage queries); everything else is Voyage query embeddings + local Chroma.

Usage: python -m analysis.calibration
Output: results/calibration/similarities.json + printed summary tables.
"""

import json
import os
from pathlib import Path

OUTPUT_PATH = Path("results/calibration/similarities.json")

# Task-type spread across Lite-22: vision, text, tabular.
CALIBRATION_COMPETITIONS = [
    "aerial-cactus-identification",
    "detecting-insults-in-social-commentary",
    "dog-breed-identification",
    "jigsaw-toxic-comment-classification-challenge",
    "leaf-classification",
    "new-york-city-taxi-fare-prediction",
    "nomad2018-predict-transparent-conductors",
    "random-acts-of-pizza",
    "spooky-author-identification",
    "tabular-playground-series-dec-2021",
]

SWEEP_GRID = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def collect(*, competitions: list[str]) -> dict:
    """Similarities per competition per query kind, thresholds disabled."""
    from harness.runner import load_description
    from pipeline.config import (
        BASELINE_CHUNKS_PER_NOTEBOOK,
        BASELINE_N_NOTEBOOKS,
        COMPETITION_METADATA,
        METADATA_K,
        NOTEBOOK_SUMMARIES,
        STAGE_CHUNKS_PER_NOTEBOOK,
        STAGE_N_NOTEBOOKS,
    )
    from pipeline.nodes import advise_query, flag_query, parse_problem, surface_query
    from pipeline.retriever import retrieve, retrieve_two_level

    results: dict = {}
    for competition_id in competitions:
        raw_problem = load_description(competition_id=competition_id)
        parse_update = parse_problem(
            {"raw_problem": raw_problem, "competition_id": competition_id, "stage_trace": []}
        )
        query_fields = {
            "task_type": parse_update["task_type"],
            "evaluation_metric": parse_update["evaluation_metric"],
            "goal": parse_update["parsed_goal"],
        }
        sims: dict[str, list[float]] = {}
        sims["metadata_flat"] = [
            doc.similarity
            for doc in retrieve(
                query=raw_problem,
                collection=COMPETITION_METADATA,
                exclude_competition=competition_id,
                k=METADATA_K,
                score_threshold=None,
            )
        ]
        sims["summaries_flat"] = [
            doc.similarity
            for doc in retrieve(
                query=raw_problem,
                collection=NOTEBOOK_SUMMARIES,
                exclude_competition=competition_id,
                k=BASELINE_N_NOTEBOOKS,
                score_threshold=None,
            )
        ]
        sims["chunks_flat"] = [
            doc.similarity
            for doc in retrieve_two_level(
                query=raw_problem,
                exclude_competition=competition_id,
                n_notebooks=BASELINE_N_NOTEBOOKS,
                chunks_per_notebook=BASELINE_CHUNKS_PER_NOTEBOOK,
                score_threshold=None,
            )
        ]
        for stage, query_builder in [
            ("surface", surface_query),
            ("flag", flag_query),
            ("advise", advise_query),
        ]:
            sims[f"chunks_{stage}"] = [
                doc.similarity
                for doc in retrieve_two_level(
                    query=query_builder(**query_fields),
                    exclude_competition=competition_id,
                    n_notebooks=STAGE_N_NOTEBOOKS,
                    chunks_per_notebook=STAGE_CHUNKS_PER_NOTEBOOK,
                    score_threshold=None,
                )
            ]
        results[competition_id] = sims
        print(f"collected: {competition_id}")
    return results


def summarize(*, results: dict) -> None:
    """Retained fraction per query kind at each sweep threshold, plus quantiles."""
    by_kind: dict[str, list[float]] = {}
    for sims in results.values():
        for kind, values in sims.items():
            by_kind.setdefault(kind, []).extend(values)

    header = ["kind", "n", "p10", "p50", "p90"] + [f">={t:.2f}" for t in SWEEP_GRID]
    rows = []
    for kind, values in sorted(by_kind.items()):
        ordered = sorted(values)
        quantile = lambda q: ordered[min(int(q * len(ordered)), len(ordered) - 1)]
        retained = [sum(v >= t for v in values) / len(values) for t in SWEEP_GRID]
        rows.append(
            [kind, str(len(values))]
            + [f"{quantile(q):.3f}" for q in (0.1, 0.5, 0.9)]
            + [f"{r:.0%}" for r in retained]
        )
    widths = [max(len(row[i]) for row in [header] + rows) for i in range(len(header))]
    for row in [header] + rows:
        print("  ".join(cell.ljust(width) for cell, width in zip(row, widths)))


def main() -> None:
    # Pre-run tooling, not experiment observability — don't burn trace quota.
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGSMITH_TRACING_V2"] = "false"
    results = collect(competitions=CALIBRATION_COMPETITIONS)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUTPUT_PATH}\n")
    summarize(results=results)


if __name__ == "__main__":
    main()
