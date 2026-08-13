"""Pre-run retrieval quality audit.

The similarity-threshold decision in docs/RESEARCH_DESIGN.md carries a revisit
trigger ("evidence of junk retrievals in run inspection") that had no mechanism
to fire it. This is that mechanism. For each competition it dumps what retrieval
actually returns, flat (B) and staged (C), so the corpus can be skimmed before
eval runs commit GPU budget to a corpus that retrieves badly.

Read the output asking two questions:
  1. Are the top hits plausibly about this problem class?
  2. How do similarities compare between the flat query and the directed ones?
     A large gap is the query-impoverishment asymmetry that makes a global
     threshold filter the staged conditions harder than flat retrieval.

Cost: one parse call per competition (pinned to the dev model), because C's
directed queries are built from parse's extracted fields and approximating them
would audit queries the pipeline never issues. Everything else is query
embeddings. Writes nothing to the corpus.

Run: PYTHONPATH=. .venv/bin/python -m analysis.retrieval_audit \
        --competitions spooky-author-identification,leaf-classification
(needs ANTHROPIC_API_KEY + VOYAGE_API_KEY; source .env first)
"""

import argparse
import statistics
from pathlib import Path

from pipeline.config import (
    BASELINE_N_NOTEBOOKS,
    COMPETITION_METADATA,
    METADATA_K,
    NOTEBOOK_SUMMARIES,
    RETRIEVAL_K,
    STAGE_N_NOTEBOOKS,
)
from pipeline.nodes import advise_query, flag_query, parse_problem, surface_query
from pipeline.retriever import RetrievedDoc, retrieve, retrieve_with_topup

DESCRIPTIONS_DIR = Path("data/raw/mlebench_descriptions")
EXCERPT_CHARS = 180


def _render(*, label: str, query: str, docs: list[RetrievedDoc]) -> list[str]:
    lines = [f"### {label}", "", f"*query:* `{' '.join(query.split())[:160]}`", ""]
    if not docs:
        return lines + ["(no documents retrieved)", ""]
    sims = [doc.similarity for doc in docs]
    lines += [
        f"*{len(docs)} docs | similarity max={max(sims):.3f} "
        f"median={statistics.median(sims):.3f} min={min(sims):.3f}*",
        "",
    ]
    for doc in docs:
        excerpt = " ".join(doc.text.split())[:EXCERPT_CHARS]
        lines.append(f"- `{doc.similarity:.3f}` **{doc.doc_id}** ({doc.competition_id}) {excerpt}")
    return lines + [""]


def audit_competition(*, competition_id: str) -> list[str]:
    """Flat (B) and staged (C) retrieval for one competition, side by side."""
    raw_problem = (DESCRIPTIONS_DIR / f"{competition_id}.md").read_text()
    out = [f"## {competition_id}", ""]

    out += ["### Condition B (flat, single query)", ""]
    out += _render(
        label="B: metadata",
        query=raw_problem,
        docs=retrieve(
            query=raw_problem,
            collection=COMPETITION_METADATA,
            exclude_competition=competition_id,
            k=METADATA_K,
        ),
    )
    out += _render(
        label="B: notebook summaries",
        query=raw_problem,
        docs=retrieve(
            query=raw_problem,
            collection=NOTEBOOK_SUMMARIES,
            exclude_competition=competition_id,
            k=BASELINE_N_NOTEBOOKS,
        ),
    )

    out += ["### Condition C (staged, directed queries)", ""]
    parsed = parse_problem(
        {"raw_problem": raw_problem, "competition_id": competition_id, "stage_trace": []}
    )
    out += _render(
        label="C: parse (metadata)",
        query=raw_problem,
        docs=[RetrievedDoc(**doc) for doc in parsed["retrieved_parse"]],
    )

    fields = {
        "task_type": parsed["task_type"],
        "evaluation_metric": parsed["evaluation_metric"],
        "goal": parsed["parsed_goal"],
    }
    seen: set[str] = set()
    for label, builder in [
        ("surface", surface_query),
        ("flag", flag_query),
        ("advise", advise_query),
    ]:
        query = builder(**fields)
        out += _render(
            label=f"C: {label}",
            query=query,
            docs=retrieve_with_topup(
                query=query,
                collection=NOTEBOOK_SUMMARIES,
                exclude_competition=competition_id,
                k=STAGE_N_NOTEBOOKS,
                seen=seen,
            ),
        )
    out += [
        f"*distinct summaries across C's stages: {len(seen)} "
        f"(parity target {BASELINE_N_NOTEBOOKS})*",
        "",
    ]
    return out


def main(*, competitions: list[str], out_path: Path | None) -> None:
    report = [
        "# Retrieval quality audit",
        "",
        "Skim for junk retrievals before eval runs. Firing mechanism for the "
        "similarity-threshold revisit trigger in docs/RESEARCH_DESIGN.md.",
        "",
    ]
    for competition_id in competitions:
        report += audit_competition(competition_id=competition_id)
    text = "\n".join(report)
    if out_path is None:
        print(text)
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    print(f"wrote audit for {len(competitions)} competition(s) to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competitions", required=True, help="comma-separated competition slugs")
    parser.add_argument("--out", default="results/retrieval_audit.md", help="'-' for stdout")
    args = parser.parse_args()
    main(
        competitions=[slug.strip() for slug in args.competitions.split(",") if slug.strip()],
        out_path=None if args.out == "-" else Path(args.out),
    )
