"""Apply the frozen rubric to a run's flags; write judgments for aggregation and validation.

The two ends of the judging chain existed without a middle: `analysis/judge.py`
applies the rubric to one flag, `analysis/judge_agreement.py` samples judged
flags for the human anchor, and nothing read a run directory to produce the
judgments file connecting them. This is that connector.

**Evidence parity.** The judge and the human reviewer must see the same
artifacts, or their disagreement measures the difference in what they were shown
rather than the judge's fidelity to the rubric (docs/JUDGE_VALIDATION.md). The
bundle is therefore built once here and used for both.

That rules out selecting logs by what the judge quoted. A judge-derived excerpt
is empty exactly when the judge saw nothing, so a human could verify evidence
the judge found but never discover evidence it missed — the false-negative case
would be invisible, and agreement inflated rather than measured.

**Why the bundle is one node, not a lineage.** The bundle is the submitted
solution plus the best node's own analysis and output — a fixed shape, identical
for every flag in every run. It was previously the best node's *ancestor chain*,
which is gone for two reasons.

First, chain depth is a property of where the best node happens to land: 1 node
in one observed run, potentially dozens at a 500-step budget. A flag judged
against a deep chain has more evidence than an identical flag judged against a
shallow one, so a rate computed that way partly measures search-tree shape
rather than agent behavior — and the distortion grows with the step budget.

Second, lineage answers the wrong question. It records how the agent responded
to its own prior results, which is credit assignment over a reward signal, not
evidence about the injected specification. H2's causal purchase comes from
between-condition comparison under the design (see the base-rate counterfactual
in RESEARCH_DESIGN.md), never from reconstructing why one instance occurred.

The rubric's action criterion is judged against the flag and the **solution**,
which is always supplied whole, so nothing H2 rests on depended on the chain.

Usage:
    python -m analysis.judge_run --run-key <key> [--run-key ...]
    python -m analysis.judge_run --all-c2                       # every C2 run in the stage
    python -m analysis.judge_run --all-c2 --combined results/judgments.json
    python -m analysis.judge_run --all-c2 --base-rate B2        # + the counterfactual
"""

import argparse
import json
import logging
from pathlib import Path

from analysis.artifacts import run_root
from analysis.judge import FlagJudgment, aggregate_by_category, judge_flags, judge_provenance

logger = logging.getLogger(__name__)

# Caps one node's captured output, since a single failing step can emit a
# traceback larger than everything else combined. Unlike the retired chain
# depth, this does not vary with the step budget: the bundle is always one node.
EVIDENCE_NODE_CHARS = 4000


def _best_node(*, nodes: list[dict], solution: str) -> dict | None:
    """The node that produced the submitted solution.

    Matched by code first — `best_solution.py` is a copy of some node's code, so
    the match is exact when it holds. Falls back to the last non-buggy scored
    node, then the last node, so a journal that omits `code` still yields a chain.
    """
    if not nodes:
        return None
    stripped = solution.strip()
    if stripped:
        for node in nodes:
            if str(node.get("code", "")).strip() == stripped:
                return node
    scored = [
        (value, maximize, node)
        for node in nodes
        if not node.get("is_buggy")
        for value, maximize in [_metric(node=node)]
        if value is not None
    ]
    if not scored:
        return nodes[-1]
    # AIDE states its own direction per node, so the best node is the extremum
    # rather than the latest — a later node routinely scores worse than an
    # earlier one (observed: 0.6625 at step 1 against 0.6352 at step 6).
    maximize = scored[0][1]
    return (max if maximize else min)(scored, key=lambda item: item[0])[2]


def _metric(*, node: dict) -> tuple[float | None, bool]:
    """(value, maximize) from a journal node.

    AIDE serializes `metric` as `{"value": ..., "maximize": ...}`, and a buggy
    node carries `{"value": None}` — which is not None as a dict, so testing the
    field itself for None treats failed nodes as scored.
    """
    metric = node.get("metric")
    if isinstance(metric, dict):
        return metric.get("value"), bool(metric.get("maximize", True))
    return metric, True


def chain_nodes(*, run_dir: Path, solution: str) -> list[dict]:
    """The evidence node for one run, normalized for rendering.

    A list of at most one element: the node that produced the submitted
    solution. Kept as a list because the review page renders a sequence, and
    because the shape is what guarantees every item is judged on equal evidence.
    """
    journal_path = run_dir / "journal.json"
    if not journal_path.is_file():
        return []
    try:
        nodes = json.loads(journal_path.read_text()).get("nodes", [])
    except Exception:
        logger.exception("journal parse failed: %s", journal_path)
        return []
    best = _best_node(nodes=nodes, solution=solution)
    if best is None:
        return []
    return [
        {
            "step": best.get("step"),
            "is_buggy": bool(best.get("is_buggy")),
            "metric": _metric(node=best)[0],
            "analysis": str(best.get("analysis", "")),
            "term_out": _term_out(node=best)[:EVIDENCE_NODE_CHARS],
        }
    ]


def _term_out(*, node: dict) -> str:
    """Execution output, under whichever key the journal used.

    aideml exposes `term_out` as a property but serializes the underlying
    `_term_out` attribute, so a journal on disk carries only the underscored
    name. Reading the property name silently yields "" for every node — and the
    rubric's evidence requirement rests on observed output, so that would look
    like a judge that never finds positive evidence rather than a missing field.
    """
    return str(node.get("_term_out") or node.get("term_out") or "")


def _render_chain(*, chain: list[dict]) -> str:
    return "\n\n".join(
        f"--- step {node['step']} (buggy={node['is_buggy']}, metric={node['metric']}) ---\n"
        f"{node['analysis']}\n\n[output]\n{node['term_out']}"
        for node in chain
    )


def evidence_bundle(*, run_dir: Path) -> tuple[str, str]:
    """(solution, logs) for one run — the identical bundle both raters see."""
    solution_path = run_dir / "best_solution.py"
    solution = solution_path.read_text() if solution_path.is_file() else ""
    return solution, _render_chain(chain=chain_nodes(run_dir=run_dir, solution=solution))


def _records(
    *,
    flag_run_key: str,
    solution_run_key: str,
    flags: list[dict],
    judgments: list[FlagJudgment],
    solution: str,
    logs: str,
) -> list[dict]:
    """One record per judged flag, shaped for analysis.judge_agreement.

    `item_id` is qualified by BOTH run keys: the same flag is judged against
    several conditions' solutions for the base-rate comparison, so qualifying by
    the flag's own run alone would collide.

    `condition` is recorded here, after judging, purely for analysis. It is never
    in the judge's prompt — see judge_flags.
    """
    condition = solution_run_key.split("_")[-2] if "_" in solution_run_key else ""
    return [
        {
            "item_id": f"{flag_run_key}#{flag.get('flag_id', index)}@{solution_run_key}",
            "run_key": flag_run_key,
            "solution_run_key": solution_run_key,
            "condition": condition,
            "is_base_rate": flag_run_key != solution_run_key,
            "category": flag["category"],
            "explanation": flag["explanation"],
            "classification": judgment.classification,
            "evidence_quote": judgment.evidence_quote,
            "reasoning": judgment.reasoning,
            "evidence_doc_ids": flag.get("evidence_doc_ids") or [],
            "solution_excerpt": solution,
            "logs_excerpt": logs,
        }
        for index, (flag, judgment) in enumerate(zip(flags, judgments))
    ]


def _flags_for(*, run_key: str) -> list[dict]:
    path = run_root() / run_key / "pipeline_output.json"
    if not path.is_file():
        return []
    return json.loads(path.read_text()).get("assumption_flags", [])


def judge_run(*, run_key: str, solution_run_key: str | None = None) -> dict:
    """Judge one run's flags against preserved artifacts; write judgments.

    `solution_run_key` defaults to the flags' own run. Naming a different run
    judges the same flags against another condition's solution, which is the
    base-rate counterfactual: the rate at which the flagged mechanisms are
    addressed by an agent that never received the specification
    (RESEARCH_DESIGN.md, H2). Those judgments are written into the *solution's*
    run directory, so a condition's directory holds what was judged against it.
    """
    solution_run_key = solution_run_key or run_key
    is_base_rate = solution_run_key != run_key
    flags = _flags_for(run_key=run_key)
    if not flags:
        logger.info("no flags to judge: %s", run_key)
        return {"run_key": run_key, "judged": 0}
    solution_dir = run_root() / solution_run_key
    solution, logs = evidence_bundle(run_dir=solution_dir)
    if not solution:
        # Without the solution the rubric's classes cannot be distinguished, so a
        # run missing its artifacts is skipped rather than judged as not_acted_on.
        logger.warning("no preserved solution, skipping: %s", solution_run_key)
        return {"run_key": solution_run_key, "judged": 0, "skipped": "no solution artifact"}
    judgments = judge_flags(flags=flags, solution=solution, logs=logs)
    payload = {
        "run_key": run_key,
        "solution_run_key": solution_run_key,
        "is_base_rate": is_base_rate,
        "provenance": judge_provenance(),
        "by_category": aggregate_by_category(flags=flags, judgments=judgments),
        "judgments": _records(
            flag_run_key=run_key,
            solution_run_key=solution_run_key,
            flags=flags,
            judgments=judgments,
            solution=solution,
            logs=logs,
        ),
    }
    name = "judgments.json" if not is_base_rate else f"judgments_baserate_{run_key}.json"
    (solution_dir / name).write_text(json.dumps(payload, indent=2))
    return {
        "run_key": run_key,
        "solution_run_key": solution_run_key,
        "judged": len(payload["judgments"]),
    }


def c2_run_keys() -> list[str]:
    """Every run in the active stage that produced structured flags."""
    return sorted(
        path.parent.name
        for path in run_root().glob("*/pipeline_output.json")
        if "_C2_" in path.parent.name
    )


def paired_run_key(*, run_key: str, condition: str) -> str:
    """The same competition and seed under another condition."""
    competition, _, seed = run_key.rsplit("_", 2)
    return f"{competition}_{condition}_{seed}"


def combine(*, run_keys: list[str], out_path: Path) -> int:
    """Concatenate every judgments file into the flat list judge_agreement reads.

    Picks up base-rate files alongside each run's own, so the agreement sample
    and the paired analysis both see conditioned and unconditioned items.
    """
    records: list[dict] = []
    for run_key in run_keys:
        for path in sorted((run_root() / run_key).glob("judgments*.json")):
            records.extend(json.loads(path.read_text())["judgments"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, indent=2))
    return len(records)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-key", action="append", default=[])
    parser.add_argument("--all-c2", action="store_true", help="judge every C2 run in the stage")
    parser.add_argument(
        "--base-rate",
        action="append",
        default=[],
        metavar="CONDITION",
        help="also judge each C2 flag set against this condition's solution "
        "for the same competition and seed (B2 is the pre-registered control)",
    )
    parser.add_argument("--combined", help="also write a flat judgments file at this path")
    args = parser.parse_args()

    run_keys = args.run_key or (c2_run_keys() if args.all_c2 else [])
    if not run_keys:
        parser.error("pass --run-key or --all-c2")
    for run_key in run_keys:
        print(judge_run(run_key=run_key))
        for condition in args.base_rate:
            print(
                judge_run(
                    run_key=run_key,
                    solution_run_key=paired_run_key(run_key=run_key, condition=condition),
                )
            )
    if args.combined:
        # Combine over every run directory touched, not just the flag-side ones:
        # base-rate judgments live in the control condition's directory.
        touched = sorted(
            {key for key in run_keys}
            | {
                paired_run_key(run_key=key, condition=condition)
                for key in run_keys
                for condition in args.base_rate
            }
        )
        print(f"combined: {combine(run_keys=touched, out_path=Path(args.combined))} judgments")
