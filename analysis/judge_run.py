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

**Why the bundle is bounded at all.** AIDE runs at `step_count: 500`
(cloudbox/agents/aide-prelude/config.yaml), so a full journal fits neither a
judge's context nor a reviewer's afternoon. The bundle is the best solution plus
its ancestor chain in the search tree — the lineage that produced the submitted
code, selected structurally rather than by any rater. That is also the evidence
`acted_on_positive` is defined on: a failure at one node, resolved at a
descendant. Discarded branches are invisible to both raters, a real limit but
arguably correct scoping — a flag addressed only in an abandoned branch did not
shape the submitted solution.

The caps below are placeholders, pinned against a measured journal before
judging begins; nothing has run at full budget yet.

Usage:
    python -m analysis.judge_run --run-key <key> [--run-key ...]
    python -m analysis.judge_run --all-c2                       # every C2 run in the stage
    python -m analysis.judge_run --all-c2 --combined results/judgments.json
"""

import argparse
import json
import logging
from pathlib import Path

from analysis.artifacts import run_root
from analysis.judge import FlagJudgment, aggregate_by_category, judge_flags, judge_provenance

logger = logging.getLogger(__name__)

# Pin against a real journal before judging (see module docstring). DEPTH is how
# many nodes of ancestry are retained; NODE_CHARS caps one node's captured
# output, since a single failing step can emit a traceback larger than the rest
# of the chain combined.
EVIDENCE_CHAIN_DEPTH = 12
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
    scored = [node for node in nodes if not node.get("is_buggy") and node.get("metric") is not None]
    # AIDE metrics carry their own direction, which the journal does not state
    # here; ordering by position is the honest fallback — later non-buggy nodes
    # supersede earlier ones in AIDE's own search.
    return scored[-1] if scored else nodes[-1]


def _ancestor_chain(*, nodes: list[dict], solution: str) -> list[dict]:
    """The submitted node and its ancestors, oldest first, capped at CHAIN_DEPTH.

    Falls back to the last CHAIN_DEPTH nodes when the journal carries no usable
    parent links. The fallback stays rater-independent — positional, not
    content-selected — so evidence parity holds either way.
    """
    best = _best_node(nodes=nodes, solution=solution)
    if best is None:
        return []
    by_id = {node["id"]: node for node in nodes if "id" in node}
    if not by_id or "id" not in best:
        return nodes[-EVIDENCE_CHAIN_DEPTH:]
    chain = [best]
    seen = {best["id"]}
    current = best
    while len(chain) < EVIDENCE_CHAIN_DEPTH:
        parent_id = current.get("parent")
        if parent_id is None or parent_id not in by_id or parent_id in seen:
            break
        current = by_id[parent_id]
        seen.add(parent_id)
        chain.append(current)
    return list(reversed(chain))


def chain_nodes(*, run_dir: Path, solution: str) -> list[dict]:
    """Ancestor-chain nodes for one run, normalized for rendering."""
    journal_path = run_dir / "journal.json"
    if not journal_path.is_file():
        return []
    try:
        nodes = json.loads(journal_path.read_text()).get("nodes", [])
    except Exception:
        logger.exception("journal parse failed: %s", journal_path)
        return []
    return [
        {
            "step": node.get("step"),
            "is_buggy": bool(node.get("is_buggy")),
            "metric": node.get("metric"),
            "analysis": str(node.get("analysis", "")),
            "term_out": str(node.get("term_out", ""))[:EVIDENCE_NODE_CHARS],
        }
        for node in _ancestor_chain(nodes=nodes, solution=solution)
    ]


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
    *, run_key: str, flags: list[dict], judgments: list[FlagJudgment], solution: str, logs: str
) -> list[dict]:
    """One record per judged flag, shaped for analysis.judge_agreement.

    `item_id` is run-qualified because the agreement sample is drawn across runs
    and flag ids repeat between them.
    """
    return [
        {
            "item_id": f"{run_key}#{flag.get('flag_id', index)}",
            "run_key": run_key,
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


def judge_run(*, run_key: str) -> dict:
    """Judge one run's flags against its preserved artifacts; write judgments.json."""
    run_dir = run_root() / run_key
    flags = json.loads((run_dir / "pipeline_output.json").read_text()).get("assumption_flags", [])
    if not flags:
        logger.info("no flags to judge: %s", run_key)
        return {"run_key": run_key, "judged": 0}
    solution, logs = evidence_bundle(run_dir=run_dir)
    if not solution:
        # Without the solution the rubric's classes cannot be distinguished, so a
        # run missing its artifacts is skipped rather than judged as not_acted_on.
        logger.warning("no preserved solution, skipping: %s", run_key)
        return {"run_key": run_key, "judged": 0, "skipped": "no solution artifact"}
    judgments = judge_flags(flags=flags, solution=solution, logs=logs)
    payload = {
        "run_key": run_key,
        "provenance": judge_provenance(),
        "by_category": aggregate_by_category(flags=flags, judgments=judgments),
        "judgments": _records(
            run_key=run_key, flags=flags, judgments=judgments, solution=solution, logs=logs
        ),
    }
    (run_dir / "judgments.json").write_text(json.dumps(payload, indent=2))
    return {"run_key": run_key, "judged": len(payload["judgments"])}


def c2_run_keys() -> list[str]:
    """Every run in the active stage that produced structured flags."""
    return sorted(
        path.parent.name
        for path in run_root().glob("*/pipeline_output.json")
        if "_C2_" in path.parent.name
    )


def combine(*, run_keys: list[str], out_path: Path) -> int:
    """Concatenate per-run judgments into the flat list judge_agreement reads."""
    records: list[dict] = []
    for run_key in run_keys:
        path = run_root() / run_key / "judgments.json"
        if path.is_file():
            records.extend(json.loads(path.read_text())["judgments"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, indent=2))
    return len(records)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-key", action="append", default=[])
    parser.add_argument("--all-c2", action="store_true", help="judge every C2 run in the stage")
    parser.add_argument("--combined", help="also write a flat judgments file at this path")
    args = parser.parse_args()

    run_keys = args.run_key or (c2_run_keys() if args.all_c2 else [])
    if not run_keys:
        parser.error("pass --run-key or --all-c2")
    for run_key in run_keys:
        print(judge_run(run_key=run_key))
    if args.combined:
        print(f"combined: {combine(run_keys=run_keys, out_path=Path(args.combined))} judgments")
