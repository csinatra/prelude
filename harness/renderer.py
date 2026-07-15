"""Render a condition's output into the spec.md artifact injected into AIDE.

Composition policy (design decision, 2026-07-15): **additive** — every
condition injects its retrieved context block, plus its synthesis artifact
when one exists. Each grid step then changes exactly one thing:

    B1: context block
    B2: context block + freeform advice
    C1: staged context block + freeform advice
    C2: staged context block + structured specification

The rendered document is the exact text injected into the agent's initial
context; it is preserved verbatim as spec.md in the run artifact.
"""

from pipeline.nodes import _format_docs
from pipeline.retriever import RetrievedDoc

CONTEXT_HEADER = "## Reference material from similar competitions"
ADVICE_HEADER = "## Advisor notes"
SPEC_HEADER = "## Problem specification"

# C2's state stores per-stage retrievals under these keys, in stage order.
C2_RETRIEVAL_KEYS = ["retrieved_parse", "retrieved_surface", "retrieved_flag", "retrieved_advise"]


def merged_context_block(*, doc_dicts_by_stage: list[list[dict]]) -> str:
    """Dedupe docs across stages by doc_id (stage order preserved) into one block."""
    seen: set[str] = set()
    merged: list[RetrievedDoc] = []
    for stage_docs in doc_dicts_by_stage:
        for doc in stage_docs:
            if doc["doc_id"] not in seen:
                seen.add(doc["doc_id"])
                merged.append(RetrievedDoc(**doc))
    return _format_docs(merged)


def _render_c2_spec(output: dict) -> str:
    lines = [SPEC_HEADER, ""]
    lines += [
        "### Problem framing",
        f"- Goal: {output.get('parsed_goal', '')}",
        f"- Task type: {output.get('task_type', '')}",
        f"- Evaluation metric: {output.get('evaluation_metric', '')}",
        f"- Target variable: {output.get('target_variable', '')}",
        f"- Framing type: {output.get('framing_type', '')}",
        "- Constraints: " + "; ".join(output.get("constraints", [])),
        "",
        "### Signals",
        "- Available: " + "; ".join(output.get("available_signals", [])),
        "- Missing but valuable: " + "; ".join(output.get("desired_signals", [])),
        "- Relevant prior work: " + "; ".join(output.get("prior_work", [])),
        "",
        "### Assumption flags",
    ]
    for flag in output.get("assumption_flags", []):
        lines.append(
            f"- [{flag['flag_id']}] {flag['category']} ({flag['confidence']} confidence): "
            f"{flag['explanation']}"
        )
    lines += ["", "### Recommendations"]
    for rec in output.get("recommendations", []):
        addresses = ", ".join(rec.get("addresses_flags", [])) or "none"
        lines += [
            f"- {rec['approach']}",
            f"  - Tradeoff: {rec['tradeoff']}",
            f"  - Likely failure mode: {rec['failure_mode']}",
            f"  - Addresses flags: {addresses}",
        ]
    return "\n".join(lines)


def render_spec(*, condition: str, output: dict) -> str:
    """Render the injectable spec document for one condition run."""
    if condition in ("B1", "B2"):
        context_block = output["context_block"]
    elif condition == "C1":
        context_block = output["context_block"]
    elif condition == "C2":
        context_block = merged_context_block(
            doc_dicts_by_stage=[output.get(key, []) for key in C2_RETRIEVAL_KEYS]
        )
    else:
        raise ValueError(f"unknown condition: {condition}")

    sections = [f"{CONTEXT_HEADER}\n\n{context_block}"]
    if condition in ("B2", "C1"):
        sections.append(f"{ADVICE_HEADER}\n\n{output['advice']}")
    elif condition == "C2":
        sections.append(_render_c2_spec(output))
    return "\n\n".join(sections) + "\n"
