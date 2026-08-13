"""Flag → action → outcome judge (scaffold — real integration once runs exist).

Given a run's preserved solution artifacts and its SpecificationFlag list,
classify each flag per the frozen rubric in docs/JUDGE_RUBRIC.md, then
aggregate per-category. Judged via the same call_llm structured-output seam
as the pipeline; tests mock it.
"""

import hashlib
import os
from collections import defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from pipeline.llm_client import call_llm

RUBRIC_PATH = Path("docs/JUDGE_RUBRIC.md")

JUDGE_SYSTEM_PREAMBLE = (
    "You are a strict evaluation judge. Classify one specification flag against a "
    "run's solution artifacts, following this frozen rubric exactly:\n\n"
)


class FlagJudgment(BaseModel):
    classification: Literal["not_acted_on", "acted_on_unclear", "acted_on_positive"]
    evidence_quote: str  # required for acted_on_*; empty for not_acted_on
    reasoning: str


def judge_provenance() -> dict:
    """What produced a judgment, so it can be reproduced or invalidated later.

    The rubric is frozen but the judge model and prompt are not pinned by the
    rubric itself, so a re-judge under a different model or a reworded preamble
    would otherwise be indistinguishable from the original. Recorded per run
    alongside the judgments (see docs/JUDGE_VALIDATION.md).
    """
    rubric = RUBRIC_PATH.read_text()
    return {
        "judge_model": os.environ.get("MODEL"),
        "rubric_sha256": hashlib.sha256(rubric.encode()).hexdigest(),
        "judge_prompt_sha256": hashlib.sha256(
            (JUDGE_SYSTEM_PREAMBLE + rubric).encode()
        ).hexdigest(),
    }


def judge_flags(*, flags: list[dict], solution: str, logs: str = "") -> list[FlagJudgment]:
    """Judge each flag independently against the frozen rubric.

    Per the rubric, the judge never sees the run's score or medal outcome.
    Call judge_provenance() and store its output beside these judgments.
    """
    rubric = RUBRIC_PATH.read_text()
    judgments = []
    for flag in flags:
        judgment = call_llm(
            system=JUDGE_SYSTEM_PREAMBLE + rubric,
            user=(
                f"Flag category: {flag['category']}\n"
                f"Flag explanation: {flag['explanation']}\n\n"
                f"Solution code:\n{solution}\n\n"
                f"Trajectory logs:\n{logs or '(none provided)'}"
            ),
            response_model=FlagJudgment,
            max_tokens=1024,
        )
        # Rubric: acted_on_* without an evidence quote is invalid.
        if judgment.classification != "not_acted_on" and not judgment.evidence_quote.strip():
            judgment = FlagJudgment(
                classification="not_acted_on",
                evidence_quote="",
                reasoning=f"invalidated (no evidence quote): {judgment.reasoning}",
            )
        judgments.append(judgment)
    return judgments


def aggregate_by_category(*, flags: list[dict], judgments: list[FlagJudgment]) -> dict[str, dict]:
    """Roll per-flag judgments up into a per-category table.

    Returns {category: {detected, acted_on, positive, retrieval_grounded}}
    where detected is a count and the rest are fractions of detected.
    """
    counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"detected": 0, "acted_on": 0, "positive": 0, "grounded": 0}
    )
    for flag, judgment in zip(flags, judgments):
        row = counts[flag["category"]]
        row["detected"] += 1
        if judgment.classification != "not_acted_on":
            row["acted_on"] += 1
        if judgment.classification == "acted_on_positive":
            row["positive"] += 1
        if flag.get("evidence_doc_ids"):
            row["grounded"] += 1
    return {
        category: {
            "detected": row["detected"],
            "action_rate": row["acted_on"] / row["detected"],
            "positive_rate": row["positive"] / row["detected"],
            "retrieval_grounded_fraction": row["grounded"] / row["detected"],
        }
        for category, row in counts.items()
    }
