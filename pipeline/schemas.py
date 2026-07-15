from typing import Literal

from pydantic import BaseModel


class ParsedProblem(BaseModel):
    goal: str
    task_type: str  # e.g. tabular regression, image classification, text normalization
    evaluation_metric: str
    target_variable: str
    framing_type: Literal["causal", "predictive", "descriptive", "ambiguous"]
    constraints: list[str]


class SurfacedSignals(BaseModel):
    available_signals: list[str]
    desired_signals: list[str]
    prior_work: list[str]


class SpecificationFlag(BaseModel):
    category: Literal[
        "iid_violation",
        "exposure_bias",
        "outcome_measurement_gap",
        "delivery_attribution_failure",
        "sequential_dependency",
        "multi_touch_attribution",
        "resource_constraint_violation",
        "other",
    ]
    explanation: str
    # doc IDs from retrieved_flag that support this specific flag; empty when
    # the flag comes from general knowledge rather than retrieved context
    evidence_doc_ids: list[str]
    confidence: Literal["low", "medium", "high"]


class AssumptionFlags(BaseModel):
    flags: list[SpecificationFlag]


class Recommendation(BaseModel):
    approach: str
    tradeoff: str
    failure_mode: str
    # flag_ids ("F0", "F1", ...) this recommendation responds to. IDs are
    # assigned programmatically in flag_assumptions (never LLM-generated), so
    # references stay self-describing in serialized artifacts even if the
    # flags list is later filtered or reordered during analysis.
    addresses_flags: list[str]


class Advice(BaseModel):
    recommendations: list[Recommendation]
