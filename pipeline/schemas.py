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


class AssumptionFlags(BaseModel):
    assumption_flags: list[str]


class Advice(BaseModel):
    recommended_approaches: list[str]
    tradeoffs: list[str]
    failure_modes: list[str]
