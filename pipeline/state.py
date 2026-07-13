from typing import TypedDict, Optional

class PipelineState(TypedDict):
    raw_problem: str
    parsed_goal: Optional[str]
    framing_type: Optional[str]  # "causal" | "predictive" | "descriptive" | "ambiguous"
    constraints: Optional[list[str]]
    available_signals: Optional[list[str]]
    desired_signals: Optional[list[str]]
    prior_work: Optional[list[str]]
    assumption_flags: Optional[list[str]]
    recommended_approaches: Optional[list[str]]
    tradeoffs: Optional[list[str]]
    failure_modes: Optional[list[str]]
    stage_trace: list[str]
