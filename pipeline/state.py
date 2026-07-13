from typing import TypedDict


class PipelineState(TypedDict, total=False):
    raw_problem: str
    competition_id: str  # slug of the competition being specified; drives leave-one-out retrieval
    parsed_goal: str
    task_type: str
    evaluation_metric: str
    target_variable: str
    framing_type: str  # "causal" | "predictive" | "descriptive" | "ambiguous"
    constraints: list[str]
    available_signals: list[str]
    desired_signals: list[str]
    prior_work: list[str]
    assumption_flags: list[str]
    recommended_approaches: list[str]
    tradeoffs: list[str]
    failure_modes: list[str]
    # RetrievedDoc dumps per stage — kept in state so every trace shows exactly
    # which corpus documents fed each stage.
    retrieved_parse: list[dict]
    retrieved_surface: list[dict]
    retrieved_flag: list[dict]
    retrieved_advise: list[dict]
    stage_trace: list[str]
