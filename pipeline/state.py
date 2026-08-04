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
    assumption_flags: list[dict]  # SpecificationFlag dumps
    recommendations: list[dict]  # Recommendation dumps, addresses_flags -> assumption_flags indices
    # RetrievedDoc dumps per stage — kept in state so every trace shows exactly
    # which corpus documents fed each stage.
    retrieved_parse: list[dict]
    retrieved_surface: list[dict]
    retrieved_flag: list[dict]
    retrieved_advise: list[dict]
    # doc_ids of notebook summaries already surfaced by an earlier stage; threaded
    # through the ordered stages so retrieve_with_topup keeps each stage's distinct
    # contribution at STAGE_N_NOTEBOOKS (cross-stage top-up, distinct-doc parity).
    retrieved_seen: set[str]
    stage_trace: list[str]
