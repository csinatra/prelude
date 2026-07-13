from pipeline.llm_client import call_llm
from pipeline.retriever import RetrievedDoc, retrieve
from pipeline.schemas import Advice, AssumptionFlags, ParsedProblem, SurfacedSignals
from pipeline.state import PipelineState

RETRIEVAL_K = 5
COMPETITION_METADATA = "competition_metadata"
PRACTITIONER_KNOWLEDGE = "practitioner_knowledge"

# Shared stance on retrieved context, appended to every stage's system prompt.
# Condition C tests *critical* integration of retrieved knowledge (AssistedDS
# showed LLMs adopt provided knowledge uncritically) — retrieval must inform
# reasoning, never bound it.
RETRIEVAL_STANCE = (
    " Retrieved excerpts from prior competitions are evidence of past practice, not a boundary "
    "on your reasoning: reason from your full ML expertise first, use the excerpts to ground or "
    "recalibrate specific claims, and explicitly disregard excerpts that are irrelevant, "
    "outdated, or low quality."
)


def _format_docs(docs: list[RetrievedDoc]) -> str:
    if not docs:
        return "(no relevant documents retrieved)"
    return "\n\n".join(
        f"[{doc.source_type} | competition: {doc.competition_id}]\n{doc.text}" for doc in docs
    )


def _dump(docs: list[RetrievedDoc]) -> list[dict]:
    return [doc.model_dump() for doc in docs]


def parse_problem(state: PipelineState) -> dict:
    docs = retrieve(
        query=state["raw_problem"],
        collection=COMPETITION_METADATA,
        exclude_competition=state.get("competition_id", ""),
        k=RETRIEVAL_K,
    )
    parsed = call_llm(
        system=(
            "You are a senior ML engineer who translates machine learning competition "
            "descriptions into rigorous ML framing. Extract the underlying ML structure from the "
            "supplied competition description: its goal, task type, evaluation metric, target "
            "variable, whether the question is causal, predictive, descriptive, or ambiguous, and "
            "its key constraints (data size, compute, time limits, submission format)."
            + RETRIEVAL_STANCE
        ),
        user=(
            f"Competition description:\n{state['raw_problem']}\n\n"
            f"Descriptions of similar past competitions:\n{_format_docs(docs)}"
        ),
        response_model=ParsedProblem,
    )
    return {
        "parsed_goal": parsed.goal,
        "task_type": parsed.task_type,
        "evaluation_metric": parsed.evaluation_metric,
        "target_variable": parsed.target_variable,
        "framing_type": parsed.framing_type,
        "constraints": parsed.constraints,
        "retrieved_parse": _dump(docs),
        "stage_trace": state["stage_trace"] + ["parse_problem"],
    }


def surface_signals(state: PipelineState) -> dict:
    goal = state.get("parsed_goal", state["raw_problem"])
    docs = retrieve(
        query=f"{state.get('task_type', '')} {state.get('evaluation_metric', '')} {goal}",
        collection=PRACTITIONER_KNOWLEDGE,
        exclude_competition=state.get("competition_id", ""),
        k=RETRIEVAL_K,
    )
    parsed = call_llm(
        system=(
            "You are a ML data scientist working with an experienced ML engineer. Given a "
            "problem framing, identify: data signals plausibly available given the stated "
            "context, signals that would materially help but are likely missing, and directly "
            "relevant prior work (approaches, features, external data, preprocessing) — drawing "
            "on both your own knowledge of this problem class and what the retrieved excerpts "
            "from similar competitions actually used." + RETRIEVAL_STANCE
        ),
        user=(
            f"Task type: {state.get('task_type', 'unknown')}\n"
            f"Framing type: {state.get('framing_type', 'unknown')}\n"
            f"Goal: {goal}\n\n"
            f"Code excerpts from similar competitions:\n{_format_docs(docs)}"
        ),
        response_model=SurfacedSignals,
    )
    return {
        "available_signals": parsed.available_signals,
        "desired_signals": parsed.desired_signals,
        "prior_work": parsed.prior_work,
        "retrieved_surface": _dump(docs),
        "stage_trace": state["stage_trace"] + ["surface_signals"],
    }


def flag_assumptions(state: PipelineState) -> dict:
    goal = state.get("parsed_goal", state["raw_problem"])
    docs = retrieve(
        query=(
            f"validation leakage overfitting pitfalls {state.get('task_type', '')} "
            f"{state.get('evaluation_metric', '')} {goal}"
        ),
        collection=PRACTITIONER_KNOWLEDGE,
        exclude_competition=state.get("competition_id", ""),
        k=RETRIEVAL_K,
    )
    parsed = call_llm(
        system=(
            "You are an ML assumptions auditor briefing an experienced ML/AI engineer — skip "
            "definitions, go straight to the specific risk. Given a problem framing and its "
            "available/desired signals, identify the most likely assumption violations. Focus "
            "on: IID violations, exposure/selection bias, outcome measurement gaps, attribution "
            "ambiguity, sequential/temporal dependencies, train/test leakage, and resource "
            "constraints. Each flag should name the concrete mechanism, not a generic category."
            + RETRIEVAL_STANCE
        ),
        user=(
            f"Task type: {state.get('task_type', 'unknown')}\n"
            f"Framing type: {state.get('framing_type', 'unknown')}\n"
            f"Evaluation metric: {state.get('evaluation_metric', 'unknown')}\n"
            f"Goal: {goal}\n"
            f"Available signals: {state.get('available_signals', [])}\n"
            f"Desired signals: {state.get('desired_signals', [])}\n\n"
            f"Code excerpts from similar competitions:\n{_format_docs(docs)}"
        ),
        response_model=AssumptionFlags,
    )
    return {
        "assumption_flags": parsed.assumption_flags,
        "retrieved_flag": _dump(docs),
        "stage_trace": state["stage_trace"] + ["flag_assumptions"],
    }


def advise_approach(state: PipelineState) -> dict:
    goal = state.get("parsed_goal", state["raw_problem"])
    docs = retrieve(
        query=(
            f"model architecture training approach {state.get('task_type', '')} "
            f"{state.get('evaluation_metric', '')} {goal}"
        ),
        collection=PRACTITIONER_KNOWLEDGE,
        exclude_competition=state.get("competition_id", ""),
        k=RETRIEVAL_K,
    )
    parsed = call_llm(
        system=(
            "You are an ML modeling advisor briefing an experienced ML/AI engineer as a technical "
            "peer. Be direct and specific — name architectures, loss formulations, or estimators "
            "rather than generic categories, and skip explanations of concepts a practitioner "
            "already knows. Given the problem framing, surfaced signals, prior work, and flagged "
            "assumption risks, recommend concrete modeling approaches optimized for the stated "
            "evaluation metric, the key tradeoffs between them, and the most likely failure modes "
            "given the flagged risks." + RETRIEVAL_STANCE
        ),
        user=(
            f"Task type: {state.get('task_type', 'unknown')}\n"
            f"Framing type: {state.get('framing_type', 'unknown')}\n"
            f"Evaluation metric: {state.get('evaluation_metric', 'unknown')}\n"
            f"Goal: {goal}\n"
            f"Constraints: {state.get('constraints', [])}\n"
            f"Available signals: {state.get('available_signals', [])}\n"
            f"Desired signals: {state.get('desired_signals', [])}\n"
            f"Prior work: {state.get('prior_work', [])}\n"
            f"Assumption flags: {state.get('assumption_flags', [])}\n\n"
            f"Code excerpts from similar competitions:\n{_format_docs(docs)}"
        ),
        response_model=Advice,
        max_tokens=2048,
    )
    return {
        "recommended_approaches": parsed.recommended_approaches,
        "tradeoffs": parsed.tradeoffs,
        "failure_modes": parsed.failure_modes,
        "retrieved_advise": _dump(docs),
        "stage_trace": state["stage_trace"] + ["advise_approach"],
    }
