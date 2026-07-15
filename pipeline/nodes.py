from pipeline.config import (
    COMPETITION_METADATA,
    PRACTITIONER_KNOWLEDGE,
    RETRIEVAL_K,
    STAGE_CHUNKS_PER_NOTEBOOK,
    STAGE_N_NOTEBOOKS,
)
from pipeline.llm_client import call_llm
from pipeline.retriever import RetrievedDoc, retrieve, retrieve_two_level
from pipeline.schemas import Advice, AssumptionFlags, ParsedProblem, SurfacedSignals
from pipeline.state import PipelineState

# Shared stance on retrieved context, appended to every stage's system prompt.
# Condition C2 tests *critical* integration of retrieved knowledge (AssistedDS
# showed LLMs adopt provided knowledge uncritically) — retrieval must inform
# reasoning, never bound it.
RETRIEVAL_STANCE = (
    " Retrieved excerpts from prior competitions are evidence of past practice, not a boundary "
    "on your reasoning: reason from your full ML expertise first, use the excerpts to ground or "
    "recalibrate specific claims, and explicitly disregard excerpts that are irrelevant, "
    "outdated, or low quality."
)


def surface_query(*, task_type: str, evaluation_metric: str, goal: str) -> str:
    return f"{task_type} {evaluation_metric} {goal}"


def flag_query(*, task_type: str, evaluation_metric: str, goal: str) -> str:
    return f"validation leakage overfitting pitfalls {task_type} {evaluation_metric} {goal}"


def advise_query(*, task_type: str, evaluation_metric: str, goal: str) -> str:
    return f"model architecture training approach {task_type} {evaluation_metric} {goal}"


def _format_docs(docs: list[RetrievedDoc]) -> str:
    if not docs:
        return "(no relevant documents retrieved)"
    formatted = []
    for doc in docs:
        notebook = f" | notebook: {doc.kaggle_id}" if doc.kaggle_id is not None else ""
        formatted.append(
            f"[{doc.doc_id} | {doc.source_type} | competition: {doc.competition_id}{notebook}]\n"
            f"{doc.text}"
        )
    return "\n\n".join(formatted)


def _dump(docs: list[RetrievedDoc]) -> list[dict]:
    return [doc.model_dump() for doc in docs]


def _format_flags(flags: list[dict]) -> str:
    if not flags:
        return "(no flags raised)"
    return "\n".join(
        f"[{flag['flag_id']}] {flag['category']} ({flag['confidence']} confidence): "
        f"{flag['explanation']}"
        for flag in flags
    )


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
    docs = retrieve_two_level(
        query=surface_query(
            task_type=state.get("task_type", ""),
            evaluation_metric=state.get("evaluation_metric", ""),
            goal=goal,
        ),
        exclude_competition=state.get("competition_id", ""),
        n_notebooks=STAGE_N_NOTEBOOKS,
        chunks_per_notebook=STAGE_CHUNKS_PER_NOTEBOOK,
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
    docs = retrieve_two_level(
        query=flag_query(
            task_type=state.get("task_type", ""),
            evaluation_metric=state.get("evaluation_metric", ""),
            goal=goal,
        ),
        exclude_competition=state.get("competition_id", ""),
        n_notebooks=STAGE_N_NOTEBOOKS,
        chunks_per_notebook=STAGE_CHUNKS_PER_NOTEBOOK,
    )
    parsed = call_llm(
        system=(
            "You are an ML assumptions auditor briefing an experienced ML/AI engineer — skip "
            "definitions, go straight to the specific risk. Given a problem framing and its "
            "available/desired signals, identify the most likely assumption violations. Each "
            "flag must name the concrete mechanism, not a generic category, and carry: a "
            "category (iid_violation, exposure_bias, outcome_measurement_gap, "
            "delivery_attribution_failure, sequential_dependency, multi_touch_attribution, "
            "resource_constraint_violation, or other), your confidence (low/medium/high), and "
            "evidence_doc_ids — the bracketed IDs of the specific retrieved documents that "
            "informed that flag. Cite a document only if it genuinely shaped the flag; a flag "
            "drawn from your general knowledge should honestly report an empty "
            "evidence_doc_ids list — that is a valid and expected answer, not a failure."
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
        max_tokens=2048,
    )
    # flag_ids assigned here, not by the LLM — stable join keys for
    # Recommendation.addresses_flags and downstream analysis.
    return {
        "assumption_flags": [
            {"flag_id": f"F{index}", **flag.model_dump()}
            for index, flag in enumerate(parsed.flags)
        ],
        "retrieved_flag": _dump(docs),
        "stage_trace": state["stage_trace"] + ["flag_assumptions"],
    }


def advise_approach(state: PipelineState) -> dict:
    goal = state.get("parsed_goal", state["raw_problem"])
    flags = state.get("assumption_flags", [])
    docs = retrieve_two_level(
        query=advise_query(
            task_type=state.get("task_type", ""),
            evaluation_metric=state.get("evaluation_metric", ""),
            goal=goal,
        ),
        exclude_competition=state.get("competition_id", ""),
        n_notebooks=STAGE_N_NOTEBOOKS,
        chunks_per_notebook=STAGE_CHUNKS_PER_NOTEBOOK,
    )
    parsed = call_llm(
        system=(
            "You are an ML modeling advisor briefing an experienced ML/AI engineer as a technical "
            "peer. Be direct and specific — name architectures, loss formulations, or estimators "
            "rather than generic categories, and skip explanations of concepts a practitioner "
            "already knows. Recommend concrete modeling approaches optimized for the stated "
            "evaluation metric. Each recommendation carries its own tradeoff, its most likely "
            "failure mode given the flagged risks, and addresses_flags — the bracketed flag IDs "
            "(e.g. F0, F2) of the assumption flags it responds to (empty if it addresses none "
            "directly)."
            + RETRIEVAL_STANCE
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
            f"Assumption flags (indexed):\n{_format_flags(flags)}\n\n"
            f"Code excerpts from similar competitions:\n{_format_docs(docs)}"
        ),
        response_model=Advice,
        max_tokens=4096,
    )
    return {
        "recommendations": [rec.model_dump() for rec in parsed.recommendations],
        "retrieved_advise": _dump(docs),
        "stage_trace": state["stage_trace"] + ["advise_approach"],
    }
