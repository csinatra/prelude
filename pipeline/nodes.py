from pipeline.llm_client import call_llm
from pipeline.schemas import Advice, AssumptionFlags, ParsedProblem, SurfacedSignals
from pipeline.state import PipelineState

def parse_problem(state: PipelineState) -> PipelineState:
    parsed = call_llm(
        system=(
            "You are a senior ML engineer who translates ambiguous stakeholder problem statements into rigorous ML framing. "
            "Extract the underlying ML structure from the supplied problem statement: its goal, whether the "
            "question is causal, predictive, descriptive, or ambiguous, and its key constraints."
        ),
        user=f"Problem statement: {state['raw_problem']}",
        response_model=ParsedProblem,
    )

    return {
        **state,
        "parsed_goal": parsed.goal,
        "framing_type": parsed.framing_type,
        "constraints": parsed.constraints,
        "stage_trace": state["stage_trace"] + ["parse_problem"],
    }


def surface_signals(state: PipelineState) -> PipelineState:
    framing = state.get("framing_type", "unknown")
    goal = state.get("parsed_goal", state["raw_problem"])

    parsed = call_llm(
        system=(
            "You are a ML data scientist working with an experienced ML engineer. "
            "Given a problem framing, identify: signals plausibly already available given the "
            "stated context, signals that would materially help but are likely missing, and "
            "directly relevant prior work (papers, known benchmarks, or standard approaches for "
            "this problem class)."
        ),
        user=f"Framing type: {framing}\nGoal: {goal}",
        response_model=SurfacedSignals,
    )

    return {
        **state,
        "available_signals": parsed.available_signals,
        "desired_signals": parsed.desired_signals,
        "prior_work": parsed.prior_work,
        "stage_trace": state["stage_trace"] + ["surface_signals"],
    }


def flag_assumptions(state: PipelineState) -> PipelineState:
    framing = state.get("framing_type", "unknown")
    goal = state.get("parsed_goal", state["raw_problem"])

    parsed = call_llm(
        system=(
            "You are an ML assumptions auditor briefing an experienced ML/AI engineer — skip "
            "definitions, go straight to the specific risk. Given a problem framing and its "
            "available/desired signals, identify the most likely assumption violations. Focus on: "
            "IID violations, exposure/selection bias, outcome measurement gaps, attribution "
            "ambiguity, sequential/temporal dependencies, and resource constraints. Each flag "
            "should name the concrete mechanism, not a generic category."
        ),
        user=(
            f"Framing type: {framing}\n"
            f"Goal: {goal}\n"
            f"Available signals: {state.get('available_signals', [])}\n"
            f"Desired signals: {state.get('desired_signals', [])}"
        ),
        response_model=AssumptionFlags,
    )

    return {
        **state,
        "assumption_flags": parsed.assumption_flags,
        "stage_trace": state["stage_trace"] + ["flag_assumptions"],
    }


def advise_approach(state: PipelineState) -> PipelineState:
    goal = state.get("parsed_goal", state["raw_problem"])
    framing = state.get("framing_type", "unknown")

    parsed = call_llm(
        system=(
            "You are an ML modeling advisor briefing an experienced ML/AI engineer as a technical "
            "peer. Be direct and specific — name architectures, loss formulations, or estimators "
            "rather than generic categories, and skip explanations of concepts a practitioner "
            "already knows. Given the problem framing, surfaced signals, prior work, and flagged "
            "assumption risks, recommend concrete modeling approaches, the key tradeoffs between "
            "them, and the most likely failure modes given the flagged risks."
        ),
        user=(
            f"Framing type: {framing}\n"
            f"Goal: {goal}\n"
            f"Constraints: {state.get('constraints', [])}\n"
            f"Available signals: {state.get('available_signals', [])}\n"
            f"Desired signals: {state.get('desired_signals', [])}\n"
            f"Prior work: {state.get('prior_work', [])}\n"
            f"Assumption flags: {state.get('assumption_flags', [])}"
        ),
        response_model=Advice,
        max_tokens=2048,
    )

    return {
        **state,
        "recommended_approaches": parsed.recommended_approaches,
        "tradeoffs": parsed.tradeoffs,
        "failure_modes": parsed.failure_modes,
        "stage_trace": state["stage_trace"] + ["advise_approach"],
    }
