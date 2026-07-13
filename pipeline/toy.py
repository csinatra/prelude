"""Two-stage toy pipeline: understand -> advise.

Validates the LangGraph + Anthropic SDK + LangSmith wiring on a trivial
problem before building the real four-stage specification chain.
"""

from __future__ import annotations

import json
import os
from typing import TypedDict

from anthropic import Anthropic
from langgraph.graph import END, START, StateGraph


DEV_MODEL = "claude-haiku-4-5-20251001"


class SpecState(TypedDict, total=False):
    problem_statement: str
    understanding: dict
    advice: dict


def _client() -> Anthropic:
    return Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def _json_call(*, system: str, user: str, schema_hint: str) -> dict:
    """Call Claude with a JSON-only output contract. Returns parsed dict."""
    client = _client()
    response = client.messages.create(
        model=DEV_MODEL,
        max_tokens=1024,
        system=system,
        messages=[
            {
                "role": "user",
                "content": (
                    f"{user}\n\n"
                    f"Respond with ONLY a JSON object matching this shape:\n"
                    f"{schema_hint}\n"
                    f"No prose, no markdown fences."
                ),
            }
        ],
    )
    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = text.strip("`").lstrip("json").strip()
    return json.loads(text)


def understand_node(state: SpecState) -> dict:
    """Stage 1: extract actual goal, framing, key constraints."""
    understanding = _json_call(
        system=(
            "You are an ML problem-framing assistant. Read a user's stated "
            "problem and extract its underlying structure."
        ),
        user=f"Problem statement:\n\n{state['problem_statement']}",
        schema_hint=(
            '{"actual_goal": str, '
            '"framing": "predictive" | "causal" | "descriptive", '
            '"key_constraints": [str, ...]}'
        ),
    )
    return {"understanding": understanding}


def advise_node(state: SpecState) -> dict:
    """Stage 2: propose modeling approach grounded in stage-1 output."""
    advice = _json_call(
        system=(
            "You are an ML modeling advisor. Given a structured understanding "
            "of a problem, propose a concrete first modeling approach and "
            "name the most likely failure mode."
        ),
        user=(
            "Structured understanding:\n\n"
            f"{json.dumps(state['understanding'], indent=2)}"
        ),
        schema_hint=(
            '{"approach": str, "rationale": str, "top_failure_mode": str}'
        ),
    )
    return {"advice": advice}


def build_graph():
    graph = StateGraph(state_schema=SpecState)
    graph.add_node(node="understand", action=understand_node)
    graph.add_node(node="advise", action=advise_node)
    graph.add_edge(start_key=START, end_key="understand")
    graph.add_edge(start_key="understand", end_key="advise")
    graph.add_edge(start_key="advise", end_key=END)
    return graph.compile()


def run(problem_statement: str) -> SpecState:
    app = build_graph()
    return app.invoke(input={"problem_statement": problem_statement})


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    example = (
        "We want to predict which customers will churn next month so the "
        "retention team can send them a discount offer. We have 18 months "
        "of subscription, billing, and support-ticket data."
    )
    result = run(problem_statement=example)
    print(json.dumps(result, indent=2))
