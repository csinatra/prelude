"""Runner for the real four-stage spec pipeline."""

from __future__ import annotations

from pipeline.graph import build_graph
from pipeline.state import PipelineState


def run(raw_problem: str) -> PipelineState:
    app = build_graph()
    return app.invoke(input={"raw_problem": raw_problem, "stage_trace": []})
