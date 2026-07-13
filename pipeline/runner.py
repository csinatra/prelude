"""Runner for the real four-stage spec pipeline."""

from __future__ import annotations

from pipeline.graph import build_graph
from pipeline.state import PipelineState


def run(raw_problem: str, competition_id: str = "") -> PipelineState:
    """Run the spec pipeline. competition_id (Kaggle slug) drives leave-one-out retrieval."""
    app = build_graph()
    return app.invoke(
        input={"raw_problem": raw_problem, "competition_id": competition_id, "stage_trace": []}
    )
