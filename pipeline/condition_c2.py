"""Runner for Condition C2 — the four-stage structured spec pipeline."""

from __future__ import annotations

from pipeline.graph import build_graph
from pipeline.state import PipelineState


def run_c2(raw_problem: str, competition_id: str = "") -> PipelineState:
    """Run the C2 spec pipeline. competition_id (Kaggle slug) drives leave-one-out retrieval."""
    app = build_graph()
    return app.invoke(
        input={"raw_problem": raw_problem, "competition_id": competition_id, "stage_trace": []}
    )
