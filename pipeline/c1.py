"""Condition C1 — staged retrieval, freeform synthesis.

Isolates whether staged retrieval alone improves over B2's flat retrieval,
holding synthesis style constant at B2's level (schema-less call_llm_text,
no RETRIEVAL_STANCE).

Design notes:
- Staged retrieval queries for surface/flag/advise are built from
  parse_problem's structured extraction (task type, metric, goal) — that is
  what makes them directed. C1 therefore runs parse's structured call to
  DIRECT retrieval, but parse's extracted fields are NOT passed to the
  synthesis prompt: synthesis input stays comparable to B2 (description +
  context block only).
- run_c1 returns both artifacts: `context_block` (the raw staged block — the
  staged analog of B1, injectable directly) and `advice` (the freeform
  synthesis — the C1 condition per the research design). The harness decides
  which artifact is injected per run.
"""

from pipeline.baseline import FREEFORM_SYSTEM
from pipeline.config import STAGE_CHUNKS_PER_NOTEBOOK, STAGE_N_NOTEBOOKS
from pipeline.llm_client import call_llm_text
from pipeline.nodes import (
    _format_docs,
    advise_query,
    flag_query,
    parse_problem,
    surface_query,
)
from pipeline.retriever import RetrievedDoc, retrieve_two_level


def run_c1(*, raw_problem: str, competition_id: str) -> dict:
    parse_update = parse_problem(
        {"raw_problem": raw_problem, "competition_id": competition_id, "stage_trace": []}
    )
    query_fields = {
        "task_type": parse_update["task_type"],
        "evaluation_metric": parse_update["evaluation_metric"],
        "goal": parse_update["parsed_goal"],
    }

    retrieved_by_stage = {"parse": [RetrievedDoc(**doc) for doc in parse_update["retrieved_parse"]]}
    for stage, query_builder in [
        ("surface", surface_query),
        ("flag", flag_query),
        ("advise", advise_query),
    ]:
        retrieved_by_stage[stage] = retrieve_two_level(
            query=query_builder(**query_fields),
            exclude_competition=competition_id,
            n_notebooks=STAGE_N_NOTEBOOKS,
            chunks_per_notebook=STAGE_CHUNKS_PER_NOTEBOOK,
        )

    # One flat block for synthesis, deduped across stages (stages can surface
    # the same chunk), stage order preserved — comparable to B2's single block.
    seen: set[str] = set()
    merged: list[RetrievedDoc] = []
    for stage in ["parse", "surface", "flag", "advise"]:
        for doc in retrieved_by_stage[stage]:
            if doc.doc_id not in seen:
                seen.add(doc.doc_id)
                merged.append(doc)
    context_block = _format_docs(merged)

    advice = call_llm_text(
        system=FREEFORM_SYSTEM,
        user=(
            f"Competition description:\n{raw_problem}\n\n"
            f"Reference material from similar competitions:\n{context_block}"
        ),
        max_tokens=4096,
    )
    return {
        "condition": "C1",
        "competition_id": competition_id,
        "retrieved": {
            stage: [doc.model_dump() for doc in docs]
            for stage, docs in retrieved_by_stage.items()
        },
        "context_block": context_block,
        "advice": advice,
    }
