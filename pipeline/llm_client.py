"""Swappable LLM backend for pipeline nodes.

LLM_PROVIDER=anthropic (default) — Claude via the Anthropic API, using native
structured outputs (messages.parse) to constrain generation to a Pydantic
schema. Model selected by MODEL (Haiku for dev iteration, Sonnet for eval
runs — see CLAUDE.md). Used for the runs whose results go in the writeup.

LLM_PROVIDER=ollama — a local open-source model via a running `ollama serve`,
using Ollama's JSON-schema `format` field for the same schema constraint.
Model selected by OLLAMA_MODEL. For free, offline smoke-testing of pipeline
wiring only — never for eval runs.
"""

import os
from datetime import datetime, timezone
from typing import TypeVar

import anthropic
import requests
from langsmith import get_current_run_tree, traceable
from pydantic import BaseModel

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "anthropic")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

_anthropic_client = anthropic.Anthropic() if LLM_PROVIDER == "anthropic" else None

ModelT = TypeVar("ModelT", bound=BaseModel)

# Process-local usage ledger, reset/read by callers (harness.runner wraps
# each condition run) — the durable record of upfront spec-build cost.
# One entry per LLM call, in call order; pipeline stages execute
# sequentially, so call order attributes usage to stages. LangSmith gets
# the same numbers per call, but traces are ephemeral and quota-bound; the
# run artifact needs them locally.
_usage_log: list[dict] = []


def reset_usage() -> None:
    _usage_log.clear()


def usage_log() -> list[dict]:
    """Per-call entries since the last reset, in call order."""
    return list(_usage_log)


def usage_snapshot() -> dict[str, int]:
    """Totals since the last reset."""
    return {
        "llm_calls": len(_usage_log),
        "input_tokens": sum(entry["input_tokens"] for entry in _usage_log),
        "output_tokens": sum(entry["output_tokens"] for entry in _usage_log),
    }


def _record_llm_trace_metadata(*, provider: str, model: str, input_tokens: int, output_tokens: int) -> None:
    """Accumulate token usage locally and attach model/usage to the LangSmith run.

    The model is resolved from env inside the provider call, so @traceable's
    input capture never sees it; without this, a trace can't say whether Haiku
    or Sonnet produced it. Trace attach is a no-op when tracing is disabled;
    the local ledger always accumulates.
    """
    _usage_log.append(
        {
            "call_index": len(_usage_log),
            "provider": provider,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "at": datetime.now(tz=timezone.utc).isoformat(),
        }
    )
    run = get_current_run_tree()
    if run is None:
        return
    run.set(
        metadata={"ls_provider": provider, "ls_model_name": model},
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    )


# Output caps are runaway guards, sized not to bind. A cap that binds truncates,
# and a truncated stage is a broken output rather than a smaller one — it is a
# measurement artifact in whichever condition happens to hit it first.
#
# This is NOT the budget confound. Whether C2's larger total generation budget is
# itself doing the work is a live question, and it is answered by the v1.5 stage
# ablations (RESEARCH_DESIGN.md roadmap), not by tuning these numbers. Per-call
# usage is recorded in llm_usage.json either way.
# Sized against sample VARIANCE, not against a typical response. The flag stage
# emitted 1,466 tokens on one sample of a problem and 5,008 on another (Haiku vs
# Sonnet, uw-madison, 2026-09-01), so a cap set at ~2x the observed mean still
# binds on a bad draw. Headroom here exceeds that ~3.4x spread on the hardest
# problem measured. Ceilings cost nothing when unused; a bound one parks a run.
DEFAULT_MAX_TOKENS = 8192
SYNTHESIS_MAX_TOKENS = 16384


@traceable(run_type="llm")


def call_llm(
    *, system: str, user: str, response_model: type[ModelT], max_tokens: int = DEFAULT_MAX_TOKENS
) -> ModelT:
    """Call the configured backend, constrained to response_model. Returns a validated instance."""
    if LLM_PROVIDER == "ollama":
        return _call_ollama(system=system, user=user, response_model=response_model, max_tokens=max_tokens)
    return _call_anthropic(system=system, user=user, response_model=response_model, max_tokens=max_tokens)


@traceable(run_type="llm")
def call_llm_text(
    *, system: str, user: str, max_tokens: int = SYNTHESIS_MAX_TOKENS, model: str | None = None
) -> str:
    """Call the configured backend with no output schema. Returns raw text.

    Used by the B2/C1 freeform synthesis (which must be free of any imposed
    structure, including a JSON constraint) and by notebook-summary ingestion,
    which pins `model` explicitly — summaries are corpus infrastructure and
    must stay homogeneous regardless of the MODEL env var (see
    ingest/ingest_summaries.py).
    """
    if LLM_PROVIDER == "ollama":
        response = requests.post(
            url=f"{OLLAMA_HOST}/api/chat",
            json={
                "model": model or os.environ["OLLAMA_MODEL"],
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
            },
        )
        response.raise_for_status()
        body = response.json()
        _record_llm_trace_metadata(
            provider="ollama",
            model=model or os.environ["OLLAMA_MODEL"],
            input_tokens=body.get("prompt_eval_count", 0),
            output_tokens=body.get("eval_count", 0),
        )
        return body["message"]["content"]
    resolved_model = model or os.environ["MODEL"]
    response = _anthropic_client.messages.create(
        model=resolved_model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    # Freeform truncation is SILENT without this: no schema means nothing fails
    # to parse, so a capped B2/C1 synthesis would ship as a spec that stops
    # mid-sentence. Since the structured path raises on the same condition, the
    # asymmetry would land as a quiet handicap on the control arm — and it would
    # bite hardest on the complex problems, where C2 has the most to gain.
    if response.stop_reason == "max_tokens":
        raise RuntimeError(
            f"freeform response hit max_tokens={max_tokens} and was truncated; "
            "raise the caller's max_tokens for this stage"
        )
    _record_llm_trace_metadata(
        provider="anthropic",
        model=resolved_model,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )
    return response.content[0].text


def _call_anthropic(*, system: str, user: str, response_model: type[ModelT], max_tokens: int) -> ModelT:
    model = os.environ["MODEL"]
    response = _anthropic_client.messages.parse(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
        output_format=response_model,
    )
    # Say what actually went wrong. A response cut off at the cap is still valid
    # JSON-so-far, so pydantic reports "EOF while parsing a string" from deep in
    # the SDK — which reads like a malformed schema rather than a budget that was
    # too small for this problem. Checked before .parsed_output, which is what
    # raises.
    if response.stop_reason == "max_tokens":
        raise RuntimeError(
            f"{response_model.__name__} response hit max_tokens={max_tokens} and was "
            "truncated; raise the caller's max_tokens for this stage"
        )
    _record_llm_trace_metadata(
        provider="anthropic",
        model=model,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )
    return response.parsed_output


def _call_ollama(*, system: str, user: str, response_model: type[ModelT], max_tokens: int) -> ModelT:
    model = os.environ["OLLAMA_MODEL"]
    response = requests.post(
        url=f"{OLLAMA_HOST}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "format": response_model.model_json_schema(),
            "stream": False,
        },
    )
    response.raise_for_status()
    body = response.json()
    _record_llm_trace_metadata(
        provider="ollama",
        model=model,
        input_tokens=body.get("prompt_eval_count", 0),
        output_tokens=body.get("eval_count", 0),
    )
    return response_model.model_validate_json(body["message"]["content"])
