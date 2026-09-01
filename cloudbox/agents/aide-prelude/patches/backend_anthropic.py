# PATCHED COPY of aideml v6.3.3 `aide/backend/backend_anthropic.py`
# (github.com/thesofakillers/aideml @ v6.3.3, MIT — see ACKNOWLEDGMENTS.md).
#
# Three changes vs upstream, none of which alters AIDE's search/coding
# algorithm; the Dockerfile overlays this file onto the pip-installed aideml.
#
# (1) The `func_spec` path (resolved on-box 2026-07-23). Upstream leaves it as
#     `raise NotImplementedError("Anthropic does not support function calling
#     for now.")`, which makes AIDE's execution-review step (parse_exec_result)
#     fail on any Claude model — the reason mle-bench's own aide/claude agent
#     runs the *code* model on Anthropic but the *feedback* model on gpt-4o.
#     Here we implement it via Anthropic tool use, reaching parity with aideml's
#     OpenAI backend so the whole AIDE loop runs single-provider on Claude.
#
# (2) Per-call token-usage side-channel (2026-07-24). aideml already computes
#     in/out tokens from message.usage and returns them, but never persists them
#     to the journal (per-step agent cost is otherwise unrecoverable). We append
#     each call's usage + timestamps to $LOGS_DIR/prelude_token_usage.jsonl for
#     post-run per-step attribution (correlated to journal node ctimes offline).
#     Best-effort and identical across all conditions: it only READS usage the
#     SDK already returns and appends to a file — it never changes what query()
#     returns or how AIDE behaves, and a logging failure is swallowed.
#
# (3) Output cap raised from upstream's 4096 (measured on-box 2026-09-01). Same
#     category as the httpx pin in requirements.txt: API drift, not a design
#     choice. 4096 was Claude 3's hard output ceiling when aideml was written;
#     current models allow far more, so the constant silently became a budget.
#     It was truncating 16 of 28 agent calls on a segmentation task, and the
#     failure is invisible rather than loud: a response cut mid-code-block never
#     emits its closing fence, so aideml's extract_code finds nothing valid,
#     plan_and_code_query burns all 3 retries, and its fallback returns the raw
#     truncated text AS THE CODE — which then fails with a SyntaxError that
#     looks like the model cannot write Python. Half the step budget was lost
#     that way. It affects every condition (Condition A truncated 2 of 3 calls
#     with no spec at all), so it is a scaffold bug rather than a treatment
#     effect, and leaving it would put extraction luck in the measurement.
"""Backend for Anthropic API."""

import json
import os
import time
import logging

import anthropic
from .utils import FunctionSpec, OutputType, backoff_create, opt_messages_to_list
from funcy import notnone, once, select_values

logger = logging.getLogger("aide")


def _log_token_usage(*, model, usage, t_start, t_end, stop_reason) -> None:
    """Append one call's token usage to $LOGS_DIR/prelude_token_usage.jsonl.

    Behavior-neutral: reads only usage the SDK already returned. Best-effort —
    swallows every error so logging can never perturb the agent run."""
    try:
        record = {
            "t_start": t_start,
            "t_end": t_end,
            "model": model,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", None),
            "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", None),
            "stop_reason": stop_reason,
        }
        path = os.path.join(os.environ.get("LOGS_DIR", "."), "prelude_token_usage.jsonl")
        with open(path, "a") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception:
        pass  # logging must never affect the agent run

_client: anthropic.Anthropic = None  # type: ignore

ANTHROPIC_TIMEOUT_EXCEPTIONS = (
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.InternalServerError,
)


@once
def _setup_anthropic_client():
    global _client
    _client = anthropic.Anthropic(max_retries=0)


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    convert_system_to_user: bool = False,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    _setup_anthropic_client()

    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore
    if "max_tokens" not in filtered_kwargs:
        # PRELUDE PATCH (3): was 4096 upstream. See the header note — that value
        # was Claude 3's ceiling, and it truncates real solutions into
        # unparseable code. Sized to not bind on a full segmentation solution.
        filtered_kwargs["max_tokens"] = 16384

    # ── PRELUDE PATCH: function calling via Anthropic tool use ──────────────
    # Build a single tool from the FunctionSpec and force it with tool_choice,
    # so the model must return one tool_use block whose parsed `input` is the
    # structured result (parity with backend_openai.py's func_spec path).
    if func_spec is not None:
        filtered_kwargs["tools"] = [
            {
                "name": func_spec.name,
                "description": func_spec.description,
                "input_schema": func_spec.json_schema,
            }
        ]
        filtered_kwargs["tool_choice"] = {"type": "tool", "name": func_spec.name}
    # ── END PRELUDE PATCH ───────────────────────────────────────────────────

    # Anthropic doesn't allow not having a user messages
    # if we only have system msg -> use it as user msg
    if system_message is not None and user_message is None:
        system_message, user_message = user_message, system_message

    # Anthropic passes the system messages as a separate argument
    if system_message is not None:
        filtered_kwargs["system"] = system_message

    messages = opt_messages_to_list(None, user_message)

    t0 = time.time()
    message = backoff_create(
        _client.messages.create,
        ANTHROPIC_TIMEOUT_EXCEPTIONS,
        messages=messages,
        **filtered_kwargs,
    )
    req_time = time.time() - t0

    # ── PRELUDE PATCH: extract the forced tool_use result (a parsed dict),
    # else the plain text block as upstream did ─────────────────────────────
    if func_spec is not None:
        tool_use = next(block for block in message.content if block.type == "tool_use")
        output: OutputType = tool_use.input
    else:
        assert len(message.content) == 1 and message.content[0].type == "text"
        output = message.content[0].text
    # ── END PRELUDE PATCH ───────────────────────────────────────────────────

    in_tokens = message.usage.input_tokens
    out_tokens = message.usage.output_tokens

    info = {
        "stop_reason": message.stop_reason,
    }

    _log_token_usage(
        model=filtered_kwargs.get("model"),
        usage=message.usage,
        t_start=t0,
        t_end=t0 + req_time,
        stop_reason=message.stop_reason,
    )

    return output, req_time, in_tokens, out_tokens, info
