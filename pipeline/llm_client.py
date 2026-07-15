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
from typing import TypeVar

import anthropic
import requests
from pydantic import BaseModel

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "anthropic")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

_anthropic_client = anthropic.Anthropic() if LLM_PROVIDER == "anthropic" else None

ModelT = TypeVar("ModelT", bound=BaseModel)


def call_llm(*, system: str, user: str, response_model: type[ModelT], max_tokens: int = 1024) -> ModelT:
    """Call the configured backend, constrained to response_model. Returns a validated instance."""
    if LLM_PROVIDER == "ollama":
        return _call_ollama(system=system, user=user, response_model=response_model, max_tokens=max_tokens)
    return _call_anthropic(system=system, user=user, response_model=response_model, max_tokens=max_tokens)


def call_llm_text(*, system: str, user: str, max_tokens: int = 4096) -> str:
    """Call the configured backend with no output schema. Returns raw text.

    Used by the B2/C1 freeform synthesis (which must be free of any imposed
    structure, including a JSON constraint) and by notebook-summary ingestion.
    """
    if LLM_PROVIDER == "ollama":
        response = requests.post(
            url=f"{OLLAMA_HOST}/api/chat",
            json={
                "model": os.environ["OLLAMA_MODEL"],
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
            },
        )
        response.raise_for_status()
        return response.json()["message"]["content"]
    response = _anthropic_client.messages.create(
        model=os.environ["MODEL"],
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
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
    text = response.json()["message"]["content"]
    return response_model.model_validate_json(text)
