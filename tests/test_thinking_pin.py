"""Spec-build thinking pin: one setting (off), spelled per model, on every call path."""

from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from pipeline import llm_client


class _Out(BaseModel):
    x: int


def _response(*, stop_reason: str = "end_turn") -> SimpleNamespace:
    return SimpleNamespace(
        stop_reason=stop_reason,
        model="m",
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        content=[SimpleNamespace(type="text", text="ok")],
        parsed_output=_Out(x=1),
    )


class _FakeMessages:
    def __init__(self, *, stop_reason: str = "end_turn") -> None:
        self.calls: list[dict] = []
        self.stop_reason = stop_reason

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _response(stop_reason=self.stop_reason)

    parse = create


@pytest.fixture
def fake_messages(monkeypatch):
    messages = _FakeMessages()
    monkeypatch.setattr(llm_client, "_anthropic_client", SimpleNamespace(messages=messages))
    return messages


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("claude-sonnet-5-5", {"type": "between_tools"}),
        ("claude-sonnet-5", {"type": "disabled"}),
        ("claude-haiku-4-5-20251001", {"type": "disabled"}),
    ],
)
def test_both_call_paths_send_the_per_model_off_setting(*, fake_messages, monkeypatch, model, expected):
    monkeypatch.setenv("MODEL", model)
    llm_client.call_llm_text(system="s", user="u")
    llm_client.call_llm(system="s", user="u", response_model=_Out)
    assert [call["thinking"] for call in fake_messages.calls] == [expected, expected]


def test_refusal_raises_on_both_call_paths(*, fake_messages, monkeypatch):
    monkeypatch.setenv("MODEL", "claude-sonnet-5-5")
    fake_messages.stop_reason = "refusal"
    with pytest.raises(RuntimeError, match="declined"):
        llm_client.call_llm_text(system="s", user="u")
    with pytest.raises(RuntimeError, match="declined"):
        llm_client.call_llm(system="s", user="u", response_model=_Out)
