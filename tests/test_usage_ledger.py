"""Local per-call LLM usage ledger at the llm_client seam."""

from pipeline import llm_client


def test_ledger_records_per_call_entries_in_order():
    llm_client.reset_usage()
    llm_client._record_llm_trace_metadata(
        provider="anthropic", model="m", input_tokens=100, output_tokens=10
    )
    llm_client._record_llm_trace_metadata(
        provider="anthropic", model="m", input_tokens=50, output_tokens=5
    )
    log = llm_client.usage_log()
    assert [entry["call_index"] for entry in log] == [0, 1]
    assert log[0]["input_tokens"] == 100 and log[1]["input_tokens"] == 50
    assert all(entry["model"] == "m" and entry["at"] for entry in log)
    assert llm_client.usage_snapshot() == {
        "llm_calls": 2,
        "input_tokens": 150,
        "output_tokens": 15,
    }


def test_reset_clears_log_and_totals():
    llm_client.reset_usage()
    llm_client._record_llm_trace_metadata(
        provider="ollama", model="q", input_tokens=1, output_tokens=1
    )
    llm_client.reset_usage()
    assert llm_client.usage_log() == []
    assert llm_client.usage_snapshot() == {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0}
