"""Queue status derived from the registry. Docker is never consulted here."""

import pytest

from harness import registry, status


@pytest.fixture
def seeded(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "RESULTS_DIR", tmp_path)
    return tmp_path


def _run(*, key, status_name, updated_at, **fields):
    return {"run_key": key, "status": status_name, "updated_at": updated_at, **fields}


NOW = "2026-08-27T12:00:00+00:00"
OLD = "2026-08-25T12:00:00+00:00"


def test_counts_split_abandoned_out_of_its_phase_status(seeded):
    """A parked run keeps its phase status, so counting by status alone hides it."""
    runs = {
        "a": _run(key="a", status_name="graded", updated_at=NOW, score=0.5),
        "b": _run(key="b", status_name="spec_built", updated_at=NOW, abandoned=True),
        "c": _run(key="c", status_name="spec_built", updated_at=NOW),
    }
    state = status.summarize(runs=runs)
    assert state["by_status"] == {"graded": 1, "abandoned": 1, "spec_built": 1}
    assert state["graded"] == 1
    # An abandoned run is not remaining work: the driver will not pick it up again.
    assert state["remaining"] == 1


def test_silence_past_a_full_budget_is_flagged(seeded):
    quiet = {"a": _run(key="a", status_name="agent_run", updated_at=OLD)}
    assert status.summarize(runs=quiet)["stalled"] is True

    fresh = {"a": _run(key="a", status_name="agent_run", updated_at=NOW)}
    assert status.summarize(runs=fresh)["stalled"] is False


def test_a_drained_queue_is_quiet_because_it_finished(seeded):
    """Regression: an old but complete queue reported STALLED.

    Silence only means something while work is outstanding; flagging a finished
    drain trains the reader to ignore the flag that matters.
    """
    done = {"a": _run(key="a", status_name="graded", updated_at=OLD, score=0.5)}
    state = status.summarize(runs=done)
    assert state["remaining"] == 0
    assert state["stalled"] is False


def test_only_abandoned_work_left_is_not_a_stall(seeded):
    parked = {"a": _run(key="a", status_name="spec_built", updated_at=OLD, abandoned=True)}
    assert status.summarize(runs=parked)["stalled"] is False


def test_empty_registry_does_not_report_a_stall(seeded):
    state = status.summarize(runs={})
    assert state["total"] == 0
    assert state["stalled"] is False
    assert state["last_activity"] is None


def test_render_names_the_container_and_how_to_tail_it(seeded):
    state = status.summarize(runs={"a": _run(key="a", status_name="graded", updated_at=NOW)})
    out = status.render(state=state, container="competition-x-2026-uuid")
    assert "docker logs -f competition-x-2026-uuid" in out


def test_unparseable_timestamp_does_not_raise(seeded):
    runs = {"a": _run(key="a", status_name="agent_run", updated_at="not-a-date")}
    assert status.summarize(runs=runs)["idle_secs"] is None
