"""Smoke test: graph compiles and state schema flows correctly.

Skips the network call. Run with: pytest tests/ -v
"""

from pipeline.toy import SpecState, build_graph


def test_graph_compiles():
    app = build_graph()
    assert app is not None


def test_state_keys():
    keys = set(SpecState.__optional_keys__) | set(SpecState.__required_keys__)
    assert {"problem_statement", "understanding", "advice"} <= keys
