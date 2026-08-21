"""Driver + evidence-bundle tests. No API calls: judge_flags is mocked."""

import json

import pytest

from analysis import judge_run
from analysis.judge import FlagJudgment

FLAGS = [
    {"flag_id": "f1", "category": "iid_violation", "explanation": "grouped rows", "evidence_doc_ids": ["d1"]},
    {"flag_id": "f2", "category": "exposure_bias", "explanation": "class imbalance", "evidence_doc_ids": []},
]

JOURNAL = {
    "nodes": [
        {"id": "a", "parent": None, "step": 0, "code": "print(0)", "is_buggy": True,
         "metric": None, "analysis": "crashed", "term_out": "Traceback"},
        {"id": "b", "parent": "a", "step": 1, "code": "print(1)", "is_buggy": False,
         "metric": 0.7, "analysis": "fixed", "term_out": "ok"},
        {"id": "z", "parent": None, "step": 2, "code": "print(9)", "is_buggy": False,
         "metric": 0.1, "analysis": "abandoned branch", "term_out": "meh"},
    ]
}


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(judge_run, "run_root", lambda: tmp_path)
    directory = tmp_path / "comp_C2_0"
    directory.mkdir()
    (directory / "pipeline_output.json").write_text(json.dumps({"assumption_flags": FLAGS}))
    (directory / "best_solution.py").write_text("print(1)")
    (directory / "journal.json").write_text(json.dumps(JOURNAL))
    return directory


def test_chain_follows_ancestry_not_step_order(run_dir):
    """The chain is the submitted node's lineage, so a sibling branch is excluded."""
    nodes = judge_run.chain_nodes(run_dir=run_dir, solution="print(1)")
    assert [node["step"] for node in nodes] == [0, 1]


def test_chain_falls_back_positionally_without_parent_links(run_dir):
    (run_dir / "journal.json").write_text(
        json.dumps({"nodes": [{"step": index, "term_out": "x"} for index in range(30)]})
    )
    nodes = judge_run.chain_nodes(run_dir=run_dir, solution="unmatched")
    assert len(nodes) == judge_run.EVIDENCE_CHAIN_DEPTH


def test_node_output_is_capped(run_dir):
    (run_dir / "journal.json").write_text(
        json.dumps({"nodes": [{"id": "a", "step": 0, "term_out": "x" * 99_999}]})
    )
    nodes = judge_run.chain_nodes(run_dir=run_dir, solution="")
    assert len(nodes[0]["term_out"]) == judge_run.EVIDENCE_NODE_CHARS


def test_unparseable_journal_yields_no_chain(run_dir):
    (run_dir / "journal.json").write_text("{not json")
    assert judge_run.chain_nodes(run_dir=run_dir, solution="") == []


def test_judge_run_writes_records_for_every_flag(run_dir, monkeypatch):
    monkeypatch.setattr(
        judge_run,
        "judge_flags",
        lambda **kwargs: [
            FlagJudgment(classification="acted_on_unclear", evidence_quote="q", reasoning="r")
            for _ in kwargs["flags"]
        ],
    )
    result = judge_run.judge_run(run_key="comp_C2_0")
    assert result["judged"] == 2
    payload = json.loads((run_dir / "judgments.json").read_text())
    assert [record["item_id"] for record in payload["judgments"]] == [
        "comp_C2_0#f1",
        "comp_C2_0#f2",
    ]
    assert payload["provenance"]["rubric_sha256"]
    assert payload["by_category"]["iid_violation"]["detected"] == 1


def test_judge_and_reviewer_get_the_same_bundle(run_dir, monkeypatch):
    """Evidence parity: the judge's logs are exactly what the page renders."""
    captured = {}
    monkeypatch.setattr(
        judge_run,
        "judge_flags",
        lambda **kwargs: captured.update(kwargs)
        or [FlagJudgment(classification="not_acted_on", evidence_quote="", reasoning="r")]
        * len(kwargs["flags"]),
    )
    judge_run.judge_run(run_key="comp_C2_0")
    payload = json.loads((run_dir / "judgments.json").read_text())
    assert payload["judgments"][0]["logs_excerpt"] == captured["logs"]
    assert payload["judgments"][0]["solution_excerpt"] == captured["solution"]


def test_run_without_solution_is_skipped_not_judged(run_dir, monkeypatch):
    (run_dir / "best_solution.py").unlink()
    monkeypatch.setattr(
        judge_run, "judge_flags", lambda **kwargs: pytest.fail("must not judge without artifacts")
    )
    assert judge_run.judge_run(run_key="comp_C2_0")["judged"] == 0
