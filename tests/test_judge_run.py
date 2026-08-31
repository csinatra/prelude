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


def test_bundle_is_the_submitted_node_only(run_dir):
    """The evidence is one node: the one that produced the submitted solution."""
    nodes = judge_run.chain_nodes(run_dir=run_dir, solution="print(1)")
    assert [node["step"] for node in nodes] == [1]


def test_bundle_size_does_not_track_journal_length(run_dir):
    """Equal evidence per item, whatever the search tree did.

    The retired ancestor chain returned more nodes for a deep best node than a
    shallow one, so an identical flag could be judged on more evidence in one run
    than another and the action rate would partly measure tree shape. Chain depth
    also grows with the step budget, so that distortion would have grown too.
    """
    for journal_length in (2, 30, 300):
        (run_dir / "journal.json").write_text(
            json.dumps(
                {
                    "nodes": [
                        {"id": str(index), "parent": str(index - 1) if index else None,
                         "step": index, "is_buggy": False,
                         "metric": {"value": index / 1000, "maximize": True}}
                        for index in range(journal_length)
                    ]
                }
            )
        )
        assert len(judge_run.chain_nodes(run_dir=run_dir, solution="unmatched")) == 1


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
            FlagJudgment(classification="acted_on", evidence_quote="q", reasoning="r")
            for _ in kwargs["flags"]
        ],
    )
    result = judge_run.judge_run(run_key="comp_C2_0")
    assert result["judged"] == 2
    payload = json.loads((run_dir / "judgments.json").read_text())
    # item_id carries BOTH runs: the same flag is judged against other conditions'
    # solutions for the base rate, so the flag's own run alone would collide.
    assert [record["item_id"] for record in payload["judgments"]] == [
        "comp_C2_0#f1@comp_C2_0",
        "comp_C2_0#f2@comp_C2_0",
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


def test_term_out_read_from_the_serialized_key(run_dir):
    """aideml serializes `_term_out`; reading `term_out` yields "" for every node.

    Regression: the evidence bundle carried no observed output at all, which the
    rubric's acted_on depends on.
    """
    (run_dir / "journal.json").write_text(
        json.dumps({"nodes": [{"id": "a", "step": 0, "_term_out": "fold auc=0.61"}]})
    )
    nodes = judge_run.chain_nodes(run_dir=run_dir, solution="")
    assert nodes[0]["term_out"] == "fold auc=0.61"


REAL_METRIC_JOURNAL = {
    "nodes": [
        {"id": "a", "parent": None, "step": 0, "is_buggy": False,
         "metric": {"value": 0.5997, "maximize": True}, "code": "a"},
        {"id": "b", "parent": "a", "step": 1, "is_buggy": False,
         "metric": {"value": 0.6625, "maximize": True}, "code": "b"},  # best
        {"id": "c", "parent": "b", "step": 3, "is_buggy": True,
         "metric": {"value": None, "maximize": None}, "code": "c"},
        {"id": "d", "parent": "c", "step": 6, "is_buggy": False,
         "metric": {"value": 0.6352, "maximize": True}, "code": "d"},  # later but worse
    ]
}


def test_best_node_uses_the_metric_not_position(run_dir):
    """AIDE serializes metric as {value, maximize}; a later node often scores worse."""
    (run_dir / "journal.json").write_text(json.dumps(REAL_METRIC_JOURNAL))
    nodes = judge_run.chain_nodes(run_dir=run_dir, solution="no match")
    assert nodes[-1]["step"] == 1  # 0.6625, not the later 0.6352
    assert nodes[-1]["metric"] == 0.6625  # scalar, not the raw dict


def test_buggy_nodes_with_null_metric_are_not_scored(run_dir):
    """{"value": None} is a dict, so testing the field for None counts it as scored."""
    (run_dir / "journal.json").write_text(json.dumps({"nodes": [
        {"id": "a", "step": 0, "is_buggy": True, "metric": {"value": None, "maximize": None}},
    ]}))
    nodes = judge_run.chain_nodes(run_dir=run_dir, solution="no match")
    assert nodes[0]["metric"] is None


def test_minimizing_metric_picks_the_lowest(run_dir):
    (run_dir / "journal.json").write_text(json.dumps({"nodes": [
        {"id": "a", "parent": None, "step": 0, "is_buggy": False,
         "metric": {"value": 0.9, "maximize": False}},
        {"id": "b", "parent": "a", "step": 1, "is_buggy": False,
         "metric": {"value": 0.2, "maximize": False}},
    ]}))
    nodes = judge_run.chain_nodes(run_dir=run_dir, solution="no match")
    assert nodes[-1]["metric"] == 0.2


# ── base-rate counterfactual (H2 attribution) ───────────────────────────

@pytest.fixture
def control_run_dir(run_dir):
    """A B2 run for the same competition and seed, with its own solution."""
    directory = run_dir.parent / "comp_B2_0"
    directory.mkdir()
    (directory / "best_solution.py").write_text("import pandas  # different solution")
    (directory / "journal.json").write_text(json.dumps(JOURNAL))
    return directory


def test_paired_run_key_swaps_only_the_condition():
    assert judge_run.paired_run_key(run_key="comp_C2_0", condition="B2") == "comp_B2_0"
    assert (
        judge_run.paired_run_key(run_key="random-acts-of-pizza_C2_2", condition="B1")
        == "random-acts-of-pizza_B1_2"
    )


def test_base_rate_judges_c2_flags_against_the_control_solution(
    run_dir, control_run_dir, monkeypatch
):
    """The flag is unchanged; only the solution differs. That difference is the estimand."""
    seen = {}
    monkeypatch.setattr(
        judge_run,
        "judge_flags",
        lambda **kwargs: seen.update(kwargs)
        or [FlagJudgment(classification="not_acted_on", evidence_quote="", reasoning="r")]
        * len(kwargs["flags"]),
    )
    result = judge_run.judge_run(run_key="comp_C2_0", solution_run_key="comp_B2_0")

    assert result["judged"] == 2
    assert "import pandas" in seen["solution"]  # the control's code, not C2's
    assert [flag["flag_id"] for flag in seen["flags"]] == ["f1", "f2"]  # C2's flags

    # Written into the CONTROL's directory: a condition's dir holds what was
    # judged against it, and C2's own judgments.json must not be overwritten.
    payload = json.loads((control_run_dir / "judgments_baserate_comp_C2_0.json").read_text())
    assert payload["is_base_rate"] is True
    assert not (control_run_dir / "judgments.json").exists()
    record = payload["judgments"][0]
    assert record["condition"] == "B2"
    assert record["is_base_rate"] is True
    assert record["item_id"] == "comp_C2_0#f1@comp_B2_0"


def test_judge_prompt_never_names_the_condition(run_dir, control_run_dir, monkeypatch):
    """Condition-blind judging (rubric procedure constraints).

    The same flags are judged against conditioned and unconditioned solutions, so
    a visible condition label would invite expectancy bias in exactly the
    comparison that carries the attribution claim.
    """
    captured = {}
    monkeypatch.setattr(
        judge_run,
        "judge_flags",
        lambda **kwargs: captured.update(kwargs)
        or [FlagJudgment(classification="not_acted_on", evidence_quote="", reasoning="r")]
        * len(kwargs["flags"]),
    )
    judge_run.judge_run(run_key="comp_C2_0", solution_run_key="comp_B2_0")
    assert set(captured) == {"flags", "solution", "logs"}  # no run_key, no condition
    for value in (captured["solution"], captured["logs"]):
        assert "comp_B2_0" not in value and "comp_C2_0" not in value


def test_combine_picks_up_base_rate_files(run_dir, control_run_dir, monkeypatch):
    monkeypatch.setattr(
        judge_run,
        "judge_flags",
        lambda **kwargs: [
            FlagJudgment(classification="acted_on", evidence_quote="q", reasoning="r")
        ]
        * len(kwargs["flags"]),
    )
    judge_run.judge_run(run_key="comp_C2_0")
    judge_run.judge_run(run_key="comp_C2_0", solution_run_key="comp_B2_0")
    out = run_dir.parent / "combined.json"
    assert judge_run.combine(run_keys=["comp_C2_0", "comp_B2_0"], out_path=out) == 4
    records = json.loads(out.read_text())
    assert sum(1 for record in records if record["is_base_rate"]) == 2
