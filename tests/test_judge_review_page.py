"""The review page must be blinded, grouped, and round-trip its labels."""

import json

from analysis.judge_review_page import parse_labels, render_review_page

SAMPLE = [
    {
        "item_id": "comp_C2_0#f1",
        "run_key": "comp_C2_0",
        "category": "iid_violation",
        "explanation": "grouped rows leak across folds",
        "classification": "acted_on_positive",
        "evidence_quote": "GroupKFold(",
        "reasoning": "used grouped CV",
        "solution_excerpt": "print('solution')",
    },
    {
        "item_id": "comp_C2_0#f2",
        "run_key": "comp_C2_0",
        "category": "exposure_bias",
        "explanation": "class imbalance",
        "classification": "not_acted_on",
        "evidence_quote": "",
        "reasoning": "nothing found",
        "solution_excerpt": "print('solution')",
    },
    {
        "item_id": "other_C2_1#f1",
        "run_key": "other_C2_1",
        "category": "iid_violation",
        "explanation": "temporal ordering",
        "classification": "acted_on_unclear",
        "evidence_quote": "sort_values",
        "reasoning": "sorted by date",
        "solution_excerpt": "print('other')",
    },
]

NODES = {"comp_C2_0": [{"step": 1, "is_buggy": False, "metric": 0.7, "analysis": "fixed", "term_out": "ok"}]}


def test_page_is_blinded():
    """No judge classification, quote, or reasoning may appear anywhere in the file."""
    page = render_review_page(sample=SAMPLE, nodes_by_run=NODES)
    body = page.split("<main>")[1]
    for item in SAMPLE:
        assert item["reasoning"] not in page
        assert item["evidence_quote"] == "" or item["evidence_quote"] not in page
    # The class names exist as label options; what must not appear is a per-item
    # binding of an item to its judged class.
    for item in SAMPLE:
        assert f"{item['item_id']}'>{item['classification']}" not in body


def test_items_are_grouped_by_run():
    page = render_review_page(sample=SAMPLE, nodes_by_run=NODES)
    assert page.count("class='run'") == 2
    assert "comp_C2_0 — 2 flag(s)" in page
    assert "other_C2_1 — 1 flag(s)" in page


def test_every_item_gets_all_three_label_options():
    page = render_review_page(sample=SAMPLE, nodes_by_run=NODES)
    for item in SAMPLE:
        for name in ("not_acted_on", "acted_on_unclear", "acted_on_positive"):
            assert f"name='{item['item_id']}' value='{name}'" in page


def test_trajectory_nodes_render_and_absence_is_stated():
    page = render_review_page(sample=SAMPLE, nodes_by_run=NODES)
    assert "step 1 · ok · metric=0.7" in page
    assert "(no trajectory preserved)" in page  # other_C2_1 has no nodes


def test_explanations_are_escaped():
    sample = [{**SAMPLE[0], "explanation": "<script>alert(1)</script>"}]
    page = render_review_page(sample=sample)
    assert "<script>alert(1)</script>" not in page.split("<main>")[1]
    assert "&lt;script&gt;" in page


def test_parse_labels_round_trip_and_drops_invalid():
    text = json.dumps(
        {
            "comp_C2_0#f1": {"label": "acted_on_positive", "quote": "GroupKFold("},
            "comp_C2_0#f2": {"label": "typo"},
            "other_C2_1#f1": {"label": ""},
        }
    )
    assert parse_labels(text=text) == {"comp_C2_0#f1": "acted_on_positive"}


def test_acted_on_without_a_quote_is_voided_like_the_llm_judge():
    """Same rule analysis.judge applies to the model, so both raters match."""
    text = json.dumps(
        {
            "a": {"label": "acted_on_positive", "quote": "   "},
            "b": {"label": "acted_on_unclear"},
            "c": {"label": "not_acted_on"},
            "d": {"label": "acted_on_unclear", "quote": "sort_values("},
        }
    )
    assert parse_labels(text=text) == {
        "a": "not_acted_on",
        "b": "not_acted_on",
        "c": "not_acted_on",
        "d": "acted_on_unclear",
    }


def test_bare_string_export_is_treated_as_quoteless():
    text = json.dumps({"a": "acted_on_positive", "b": "not_acted_on"})
    assert parse_labels(text=text) == {"a": "not_acted_on", "b": "not_acted_on"}


def test_every_item_has_a_quote_field():
    page = render_review_page(sample=SAMPLE, nodes_by_run=NODES)
    for item in SAMPLE:
        assert f"data-for='{item['item_id']}'" in page


def test_code_is_highlighted():
    page = render_review_page(sample=SAMPLE, nodes_by_run=NODES)
    assert "class='t-str'" in page  # 'solution' string literal


def test_highlighting_is_byte_faithful():
    """Stripping tags must return the exact source the judge was given."""
    import re

    from analysis.judge_review_page import _highlight

    source = "def f(x=1):  # note\n    return 'a' if x else None\n"
    rendered = _highlight(source=source)
    stripped = re.sub(r"<[^>]+>", "", rendered)
    assert stripped.replace("&#x27;", "'").replace("&quot;", '"') == source


def test_unparseable_code_still_renders():
    """Buggy agent snippets are often truncated mid-statement."""
    from analysis.judge_review_page import _highlight

    assert "def broken(" in _highlight(source="def broken(x:\n    return")


def test_rubric_panel_embeds_the_frozen_text_and_its_hash():
    import hashlib
    from pathlib import Path

    rubric = Path("docs/JUDGE_RUBRIC.md").read_text()
    page = render_review_page(sample=SAMPLE, rubric_text=rubric)
    assert "acted_on_positive" in page and "id='rubric'" in page
    assert hashlib.sha256(rubric.encode()).hexdigest()[:12] in page


def test_rubric_panel_omitted_when_no_rubric_supplied():
    assert "id='rubric'" not in render_review_page(sample=SAMPLE)


def test_node_fields_are_labeled_by_evidentiary_weight():
    """The rubric treats agent self-report and observed output differently."""
    page = render_review_page(sample=SAMPLE, nodes_by_run=NODES)
    assert "agent self-report — not evidence on its own" in page
    assert "observed output" in page


def test_item_order_does_not_track_the_judged_class():
    """stratified_sample orders by (category, classification); position must not leak it."""
    sample = [
        {
            "item_id": f"r_C2_0#{index}",
            "run_key": "r_C2_0",
            "category": "iid_violation",
            "explanation": f"flag {index}",
            "classification": cls,
            "solution_excerpt": "x = 1",
        }
        for index, cls in enumerate(
            ["acted_on_positive"] * 4 + ["acted_on_unclear"] * 4 + ["not_acted_on"] * 4
        )
    ]
    import re

    page = render_review_page(sample=sample)
    rendered = [int(match) for match in re.findall(r"data-item='r_C2_0#(\d+)'", page)]
    assert len(rendered) == 12
    assert rendered != sorted(rendered), "items rendered in sample (class-correlated) order"


def test_rendering_is_deterministic():
    first = render_review_page(sample=SAMPLE, nodes_by_run=NODES)
    second = render_review_page(sample=SAMPLE, nodes_by_run=NODES)
    assert first == second


def test_run_navigator_lists_every_run():
    page = render_review_page(sample=SAMPLE, nodes_by_run=NODES)
    assert "id=\"runs\"" in page
    for run_key in ("comp_C2_0", "other_C2_1"):
        assert f"data-target='{run_key}'" in page
        assert f"data-count='{run_key}'" in page


def test_runs_are_collapsed_by_default():
    """A cross-competition sample must not open every solution at once."""
    page = render_review_page(sample=SAMPLE, nodes_by_run=NODES)
    assert "<details class='run' id=" in page
    assert "class='run' open" not in page
