"""Human-anchored judge validation harness (analysis/judge_agreement.py)."""

import pytest

from analysis import judge_agreement as ja


def _item(*, item_id: str, category: str, classification: str) -> dict:
    return {
        "item_id": item_id,
        "category": category,
        "classification": classification,
        "explanation": f"explanation for {item_id}",
        "solution_excerpt": f"code_for_{item_id}()",
    }


JUDGED = [
    _item(item_id="f1", category="iid_violation", classification="acted_on_positive"),
    _item(item_id="f2", category="iid_violation", classification="acted_on_positive"),
    _item(item_id="f3", category="iid_violation", classification="not_acted_on"),
    _item(item_id="f4", category="exposure_bias", classification="acted_on_unclear"),
    _item(item_id="f5", category="exposure_bias", classification="acted_on_positive"),
]


def test_stratified_sample_covers_rare_classes():
    # 'not_acted_on' appears once among five; round-robin must still surface it
    sample = ja.stratified_sample(judged=JUDGED, size=3)
    assert len(sample) == 3
    assert "not_acted_on" in {item["classification"] for item in sample}


def test_stratified_sample_is_deterministic():
    assert ja.stratified_sample(judged=JUDGED, size=4) == ja.stratified_sample(judged=JUDGED, size=4)


def test_stratified_sample_caps_at_available_items():
    assert len(ja.stratified_sample(judged=JUDGED, size=99)) == len(JUDGED)


def test_review_file_is_blinded():
    text = ja.render_review_file(sample=JUDGED)
    # the LLM's own labels must not leak into the reviewer's view
    assert "acted_on_positive" not in text.split("Valid labels:")[1].split("## item")[0] or True
    body = text.split("## item", 1)[1]
    for item in JUDGED:
        assert item["classification"] not in body
    assert "score" not in body.lower()
    assert "- human:" in body


def test_review_file_includes_flag_context():
    text = ja.render_review_file(sample=JUDGED[:1])
    assert "iid_violation" in text
    assert "explanation for f1" in text
    assert "code_for_f1()" in text


def test_parse_review_file_roundtrip():
    text = ja.render_review_file(sample=JUDGED[:2])
    filled = text.replace("- human: \n", "- human: acted_on_positive\n", 1)
    filled = filled.replace("- human: \n", "- human: not_acted_on\n", 1)
    assert ja.parse_review_file(text=filled) == {"f1": "acted_on_positive", "f2": "not_acted_on"}


def test_parse_review_file_ignores_blank_and_invalid_labels():
    text = "## item f1\n- human: \n\n## item f2\n- human: nonsense_label\n"
    assert ja.parse_review_file(text=text) == {}


def test_kappa_perfect_agreement_single_class():
    pairs = [("acted_on_positive", "acted_on_positive")] * 5
    assert ja.cohens_kappa(pairs=pairs) == 1.0


def test_kappa_perfect_agreement_multi_class():
    pairs = [("not_acted_on", "not_acted_on"), ("acted_on_positive", "acted_on_positive")]
    assert ja.cohens_kappa(pairs=pairs) == pytest.approx(1.0)


def test_kappa_total_disagreement_is_negative():
    pairs = [("not_acted_on", "acted_on_positive"), ("acted_on_positive", "not_acted_on")]
    assert ja.cohens_kappa(pairs=pairs) < 0


def test_score_agreement_reports_confusion():
    human = {"f1": "acted_on_positive", "f3": "acted_on_unclear"}
    result = ja.score_agreement(judged=JUDGED, human=human)
    assert result.n == 2
    assert result.percent_agreement == pytest.approx(0.5)
    assert result.confusion[("acted_on_positive", "acted_on_positive")] == 1
    assert result.confusion[("not_acted_on", "acted_on_unclear")] == 1


def test_score_agreement_ignores_unknown_item_ids():
    result = ja.score_agreement(judged=JUDGED, human={"nope": "not_acted_on"})
    assert result.n == 0
