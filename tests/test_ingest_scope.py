"""Corpus-slice selection — the CSV reader is faked, no data/raw needed.

These cover the two knobs that decide what a full-corpus ingest actually buys:
--scope (which competitions) and --scored-only (drop notebooks with no evidence
of a real submission). Both are load-bearing for cost and for corpus
composition, and neither is checked anywhere else.
"""

import pandas as pd
import pytest

from ingest import ingest_summaries as ing

ROWS = [
    # kaggle_id, data_sources, kaggle_score, code_block
    (1, "spooky-author-identification", 0.42, "import pandas"),
    (1, "spooky-author-identification", 0.42, "model.fit(X, y)"),
    (2, "spooky-author-identification", 0.0, "import numpy"),
    (3, "some-other-competition", 0.91, "import torch"),
    (4, "some-other-competition", None, "print(1)"),
]


@pytest.fixture(autouse=True)
def fake_csv(monkeypatch):
    """Replace the chunked CSV reader with a single in-memory frame."""
    frame = pd.DataFrame(ROWS, columns=["kaggle_id", "data_sources", "kaggle_score", "code_block"])

    def _read_blocks(*, columns):
        yield frame[columns].copy()

    monkeypatch.setattr(ing, "_read_blocks", _read_blocks)


def test_lite_scope_filters_to_allowlist():
    selected = ing._select(scope="lite", scored_only=False)
    assert set(selected) == {1, 2}  # notebooks 3 and 4 are outside Lite-22


def test_full_scope_keeps_every_competition():
    selected = ing._select(scope="full", scored_only=False)
    assert set(selected) == {1, 2, 3, 4}


def test_scored_only_drops_zero_and_null_scores():
    selected = ing._select(scope="full", scored_only=True)
    # 2 scores 0.0 (Code4ML's unscored sentinel), 4 has no score at all
    assert set(selected) == {1, 3}


def test_score_is_max_over_a_notebooks_rows():
    selected = ing._select(scope="full", scored_only=False)
    assert selected[1]["kaggle_score"] == 0.42
    assert selected[4]["kaggle_score"] is None


def test_load_preserves_block_order_within_a_notebook():
    notebooks = ing._load_notebooks(scope="lite", scored_only=False)
    assert notebooks[1]["blocks"] == ["import pandas", "model.fit(X, y)"]


def test_load_stops_appending_past_the_char_cap(monkeypatch):
    """Truncation during load must leave the same prefix as truncating at join time.

    The loader stops reading a notebook's blocks once it is past the cap, which
    only holds if _notebook_text would have cut at the same point.
    """
    monkeypatch.setattr(ing, "MAX_NOTEBOOK_CHARS", 10)
    notebooks = ing._load_notebooks(scope="lite", scored_only=False)
    # "import pandas" alone exceeds 10 chars, so the second block is never read
    assert notebooks[1]["blocks"] == ["import pandas"]
    # and the retained prefix matches what joining every block would have produced
    every_block = "\n\n".join(["import pandas", "model.fit(X, y)"])[:10]
    assert ing._notebook_text(notebooks[1]) == every_block == "import pan"
