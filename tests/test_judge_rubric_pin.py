"""Mechanical enforcement of the rubric freeze (CLAUDE.md core constraint 7).

docs/JUDGE_RUBRIC.md was written before any evaluation run and must not change
after results exist. "Must not" was until now only a convention in a doc; this
pins it to a hash, so an edit fails the suite instead of passing unnoticed and
silently invalidating every judgment made under the previous text.

A legitimate amendment updates PINNED_SHA256 *in the same commit* as a dated
amendment entry in the rubric, and requires re-judging every run judged under
the old text. Updating this constant to make a red test green, without that, is
the failure this exists to prevent.
"""

import hashlib
from pathlib import Path

RUBRIC_PATH = Path("docs/JUDGE_RUBRIC.md")
PINNED_SHA256 = "6f668c75067098b6e1124943ac6337a604a9bd4979981677a588caf26ecdc745"


def test_rubric_is_unchanged():
    digest = hashlib.sha256(RUBRIC_PATH.read_bytes()).hexdigest()
    assert digest == PINNED_SHA256, (
        "docs/JUDGE_RUBRIC.md changed. The rubric is frozen: amend it only with a "
        "dated amendment entry and re-judging of all prior runs, then update "
        "PINNED_SHA256 in the same commit."
    )


def test_judge_prompt_carries_the_pinned_rubric():
    """The provenance hash must be of the same bytes this test pins."""
    from analysis.judge import judge_provenance

    assert judge_provenance()["rubric_sha256"] == PINNED_SHA256
