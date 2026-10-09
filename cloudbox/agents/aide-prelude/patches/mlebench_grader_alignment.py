"""Align submission rows to the answers by id in the two mle-bench graders that
score by row position.

Every other Lite-22 grader sorts or merges on id before scoring, directly or via
`competitions/utils.py`. These two do not:

- uw-madison-gi-tract-image-segmentation decodes each submitted RLE with the
  image size from the same row *position* in the answers, so a correct
  submission in another row order decodes at the wrong sizes and is graded
  invalid. Observed 2026-09-11: all 20,400 (id, class) pairs correct, graded
  invalid; realigned, it grades.
- nomad2018-predict-transparent-conductors checks only the row count and hands
  positional Series to sklearn, so a reordered submission is scored against the
  wrong rows with no error at all.

Neither competition states a row order, and Kaggle joins on id. This brings the
two into line with their siblings: grading-side only, identical for every
condition, and a submission already in answers order scores exactly as before.
The one behavioral addition is the siblings' own: nomad2018 now rejects a
submission with no `id` column.

Run against the host checkout (final grading) and inside the aide-prelude image
(the agent's validation endpoint) so the two agree. Idempotent, because the host
checkout persists across provisions; asserts on its anchors so an mle-bench bump
fails setup instead of silently skipping.
"""

from pathlib import Path

import mlebench

MARKER = "# PRELUDE: align by id"

PATCHES = {
    "uw-madison-gi-tract-image-segmentation": (
        '    if not {"id", "class", "predicted"}.issubset(set(submission.columns)):\n'
        '        raise InvalidSubmissionError("Submission must have columns: id, class, predicted")\n',
        f"    {MARKER} (prelude/cloudbox/agents/aide-prelude/patches)\n"
        '    key = ["id", "class"]\n'
        "    if len(submission) != len(answers) or set(map(tuple, submission[key].values)) != set(\n"
        "        map(tuple, answers[key].values)\n"
        "    ):\n"
        '        raise InvalidSubmissionError("Submission (id, class) pairs must match the test set")\n'
        '    submission = answers[key].merge(submission, on=key, how="left")\n',
    ),
    "nomad2018-predict-transparent-conductors": (
        "    if len(submission) != len(answers):\n"
        "        raise InvalidSubmissionError(\n"
        '            f"Expected {len(answers)} rows in submission, got {len(submission)}"\n'
        "        )\n",
        f"    {MARKER} (prelude/cloudbox/agents/aide-prelude/patches)\n"
        '    if "id" not in submission.columns:\n'
        "        raise InvalidSubmissionError(\"Expected 'id' column in submission\")\n"
        '    submission = submission.sort_values("id").reset_index(drop=True)\n'
        '    answers = answers.sort_values("id").reset_index(drop=True)\n'
        '    if (submission["id"].values != answers["id"].values).any():\n'
        '        raise InvalidSubmissionError("Submission and answers should have the same id values")\n',
    ),
}


def main() -> None:
    competitions = Path(mlebench.__file__).parent / "competitions"
    for competition_id, (anchor, addition) in PATCHES.items():
        path = competitions / competition_id / "grade.py"
        src = path.read_text()
        if MARKER in src:
            print(f"mlebench_grader_alignment: {competition_id} already applied")
            continue
        assert src.count(anchor) == 1, f"{competition_id} grader changed — re-check this patch"
        path.write_text(src.replace(anchor, anchor + addition))
        print(f"mlebench_grader_alignment: {competition_id} applied")


if __name__ == "__main__":
    main()
