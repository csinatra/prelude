"""Print-only QA of the notebook-summary prompt on a task-diverse sample.

Runs the production summarizer (`ingest.ingest_summaries._summarize` — current
`SUMMARY_SYSTEM`, pinned Haiku) over a sample and prints each abstract with word
count and format checks (stray markdown? truncated mid-sentence?). Writes nothing
to the corpus.

Two samples, for two different questions:

  default   one notebook per Lite-22 competition across NLP / tabular / vision.
            Checks the prompt on competitions we have summarized before.
  --unseen N  N random competitions NOT in Lite-22, drawn from the scored
            full-Code4ML slice. Checks the prompt on the ~558 competitions the
            expansion adds and no prior validation has touched — unfamiliar
            domains, notebook conventions, and possibly non-English text. Run
            this before committing to a full-corpus ingest, where a systematic
            problem would otherwise surface only after the spend.

Run: PYTHONPATH=. .venv/bin/python -m analysis.review_summaries [--unseen 20]
(needs ANTHROPIC_API_KEY; source .env first).
"""
import argparse
import random

from ingest.config import DEFAULT_SCOPE, LITE_COMPETITIONS
from ingest.ingest_summaries import MAX_NOTEBOOK_CHARS, _load_notebooks, _summarize

SEED = 0

TARGETS = [
    "spooky-author-identification",                    # NLP (text classification)
    "jigsaw-toxic-comment-classification-challenge",   # NLP (multi-label)
    "text-normalization-challenge-english-language",   # NLP (sequence)
    "new-york-city-taxi-fare-prediction",              # tabular regression
    "nomad2018-predict-transparent-conductors",        # tabular regression (materials)
    "dogs-vs-cats-redux-kernels-edition",              # vision
    "aerial-cactus-identification",                    # vision
    "leaf-classification",                             # tabular/vision hybrid
]


def _has_markdown(text: str) -> bool:
    stripped = text.lstrip()
    return (
        stripped.startswith("#")
        or "\n#" in text
        or "**" in text
        or "\n- " in text
        or "\n* " in text
        or "\n1." in text
    )


def _in_range(entry: dict) -> bool:
    """Substantive enough to be worth reading: not a fragment, not an outlier."""
    return 8_000 <= entry["chars"] <= MAX_NOTEBOOK_CHARS and len(entry["blocks"]) >= 6


def _by_competition(notebooks: dict[int, dict]) -> dict[str, list]:
    grouped: dict[str, list] = {}
    for kaggle_id, entry in notebooks.items():
        if _in_range(entry):
            grouped.setdefault(entry["competition_id"], []).append((kaggle_id, entry))
    return grouped


def _best(candidates: list) -> tuple[int, dict]:
    """Highest-scoring, then longest — the notebook most likely to have content."""
    return sorted(
        candidates, key=lambda pair: (pair[1]["kaggle_score"] or -1, len(pair[1]["blocks"])), reverse=True
    )[0]


def sample_lite() -> list[tuple[str, int, dict]]:
    grouped = _by_competition(_load_notebooks(scope=DEFAULT_SCOPE, scored_only=False))
    picked = []
    for competition_id in TARGETS:
        candidates = grouped.get(competition_id)
        if not candidates:
            print(f"(no in-range notebook for {competition_id})\n")
            continue
        kaggle_id, entry = _best(candidates)
        picked.append((competition_id, kaggle_id, entry))
    return picked


def sample_unseen(*, n: int) -> list[tuple[str, int, dict]]:
    """One notebook each from n random competitions outside Lite-22."""
    grouped = _by_competition(_load_notebooks(scope="full", scored_only=True))
    lite = set(LITE_COMPETITIONS)
    unseen = sorted(c for c in grouped if c not in lite)
    print(f"{len(unseen)} unseen competitions with an in-range scored notebook; sampling {n}\n")
    picked = []
    for competition_id in random.Random(SEED).sample(unseen, min(n, len(unseen))):
        kaggle_id, entry = _best(grouped[competition_id])
        picked.append((competition_id, kaggle_id, entry))
    return picked


def main(*, unseen: int | None = None) -> None:
    picked = sample_unseen(n=unseen) if unseen else sample_lite()
    for competition_id, kaggle_id, entry in picked:
        _, abstract = _summarize(kaggle_id=kaggle_id, entry=entry)
        ends_clean = abstract.rstrip()[-1] in ".!?\"']"
        print("=" * 92)
        print(
            f"{competition_id} | nb {kaggle_id} | score={entry['kaggle_score']} | "
            f"{len(abstract.split())} words | markdown={_has_markdown(abstract)} | "
            f"ends_clean={ends_clean}"
        )
        print("-" * 92)
        print(abstract)
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--unseen",
        type=int,
        default=None,
        help="sample N competitions outside Lite-22 (the slice the expansion adds)",
    )
    main(unseen=parser.parse_args().unseen)
