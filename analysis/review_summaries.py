"""Print-only QA of the notebook-summary prompt on a task-diverse sample.

Picks one substantive notebook per competition across NLP / tabular / vision, runs
the production summarizer (`ingest.ingest_summaries._summarize` — current
`SUMMARY_SYSTEM`, pinned Haiku), and prints each abstract with word count + format
checks (stray markdown? truncated mid-sentence?). Use it to eyeball prompt changes
before a full re-ingest — it writes nothing to the corpus.

Run: PYTHONPATH=. .venv/bin/python -m analysis.review_summaries
(needs ANTHROPIC_API_KEY; source .env first).
"""
from ingest.config import DEFAULT_SCOPE
from ingest.ingest_summaries import MAX_NOTEBOOK_CHARS, _load_notebooks, _summarize

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


def main() -> None:
    notebooks = _load_notebooks(scope=DEFAULT_SCOPE, scored_only=False)
    by_comp: dict[str, list[tuple[int, dict]]] = {}
    for kid, entry in notebooks.items():
        total = entry["chars"]
        if entry["competition_id"] in TARGETS and 8_000 <= total <= MAX_NOTEBOOK_CHARS and len(entry["blocks"]) >= 6:
            by_comp.setdefault(entry["competition_id"], []).append((kid, entry))

    for comp in TARGETS:
        cands = by_comp.get(comp)
        if not cands:
            print(f"(no in-range notebook for {comp})\n")
            continue
        cands.sort(key=lambda x: (x[1]["kaggle_score"] or -1, len(x[1]["blocks"])), reverse=True)
        kid, entry = cands[0]
        _, abstract = _summarize(kaggle_id=kid, entry=entry)
        ends_clean = abstract.rstrip()[-1] in ".!?\"']"
        print("=" * 92)
        print(f"{comp} | nb {kid} | {len(abstract.split())} words | "
              f"markdown={_has_markdown(abstract)} | ends_clean={ends_clean}")
        print("-" * 92)
        print(abstract)
        print()


if __name__ == "__main__":
    main()
