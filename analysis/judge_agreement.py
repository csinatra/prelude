"""Human-anchored validation of the LLM judge.

The evaluation pipeline is otherwise a closed loop: an LLM writes the spec, an
LLM agent acts on it, and an LLM judges whether it acted. Nothing in that chain
touches an outside reference, so more runs cannot establish that the judge's
classifications mean what the rubric says they mean. Only a human anchor can.

Protocol and reporting live in docs/JUDGE_VALIDATION.md. This module is the
mechanism:

    export  -> writes a BLINDED review file (flag + artifacts, no LLM label)
    score   -> reads the filled-in file back, computes agreement vs the LLM

Blinding matters. The reviewer must not see the LLM's classification (which
would anchor the judgment) or the run's score (which the rubric forbids the
judge itself from seeing). The export therefore carries neither.

Sampling is stratified over (category, LLM classification) so both classes
appear even when one is rare.

**Read kappa beside the marginals, not alone.** With two classes and skewed
marginals — most flags landing `not_acted_on` is the expected case — chance
agreement is high, so kappa can read low while raters agree on nearly every
item. That is the prevalence paradox, not a weak judge. `Agreement` therefore
reports percent agreement, the class balance, and PABAK alongside kappa, and a
low kappa with high agreement and lopsided marginals must be reported as such
rather than as a validation failure.

No API cost: this reads existing judgments and writes files.

Usage:
    python -m analysis.judge_agreement export --judgments PATH --out review.md
    python -m analysis.judge_agreement score  --review review.md
"""

import argparse
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

CLASSES = ("not_acted_on", "acted_on")
DEFAULT_SAMPLE_SIZE = 24
SAMPLE_SEED = 0

_LABEL_RE = re.compile(r"^-\s*human:\s*(\S+)", re.MULTILINE)
_ID_RE = re.compile(r"^##\s+item\s+(\S+)", re.MULTILINE)


@dataclass(frozen=True)
class Agreement:
    n: int
    percent_agreement: float
    cohens_kappa: float
    confusion: dict[tuple[str, str], int]
    acted_on_prevalence: float

    @property
    def pabak(self) -> float:
        """Prevalence-adjusted bias-adjusted kappa: 2 * observed - 1.

        Reported beside kappa because it depends only on how often the raters
        agree, not on the class balance. A large gap between the two is the
        signature of the prevalence paradox and is the thing to report, since
        kappa alone would understate agreement on a lopsided sample.
        """
        return 2.0 * self.percent_agreement - 1.0

    def summary(self) -> str:
        return (
            f"n={self.n} | percent agreement={self.percent_agreement:.1%} | "
            f"Cohen's kappa={self.cohens_kappa:.3f} | PABAK={self.pabak:.3f} | "
            f"acted_on prevalence={self.acted_on_prevalence:.1%}"
        )


def stratified_sample(
    *, judged: list[dict], size: int = DEFAULT_SAMPLE_SIZE, seed: int = SAMPLE_SEED
) -> list[dict]:
    """Sample across (category, classification, conditioned?) strata, round-robin.

    Round-robin rather than proportional allocation: the point is to cover the
    rare classes, and a proportional sample of a skewed judgment distribution
    would return almost entirely one class.

    Base-rate items are a third stratum dimension so the anchor validates the
    judge on both sides of the comparison H2 rests on. It is a boolean rather
    than the condition itself, keeping the split coarse: the sample is ~24 items
    and stratifying on four conditions would leave cells too thin to cover the
    classes. Condition-blind judging is what lets the validated instrument be
    applied to conditions the anchor did not separately cover.
    """
    strata: dict[tuple[str, str, bool], list[dict]] = defaultdict(list)
    class_counts: dict[str, int] = defaultdict(int)
    for item in judged:
        strata[
            (item["category"], item["classification"], bool(item.get("is_base_rate")))
        ].append(item)
        class_counts[item["classification"]] += 1

    rng = random.Random(seed)
    for items in strata.values():
        rng.shuffle(items)

    sampled: list[dict] = []
    # Rarest classification first, so a sample smaller than the number of strata
    # still covers every class. Key breaks ties for determinism.
    keys = sorted(strata, key=lambda key: (class_counts[key[1]], key))
    while len(sampled) < size and any(strata[key] for key in keys):
        for key in keys:
            if strata[key] and len(sampled) < size:
                sampled.append(strata[key].pop())
    return sampled


def render_review_file(*, sample: list[dict]) -> str:
    """Blinded review document: flag + artifacts only, no LLM label, no score."""
    lines = [
        "# Judge validation review (blinded)",
        "",
        "For each item below, classify the flag against `docs/JUDGE_RUBRIC.md` and",
        "write your label on the `- human:` line. Valid labels:",
        "",
        f"  {', '.join(CLASSES)}",
        "",
        "The LLM judge's classification and the run's score are deliberately not shown.",
        "Do not look them up before finishing; anchoring defeats the purpose.",
        "",
    ]
    for item in sample:
        lines += [
            f"## item {item['item_id']}",
            "",
            f"**Flag category:** {item['category']}",
            "",
            f"**Flag explanation:** {item['explanation']}",
            "",
            "**Solution artifact (excerpt):**",
            "",
            "```python",
            item.get("solution_excerpt", "(none)"),
            "```",
            "",
            "- human: ",
            "",
        ]
    return "\n".join(lines)


def parse_review_file(*, text: str) -> dict[str, str]:
    """Read {item_id: human_label} back out of the filled-in review file."""
    item_ids = _ID_RE.findall(text)
    labels = [match.strip() for match in _LABEL_RE.findall(text)]
    parsed = {
        item_id: label
        for item_id, label in zip(item_ids, labels)
        if label and label in CLASSES
    }
    return parsed


def cohens_kappa(*, pairs: list[tuple[str, str]]) -> float:
    """Chance-corrected agreement over the three rubric classes.

    Returns 1.0 for perfect agreement on a single class (where the standard
    formula is 0/0) since raters agreeing on everything is not chance.
    """
    if not pairs:
        return float("nan")
    n = len(pairs)
    observed = sum(1 for left, right in pairs if left == right) / n
    expected = 0.0
    for label in CLASSES:
        p_left = sum(1 for left, _ in pairs if left == label) / n
        p_right = sum(1 for _, right in pairs if right == label) / n
        expected += p_left * p_right
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def score_agreement(*, judged: list[dict], human: dict[str, str]) -> Agreement:
    """Compare human labels against the LLM judge's on the same items."""
    llm_by_id = {item["item_id"]: item["classification"] for item in judged}
    pairs = [
        (llm_by_id[item_id], label) for item_id, label in human.items() if item_id in llm_by_id
    ]
    confusion: dict[tuple[str, str], int] = defaultdict(int)
    for pair in pairs:
        confusion[pair] += 1
    agreed = sum(1 for left, right in pairs if left == right)
    # Prevalence over both raters' labels, not just the judge's: it describes the
    # sample kappa was computed on, and either rater's skew inflates chance
    # agreement.
    labels = [label for pair in pairs for label in pair]
    return Agreement(
        n=len(pairs),
        percent_agreement=agreed / len(pairs) if pairs else float("nan"),
        cohens_kappa=cohens_kappa(pairs=pairs),
        confusion=dict(confusion),
        acted_on_prevalence=(
            sum(1 for label in labels if label == "acted_on") / len(labels)
            if labels
            else float("nan")
        ),
    )


def _load_judged(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def _chain_nodes(sample: list[dict]) -> dict[str, list[dict]]:
    """Trajectory nodes per solution for the HTML page, rebuilt from the same
    seam the judge's bundle came from — so the page shows the judge's evidence,
    not a second selection of it.

    Keyed by the SOLUTION's run, not the flag's: a base-rate item's flags come
    from a C2 run but were judged against a control run's code, and keying by
    the flag's run would pair it with the wrong trajectory."""
    from analysis.artifacts import run_root
    from analysis.judge_review_page import solution_key

    from analysis.judge_run import chain_nodes

    runs = {solution_key(item=item) for item in sample}
    nodes: dict[str, list[dict]] = {}
    for run_key in runs:
        run_dir = run_root() / run_key
        solution_path = run_dir / "best_solution.py"
        solution = solution_path.read_text() if solution_path.is_file() else ""
        nodes[run_key] = chain_nodes(run_dir=run_dir, solution=solution)
    return nodes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="write the blinded review file")
    export_parser.add_argument("--judgments", required=True, help="JSON list of judged flags")
    export_parser.add_argument("--out", default="results/judge_review.md")
    export_parser.add_argument("--size", type=int, default=DEFAULT_SAMPLE_SIZE)

    score_parser = subparsers.add_parser("score", help="score a filled-in review file")
    score_parser.add_argument("--judgments", required=True)
    score_parser.add_argument("--review", required=True)

    args = parser.parse_args()
    judged = _load_judged(Path(args.judgments))

    if args.command == "export":
        sample = stratified_sample(judged=judged, size=args.size)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix == ".html":
            from analysis.judge_review_page import render_review_page

            from analysis.judge import RUBRIC_PATH

            out.write_text(
                render_review_page(
                    sample=sample,
                    nodes_by_run=_chain_nodes(sample),
                    rubric_text=RUBRIC_PATH.read_text(),
                )
            )
        else:
            out.write_text(render_review_file(sample=sample))
        print(f"wrote {len(sample)} blinded items to {out}")
    else:
        review_path = Path(args.review)
        if review_path.suffix == ".json":
            from analysis.judge_review_page import parse_labels

            human = parse_labels(text=review_path.read_text())
        else:
            human = parse_review_file(text=review_path.read_text())
        result = score_agreement(judged=judged, human=human)
        print(result.summary())
        for (llm_label, human_label), count in sorted(result.confusion.items()):
            marker = "  " if llm_label == human_label else "x "
            print(f"  {marker}llm={llm_label:<18} human={human_label:<18} {count}")
