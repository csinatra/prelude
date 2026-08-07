"""Pre-registered statistical analysis for cross-condition comparison.

Implements the analysis plan in docs/RESEARCH_DESIGN.md (Outcome metrics,
pre-registered 2026-08-07). The plan is fixed before any eval run so that H1's
"beyond seed noise" has an operational meaning committed in advance.

The comparison is PAIRED per competition. Between-competition variance on
MLE-bench is far larger than the effect under test, so at n=3-5 competitions an
unpaired comparison is unreadable whatever the true effect. Pairing removes that
variance; it does not remove within-competition seed variance, which the
bootstrap resamples.

Three numbers are reported per contrast:
  - mean paired delta across competitions (the effect estimate)
  - a 90% bootstrap interval (90 rather than 95 is a deliberate POC choice: at
    this n a 95% interval is nearly certain to span zero and carry no signal)
  - the direction summary, the fraction of competition-seed pairs where the
    treatment beats the baseline, which does not depend on medal thresholds

H1 is supported only if all three agree: positive mean delta, interval
excluding zero, direction summary above one half.

numpy only, no scipy: the bootstrap is a plain percentile bootstrap and adding a
dependency for it is not worth the install surface.
"""

from dataclasses import dataclass, field

import numpy as np

# Pre-registered: 90% interval, 10k resamples, fixed seed so a reported interval
# is reproducible from the same registry rows.
CI_LEVEL = 0.90
N_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0


@dataclass(frozen=True)
class PairedResult:
    """One contrast (treatment vs baseline) under the pre-registered plan."""

    metric: str
    treatment: str
    baseline: str
    n_competitions: int
    n_pairs: int
    mean_delta: float
    ci_low: float
    ci_high: float
    direction_fraction: float
    per_competition: dict[str, float] = field(default_factory=dict)

    @property
    def interval_excludes_zero(self) -> bool:
        return self.ci_low > 0.0 or self.ci_high < 0.0

    @property
    def supports_h1(self) -> bool:
        """All three pre-registered conditions met (positive, interval clear, majority)."""
        return (
            self.mean_delta > 0.0
            and self.interval_excludes_zero
            and self.direction_fraction > 0.5
        )


def _by_competition(
    *, runs: list[dict], condition: str, metric: str
) -> dict[str, list[float]]:
    """Metric values per competition for one condition, skipping missing values."""
    grouped: dict[str, list[float]] = {}
    for run in runs:
        if run.get("condition") != condition:
            continue
        value = run.get(metric)
        if value is None:
            continue
        grouped.setdefault(run["competition_id"], []).append(float(value))
    return grouped


def paired_deltas(
    *, runs: list[dict], treatment: str, baseline: str, metric: str
) -> dict[str, float]:
    """Per-competition delta (treatment minus baseline), seed-averaged within competition.

    Only competitions where BOTH conditions have at least one run contribute —
    an unpaired competition would reintroduce the between-competition variance
    the pairing exists to remove.
    """
    treated = _by_competition(runs=runs, condition=treatment, metric=metric)
    control = _by_competition(runs=runs, condition=baseline, metric=metric)
    return {
        competition: float(np.mean(treated[competition]) - np.mean(control[competition]))
        for competition in sorted(set(treated) & set(control))
    }


def direction_fraction(
    *, runs: list[dict], treatment: str, baseline: str, metric: str
) -> float:
    """Fraction of competition-seed pairs where treatment beats baseline.

    Pairs are matched on (competition, seed) so the comparison stays within a
    competition. Ties count as half, the usual sign-test convention, so a metric
    that is frequently equal (e.g. a binary medal at 0) is not scored as a win.
    """
    keyed = {
        (run["competition_id"], run["seed"], run["condition"]): run.get(metric)
        for run in runs
    }
    wins = 0.0
    total = 0
    for (competition, seed, condition), value in keyed.items():
        if condition != treatment or value is None:
            continue
        control = keyed.get((competition, seed, baseline))
        if control is None:
            continue
        total += 1
        if float(value) > float(control):
            wins += 1.0
        elif float(value) == float(control):
            wins += 0.5
    return wins / total if total else float("nan")


def bootstrap_interval(
    *,
    deltas: list[float],
    level: float = CI_LEVEL,
    n_resamples: int = N_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Percentile bootstrap over per-competition paired deltas.

    Resamples competitions with replacement. Seed-level variance is already
    folded into each competition's delta by seed-averaging upstream.
    """
    if not deltas:
        return (float("nan"), float("nan"))
    if len(deltas) == 1:
        return (deltas[0], deltas[0])
    rng = np.random.default_rng(seed)
    sample = np.asarray(deltas, dtype=float)
    draws = rng.choice(sample, size=(n_resamples, sample.size), replace=True)
    means = draws.mean(axis=1)
    tail = (1.0 - level) / 2.0
    return (
        float(np.quantile(means, tail)),
        float(np.quantile(means, 1.0 - tail)),
    )


def compare(
    *, runs: list[dict], treatment: str, baseline: str, metric: str
) -> PairedResult:
    """Run the full pre-registered contrast for one metric."""
    deltas = paired_deltas(
        runs=runs, treatment=treatment, baseline=baseline, metric=metric
    )
    values = list(deltas.values())
    ci_low, ci_high = bootstrap_interval(deltas=values)
    pairs = {
        (run["competition_id"], run["seed"])
        for run in runs
        if run.get("condition") == treatment and run.get(metric) is not None
    }
    return PairedResult(
        metric=metric,
        treatment=treatment,
        baseline=baseline,
        n_competitions=len(values),
        n_pairs=len(pairs),
        mean_delta=float(np.mean(values)) if values else float("nan"),
        ci_low=ci_low,
        ci_high=ci_high,
        direction_fraction=direction_fraction(
            runs=runs, treatment=treatment, baseline=baseline, metric=metric
        ),
        per_competition=deltas,
    )


def format_result(*, result: PairedResult) -> str:
    """One-line readout for the writeup tables."""
    return (
        f"{result.metric}: {result.treatment} vs {result.baseline} | "
        f"delta={result.mean_delta:+.3f} "
        f"[{int(CI_LEVEL * 100)}% CI {result.ci_low:+.3f}, {result.ci_high:+.3f}] | "
        f"direction={result.direction_fraction:.2f} | "
        f"n_comp={result.n_competitions} n_pairs={result.n_pairs} | "
        f"H1={'supported' if result.supports_h1 else 'not supported'}"
    )
