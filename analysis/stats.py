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

A positive signal is all three agreeing: positive mean delta, interval excluding
zero, direction summary above one half. Read together as a direction worth
powering at a larger n, NOT as a pass/fail verdict on H1 (revised 2026-08-31 —
the plan previously adjudicated, which over-formalized what 3-5 competitions can
decide).

Missing data is a policy, not an accident. A run whose agent finished but which
never produced a gradeable submission stops at status `agent_run`, because
harness.batch raises rather than recording a grade, so it carries no outcome
fields at all. Skipping those silently would make whichever condition caused
more of them look better, so metrics in NONSUBMISSION_FLOOR are floored rather
than dropped, and every result reports how many were floored on each side.

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

# Metrics where a non-submission is a real value rather than absent data, mapped
# to the value it takes. Any-Medal only: a run that submitted nothing did not
# medal, so 0 is the honest entry. Leaderboard percentile is deliberately absent
# — it is undefined without a score, and inventing a floor there would fabricate
# a placement. Everything else is skipped as before.
NONSUBMISSION_FLOOR = {"any_medal": 0.0}


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
    # Non-submissions entering at the floor, per side. Reported so the
    # denominator behind a delta is always visible.
    n_floored_treatment: int = 0
    n_floored_baseline: int = 0

    @property
    def interval_excludes_zero(self) -> bool:
        return self.ci_low > 0.0 or self.ci_high < 0.0

    @property
    def positive_signal(self) -> bool:
        """All three pre-registered indicators agree (positive, interval clear, majority).

        A direction worth powering at a larger n, not a verdict that H1 passed.
        The distinction is the point of the 2026-08-31 revision: at 3-5
        competitions a pass/fail bar claims more than the design can support.
        """
        return (
            self.mean_delta > 0.0
            and self.interval_excludes_zero
            and self.direction_fraction > 0.5
        )


def is_nonsubmission(*, run: dict) -> bool:
    """The agent finished and no grade was ever recorded.

    harness.batch raises rather than recording a grade when there is no
    submission to grade, so such a run stops at `agent_run` holding no outcome
    fields. An abandoned run is excluded: it was parked deliberately and is not
    evidence about the condition.
    """
    return run.get("status") == "agent_run" and not run.get("abandoned")


def _value(*, run: dict, metric: str) -> float | None:
    """One run's value for a metric under the pre-registered missing-data rule.

    The single seam through which every statistic reads a metric, so the paired
    deltas and the direction summary always agree on which runs count.
    """
    value = run.get(metric)
    if value is not None:
        return float(value)
    if metric in NONSUBMISSION_FLOOR and is_nonsubmission(run=run):
        return NONSUBMISSION_FLOOR[metric]
    return None


def _by_competition(
    *, runs: list[dict], condition: str, metric: str
) -> tuple[dict[str, list[float]], int]:
    """Metric values per competition for one condition, plus the floored count.

    Missing values are skipped, EXCEPT non-submissions on a NONSUBMISSION_FLOOR
    metric, which enter at the floor. See the module docstring for why dropping
    them is not neutral.
    """
    grouped: dict[str, list[float]] = {}
    floored = 0
    for run in runs:
        if run.get("condition") != condition:
            continue
        value = _value(run=run, metric=metric)
        if value is None:
            continue
        if run.get(metric) is None:
            floored += 1
        grouped.setdefault(run["competition_id"], []).append(value)
    return grouped, floored


def paired_deltas(
    *, runs: list[dict], treatment: str, baseline: str, metric: str
) -> dict[str, float]:
    """Per-competition delta (treatment minus baseline), seed-averaged within competition.

    Only competitions where BOTH conditions have at least one run contribute —
    an unpaired competition would reintroduce the between-competition variance
    the pairing exists to remove.
    """
    treated, _ = _by_competition(runs=runs, condition=treatment, metric=metric)
    control, _ = _by_competition(runs=runs, condition=baseline, metric=metric)
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
        (run["competition_id"], run["seed"], run["condition"]): _value(run=run, metric=metric)
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
    _, floored_treatment = _by_competition(runs=runs, condition=treatment, metric=metric)
    _, floored_baseline = _by_competition(runs=runs, condition=baseline, metric=metric)
    pairs = {
        (run["competition_id"], run["seed"])
        for run in runs
        if run.get("condition") == treatment and _value(run=run, metric=metric) is not None
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
        n_floored_treatment=floored_treatment,
        n_floored_baseline=floored_baseline,
    )


def format_result(*, result: PairedResult) -> str:
    """One-line readout for the writeup tables."""
    floored = result.n_floored_treatment + result.n_floored_baseline
    return (
        f"{result.metric}: {result.treatment} vs {result.baseline} | "
        f"delta={result.mean_delta:+.3f} "
        f"[{int(CI_LEVEL * 100)}% CI {result.ci_low:+.3f}, {result.ci_high:+.3f}] | "
        f"direction={result.direction_fraction:.2f} | "
        f"n_comp={result.n_competitions} n_pairs={result.n_pairs}"
        # Only when non-zero: a floored run is an exception worth seeing, and a
        # constant "floored=0" would train the reader to skip past it.
        + (
            f" | floored={result.n_floored_treatment}/{result.n_floored_baseline}"
            if floored
            else ""
        )
    )
