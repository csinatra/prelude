"""Pre-registered analysis plan (analysis/stats.py) against synthetic fixtures."""

import math

import pytest

from analysis import stats


def _run(*, comp: str, condition: str, seed: int, **metrics) -> dict:
    return {"competition_id": comp, "condition": condition, "seed": seed, **metrics}


def _grid(*, c2_values: dict[str, list[float]], b2_values: dict[str, list[float]]) -> list[dict]:
    runs = []
    for comp, values in c2_values.items():
        runs += [_run(comp=comp, condition="C2", seed=i, score=v) for i, v in enumerate(values)]
    for comp, values in b2_values.items():
        runs += [_run(comp=comp, condition="B2", seed=i, score=v) for i, v in enumerate(values)]
    return runs


def test_paired_deltas_seed_average_within_competition():
    runs = _grid(c2_values={"a": [1.0, 3.0]}, b2_values={"a": [1.0, 1.0]})
    assert stats.paired_deltas(runs=runs, treatment="C2", baseline="B2", metric="score") == {"a": 1.0}


def test_paired_deltas_skips_unpaired_competitions():
    # 'b' has no B2 run; pairing must drop it rather than compare across competitions
    runs = _grid(c2_values={"a": [1.0], "b": [9.0]}, b2_values={"a": [0.0]})
    deltas = stats.paired_deltas(runs=runs, treatment="C2", baseline="B2", metric="score")
    assert set(deltas) == {"a"}


def test_direction_fraction_matches_on_competition_and_seed():
    runs = _grid(c2_values={"a": [1.0, 0.0], "b": [1.0]}, b2_values={"a": [0.0, 1.0], "b": [0.0]})
    # a/seed0 win, a/seed1 loss, b/seed0 win -> 2/3
    assert stats.direction_fraction(
        runs=runs, treatment="C2", baseline="B2", metric="score"
    ) == pytest.approx(2 / 3)


def test_direction_fraction_counts_ties_as_half():
    runs = _grid(c2_values={"a": [1.0, 1.0]}, b2_values={"a": [1.0, 0.0]})
    # tie + win -> 1.5/2
    assert stats.direction_fraction(
        runs=runs, treatment="C2", baseline="B2", metric="score"
    ) == pytest.approx(0.75)


def test_bootstrap_interval_brackets_mean_for_clear_effect():
    deltas = [0.4, 0.5, 0.6, 0.55, 0.45]
    low, high = stats.bootstrap_interval(deltas=deltas)
    assert low > 0.0  # a consistent positive effect must exclude zero
    assert low < sum(deltas) / len(deltas) < high


def test_bootstrap_interval_spans_zero_for_noise():
    deltas = [0.5, -0.5, 0.4, -0.6]
    low, high = stats.bootstrap_interval(deltas=deltas)
    assert low < 0.0 < high


def test_bootstrap_interval_is_deterministic():
    deltas = [0.2, -0.1, 0.4]
    assert stats.bootstrap_interval(deltas=deltas) == stats.bootstrap_interval(deltas=deltas)


def test_bootstrap_interval_degenerate_cases():
    assert all(math.isnan(x) for x in stats.bootstrap_interval(deltas=[]))
    assert stats.bootstrap_interval(deltas=[0.3]) == (0.3, 0.3)


def test_compare_flags_a_positive_signal_on_consistent_positive_effect():
    runs = _grid(
        c2_values={"a": [1.0, 1.0], "b": [1.0, 1.0], "c": [1.0, 1.0]},
        b2_values={"a": [0.0, 0.0], "b": [0.0, 0.0], "c": [0.0, 0.0]},
    )
    result = stats.compare(runs=runs, treatment="C2", baseline="B2", metric="score")
    assert result.n_competitions == 3
    assert result.mean_delta == pytest.approx(1.0)
    assert result.interval_excludes_zero
    assert result.direction_fraction == pytest.approx(1.0)
    assert result.positive_signal


def test_compare_reports_no_signal_on_noise():
    runs = _grid(
        c2_values={"a": [1.0], "b": [0.0], "c": [1.0], "d": [0.0]},
        b2_values={"a": [0.0], "b": [1.0], "c": [0.0], "d": [1.0]},
    )
    result = stats.compare(runs=runs, treatment="C2", baseline="B2", metric="score")
    assert result.mean_delta == pytest.approx(0.0)
    assert not result.positive_signal


def test_compare_reports_no_signal_when_direction_is_split_despite_positive_mean():
    # one huge win, three small losses: positive mean but no majority direction
    runs = _grid(
        c2_values={"a": [10.0], "b": [0.0], "c": [0.0], "d": [0.0]},
        b2_values={"a": [0.0], "b": [1.0], "c": [1.0], "d": [1.0]},
    )
    result = stats.compare(runs=runs, treatment="C2", baseline="B2", metric="score")
    assert result.mean_delta > 0
    assert result.direction_fraction < 0.5
    assert not result.positive_signal


def test_nonsubmission_is_floored_not_dropped_for_any_medal():
    """The pre-registered rule: an agent that submitted nothing did not medal.

    Dropping it instead would erase a failure from the average of whichever
    condition caused it, which is the bias the rule exists to prevent.
    """
    runs = [
        _run(comp="a", condition="C2", seed=0, status="graded", any_medal=1.0),
        # Agent finished, grading never recorded: harness.batch raises when
        # there is no submission, so the run holds no outcome fields.
        _run(comp="a", condition="C2", seed=1, status="agent_run"),
        _run(comp="a", condition="B2", seed=0, status="graded", any_medal=0.0),
        _run(comp="a", condition="B2", seed=1, status="graded", any_medal=0.0),
    ]
    result = stats.compare(runs=runs, treatment="C2", baseline="B2", metric="any_medal")
    # Floored to 0, so C2 averages 0.5 rather than the 1.0 dropping would give.
    assert result.mean_delta == pytest.approx(0.5)
    assert result.n_floored_treatment == 1
    assert result.n_floored_baseline == 0


def test_percentile_has_no_floor_because_it_is_undefined_without_a_score():
    runs = [
        _run(comp="a", condition="C2", seed=0, status="graded", leaderboard_percentile=0.8),
        _run(comp="a", condition="C2", seed=1, status="agent_run"),
        _run(comp="a", condition="B2", seed=0, status="graded", leaderboard_percentile=0.4),
    ]
    result = stats.compare(
        runs=runs, treatment="C2", baseline="B2", metric="leaderboard_percentile"
    )
    assert result.mean_delta == pytest.approx(0.4)
    assert result.n_floored_treatment == 0


def test_abandoned_run_is_not_a_nonsubmission():
    """A parked run is not evidence about its condition, so it must not score 0."""
    parked = _run(comp="a", condition="C2", seed=0, status="agent_run", abandoned=True)
    assert not stats.is_nonsubmission(run=parked)


def test_direction_summary_applies_the_same_floor_as_the_deltas():
    """Both statistics read metrics through one seam, so they cannot disagree."""
    runs = [
        _run(comp="a", condition="C2", seed=0, status="agent_run"),
        _run(comp="a", condition="B2", seed=0, status="graded", any_medal=1.0),
    ]
    # The floored C2 run loses to B2's medal; dropping it would leave no pairs.
    assert stats.direction_fraction(
        runs=runs, treatment="C2", baseline="B2", metric="any_medal"
    ) == pytest.approx(0.0)


def test_format_result_is_one_line():
    runs = _grid(c2_values={"a": [1.0]}, b2_values={"a": [0.0]})
    line = stats.format_result(result=stats.compare(runs=runs, treatment="C2", baseline="B2", metric="score"))
    assert "\n" not in line
    assert "C2 vs B2" in line
