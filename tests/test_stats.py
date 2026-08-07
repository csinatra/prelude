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


def test_compare_supports_h1_on_consistent_positive_effect():
    runs = _grid(
        c2_values={"a": [1.0, 1.0], "b": [1.0, 1.0], "c": [1.0, 1.0]},
        b2_values={"a": [0.0, 0.0], "b": [0.0, 0.0], "c": [0.0, 0.0]},
    )
    result = stats.compare(runs=runs, treatment="C2", baseline="B2", metric="score")
    assert result.n_competitions == 3
    assert result.mean_delta == pytest.approx(1.0)
    assert result.interval_excludes_zero
    assert result.direction_fraction == pytest.approx(1.0)
    assert result.supports_h1


def test_compare_rejects_h1_on_noise():
    runs = _grid(
        c2_values={"a": [1.0], "b": [0.0], "c": [1.0], "d": [0.0]},
        b2_values={"a": [0.0], "b": [1.0], "c": [0.0], "d": [1.0]},
    )
    result = stats.compare(runs=runs, treatment="C2", baseline="B2", metric="score")
    assert result.mean_delta == pytest.approx(0.0)
    assert not result.supports_h1


def test_compare_rejects_h1_when_direction_is_split_despite_positive_mean():
    # one huge win, three small losses: positive mean but no majority direction
    runs = _grid(
        c2_values={"a": [10.0], "b": [0.0], "c": [0.0], "d": [0.0]},
        b2_values={"a": [0.0], "b": [1.0], "c": [1.0], "d": [1.0]},
    )
    result = stats.compare(runs=runs, treatment="C2", baseline="B2", metric="score")
    assert result.mean_delta > 0
    assert result.direction_fraction < 0.5
    assert not result.supports_h1


def test_format_result_is_one_line():
    runs = _grid(c2_values={"a": [1.0]}, b2_values={"a": [0.0]})
    line = stats.format_result(result=stats.compare(runs=runs, treatment="C2", baseline="B2", metric="score"))
    assert "\n" not in line
    assert "C2 vs B2" in line
