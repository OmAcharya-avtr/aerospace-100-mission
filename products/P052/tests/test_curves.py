"""Sample-efficiency curves, bootstrap bands and the binomial interval."""

from __future__ import annotations

import numpy as np
import pytest

from falsifyloop.curves import (
    DEFAULT_BOOTSTRAP,
    aggregate_curve,
    area_under_curve,
    bootstrap_aggregate_band,
    bootstrap_band,
    clopper_pearson,
    efficiency_curve,
    median_first_violation,
    success_rate,
)

# Four runs on a budget of 10: violations at 2, 5, and none twice.
RUNS: tuple[int | None, ...] = (2, 5, None, None)
BUDGET = 10


def test_efficiency_curve_known_answer() -> None:
    # P(found by n) = 0 for n = 1; 0.25 for n = 2..4; 0.5 for n = 5..10.
    curve = efficiency_curve(RUNS, BUDGET)
    assert curve.shape == (BUDGET,)
    np.testing.assert_allclose(curve[0], 0.0)
    np.testing.assert_allclose(curve[1:4], 0.25)
    np.testing.assert_allclose(curve[4:], 0.5)


def test_efficiency_curve_is_non_decreasing() -> None:
    curve = efficiency_curve((1, 3, 7, None, 9, 2), 12)
    assert np.all(np.diff(curve) >= 0.0)


def test_success_rate_and_auc_known_answers() -> None:
    assert success_rate(RUNS, BUDGET) == pytest.approx(0.5)
    # Mean of [0, .25, .25, .25, .5, .5, .5, .5, .5, .5] = 3.75/10 = 0.375.
    assert area_under_curve(RUNS, BUDGET) == pytest.approx(0.375)


def test_median_first_violation_known_answers() -> None:
    # Half the runs found nothing, so the median of [2, 5, inf, inf] is
    # (5 + inf)/2 = inf, which is reported as None rather than a number.
    assert median_first_violation(RUNS, BUDGET) is None
    # Three of four found something, median of [2, 4, 5, inf] = 4.5.
    assert median_first_violation((2, 4, 5, None), BUDGET) == pytest.approx(4.5)
    # All found something, median of [1, 3, 5] = 3.
    assert median_first_violation((1, 3, 5), BUDGET) == pytest.approx(3.0)


def test_median_returns_none_when_nothing_was_found() -> None:
    assert median_first_violation((None, None), BUDGET) is None


def test_bootstrap_band_brackets_the_curve_and_is_reproducible() -> None:
    curve = efficiency_curve(RUNS, BUDGET)
    lower, upper = bootstrap_band(RUNS, BUDGET, n_boot=500, seed=1)
    again = bootstrap_band(RUNS, BUDGET, n_boot=500, seed=1)
    np.testing.assert_array_equal(lower, again[0])
    np.testing.assert_array_equal(upper, again[1])
    assert np.all(lower <= curve + 1e-12)
    assert np.all(upper >= curve - 1e-12)
    assert lower.shape == upper.shape == (BUDGET,)


def test_bootstrap_band_is_non_decreasing() -> None:
    lower, upper = bootstrap_band((1, 4, 6, None, 8, 2, 9, None), 12, n_boot=400, seed=2)
    assert np.all(np.diff(lower) >= 0.0)
    assert np.all(np.diff(upper) >= 0.0)


def test_bootstrap_band_of_identical_runs_has_zero_width() -> None:
    lower, upper = bootstrap_band((3, 3, 3), BUDGET, n_boot=200, seed=3)
    np.testing.assert_allclose(lower, upper)


def test_a_wider_alpha_gives_a_narrower_band() -> None:
    runs = (1, 4, 6, None, 8, 2, 9, None)
    tight = bootstrap_band(runs, 12, n_boot=800, alpha=0.5 - 1e-6, seed=4)
    loose = bootstrap_band(runs, 12, n_boot=800, alpha=0.01, seed=4)
    assert np.mean(tight[1] - tight[0]) <= np.mean(loose[1] - loose[0])


def test_aggregate_curve_known_answer() -> None:
    # Instance A: found at 1 always -> curve all ones.
    # Instance B: never found -> curve all zeros. Unweighted mean is 0.5.
    curve = aggregate_curve({"a": (1, 1), "b": (None, None)}, BUDGET)
    np.testing.assert_allclose(curve, 0.5)


def test_aggregate_weights_every_instance_equally() -> None:
    # Instance A contributes 6 runs and B contributes 2; the mean of the two
    # curves must still be the average of 1.0 and 0.0.
    curve = aggregate_curve({"a": (1,) * 6, "b": (None, None)}, BUDGET)
    np.testing.assert_allclose(curve, 0.5)


def test_aggregate_band_brackets_the_aggregate_curve() -> None:
    per_instance = {"a": (1, 4, None, 7), "b": (2, None, None, 9)}
    curve = aggregate_curve(per_instance, BUDGET)
    lower, upper = bootstrap_aggregate_band(per_instance, BUDGET, n_boot=500, seed=5)
    assert np.all(lower <= curve + 1e-12)
    assert np.all(upper >= curve - 1e-12)


def test_clopper_pearson_known_answers() -> None:
    # A zero count gives an exactly zero lower limit and 1 - alpha/2 upper
    # limit 1 - (alpha/2)**(1/n): for n = 10, alpha = 0.05 that is
    # 1 - 0.025**0.1 = 0.30850...
    lower, upper = clopper_pearson(0, 10)
    assert lower == 0.0
    assert upper == pytest.approx(1.0 - 0.025 ** (1.0 / 10.0), rel=1e-9)
    # All successes mirrors it.
    lower, upper = clopper_pearson(10, 10)
    assert upper == 1.0
    assert lower == pytest.approx(0.025 ** (1.0 / 10.0), rel=1e-9)


def test_clopper_pearson_brackets_the_point_estimate() -> None:
    for k, n in [(1, 100), (56, 40000), (500, 1000)]:
        lower, upper = clopper_pearson(k, n)
        assert lower <= k / n <= upper


def test_clopper_pearson_narrows_with_more_trials() -> None:
    narrow = clopper_pearson(100, 10000)
    wide = clopper_pearson(10, 1000)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_default_bootstrap_count_is_the_documented_value() -> None:
    assert DEFAULT_BOOTSTRAP == 2000


@pytest.mark.parametrize(
    ("runs", "budget", "match"),
    [
        ((), 10, "at least one run"),
        ((1,), 0, "budget must be at least 1"),
        ((0,), 10, "outside"),
        ((11,), 10, "outside"),
        ((-3,), 10, "outside"),
    ],
)
def test_curve_input_validation(runs, budget, match) -> None:
    with pytest.raises(ValueError, match=match):
        efficiency_curve(runs, budget)


def test_bootstrap_validation() -> None:
    with pytest.raises(ValueError, match="n_boot must be at least"):
        bootstrap_band(RUNS, BUDGET, n_boot=10)
    with pytest.raises(ValueError, match="alpha must lie"):
        bootstrap_band(RUNS, BUDGET, alpha=0.9)
    with pytest.raises(ValueError, match="at least one instance"):
        aggregate_curve({}, BUDGET)
    with pytest.raises(ValueError, match="at least one instance"):
        bootstrap_aggregate_band({}, BUDGET)
    with pytest.raises(ValueError, match="n_boot must be at least"):
        bootstrap_aggregate_band({"a": RUNS}, BUDGET, n_boot=1)
    with pytest.raises(ValueError, match="alpha must lie"):
        bootstrap_aggregate_band({"a": RUNS}, BUDGET, alpha=0.75)


def test_clopper_pearson_validation() -> None:
    with pytest.raises(ValueError, match="trials must be at least 1"):
        clopper_pearson(0, 0)
    with pytest.raises(ValueError, match="successes must lie"):
        clopper_pearson(11, 10)
    with pytest.raises(ValueError, match="successes must lie"):
        clopper_pearson(-1, 10)
    with pytest.raises(ValueError, match="alpha must lie"):
        clopper_pearson(1, 10, alpha=0.0)
