"""Pinned counterexamples and frozen numbers. Nothing here is decorative.

``.hypothesis/`` is gitignored, so a falsifying example found by one run is
LOST on a fresh clone. Every counterexample Hypothesis found during this build
is pinned below with its literal inputs, as is every defect found by hand.

The last two tests in this file are the benchmark and regression tests the
medium-class testing minimum asks for: a set of bound values frozen at six
significant figures against a declared sample and seed, and a wall-clock
budget on the full bound stack.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from transfergap import synthetic
from transfergap.bounds import (
    empirical_bernstein_mean_upper,
    hoeffding_mean_upper,
    normal_tolerance_upper,
    wilks_tolerance_upper,
)
from transfergap.exchangeability import trend_statistic
from transfergap.resample import (
    bootstrap_mean_upper,
    conformal_order_index,
    conformal_required_n,
    split_conformal_upper,
)


def test_regression_conformal_required_n_at_alpha_one_tenth():
    """PINNED. A float round trip through the complement returned 10, not 9.

    The first implementation computed ``ceil(1 / (1 - level)) - 1`` with
    ``level = 1 - alpha``. At the literal input ``alpha = 0.10`` that is
    ``1 - (1 - 0.10) = 0.09999999999999998``, whose reciprocal is
    ``10.000000000000002``, whose ceiling is 11, giving 10 where the answer is
    9. The reciprocal is now taken from ``alpha`` directly.
    """
    assert conformal_required_n(0.10) == 9
    assert conformal_order_index(9, 0.10) == 9
    assert conformal_order_index(8, 0.10) == 9


def test_regression_conformal_required_n_at_alpha_one_fifth():
    """PINNED. The same defect at ``alpha = 0.20``, which returned 5 for 4.

    ``1 - (1 - 0.20) = 0.19999999999999996``, reciprocal
    ``5.000000000000001``, ceiling 6, result 5. The answer is 4.
    """
    assert conformal_required_n(0.20) == 4
    assert conformal_order_index(4, 0.20) == 4
    assert conformal_order_index(3, 0.20) == 4


def test_regression_trend_statistic_on_a_constant_sample():
    """PINNED. Position-broken ties made a constant sample look like a drift.

    The first implementation ranked by ``argsort(argsort(x))``. On the literal
    input ``numpy.full(10, 3.0)`` that returns ``[0, 1, ..., 9]`` as the rank
    vector, so the correlation with the index is exactly 1.0 and the statistic
    reports a perfect monotone drift in a sample with no variation at all.
    Average ranks give a constant rank vector, a zero denominator and 0.0.
    """
    assert trend_statistic(np.full(10, 3.0)) == 0.0


def test_regression_trend_statistic_under_an_underflowing_affine_map():
    """PINNED, found by Hypothesis at 300 examples.

    Literal inputs: ``values = [5e-324, 0.0, 0.0]``, ``scale = 0.5``,
    ``shift = 0.0``. ``0.5 * 5e-324`` underflows the denormal minimum to
    exactly 0.0, so the three values become tied, the average ranks go
    constant and the statistic falls from -0.8660254037844387 to 0.0.

    This is pinned as a STATEMENT ABOUT THE PRECONDITION, not as a defect: a
    strictly increasing real map is not strictly increasing in float64 once it
    collapses two representable numbers onto one. The property test in
    ``tests/test_properties.py`` now assumes distinctness is preserved, and
    this test records what happens when it is not.
    """
    values = np.array([5e-324, 0.0, 0.0])
    assert trend_statistic(values) == pytest.approx(
        -0.8660254037844387, rel=1e-13, abs=0.0
    )
    assert 0.5 * 5e-324 == 0.0
    assert trend_statistic(0.5 * values) == 0.0


def test_regression_wilks_at_exactly_the_required_n():
    """PINNED boundary. n = 59 works and n = 58 does not, at p = delta = 0.05.

    ``ln(0.05)/ln(0.95) = 58.40397481431972``, so the boundary sits between
    58 and 59 and a strict-versus-non-strict comparison error would move it.
    """
    gaps = np.linspace(0.1, 9.0, 59)
    assert wilks_tolerance_upper(gaps, 10.0, 0.95, 0.95).upper == pytest.approx(
        9.0, rel=0.0, abs=1e-14
    )
    with pytest.raises(ValueError, match="needs n >= 59"):
        wilks_tolerance_upper(gaps[:58], 10.0, 0.95, 0.95)


def test_regression_split_conformal_at_exactly_nineteen_scores():
    """PINNED boundary. 19 calibration scores is the first n that works at 0.05."""
    gaps = np.linspace(0.0, 9.0, 19)
    bound = split_conformal_upper(gaps, 10.0, 0.95)
    assert bound.upper == pytest.approx(9.0, rel=0.0, abs=1e-14)
    with pytest.raises(ValueError, match="needs n >= 19"):
        split_conformal_upper(gaps[:18], 10.0, 0.95)


def test_benchmark_frozen_bound_values_on_a_declared_sample():
    """BENCHMARK/REGRESSION. Six bound values frozen against a declared sample.

    Sample: ``synthetic.generate(60, "random", seed=202)``, R = 40 kN.m,
    confidence 0.95, proportion 0.95, bootstrap seed 202 with 2000 resamples.
    These are the numbers the implementation produced at build time, to 12
    significant figures, and they are here so that a silent change in a
    quantile rule, a random-number stream or a library default is caught by a
    test rather than discovered in a validation output diff.

    This is the class of defect that P025 of this mission shipped: a suite
    that passes while its validation numbers move. A frozen-number test is
    what catches it.
    """
    sample = synthetic.generate(60, "random", seed=202)
    frozen = {
        "hoeffding": 10.9750920366,
        "empirical_bernstein": 12.0130859561,
        "bootstrap_percentile": 5.55961654509,
        "normal_tolerance": 13.4350590317,
        "split_conformal": 14.2450127814,
        "wilks_tolerance": 25.5521561097,
    }
    measured = {
        "hoeffding": hoeffding_mean_upper(sample.gaps, 40.0, 0.95).upper,
        "empirical_bernstein": empirical_bernstein_mean_upper(sample.gaps, 40.0, 0.95).upper,
        "bootstrap_percentile": bootstrap_mean_upper(
            sample.gaps, 40.0, 0.95, resamples=2000, seed=202
        ).upper,
        "normal_tolerance": normal_tolerance_upper(sample.gaps, 40.0, 0.95, 0.95).upper,
        "split_conformal": split_conformal_upper(sample.gaps, 40.0, 0.95).upper,
        "wilks_tolerance": wilks_tolerance_upper(sample.gaps, 40.0, 0.95, 0.95).upper,
    }
    for name, expected in frozen.items():
        assert measured[name] == pytest.approx(expected, rel=1e-9, abs=0.0), name


def test_benchmark_the_full_bound_stack_is_fast_enough_to_monte_carlo():
    """BENCHMARK. The whole analytic stack on n = 60, 50 times, under 10 seconds.

    The coverage measurements in ``validation/`` run this stack 300 times at
    each of four sample sizes, so a regression that makes one evaluation cost
    a tenth of a second would push those scripts past the three-minute budget
    this product declares. The budget here is deliberately loose -- a factor of
    about 50 above the measured time on two cores -- because a wall-clock
    assertion on shared hardware that is tight is a flaky test, and a flaky
    test is worse than no test.
    """
    sample = synthetic.generate(60, "random", seed=202)
    started = time.perf_counter()
    for _ in range(50):
        hoeffding_mean_upper(sample.gaps, 40.0, 0.95)
        empirical_bernstein_mean_upper(sample.gaps, 40.0, 0.95)
        normal_tolerance_upper(sample.gaps, 40.0, 0.95, 0.95)
        wilks_tolerance_upper(sample.gaps, 40.0, 0.95, 0.95)
        split_conformal_upper(sample.gaps, 40.0, 0.95)
        bootstrap_mean_upper(sample.gaps, 40.0, 0.95, resamples=500, seed=1)
    elapsed = time.perf_counter() - started
    assert elapsed < 10.0, f"the analytic stack took {elapsed:.2f} s for 50 evaluations"
