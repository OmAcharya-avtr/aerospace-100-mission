"""Property-based tests. Every falsifier Hypothesis has found is pinned as a
literal regression test in ``tests/test_regressions.py``, because
``.hypothesis/`` is gitignored and a found counterexample is otherwise lost on
a fresh clone.
"""

from __future__ import annotations

import math

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from acqsearch import (
    PointingUncertainty,
    PoissonDwellDetector,
    TerminalGeometry,
    cell_visit_order,
    poisson_upper_tail,
    random_restart_path,
    spiral_closed_form_coverage,
    update_posterior,
)

SIGMA = st.floats(min_value=1e-6, max_value=1e-3, allow_nan=False, allow_infinity=False)
RADIUS = st.floats(min_value=0.0, max_value=1e-2, allow_nan=False, allow_infinity=False)
MEAN = st.floats(min_value=0.0, max_value=60.0, allow_nan=False, allow_infinity=False)


@given(sigma=SIGMA, r=RADIUS)
def test_radial_cdf_is_a_probability(sigma, r):
    assert 0.0 <= PointingUncertainty(sigma).radial_cdf(r) <= 1.0


@given(sigma=SIGMA, a=RADIUS, b=RADIUS)
def test_radial_cdf_is_monotone(sigma, a, b):
    lo, hi = sorted((a, b))
    u = PointingUncertainty(sigma)
    assert u.radial_cdf(lo) <= u.radial_cdf(hi) + 1e-15


@given(sigma=SIGMA, x=RADIUS, y=RADIUS, cut=st.floats(min_value=0.0, max_value=1.0))
def test_rectangle_mass_is_additive_under_a_vertical_cut(sigma, x, y, cut):
    assume(x > 0.0 and y > 0.0)
    u = PointingUncertainty(sigma)
    split = -x + cut * (2.0 * x)
    whole = u.rectangle_mass(-x, x, -y, y)
    left = u.rectangle_mass(-x, split, -y, y)
    right = u.rectangle_mass(split, x, -y, y)
    assert math.isclose(left + right, whole, rel_tol=1e-9, abs_tol=1e-15)


@given(k=st.integers(min_value=0, max_value=25), mean=MEAN)
def test_poisson_upper_tail_is_a_probability(k, mean):
    assert 0.0 <= poisson_upper_tail(k, mean) <= 1.0


@given(k=st.integers(min_value=1, max_value=20), a=MEAN, b=MEAN)
def test_poisson_upper_tail_is_monotone_in_the_mean(k, a, b):
    lo, hi = sorted((a, b))
    assert poisson_upper_tail(k, lo) <= poisson_upper_tail(k, hi) + 1e-12


@given(
    signal=st.floats(min_value=0.0, max_value=1e5),
    background=st.floats(min_value=0.0, max_value=1e4),
    k=st.integers(min_value=1, max_value=12),
    rytov=st.floats(min_value=0.0, max_value=0.9),
    dwell=st.floats(min_value=1e-5, max_value=1e-2),
    offset=st.floats(min_value=0.0, max_value=5e-4),
)
@settings(max_examples=60, deadline=None)
def test_detection_probability_never_falls_below_the_false_alarm_floor(
    signal, background, k, rytov, dwell, offset
):
    detector = PoissonDwellDetector(signal, background, k, rytov)
    detected = detector.detection_probability(dwell, offset, 5.0e-5)
    assert detected >= detector.false_alarm_probability(dwell) - 1e-12
    assert 0.0 <= detected <= 1.0


@given(
    mass=st.lists(st.floats(min_value=1e-6, max_value=1.0), min_size=2, max_size=40),
    q=st.floats(min_value=0.0, max_value=0.999),
    index=st.integers(min_value=0, max_value=39),
)
def test_posterior_update_stays_normalised_and_preserves_ratios(mass, q, index):
    posterior = np.asarray(mass, dtype=float)
    posterior = posterior / posterior.sum()
    cell = index % posterior.size
    updated = update_posterior(posterior, cell, q)
    assert math.isclose(float(updated.sum()), 1.0, rel_tol=1e-12)
    others = [i for i in range(posterior.size) if i != cell]
    if len(others) >= 2:
        i, j = others[0], others[1]
        assert math.isclose(
            updated[i] * posterior[j], updated[j] * posterior[i], rel_tol=1e-9, abs_tol=1e-18
        )


@given(
    pitch=st.floats(min_value=1e-6, max_value=2e-4),
    turns=st.integers(min_value=1, max_value=12),
    a=st.floats(min_value=1e-6, max_value=2e-4),
    sigma=st.floats(min_value=2e-5, max_value=5e-4),
)
def test_spiral_closed_form_lies_between_its_bounding_discs(pitch, turns, a, sigma):
    u = PointingUncertainty(sigma)
    value = spiral_closed_form_coverage(pitch, turns, a, u)
    low = u.radial_cdf((turns - 1) * pitch + a)
    high = u.radial_cdf(turns * pitch + a)
    assert low - 1e-9 <= value <= high + 1e-9


@given(
    distance=st.floats(min_value=0.0, max_value=1e-2),
    rate=st.floats(min_value=1e-4, max_value=1.0),
    settle=st.floats(min_value=0.0, max_value=1.0),
)
def test_slew_time_is_non_negative_and_at_least_the_settle_time(distance, rate, settle):
    g = TerminalGeometry(1e-4, 2e-4, rate, settle)
    value = g.slew_time_s(distance)
    assert value >= settle - 1e-15
    assert value >= 0.0


@given(n=st.integers(min_value=1, max_value=200), seed=st.integers(min_value=0, max_value=10**6))
@settings(max_examples=25, deadline=None)
def test_any_random_path_induces_a_permutation_of_the_grid(grid, n, seed):
    order = cell_visit_order(random_restart_path(grid.extent_rad, n, seed), grid)
    assert np.array_equal(np.sort(order), np.arange(grid.n_cells))
