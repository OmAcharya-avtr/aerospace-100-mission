"""Property-based tests for the algebraic identities the package relies on.

Every falsifying example Hypothesis has found in this repository is pinned as a
named regression test in ``tests/test_regression.py`` with literal inputs,
because ``.hypothesis/`` is gitignored and a counterexample found by one run is
otherwise lost on a fresh clone.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from twincalib.absorption import decompose_residual
from twincalib.benchmark import coverage_fraction
from twincalib.identifiability import analyse, collinearity_index
from twincalib.model import LinearNodeModel, rk4_steps
from twincalib.profile import chi2_threshold
from twincalib.scenarios import (
    LINEAR_THETA_TRUE,
    build_inputs,
    calibration_model,
    design,
    linear_design,
    linear_model,
)
from twincalib.sensitivity import log_sensitivity

FAST = settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
SLOW = settings(max_examples=12, deadline=None)


@pytest.fixture(scope="module")
def baseline_sensitivity():
    d = design("baseline-eclipse")
    return log_sensitivity(
        calibration_model(),
        d.theta_true,
        build_inputs(d),
        d.initial_state,
        d.channels,
        d.sigma_k,
        d.sample_stride,
    )


@FAST
@given(scale=st.floats(min_value=0.2, max_value=5.0))
def test_linear_model_is_invariant_along_its_null_direction(scale):
    # The exact structural statement: T(t) depends on (C, k, q) only through
    # q/k and k/C, so a common scaling changes nothing. The comparison is to a
    # tolerance, not bit-exact, because both sides are Runge-Kutta solutions.
    model = linear_model()
    inputs = linear_design(81, 25.0)
    base = model.observe(LINEAR_THETA_TRUE, inputs, [300.0], [0])
    scaled = model.observe(
        np.asarray(LINEAR_THETA_TRUE, dtype=float) * scale, inputs, [300.0], [0]
    )
    assert float(np.max(np.abs(base - scaled))) < 1e-5


@FAST
@given(
    t_sink=st.floats(min_value=150.0, max_value=350.0),
    t0=st.floats(min_value=150.0, max_value=350.0),
    cap=st.floats(min_value=100.0, max_value=5000.0),
    cond=st.floats(min_value=0.5, max_value=8.0),
    load=st.floats(min_value=1.0, max_value=100.0),
)
def test_linear_closed_form_matches_the_integrator(t_sink, t0, cap, cond, load):
    # The grid is set from the problem's own time constant, twenty steps per
    # tau over ten tau, because the comparison is between an exact solution and
    # a fixed-step scheme and a fixed grid would be testing the step size
    # rather than the closed form. Hypothesis found that at cap = 100 J/K and
    # cond = 2 W/K a 10 s grid is only h/tau = 0.2 and the agreement is
    # 2.9e-06 K. Both that case and the worst case on the time-constant grid are
    # pinned in tests/test_regression.py.
    from twincalib.model import Inputs

    model = LinearNodeModel(t_sink=t_sink)
    tau = cap / cond
    times = np.arange(0.0, 10.0 * tau, tau / 20.0)
    inputs = Inputs(times, np.zeros(times.size), np.zeros(times.size))
    numeric = model.simulate([cap, cond, load], inputs, [t0])[:, 0]
    exact = model.analytic_solution([cap, cond, load], times, t0)
    span = max(float(np.max(np.abs(exact - t_sink))), 1e-6)
    assert float(np.max(np.abs(numeric - exact))) / span < 1e-6


@FAST
@given(
    rate=st.floats(min_value=0.05, max_value=2.0),
    y0=st.floats(min_value=-10.0, max_value=10.0),
)
def test_rk4_matches_the_exponential_closed_form(rate, y0):
    # Hypothesis falsified a 1e-08 bound at rate = 2.0, y0 = 2.0, where the
    # accumulated error over 80 steps of h = 0.025 is 3.995e-08; pinned in
    # tests/test_regression.py. The bound below is what the scheme actually
    # delivers on this grid, not what it would be nice for it to deliver.
    times = np.linspace(0.0, 2.0, 81)
    out = rk4_steps(lambda _t, y: -rate * y, np.array([y0]), times)
    exact = y0 * np.exp(-rate * times)
    assert float(np.max(np.abs(out[:, 0] - exact))) < 1e-6 * (abs(y0) + 1.0)


@FAST
@given(
    weights=st.lists(
        st.floats(min_value=-3.0, max_value=3.0), min_size=6, max_size=6
    )
)
def test_decomposition_fractions_sum_to_one(weights, baseline_sensitivity):
    field = baseline_sensitivity.jacobian_log @ np.asarray(weights, dtype=float)
    assume(float(np.max(np.abs(field))) > 1e-9)
    split = decompose_residual(baseline_sensitivity, field)
    total = split.strong_fraction + split.weak_fraction + split.orthogonal_fraction
    assert total == pytest.approx(1.0, abs=1e-9)
    assert all(
        f >= -1e-12
        for f in (
            split.strong_fraction,
            split.weak_fraction,
            split.orthogonal_fraction,
        )
    )


@FAST
@given(
    weights=st.lists(
        st.floats(min_value=-3.0, max_value=3.0), min_size=6, max_size=6
    )
)
def test_a_parameter_perturbation_has_no_orthogonal_energy(
    weights, baseline_sensitivity
):
    field = baseline_sensitivity.jacobian_log @ np.asarray(weights, dtype=float)
    assume(float(np.max(np.abs(field))) > 1e-6)
    split = decompose_residual(baseline_sensitivity, field, strong_cut=1e-9)
    assert split.orthogonal_fraction < 1e-8


@FAST
@given(scale=st.floats(min_value=1e-3, max_value=1e3))
def test_decomposition_is_scale_invariant(scale, baseline_sensitivity):
    rng = np.random.default_rng(5)
    field = rng.normal(0.0, 1.0, baseline_sensitivity.n_observations)
    a = decompose_residual(baseline_sensitivity, field)
    b = decompose_residual(baseline_sensitivity, field * scale)
    assert b.strong_fraction == pytest.approx(a.strong_fraction, abs=1e-9)
    assert b.weak_fraction == pytest.approx(a.weak_fraction, abs=1e-9)
    assert b.orthogonal_fraction == pytest.approx(a.orthogonal_fraction, abs=1e-9)
    assert b.total_energy == pytest.approx(a.total_energy * scale**2, rel=1e-9)


@FAST
@given(
    cut_a=st.floats(min_value=1e-8, max_value=1e-1),
    cut_b=st.floats(min_value=1e-8, max_value=1e-1),
)
def test_a_larger_cut_moves_energy_from_strong_to_weak(
    cut_a, cut_b, baseline_sensitivity
):
    lo, hi = sorted((cut_a, cut_b))
    assume(hi > lo * 1.5)
    rng = np.random.default_rng(6)
    field = rng.normal(0.0, 1.0, baseline_sensitivity.n_observations)
    a = decompose_residual(baseline_sensitivity, field, strong_cut=lo)
    b = decompose_residual(baseline_sensitivity, field, strong_cut=hi)
    assert b.n_strong <= a.n_strong
    assert b.strong_fraction <= a.strong_fraction + 1e-12
    assert b.weak_fraction >= a.weak_fraction - 1e-12
    assert b.orthogonal_fraction == pytest.approx(a.orthogonal_fraction, abs=1e-9)


@FAST
@given(
    subset=st.lists(
        st.integers(min_value=0, max_value=5), min_size=1, max_size=6, unique=True
    )
)
def test_collinearity_index_is_at_least_one(subset, baseline_sensitivity):
    value = collinearity_index(baseline_sensitivity, sorted(subset))
    assert value >= 1.0 - 1e-9


@FAST
@given(
    subset=st.lists(
        st.integers(min_value=0, max_value=5), min_size=2, max_size=5, unique=True
    )
)
def test_adding_a_parameter_cannot_reduce_the_collinearity_index(
    subset, baseline_sensitivity
):
    chosen = sorted(subset)
    missing = [j for j in range(6) if j not in chosen]
    assume(missing)
    smaller = collinearity_index(baseline_sensitivity, chosen[:-1])
    larger = collinearity_index(baseline_sensitivity, chosen)
    # The index is 1/s_min of a column-normalised matrix; adding a column can
    # only lower the smallest singular value, so the index cannot fall.
    assert larger >= smaller * (1.0 - 1e-9)


@FAST
@given(
    rank_tol=st.floats(min_value=1e-12, max_value=1e-4),
)
def test_rank_never_exceeds_the_parameter_count(rank_tol, baseline_sensitivity):
    report = analyse(baseline_sensitivity, rank_tol=rank_tol)
    assert 0 <= report.rank <= report.n_parameters
    assert report.condition_number >= 1.0 - 1e-9


@FAST
@given(level=st.floats(min_value=0.01, max_value=0.999))
def test_chi2_threshold_is_increasing_in_the_level(level):
    assume(level < 0.99)
    assert chi2_threshold(level) < chi2_threshold(level + 0.001)
    assert chi2_threshold(level) > 0.0


@FAST
@given(
    errors=st.lists(
        st.floats(min_value=-10.0, max_value=10.0), min_size=1, max_size=60
    ),
    z=st.floats(min_value=0.1, max_value=4.0),
)
def test_coverage_fraction_is_a_fraction(errors, z):
    err = np.asarray(errors, dtype=float)
    std = np.ones(err.size)
    value = coverage_fraction(err, std, z=z)
    assert 0.0 <= value <= 1.0
    bigger = coverage_fraction(err, std, z=z * 2.0)
    assert bigger >= value


@SLOW
@given(
    factor=st.floats(min_value=0.6, max_value=1.6),
)
def test_full_sun_remains_rank_deficient_at_nearby_parameter_points(factor):
    # Structural non-identifiability is a property of the design and the model,
    # not of one parameter point: the rank deficiency must survive moving every
    # parameter by up to 60 per cent.
    d = design("full-sun")
    theta = np.asarray(d.theta_true, dtype=float) * factor
    sens = log_sensitivity(
        calibration_model(),
        theta,
        build_inputs(d),
        d.initial_state,
        d.channels,
        d.sigma_k,
        d.sample_stride,
    )
    report = analyse(sens)
    assert report.rank == 5
    assert not report.structurally_identifiable
