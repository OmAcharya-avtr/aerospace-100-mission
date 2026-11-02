"""Property-based tests, where an algebraic identity or an inequality exists.

Every property here is a statement that holds for a RANGE of inputs rather
than at a point, which is what Hypothesis is for. Floating-point comparisons
carry an explicit tolerance on both sides, and no boolean produced by a strict
threshold comparison is asserted: a boolean is a step function of a rounded
quantity and is the one thing floating point cannot promise. Where a
counterexample was found during the build it is pinned as a named regression
in ``tests/test_regressions.py`` rather than left in the Hypothesis database,
which is not committed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from scipy.linalg import expm

from hilverdict.certificate import (
    CertificateModel,
    certified_overrun_s,
    divergence_bound,
    exponential_integral,
    gronwall_horizon_factor,
    horizon_gains,
    log_norm,
)
from hilverdict.clock import detect_resets, least_squares_rate, theil_sen_rate
from hilverdict.plant import SHORT_PERIOD, DivergenceMetric, LoopModel
from hilverdict.scoring import clopper_pearson, confusion, operating_points

FINITE = {"allow_nan": False, "allow_infinity": False}
SLOW = settings(max_examples=60, deadline=None)
FAST = settings(max_examples=150, deadline=None)


@FAST
@given(
    entries=st.lists(
        st.floats(min_value=-20.0, max_value=20.0, **FINITE),
        min_size=4,
        max_size=4,
    )
)
def test_log_norm_ignores_the_antisymmetric_part(entries):
    matrix = np.array(entries, dtype=float).reshape(2, 2)
    symmetric = 0.5 * (matrix + matrix.T)
    skew = 0.5 * (matrix - matrix.T)
    assert log_norm(matrix) == pytest.approx(
        log_norm(symmetric + 10.0 * skew), rel=1e-9, abs=1e-12
    )


@FAST
@given(
    entries=st.lists(
        st.floats(min_value=-10.0, max_value=10.0, **FINITE),
        min_size=4,
        max_size=4,
    ),
    shift=st.floats(min_value=-5.0, max_value=5.0, **FINITE),
)
def test_log_norm_is_translation_equivariant(entries, shift):
    matrix = np.array(entries, dtype=float).reshape(2, 2)
    assert log_norm(matrix + shift * np.eye(2)) == pytest.approx(
        log_norm(matrix) + shift, rel=1e-9, abs=1e-12
    )


@FAST
@given(
    entries=st.lists(
        st.floats(min_value=-10.0, max_value=10.0, **FINITE),
        min_size=4,
        max_size=4,
    ),
    scale=st.floats(min_value=0.01, max_value=10.0, **FINITE),
)
def test_log_norm_is_positively_homogeneous(entries, scale):
    matrix = np.array(entries, dtype=float).reshape(2, 2)
    assert log_norm(scale * matrix) == pytest.approx(
        scale * log_norm(matrix), rel=1e-9, abs=1e-12
    )


@SLOW
@given(
    entries=st.lists(
        st.floats(min_value=-8.0, max_value=8.0, **FINITE),
        min_size=4,
        max_size=4,
    ),
    duration=st.floats(min_value=0.0, max_value=0.5, **FINITE),
)
def test_dahlquist_inequality_holds(entries, duration):
    matrix = np.array(entries, dtype=float).reshape(2, 2)
    left = float(np.linalg.norm(expm(matrix * duration), 2))
    right = math.exp(log_norm(matrix) * duration)
    assert left <= right * (1.0 + 1e-9) + 1e-12


@FAST
@given(
    mu=st.floats(min_value=-200.0, max_value=200.0, **FINITE),
    first=st.floats(min_value=0.0, max_value=0.05, **FINITE),
    second=st.floats(min_value=0.0, max_value=0.05, **FINITE),
)
def test_exponential_integral_is_monotone_in_duration(mu, first, second):
    low, high = sorted((first, second))
    assert exponential_integral(mu, high) >= exponential_integral(mu, low) - 1e-18


@FAST
@given(
    mu=st.floats(min_value=-200.0, max_value=200.0, **FINITE),
    duration=st.floats(min_value=1e-9, max_value=0.05, **FINITE),
)
def test_exponential_integral_straddles_the_duration_by_the_sign_of_mu(mu, duration):
    # phi(mu, t) = t (1 + mu t / 2 + ...), so phi >= t for mu > 0 and phi <= t
    # for mu < 0. Hypothesis found two subnormal-mu cases where the naive
    # expm1(mu t)/mu breaks this -- one by a single ulp, one by returning zero --
    # and both are pinned in tests/test_regressions.py. The implementation takes
    # the first-order series for |mu t| < 1e-8, so the statement now holds to
    # within 8 ulps, which is the tolerance asserted here.
    slack = 8.0 * np.finfo(float).eps
    value = exponential_integral(mu, duration)
    if mu > 0.0:
        assert value >= duration * (1.0 - slack)
    elif mu < 0.0:
        assert value <= duration * (1.0 + slack)
    else:
        assert value == pytest.approx(duration, rel=0.0, abs=0.0)


@FAST
@given(
    mu=st.floats(min_value=-500.0, max_value=500.0, **FINITE),
    duration=st.floats(min_value=0.0, max_value=0.05, **FINITE),
)
def test_exponential_integral_is_non_negative(mu, duration):
    assert exponential_integral(mu, duration) >= 0.0


@FAST
@given(steps=st.integers(min_value=0, max_value=40))
def test_horizon_gains_start_at_one_and_are_non_negative(steps):
    gains = horizon_gains(LoopModel().scaled_monodromy(), steps)
    assert gains.shape == (steps + 1,)
    assert gains[0] == pytest.approx(1.0, rel=0.0, abs=0.0)
    assert float(np.min(gains)) >= 0.0


@FAST
@given(steps=st.integers(min_value=0, max_value=60))
def test_gronwall_dominates_the_horizon_factor(steps):
    model = LoopModel()
    gains = horizon_gains(model.scaled_monodromy(), steps)
    assert float(np.max(gains)) <= gronwall_horizon_factor(model, steps) * (
        1.0 + 1e-9
    )


@FAST
@given(
    overrun=st.floats(min_value=0.0, max_value=0.05, **FINITE),
    step=st.floats(min_value=0.0, max_value=0.5, **FINITE),
    horizon=st.integers(min_value=0, max_value=60),
)
def test_divergence_bound_is_non_negative_and_finite(overrun, step, horizon):
    constants = CertificateModel.build(LoopModel(), 60)
    value = divergence_bound(constants, overrun, step, horizon)
    assert value >= 0.0
    assert math.isfinite(value)


@FAST
@given(
    overrun=st.floats(min_value=0.0, max_value=0.05, **FINITE),
    step=st.floats(min_value=0.0, max_value=0.5, **FINITE),
    scale=st.floats(min_value=0.1, max_value=10.0, **FINITE),
)
def test_divergence_bound_is_homogeneous_in_the_input_step(overrun, step, scale):
    constants = CertificateModel.build(LoopModel(), 30)
    base = divergence_bound(constants, overrun, step, 30)
    scaled = divergence_bound(constants, overrun, scale * step, 30)
    assert scaled == pytest.approx(scale * base, rel=1e-9, abs=1e-15)


@FAST
@given(
    first=st.floats(min_value=0.0, max_value=0.05, **FINITE),
    second=st.floats(min_value=0.0, max_value=0.05, **FINITE),
    step=st.floats(min_value=1e-6, max_value=0.5, **FINITE),
)
def test_divergence_bound_is_monotone_in_the_overrun(first, second, step):
    constants = CertificateModel.build(LoopModel(), 30)
    low, high = sorted((first, second))
    assert divergence_bound(constants, high, step, 30) >= divergence_bound(
        constants, low, step, 30
    ) - 1e-15


@SLOW
@given(
    step=st.floats(min_value=1e-5, max_value=0.5, **FINITE),
    horizon=st.integers(min_value=0, max_value=60),
    tolerance=st.floats(min_value=0.05, max_value=20.0, **FINITE),
)
def test_equation_fourteen_inverts_equation_thirteen(step, horizon, tolerance):
    constants = CertificateModel.build(LoopModel(), 60)
    threshold = certified_overrun_s(constants, step, horizon, tolerance)
    assume(math.isfinite(threshold))
    assert divergence_bound(constants, threshold, step, horizon) == pytest.approx(
        tolerance, rel=1e-9, abs=0.0
    )


@FAST
@given(
    alpha=st.floats(min_value=-0.2, max_value=0.2, **FINITE),
    rate=st.floats(min_value=-2.0, max_value=2.0, **FINITE),
    scale=st.floats(min_value=0.01, max_value=100.0, **FINITE),
)
def test_metric_is_absolutely_homogeneous(alpha, rate, scale):
    metric = DivergenceMetric(
        alpha_tolerance_rad=math.radians(0.01), rate_tolerance_rad_s=math.radians(2.0)
    )
    vector = np.array([alpha, rate])
    assert float(metric.norm(scale * vector)) == pytest.approx(
        scale * float(metric.norm(vector)), rel=1e-12, abs=1e-15
    )


@FAST
@given(
    first=st.lists(
        st.floats(min_value=-1.0, max_value=1.0, **FINITE), min_size=2, max_size=2
    ),
    second=st.lists(
        st.floats(min_value=-1.0, max_value=1.0, **FINITE), min_size=2, max_size=2
    ),
)
def test_metric_satisfies_the_triangle_inequality(first, second):
    metric = DivergenceMetric(
        alpha_tolerance_rad=math.radians(0.01), rate_tolerance_rad_s=math.radians(2.0)
    )
    left = np.array(first)
    right = np.array(second)
    total = float(metric.norm(left + right))
    parts = float(metric.norm(left)) + float(metric.norm(right))
    assert total <= parts * (1.0 + 1e-12) + 1e-15


@FAST
@given(
    degrees=st.floats(min_value=-10.0, max_value=10.0, **FINITE),
)
def test_trim_is_an_equilibrium_at_every_declared_amplitude(degrees):
    state, control = SHORT_PERIOD.trim(math.radians(degrees))
    residual = SHORT_PERIOD.derivative(state, control)
    assert float(np.max(np.abs(residual))) < 1e-12


@FAST
@given(
    degrees=st.floats(min_value=-15.0, max_value=15.0, **FINITE),
)
def test_trim_is_odd_in_the_commanded_incidence(degrees):
    _state, positive = SHORT_PERIOD.trim(math.radians(degrees))
    _other, negative = SHORT_PERIOD.trim(-math.radians(degrees))
    assert positive == pytest.approx(-negative, rel=1e-12, abs=1e-18)


@FAST
@given(
    drift=st.floats(min_value=-50000.0, max_value=50000.0, **FINITE),
    samples=st.integers(min_value=5, max_value=120),
)
def test_least_squares_recovers_an_exact_rate(drift, samples):
    host = np.arange(samples, dtype=float) * 0.010
    device = (1.0 + drift * 1e-6) * host
    rate, _error = least_squares_rate(host, device)
    assert (rate - 1.0) * 1e6 == pytest.approx(drift, rel=1e-6, abs=1e-6)


@FAST
@given(
    drift=st.floats(min_value=-50000.0, max_value=50000.0, **FINITE),
    offset=st.floats(min_value=-10.0, max_value=10.0, **FINITE),
    samples=st.integers(min_value=5, max_value=80),
)
def test_theil_sen_is_invariant_to_an_offset(drift, offset, samples):
    host = np.arange(samples, dtype=float) * 0.010
    device = (1.0 + drift * 1e-6) * host
    assert theil_sen_rate(host, device) == pytest.approx(
        theil_sen_rate(host, device + offset), rel=1e-9, abs=1e-12
    )


@FAST
@given(
    samples=st.integers(min_value=5, max_value=120),
    drift=st.floats(min_value=-1000.0, max_value=1000.0, **FINITE),
)
def test_no_reset_is_found_in_a_noiseless_linear_series(samples, drift):
    host = np.arange(samples, dtype=float) * 0.010
    device = (1.0 + drift * 1e-6) * host
    assert detect_resets(host, device) == ()


@FAST
@given(
    samples=st.integers(min_value=20, max_value=200),
    position=st.floats(min_value=0.2, max_value=0.8, **FINITE),
    jump=st.floats(min_value=0.01, max_value=10.0, **FINITE),
)
def test_a_large_reset_is_always_found(samples, position, jump):
    host = np.arange(samples, dtype=float) * 0.010
    device = host.copy()
    index = int(position * samples)
    assume(1 < index < samples - 1)
    device[index:] -= jump
    assert index in detect_resets(host, device)


@FAST
@given(
    successes=st.integers(min_value=0, max_value=200),
    extra=st.integers(min_value=0, max_value=200),
)
def test_clopper_pearson_brackets_the_point_estimate(successes, extra):
    trials = successes + extra
    assume(trials > 0)
    lower, upper = clopper_pearson(successes, trials)
    assert lower <= successes / trials <= upper


@FAST
@given(
    successes=st.integers(min_value=0, max_value=200),
    extra=st.integers(min_value=0, max_value=200),
)
def test_clopper_pearson_is_ordered(successes, extra):
    trials = successes + extra
    assume(trials > 0)
    lower, upper = clopper_pearson(successes, trials)
    assert 0.0 <= lower <= upper <= 1.0


@FAST
@given(
    flags=st.lists(st.booleans(), min_size=1, max_size=60),
    truth=st.lists(st.booleans(), min_size=1, max_size=60),
)
def test_confusion_counts_partition_the_sample(flags, truth):
    size = min(len(flags), len(truth))
    counts = confusion(
        np.array(flags[:size], dtype=bool), np.array(truth[:size], dtype=bool)
    )
    assert counts.total == size


@FAST
@given(
    scores=st.lists(
        st.floats(min_value=0.0, max_value=1.0, **FINITE), min_size=2, max_size=40
    ),
    truth=st.lists(st.booleans(), min_size=2, max_size=40),
)
def test_operating_points_are_monotone_in_the_threshold(scores, truth):
    size = min(len(scores), len(truth))
    assume(size >= 2)
    values = np.array(scores[:size], dtype=float)
    labels = np.array(truth[:size], dtype=bool)
    assume(bool(np.any(labels)))
    _thresholds, recall, flag_rate = operating_points(values, labels)
    assert all(
        second <= first + 1e-15
        for first, second in zip(recall, recall[1:], strict=False)
    )
    assert all(
        second <= first + 1e-15
        for first, second in zip(flag_rate, flag_rate[1:], strict=False)
    )
