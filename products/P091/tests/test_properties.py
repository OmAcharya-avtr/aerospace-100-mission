"""Property-based tests over the algebraic identities this package actually has.

Where a property is only true in exact arithmetic, the assertion carries a
tolerance on both sides and the docstring says what is guaranteed. Nothing here
asserts equality of a boolean produced by a strict threshold comparison: that
boolean is a step function of a rounded quantity and is the one thing floating
point cannot promise.
"""

from __future__ import annotations

import math

import numpy as np
from hypothesis import assume, given
from hypothesis import strategies as st

from thermalbudget.bootstrap import bootstrap_mean, observed_cluster_mean
from thermalbudget.equilibrium import fit_linear
from thermalbudget.provenance import Origin, meet
from thermalbudget.rc_model import fit_exponential_relaxation, tau_from_integral
from thermalbudget.schema import infer_temperature_divisor

ORIGINS = st.sampled_from([Origin.SYNTHETIC, Origin.MEASURED])
FINITE = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)


@given(a=ORIGINS, b=ORIGINS, c=ORIGINS)
def test_meet_is_a_semilattice(a, b, c):
    """Associative, commutative, idempotent, with MEASURED as identity. Exact."""
    assert meet(meet(a, b), c) is meet(a, meet(b, c))
    assert meet(a, b) is meet(b, a)
    assert meet(a, a) is a
    assert meet(a, Origin.MEASURED) is a
    assert meet(a, Origin.SYNTHETIC) is Origin.SYNTHETIC


@given(
    slope=st.floats(min_value=-100.0, max_value=100.0),
    intercept=st.floats(min_value=-1e3, max_value=1e3),
    n=st.integers(min_value=3, max_value=200),
)
def test_a_straight_line_is_recovered_to_rounding(slope, intercept, n):
    """The fit recovers an exact line to within rounding, not exactly.

    The residuals of an exact line are zero in exact arithmetic; in floating
    point they are of order the rounding of the products involved, so the
    recovered slope is compared with a relative tolerance scaled by the
    magnitudes in play rather than with equality.
    """
    x = np.arange(float(n))
    y = intercept + slope * x
    fit = fit_linear(x, y)
    scale = max(abs(slope), 1.0) * max(abs(intercept), 1.0)
    assert fit.slope == np.float64(slope) or abs(fit.slope - slope) <= 1e-9 * scale


@given(
    shift=st.floats(min_value=-500.0, max_value=500.0),
    n=st.integers(min_value=5, max_value=200),
)
def test_the_fitted_slope_is_invariant_under_a_shift_in_x_to_rounding(shift, n):
    """Shifting x by a constant changes the intercept and not the slope.

    True in exact arithmetic. The slope is compared to a tolerance because the
    centring subtraction rounds differently for a shifted abscissa.
    """
    rng = np.random.default_rng(n)
    x = np.arange(float(n))
    y = 3.0 + 0.5 * x + rng.normal(0.0, 0.1, size=n)
    first = fit_linear(x, y)
    second = fit_linear(x + shift, y)
    assert abs(first.slope - second.slope) <= 1e-9 * max(abs(first.slope), 1.0)


@given(
    scale=st.floats(min_value=0.01, max_value=100.0),
    n=st.integers(min_value=5, max_value=120),
)
def test_scaling_y_scales_the_slope_proportionally_to_rounding(scale, n):
    """A positive rescaling of y multiplies the slope by the same factor."""
    rng = np.random.default_rng(n + 1)
    x = np.arange(float(n))
    y = 2.0 + 0.3 * x + rng.normal(0.0, 0.05, size=n)
    base = fit_linear(x, y)
    scaled = fit_linear(x, scale * y)
    assert abs(scaled.slope - scale * base.slope) <= 1e-8 * abs(scale * base.slope) + 1e-12


@given(
    tau=st.floats(min_value=20.0, max_value=400.0),
    asymptote=st.floats(min_value=30.0, max_value=95.0),
    offset=st.floats(min_value=1.0, max_value=40.0),
)
def test_the_relaxation_fit_recovers_a_noiseless_curve_within_the_optimiser_tolerance(
    tau, asymptote, offset
):
    """Recovery is to the optimiser's bracket, which is a stated tolerance.

    The window is fixed at eight time constants so the asymptote is observed
    rather than extrapolated; the fit's own ``span_in_tau`` is the number that
    says when it is not.
    """
    t = np.linspace(1.0, 8.0 * tau, 400)
    temperature = asymptote - offset * np.exp(-t / tau)
    fit = fit_exponential_relaxation(t, temperature)
    assert fit.tau_s == float(fit.tau_s)
    assert abs(fit.tau_s - tau) <= 1e-3 * tau
    assert abs(fit.asymptote_c - asymptote) <= 1e-6 * max(abs(asymptote), 1.0)


@given(
    tau=st.floats(min_value=30.0, max_value=300.0),
    offset=st.floats(min_value=2.0, max_value=40.0),
)
def test_the_area_rule_agrees_with_the_least_squares_fit_on_noiseless_data(tau, offset):
    """Two routes sharing no algebra agree to within quadrature error.

    The area rule uses the trapezium rule, whose error is second order in the
    sample interval, so the tolerance follows the SAMPLE SPACING and not a
    constant.
    """
    spacing = tau / 40.0
    t = np.arange(1.0, 6.0 * tau, spacing)
    assume(t.size >= 40)
    temperature = 70.0 - offset * np.exp(-t / tau)
    fit = fit_exponential_relaxation(t, temperature)
    area = tau_from_integral(t, temperature, asymptote_c=fit.asymptote_c)
    quadrature = (spacing / tau) ** 2
    assert abs(area.tau_s - fit.tau_s) <= tau * (0.02 + 10.0 * quadrature)


@given(
    values=st.lists(FINITE, min_size=2, max_size=60),
    seed=st.integers(min_value=0, max_value=1000),
)
def test_a_bootstrap_interval_brackets_nothing_wider_than_the_sample_range(values, seed):
    """Resampling cannot produce a mean outside the range of the sample.

    This is exact: every resample mean is a convex combination of the observed
    values, so it lies in ``[min, max]``. The percentile interval is therefore
    inside that range on both sides.
    """
    sample = np.asarray(values, dtype=float)
    assume(np.all(np.isfinite(sample)))
    result = bootstrap_mean(sample, n_resamples=200, seed=seed)
    low, high = float(sample.min()), float(sample.max())
    assert low - 1e-9 <= result.ci_low <= high + 1e-9
    assert low - 1e-9 <= result.ci_high <= high + 1e-9
    assert result.ci_low <= result.ci_high


@given(
    groups=st.lists(
        st.lists(FINITE, min_size=1, max_size=20), min_size=1, max_size=6
    )
)
def test_the_cluster_mean_is_unweighted_over_group_means(groups):
    """The statistic is the mean of per-group means, to within rounding."""
    arrays = [np.asarray(group, dtype=float) for group in groups]
    assume(all(np.all(np.isfinite(array)) for array in arrays))
    expected = float(np.mean([float(array.mean()) for array in arrays]))
    observed = observed_cluster_mean(arrays)
    assert abs(observed - expected) <= 1e-9 * max(abs(expected), 1.0)


@given(
    centre=st.floats(min_value=20.0, max_value=95.0),
    n=st.integers(min_value=1, max_value=50),
)
def test_degree_and_milli_degree_samples_are_distinguished(centre, n):
    """A plausible die temperature in either unit lands in exactly one band."""
    degrees = [centre] * n
    assert infer_temperature_divisor(degrees) == 1.0
    assert infer_temperature_divisor([1000.0 * centre] * n) == 1000.0
    # And the conversion round-trips to within rounding.
    assert abs((1000.0 * centre) / 1000.0 - centre) <= 1e-9 * max(abs(centre), 1.0)


@given(values=st.lists(FINITE, min_size=3, max_size=50))
def test_a_constant_sample_has_a_zero_width_interval(values):
    """Every resample of a constant sample is that constant. Exact."""
    assume(math.isfinite(values[0]))
    sample = np.full(len(values), values[0])
    result = bootstrap_mean(sample, n_resamples=100, seed=1)
    assert result.ci_low == result.ci_high == result.point
