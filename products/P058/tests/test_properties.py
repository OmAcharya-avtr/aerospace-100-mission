"""Hypothesis property tests for the algebraic identities that actually hold.

Only identities that are true are tested. Several plausible-looking ones are
false and are pinned as *non*-properties below, because a test suite that
asserts a false identity with a loose tolerance is worse than no test.
"""

from __future__ import annotations

import math

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from scipy.stats import ks_2samp

from telemdrift.detectors import CUSUM, EWMA, PageHinkley, ks_two_sample_statistic
from telemdrift.features import window_features, window_features_single
from telemdrift.reference import siegmund_one_sided_arl
from telemdrift.scoring import wilson_interval

FINITE = st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False)
SAMPLES = st.lists(FINITE, min_size=1, max_size=120)


@given(SAMPLES, st.floats(min_value=0.5, max_value=20.0), st.floats(min_value=0.0, max_value=2.0))
@settings(max_examples=150, deadline=None)
def test_cusum_arms_are_never_negative(xs, h, k):
    det = CUSUM(h=h, k=k)
    for x in xs:
        det.update(x)
        assert det.s_plus >= 0.0
        assert det.s_minus >= 0.0


@given(SAMPLES, st.floats(min_value=0.5, max_value=20.0))
@settings(max_examples=150, deadline=None)
def test_cusum_is_odd_symmetric(xs, h):
    """Negating every sample swaps the two arms and leaves the alarm unchanged."""
    a, b = CUSUM(h=h), CUSUM(h=h)
    for x in xs:
        fa = a.update(x)
        fb = b.update(-x)
        assert fa == fb
    assert a.s_plus == b.s_minus
    assert a.s_minus == b.s_plus


@given(SAMPLES, st.floats(min_value=0.5, max_value=20.0), st.floats(min_value=0.1, max_value=10.0))
@settings(max_examples=150, deadline=None)
def test_cusum_is_invariant_under_an_affine_rescaling_of_the_channel(xs, h, scale):
    """CUSUM standardises by mu0 and sigma0, so feeding a + s*x with mu0 = a and
    sigma0 = s must reproduce the plain chart exactly. This is the property that
    lets a threshold calibrated on one channel transfer to another."""
    shift = 3.0
    plain, scaled = CUSUM(h=h), CUSUM(h=h, mu0=shift, sigma0=scale)
    for x in xs:
        assert plain.update(x) == scaled.update(shift + scale * x)


@given(SAMPLES, st.floats(min_value=0.5, max_value=20.0),
       st.floats(min_value=0.01, max_value=1.0))
@settings(max_examples=150, deadline=None)
def test_ewma_statistic_stays_within_the_convex_hull_of_the_inputs(xs, L, r):
    """z is a convex combination of 0 and the samples, so it cannot exceed them."""
    det = EWMA(L=L, r=r)
    lo = min(0.0, min(xs))
    hi = max(0.0, max(xs))
    for x in xs:
        det.update(x)
        assert lo - 1e-9 <= det.z <= hi + 1e-9


@given(SAMPLES, st.floats(min_value=0.5, max_value=20.0))
@settings(max_examples=150, deadline=None)
def test_ewma_is_odd_symmetric(xs, L):
    a, b = EWMA(L=L), EWMA(L=L)
    for x in xs:
        assert a.update(x) == b.update(-x)
    assert a.z == -b.z


@given(SAMPLES, st.floats(min_value=0.01, max_value=1.0))
@settings(max_examples=100, deadline=None)
def test_ewma_exact_variance_never_exceeds_the_asymptotic_one(xs, r):
    det = EWMA(L=3.0, r=r)
    asym = 3.0 * math.sqrt(r / (2.0 - r))
    for x in xs:
        det.update(x)
        assert det._limit() <= asym + 1e-12


@given(SAMPLES, st.floats(min_value=0.5, max_value=100.0))
@settings(max_examples=150, deadline=None)
def test_page_hinkley_statistic_bounds_are_ordered(xs, lam):
    det = PageHinkley(lambda_=lam, delta=0.0)
    for x in xs:
        det.update(x)
        assert det.m_min <= det.m <= det.m_max


@given(SAMPLES)
@settings(max_examples=100, deadline=None)
def test_page_hinkley_is_translation_invariant(xs):
    """Subtracting the running mean removes any constant offset exactly."""
    a, b = PageHinkley(lambda_=10.0, delta=0.0), PageHinkley(lambda_=10.0, delta=0.0)
    for x in xs:
        assert a.update(x) == b.update(x + 17.0)
    assert a.m == b.m or abs(a.m - b.m) < 1e-9


@given(
    st.lists(FINITE, min_size=2, max_size=60),
    st.lists(FINITE, min_size=2, max_size=60),
)
@settings(max_examples=200, deadline=None)
def test_ks_statistic_is_bounded_symmetric_and_matches_scipy(a, b):
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    d_ab = ks_two_sample_statistic(np.sort(arr_a), arr_b)
    d_ba = ks_two_sample_statistic(np.sort(arr_b), arr_a)
    assert 0.0 <= d_ab <= 1.0
    assert d_ab == d_ba
    assert abs(d_ab - ks_2samp(arr_a, arr_b).statistic) < 1e-12


#: A lattice of well-separated values. The rank-invariance property below is
#: exact in real arithmetic but not in floating point: with raw floats,
#: ``0.5 * 2.22e-16 + 1.0`` rounds to ``1.0``, so two distinct samples become a
#: tie and the statistic legitimately changes. Restricting the inputs to a
#: lattice tests the mathematical property instead of the rounding behaviour,
#: and the rounding behaviour is documented in the README limitations.
LATTICE = st.integers(min_value=-80, max_value=80).map(lambda i: i * 0.25)


@given(
    st.lists(LATTICE, min_size=2, max_size=40),
    st.sampled_from([0.25, 0.5, 1.0, 2.0, 4.0]),
    st.sampled_from([-8.0, -1.0, 0.0, 1.0, 8.0]),
)
@settings(max_examples=150, deadline=None)
def test_ks_statistic_is_invariant_under_a_common_monotone_rescaling(a, scale, shift):
    """The KS statistic depends only on ranks, so any strictly increasing map
    applied to both samples leaves it unchanged."""
    arr = np.asarray(a, dtype=float)
    half = max(1, arr.size // 2)
    x, y = arr[:half], arr[half:]
    assume(y.size >= 1)
    base = ks_two_sample_statistic(np.sort(x), y)
    mapped = ks_two_sample_statistic(np.sort(scale * x + shift), scale * y + shift)
    assert abs(base - mapped) < 1e-12


@given(st.lists(FINITE, min_size=8, max_size=200), st.integers(min_value=4, max_value=8))
@settings(max_examples=100, deadline=None)
def test_batch_and_single_window_features_are_identical(xs, window):
    arr = np.asarray(xs, dtype=float)
    assume(arr.size >= window)
    batch = window_features(arr, window)
    for i in range(batch.shape[0]):
        single = window_features_single(arr[i : i + window])
        assert np.array_equal(batch[i], single[0])


@given(st.lists(FINITE, min_size=4, max_size=60))
@settings(max_examples=150, deadline=None)
def test_features_are_always_finite(xs):
    arr = np.asarray(xs, dtype=float)
    f = window_features_single(arr)
    assert np.isfinite(f).all()


@given(st.lists(FINITE, min_size=4, max_size=60), FINITE)
@settings(max_examples=150, deadline=None)
def test_mean_feature_is_translation_equivariant_and_std_is_invariant(xs, shift):
    arr = np.asarray(xs, dtype=float)
    base = window_features_single(arr)[0]
    moved = window_features_single(arr + shift)[0]
    assert moved[0] == __import__("pytest").approx(base[0] + shift, abs=1e-9, rel=1e-9)
    assert moved[1] == __import__("pytest").approx(base[1], abs=1e-9, rel=1e-9)


@given(st.integers(min_value=0, max_value=500), st.integers(min_value=1, max_value=500))
@settings(max_examples=300, deadline=None)
def test_wilson_interval_contains_the_point_estimate(successes, trials):
    assume(successes <= trials)
    lo, hi = wilson_interval(successes, trials)
    p = successes / trials
    assert 0.0 <= lo <= p <= hi <= 1.0


@given(st.floats(min_value=0.5, max_value=12.0), st.floats(min_value=0.1, max_value=2.0))
@settings(max_examples=200, deadline=None)
def test_siegmund_arl_is_positive_and_increases_with_the_decision_interval(h, k):
    a = siegmund_one_sided_arl(h, k, 0.0)
    b = siegmund_one_sided_arl(h + 0.5, k, 0.0)
    assert a > 0.0
    assert b > a


@given(st.floats(min_value=1.0, max_value=8.0))
@settings(max_examples=100, deadline=None)
def test_siegmund_arl_decreases_as_the_true_shift_grows(h):
    """A larger shift is detected sooner, for every threshold."""
    arls = [siegmund_one_sided_arl(h, 0.5, d) for d in (0.75, 1.0, 1.5, 2.0, 3.0)]
    assert arls == sorted(arls, reverse=True)


# --------------------------------------------------------------------------
# Non-properties: identities that look true and are not.
# --------------------------------------------------------------------------


def test_page_hinkley_is_not_scale_invariant_unlike_cusum():
    """A deliberate negative property.

    Page-Hinkley subtracts the running mean but never divides by a scale, so its
    statistic scales with the channel and its lambda does not transfer between
    channels of different variance. CUSUM's h does, because CUSUM standardises.
    Assuming otherwise is the most likely way to misuse this package.
    """
    xs = np.linspace(-2.0, 2.0, 60)
    a, b = PageHinkley(lambda_=5.0, delta=0.0), PageHinkley(lambda_=5.0, delta=0.0)
    for x in xs:
        a.update(x)
        b.update(10.0 * x)
    assert abs(b.m) > 5.0 * abs(a.m)


def test_arl0_is_not_the_reciprocal_of_a_per_sample_false_alarm_probability():
    """Another deliberate negative.

    For a memoryless detector ARL0 = 1/p would hold. None of these detectors is
    memoryless: the statistic carries state across samples, so the run-length
    distribution is not geometric and the per-sample alarm probability is not
    constant. The measured coefficient of variation of the CUSUM run length is
    close to but not equal to 1, which is the quantitative version of this
    statement; see validation/outputs/validate_arl_calibration.txt.
    """
    from telemdrift.scoring import measure_arl0
    from telemdrift.streams import stationary

    res = measure_arl0(
        lambda: CUSUM(h=4.0), lambda length, seed: stationary(length, seed),
        [1, 2, 3, 4], 40_000,
    )
    cv = res.run_lengths.std(ddof=1) / res.run_lengths.mean()
    assert 0.6 < cv < 1.3
    assert cv != 1.0
