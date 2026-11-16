"""Property-based tests for the algebraic identities this package rests on.

Hypothesis's example database is turned off in ``conftest.py``: a falsifying
example found by one run would be lost on a fresh clone, so every
counterexample this suite has found is pinned below as an explicit regression
test with the literal inputs in the test body.

No test here asserts bit-exact equality over a floating-point pipeline, and no
test asserts a boolean produced by a strict threshold comparison. Both are
defect classes this mission has shipped.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from hiltimingdiff import declared_ops, kendall_tau_b, linear_fit, ordering
from hiltimingdiff.bootstrap import percentile_interval, unit_resamples

FINITE = st.floats(
    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False, width=64
)
POSITIVE = st.floats(
    min_value=1e-6, max_value=1e6, allow_nan=False, allow_infinity=False, width=64
)


@given(st.lists(FINITE, min_size=2, max_size=12))
def test_tau_b_with_itself_is_one_or_undefined(values):
    array = np.asarray(values, dtype=np.float64)
    result = kendall_tau_b(array, array)
    if np.isnan(result):
        # tau-b is undefined when every pair is tied, which is the case when
        # the series is constant. Undefined is reported as nan, not as zero.
        assert float(np.ptp(array)) == 0.0
    else:
        assert result == pytest.approx(1.0, rel=1e-9, abs=0.0)


@given(st.lists(FINITE, min_size=2, max_size=10), st.lists(FINITE, min_size=2, max_size=10))
def test_tau_b_is_symmetric_in_its_arguments(first, second):
    assume(len(first) == len(second))
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    forward = kendall_tau_b(left, right)
    backward = kendall_tau_b(right, left)
    if np.isnan(forward):
        assert np.isnan(backward)
    else:
        assert forward == pytest.approx(backward, rel=1e-9, abs=1e-12)


@given(st.lists(FINITE, min_size=2, max_size=10), POSITIVE)
def test_tau_b_is_invariant_under_a_positive_rescaling_of_one_side(values, factor):
    # The invariance holds in exact arithmetic. In floating point it holds only
    # while the rescaled values keep the DISTINCTNESS pattern of the originals:
    # scaling [0.0, 5e-324] by 0.5 underflows the second entry to 0.0 and turns
    # a concordant pair into a tied one. The assumption below is the honest
    # precondition; the counterexample is pinned in
    # test_regression_tau_b_invariance_needs_distinctness_to_survive_rescaling.
    array = np.asarray(values, dtype=np.float64)
    scaled_array = factor * array
    # Both guards act on the RESCALED array. Distinctness is the one the
    # counterexample below needs; finiteness is the mirror case, added when the
    # same hole was found in the `ordering` property, because an overflow to
    # inf makes a difference of two scaled values nan rather than merely tied.
    assume(bool(np.all(np.isfinite(scaled_array))))
    assume(_distinctness_pattern(array) == _distinctness_pattern(scaled_array))
    other = np.arange(array.size, dtype=np.float64)
    base = kendall_tau_b(other, array)
    scaled = kendall_tau_b(other, scaled_array)
    if np.isnan(base):
        assert np.isnan(scaled)
    else:
        assert scaled == pytest.approx(base, rel=1e-9, abs=1e-12)


@given(st.lists(FINITE, min_size=2, max_size=10))
def test_tau_b_negates_under_a_sign_flip_of_one_side(values):
    array = np.asarray(values, dtype=np.float64)
    other = np.arange(array.size, dtype=np.float64)
    base = kendall_tau_b(other, array)
    flipped = kendall_tau_b(other, -array)
    if np.isnan(base):
        assert np.isnan(flipped)
    else:
        assert flipped == pytest.approx(-base, rel=1e-9, abs=1e-12)


@given(st.lists(FINITE, min_size=2, max_size=12))
def test_tau_b_lies_in_the_closed_unit_interval(values):
    array = np.asarray(values, dtype=np.float64)
    other = np.arange(array.size, dtype=np.float64)[::-1]
    result = kendall_tau_b(other, array)
    assume(not np.isnan(result))
    assert -1.0 - 1e-12 <= result <= 1.0 + 1e-12


@given(st.lists(FINITE, min_size=1, max_size=12))
def test_ordering_is_a_permutation(values):
    array = np.asarray(values, dtype=np.float64)
    assert sorted(ordering(array).tolist()) == list(range(array.size))


@given(st.lists(FINITE, min_size=1, max_size=12))
def test_ordering_sorts_ascending(values):
    array = np.asarray(values, dtype=np.float64)
    sorted_values = array[ordering(array)]
    assert np.all(np.diff(sorted_values) >= 0.0)


def _distinctness_pattern(values: np.ndarray) -> list[bool]:
    """Pairwise equality pattern of ``values``, as a flat list.

    Two arrays with the same pattern tie in exactly the same places, which is
    the precondition every rescaling invariance in this file actually needs.
    An ``assume`` on the INPUT array cannot express it: the operation that
    destroys distinctness acts on the OUTPUT.
    """
    return [
        bool(values[i] == values[j])
        for i in range(values.size)
        for j in range(i + 1, values.size)
    ]


@given(st.lists(FINITE, min_size=1, max_size=12), POSITIVE)
def test_ordering_is_invariant_under_a_positive_rescaling(values, factor):
    # A positive multiplicative rescaling is monotone non-decreasing, so it can
    # never INVERT an order. It can still change the permutation, by destroying
    # DISTINCTNESS: two distinct finite values can overflow to the same inf or
    # underflow to the same subnormal, and the stable tie-break then orders the
    # tied pair by index, which flips a descending input. Guarding the spread of
    # the INPUT does not express that, because the loss happens in the OUTPUT.
    # Three counterexamples are pinned below, two overflow and one underflow.
    #
    # An additive OFFSET fails the same way and is pinned below as well.
    array = np.asarray(values, dtype=np.float64)
    assume(float(np.ptp(array)) > 1e-9 * max(1.0, float(np.abs(array).max())))
    scaled = factor * array
    # The two guards that make the claim true, both on the RESCALED array.
    assume(bool(np.all(np.isfinite(scaled))))
    assume(_distinctness_pattern(array) == _distinctness_pattern(scaled))
    assert ordering(array).tolist() == ordering(scaled).tolist()


@given(
    st.integers(min_value=1, max_value=512),
    st.floats(min_value=0.0, max_value=1e4, allow_nan=False, allow_infinity=False),
)
def test_declared_ops_is_quadratic_in_width_for_the_vectorised_stages(width, candidates):
    counts = declared_ops(width, candidates)
    area = float(width) * float(width)
    assert counts["ingest"] == pytest.approx(2.0 * area, rel=1e-12, abs=0.0)
    assert counts["background"] == pytest.approx(4.0 * area, rel=1e-12, abs=0.0)
    assert counts["threshold"] == pytest.approx(3.0 * area, rel=1e-12, abs=0.0)


@given(
    st.integers(min_value=1, max_value=512),
    st.floats(min_value=0.0, max_value=1e4, allow_nan=False, allow_infinity=False),
)
def test_declared_ops_is_linear_in_candidates_for_the_interpreted_stages(
    width, candidates
):
    counts = declared_ops(width, candidates)
    assert counts["centroid"] == pytest.approx(150.0 * candidates, rel=1e-12, abs=0.0)
    assert counts["gate"] == pytest.approx(12.0 * candidates, rel=1e-12, abs=0.0)


@given(
    st.integers(min_value=1, max_value=512),
    st.floats(min_value=0.0, max_value=1e4, allow_nan=False, allow_infinity=False),
)
def test_declared_ops_track_is_constant(width, candidates):
    assert declared_ops(width, candidates)["track"] == 180.0


@given(
    st.integers(min_value=1, max_value=256),
    st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    POSITIVE,
)
def test_declared_ops_scales_additively_in_candidates(width, candidates, delta):
    base = declared_ops(width, candidates)
    shifted = declared_ops(width, candidates + delta)
    assert shifted["centroid"] - base["centroid"] == pytest.approx(
        150.0 * delta, rel=1e-9, abs=1e-9
    )
    assert shifted["gate"] - base["gate"] == pytest.approx(12.0 * delta, rel=1e-9, abs=1e-9)
    assert shifted["ingest"] == pytest.approx(base["ingest"], rel=1e-12, abs=0.0)


@given(
    st.lists(FINITE, min_size=3, max_size=40),
    st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
)
def test_linear_fit_recovers_an_exact_line_on_arbitrary_abscissae(xs, slope, intercept):
    # AUDITED for the same hole as the `ordering` rescaling property: the guard
    # is on the INPUT `x` while the operation forms `y = slope*x + intercept`,
    # which is an output. It is safe here, and the reason is the absolute
    # tolerance floor rather than luck. `y` can lose the slope entirely only
    # when |slope| * ptp(x) falls below one ulp of `intercept`; with
    # |intercept| <= 100 and the guard forcing ptp(x) > 1e-3 * max|x|, that
    # needs |slope| < 100 * 2.2e-16 / 1e-3 = 2.2e-11, which the `abs=1e-6` floor
    # already admits. Between those regimes the guard bounds the conditioning of
    # the fit to about 1e3, so the relative slope error stays near 1e-13 against
    # a 1e-6 bound. The degenerate end is pinned in
    # test_regression_linear_fit_on_a_slope_too_small_to_survive_the_intercept.
    x = np.asarray(sorted(set(xs)), dtype=np.float64)
    assume(x.size >= 3)
    scale = max(1.0, float(np.abs(x).max()))
    assume(float(np.ptp(x)) > 1e-3 * scale)
    y = slope * x + intercept
    fit = linear_fit(x, y)
    assert fit.slope == pytest.approx(slope, rel=1e-6, abs=1e-6)
    assert fit.intercept == pytest.approx(intercept, rel=1e-6, abs=1e-6)


@given(st.lists(FINITE, min_size=3, max_size=40))
def test_linear_fit_r_squared_never_exceeds_one(values):
    x = np.arange(len(values), dtype=np.float64)
    fit = linear_fit(x, np.asarray(values, dtype=np.float64))
    assume(not np.isnan(fit.r_squared))
    assert fit.r_squared <= 1.0 + 1e-9


@given(
    st.lists(POSITIVE, min_size=3, max_size=30),
    st.floats(min_value=0.5, max_value=0.999, allow_nan=False, allow_infinity=False),
)
def test_percentile_interval_bounds_are_ordered(values, confidence):
    array = np.asarray(values, dtype=np.float64)
    interval = percentile_interval(
        float(np.median(array)), array, confidence=confidence, units=array.size
    )
    assert interval.low <= interval.high


@given(st.lists(POSITIVE, min_size=3, max_size=30))
def test_percentile_interval_lies_inside_the_replicate_range(values):
    array = np.asarray(values, dtype=np.float64)
    interval = percentile_interval(1.0, array, units=array.size)
    assert float(array.min()) - 1e-9 <= interval.low
    assert interval.high <= float(array.max()) + 1e-9


@given(st.integers(min_value=2, max_value=200), st.integers(min_value=1, max_value=50))
def test_unit_resamples_shape_and_range(n_units, resamples):
    indices = unit_resamples(n_units, resamples=resamples, seed=5)
    assert indices.shape == (resamples, n_units)
    assert int(indices.min()) >= 0
    assert int(indices.max()) < n_units


# --- pinned counterexamples ----------------------------------------------
#
# Every falsifying example Hypothesis found while this suite was written is
# pinned here with literal inputs, because .hypothesis/ is not committed and a
# counterexample found by one run is otherwise lost.


def test_regression_tau_b_is_nan_for_a_constant_side():
    # Found by test_tau_b_with_itself_is_one_or_undefined at values=[0.0, 0.0].
    # tau-b divides by sqrt((n0 - n1)(n0 - n2)), which is zero when every pair
    # is tied, so the statistic is undefined and nan is reported rather than a
    # spurious 1.0 or 0.0.
    assert np.isnan(kendall_tau_b(np.array([0.0, 0.0]), np.array([0.0, 0.0])))


def test_regression_tau_b_survives_a_difference_whose_square_underflows():
    # Found by test_tau_b_with_itself_is_one_or_undefined at
    # values=[0.0, 1.3637850869171467e-163]. The difference 1.36e-163 is
    # representable, but its SQUARE is 1.86e-326, which underflows to exactly
    # 0.0 in float64. An implementation that decides concordance on the product
    # of the two differences therefore reads a concordant pair as a tied one and
    # returns 0.0 where the answer is 1.0. THIS WAS A REAL DEFECT in this
    # package's tau-b and was fixed by deciding concordance on the product of
    # SIGNS, which cannot underflow.
    values = np.array([0.0, 1.3637850869171467e-163])
    assert kendall_tau_b(values, values) == pytest.approx(1.0, rel=1e-12, abs=0.0)


def test_regression_tau_b_survives_a_difference_whose_square_overflows():
    # The other end of the same mechanism. The product of two differences of
    # order 1e200 overflows to +inf; the sign product does not.
    values = np.array([-1e200, 0.0, 1e200])
    assert kendall_tau_b(values, values) == pytest.approx(1.0, rel=1e-12, abs=0.0)
    assert kendall_tau_b(values, -values) == pytest.approx(-1.0, rel=1e-12, abs=0.0)


def test_regression_tau_b_invariance_needs_distinctness_to_survive_rescaling():
    # Found by test_tau_b_is_invariant_under_a_positive_rescaling_of_one_side at
    # values=[0.0, 5e-324], factor=0.5. 5e-324 is the smallest positive double;
    # halving it underflows to exactly 0.0, so two distinct values become tied
    # and the statistic moves from 1.0 to undefined. The invariance is true in
    # exact arithmetic and the precondition in floating point is that the
    # rescaling preserves distinctness.
    values = np.array([0.0, 5e-324])
    order = np.arange(2, dtype=np.float64)
    assert kendall_tau_b(order, values) == pytest.approx(1.0, rel=1e-12, abs=0.0)
    assert np.isnan(kendall_tau_b(order, 0.5 * values))


def test_regression_ordering_flips_on_a_plain_ROUNDING_COLLISION_in_range():
    # THE DRAW THAT ACTUALLY FAILED IN THIS SUITE, and it is neither overflow
    # nor underflow. The coordinator's report diagnosed the class correctly and
    # named an overflow example at factor 1e308, but this file's strategies cap
    # `factor` at 1e6 and `values` at 1e6, so 1e308 is not generatable here and
    # cannot have been the draw that failed. Searching the declared strategies
    # finds the real route: an ordinary ROUNDING COLLISION.
    #
    # 1000000.0 and 999999.9999999999 are distinct, one ulp apart at 1e6. Their
    # exact products with 292915.2677047961 differ by 3.41e-05, while the float64
    # ulp at 2.9e11 is 6.10e-05 -- less than half an ulp -- so BOTH ROUND TO THE
    # SAME FLOAT. Nothing overflows, nothing underflows, every value is finite
    # and of utterly ordinary magnitude.
    #
    # The third element is not decoration. It gives the array a spread of 1.8e6
    # so the OLD `assume` on the input's range passes comfortably, which is the
    # hole exactly: the guard measured the array-wide spread while the tie
    # happened between one specific pair. The two-element version is rejected by
    # that guard and so could never have been the failing draw.
    values = np.array([-811260.5389416472, 1000000.0, 999999.9999999999])
    factor = 292915.2677047961
    scaled = factor * values
    assert values[1] != values[2]
    assert scaled[1] == scaled[2]
    assert bool(np.all(np.isfinite(scaled)))
    assert float(np.ptp(values)) > 1e-9 * max(1.0, float(np.abs(values).max()))
    assert ordering(values).tolist() == [0, 2, 1]
    assert ordering(scaled).tolist() == [0, 1, 2]


def test_regression_the_rounding_collision_is_rejected_by_the_repaired_guard():
    # The repaired guard must REJECT the draw above, or the property would still
    # be asserting something false. Checked directly rather than inferred from
    # the suite passing.
    values = np.array([-811260.5389416472, 1000000.0, 999999.9999999999])
    scaled = 292915.2677047961 * values
    assert _distinctness_pattern(values) != _distinctness_pattern(scaled)


def test_regression_ordering_flips_when_a_rescaling_OVERFLOWS_two_values():
    # Found by the COORDINATOR's independent re-run of the suite, which hit one
    # failure in six full-suite runs and then five clean runs that did not
    # reproduce it; the cause was then derived deterministically rather than by
    # replaying. values=[3.0, 2.0], factor=1e308. Both entries overflow to the
    # same inf, so two distinct values tie, and the stable tie-break reorders a
    # DESCENDING input. An ascending input is unaffected, which is why a random
    # search over mostly-ascending draws finds this only sometimes.
    values = np.array([3.0, 2.0])
    with np.errstate(over="ignore"):
        scaled = 1e308 * values
    assert scaled.tolist() == [float("inf"), float("inf")]
    assert ordering(values).tolist() == [1, 0]
    assert ordering(scaled).tolist() == [0, 1]


def test_regression_ordering_flips_on_a_three_element_overflow():
    # The second counterexample from the same report, at three elements:
    # values=[5.0, 4.0, 3.0], factor=1e308 turns [2, 1, 0] into [0, 1, 2].
    values = np.array([5.0, 4.0, 3.0])
    with np.errstate(over="ignore"):
        scaled = 1e308 * values
    assert scaled.tolist() == [float("inf")] * 3
    assert ordering(values).tolist() == [2, 1, 0]
    assert ordering(scaled).tolist() == [0, 1, 2]


def test_regression_ordering_flips_when_a_rescaling_UNDERFLOWS_two_values():
    # The underflow mirror, searched for after the overflow pair was reported,
    # because the same hole has two sides. [2.5e-323, 1.5e-323] scaled by 0.5
    # both round to 1e-323, so a descending input again has its permutation
    # flipped. Note the factor here is an ordinary 0.5, not an extreme one.
    values = np.array([2.5e-323, 1.5e-323])
    scaled = 0.5 * values
    assert scaled[0] == scaled[1]
    assert ordering(values).tolist() == [1, 0]
    assert ordering(scaled).tolist() == [0, 1]


def test_regression_a_rescaling_that_keeps_finiteness_and_distinctness_is_safe():
    # The positive half of the same statement, so the guards are shown to admit
    # the cases the property is about rather than rejecting everything.
    values = np.array([3.0, 2.0, 5.0, -1.0])
    for factor in (1e-6, 0.5, 2.0, 1e6):
        scaled = factor * values
        assert np.all(np.isfinite(scaled))
        assert ordering(values).tolist() == ordering(scaled).tolist()


def test_regression_ordering_is_not_invariant_under_an_additive_offset():
    # Found by the affine form of test_ordering_is_invariant_under_a_positive_
    # rescaling at values=[0.0, 1.0, -1.1754943508222875e-38]. Adding 3.0 makes
    # the first and third entries both exactly 3.0, so two distinct values
    # become tied and the stable tie-break reorders them. The property is stated
    # for a multiplicative rescaling only, which is what actually holds.
    values = np.array([0.0, 1.0, -1.1754943508222875e-38])
    assert ordering(values).tolist() == [2, 0, 1]
    assert ordering(values + 3.0).tolist() == [0, 2, 1]


def test_regression_linear_fit_needs_a_spread_in_the_predictor():
    # Found by test_linear_fit_recovers_an_exact_line_on_arbitrary_abscissae
    # when the generated abscissae collapsed to two distinct values separated by
    # less than the float spacing of their magnitude: polyfit's normal equations
    # are then singular. The guard raises instead of returning a slope.
    with pytest.raises(ValueError, match="constant"):
        linear_fit(np.array([1.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0]))


def test_regression_percentile_interval_on_a_single_distinct_replicate():
    # Found by test_percentile_interval_lies_inside_the_replicate_range at
    # values=[1e-06, 1e-06, 1e-06]: both endpoints coincide and the interval has
    # zero width, which is correct and must not be treated as an error.
    interval = percentile_interval(1e-6, np.full(3, 1e-6), units=3)
    assert interval.low == pytest.approx(1e-6, rel=1e-12, abs=0.0)
    assert interval.width == pytest.approx(0.0, rel=0.0, abs=1e-18)


def test_regression_linear_fit_on_a_slope_too_small_to_survive_the_intercept():
    # The degenerate end of the audited property above, asserted rather than
    # argued. A slope of 1e-20 over x in [0, 1000] moves y by 1e-17, far below
    # one ulp of an intercept of 50.0, so y is EXACTLY constant and the slope is
    # gone from the data. The claim survives only because `abs=1e-6` admits the
    # result, and that is recorded here so a future tightening of the tolerance
    # fails loudly instead of becoming intermittent.
    #
    # The recovered slope is NOT asserted to be exactly 0.0. Writing that first
    # was an overclaim of exactly the kind this file exists to catch: polyfit's
    # own normal equations leave a rounding-level residue, measured here at
    # 7.462276454104577e-18. What is guaranteed is that it is negligible against
    # the tolerance, and that is what is asserted.
    x = np.array([0.0, 500.0, 1000.0])
    y = 1e-20 * x + 50.0
    assert y.tolist() == [50.0, 50.0, 50.0]
    fit = linear_fit(x, y)
    assert abs(fit.slope) < 1e-9
    assert fit.slope == pytest.approx(1e-20, rel=1e-6, abs=1e-6)
    assert fit.intercept == pytest.approx(50.0, rel=1e-6, abs=1e-6)
    # SST is zero on a constant y, so the explained-variance fraction is
    # undefined and is reported as nan rather than as 1.0 or 0.0.
    assert np.isnan(fit.r_squared)


def test_regression_declared_ops_at_a_single_pixel_frame():
    # Found by test_declared_ops_is_quadratic_in_width_for_the_vectorised_stages
    # at width=1: a 1x1 frame is degenerate as an image and the declared model
    # is still defined on it, which is what makes the model's independence from
    # the data visible.
    counts = declared_ops(1, 0.0)
    assert counts["ingest"] == 2.0
    assert counts["centroid"] == 0.0
    assert counts["track"] == 180.0
