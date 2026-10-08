"""Hypothesis property tests for the interval algebra.

The properties tested here are the ones that can be proved, so a failure is a
code defect rather than a statistical fluctuation:

- both intervals lie in ``[0, 1]`` and contain the point estimate ``k / n``;
- both limits are monotone non-decreasing in ``k`` at fixed ``n``;
- the Clopper-Pearson upper limit is non-increasing in ``n`` at fixed ``k``;
- the zero-failure closed forms agree with the general code path;
- Clopper-Pearson coverage, computed exactly, is at least nominal for every
  ``p`` -- the conservativeness that makes it usable as evidence.

The last one is a statement about an exact sum over the binomial mass, not a
simulation, so it is a deterministic property of the method.
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from rareverify.intervals import clopper_pearson, exact_coverage, wilson, zero_failure_upper

counts = st.integers(min_value=1, max_value=400)
confidences = st.sampled_from([0.80, 0.90, 0.95, 0.99])


@st.composite
def count_pairs(draw, max_n: int = 400, require_room: int = 0):
    """Draw ``(k, n)`` with ``0 <= k <= n - require_room`` without filtering."""
    n = draw(st.integers(min_value=max(1, require_room), max_value=max_n))
    k = draw(st.integers(min_value=0, max_value=n - require_room))
    return k, n


@given(pair=count_pairs(), confidence=confidences)
def test_limits_are_in_unit_interval_and_bracket_the_point_estimate(pair, confidence):
    k, n = pair
    for interval in (
        clopper_pearson(k, n, confidence=confidence),
        wilson(k, n, confidence=confidence),
    ):
        assert 0.0 <= interval.lower <= interval.upper <= 1.0
        # The score interval contains p_hat because the score statistic is 0
        # there; Clopper-Pearson contains it by construction.
        assert interval.lower <= interval.point + 1e-15
        assert interval.point <= interval.upper + 1e-15


@given(pair=count_pairs(require_room=1), confidence=confidences)
def test_both_limits_are_monotone_in_k(pair, confidence):
    k, n = pair
    for method in (clopper_pearson, wilson):
        low = method(k, n, confidence=confidence)
        high = method(k + 1, n, confidence=confidence)
        assert high.lower >= low.lower - 1e-12
        assert high.upper >= low.upper - 1e-12


@given(pair=count_pairs(), confidence=confidences)
def test_clopper_pearson_upper_is_non_increasing_in_n(pair, confidence):
    k, n = pair
    here = clopper_pearson(k, n, confidence=confidence).upper
    there = clopper_pearson(k, n + 1, confidence=confidence).upper
    assert there <= here + 1e-12


@given(pair=count_pairs(require_room=1), confidence=confidences)
def test_clopper_pearson_lower_is_non_increasing_in_n(pair, confidence):
    """The lower limit is non-increasing in n at fixed k, the mirror property."""
    k, n = pair
    k = max(k, 1)
    here = clopper_pearson(k, n, confidence=confidence).lower
    there = clopper_pearson(k, n + 1, confidence=confidence).lower
    assert there <= here + 1e-12


@given(n=counts, confidence=confidences)
def test_zero_failure_closed_forms_match_the_general_path(n, confidence):
    for method, builder in (
        ("clopper-pearson", clopper_pearson),
        ("wilson", wilson),
    ):
        closed = zero_failure_upper(n, confidence=confidence, method=method, side="upper")
        general = builder(0, n, confidence=confidence, side="upper").upper
        assert abs(closed - general) <= 1e-12 * max(1.0, abs(general))


@given(
    n=st.integers(min_value=2, max_value=60),
    p=st.floats(min_value=0.001, max_value=0.999),
    confidence=confidences,
)
def test_clopper_pearson_coverage_is_conservative(n, p, confidence):
    """Exact coverage is at or above nominal for every p: the defining property."""
    coverage = exact_coverage(n, p, confidence=confidence, method="clopper-pearson")
    assert coverage >= confidence - 1e-12


@given(
    n=st.integers(min_value=2, max_value=60),
    p=st.floats(min_value=0.001, max_value=0.999),
    confidence=confidences,
)
def test_one_sided_clopper_pearson_coverage_is_conservative(n, p, confidence):
    coverage = exact_coverage(
        n, p, confidence=confidence, method="clopper-pearson", side="upper"
    )
    assert coverage >= confidence - 1e-12


@given(pair=count_pairs(), confidence=confidences)
def test_wider_confidence_gives_a_wider_interval(pair, confidence):
    k, n = pair
    narrow = clopper_pearson(k, n, confidence=confidence)
    wide = clopper_pearson(k, n, confidence=min(0.999, confidence + 0.009))
    assert wide.width >= narrow.width - 1e-12
