"""Rank statistics: hand-calculated known answers, scipy cross-check, properties.

Every known answer below is computed by hand in the test comment. ``scipy``'s
``kendalltau(variant="b")`` and ``spearmanr`` are a separately written
implementation of both statistics; agreement with them is a cross-check against
an independent implementation and is not the same thing as the independently
derived bound check, which lives in ``tests/test_bounds.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats

from flakerank.disagreement import (
    kendall_tau_b,
    midranks,
    rank_disagreement,
    spearman_rho,
)

# ---------------------------------------------------------------------------
# Hand-calculated known answers
# ---------------------------------------------------------------------------


def test_midranks_without_ties() -> None:
    """Strictly increasing input gets ranks 1, 2, 3, 4."""
    assert midranks(np.asarray([10.0, 20.0, 30.0, 40.0])).tolist() == [1.0, 2.0, 3.0, 4.0]


def test_midranks_with_one_tie_group() -> None:
    """x = [1, 1, 2, 3]: the two 1s span ranks 1 and 2, so both get (1+2)/2 = 1.5.

    Hand arithmetic: sorted positions 1,2 are tied -> midrank 1.5 each; 2 takes
    rank 3; 3 takes rank 4. Result [1.5, 1.5, 3, 4].
    """
    assert midranks(np.asarray([1.0, 1.0, 2.0, 3.0])).tolist() == [1.5, 1.5, 3.0, 4.0]


def test_midranks_all_tied() -> None:
    """Four equal values span ranks 1..4, so each gets (1+2+3+4)/4 = 2.5."""
    assert midranks(np.asarray([7.0] * 4)).tolist() == [2.5] * 4


def test_kendall_tau_b_no_ties_one_inversion() -> None:
    """x = [1,2,3,4], y = [1,2,4,3]: C = 5, D = 1, no ties, tau = 4/6 = 0.6666...

    Hand enumeration of the 6 pairs:
      (1,2) x 1<2, y 1<2 -> concordant
      (1,3) x 1<3, y 1<4 -> concordant
      (1,4) x 1<4, y 1<3 -> concordant
      (2,3) x 2<3, y 2<4 -> concordant
      (2,4) x 2<4, y 2<3 -> concordant
      (3,4) x 3<4, y 4>3 -> DISCORDANT
    n0 = 4*3/2 = 6, n1 = n2 = 0, so tau_b = (5-1)/sqrt(6*6) = 4/6.
    """
    x = np.asarray([1.0, 2.0, 3.0, 4.0])
    y = np.asarray([1.0, 2.0, 4.0, 3.0])
    assert kendall_tau_b(x, y) == pytest.approx(4.0 / 6.0, rel=1e-15, abs=0.0)


def test_kendall_tau_b_with_ties_both_sides() -> None:
    """x = [1,1,2,3], y = [1,2,2,3]: tau_b = 4/sqrt((6-1)(6-1)) = 4/5 = 0.8.

    Hand enumeration of the 6 pairs:
      (0,1) x tied        -> neither
      (0,2) 1<2 and 1<2   -> concordant
      (0,3) 1<3 and 1<3   -> concordant
      (1,2) 1<2 and y tied-> neither
      (1,3) 1<3 and 2<3   -> concordant
      (2,3) 2<3 and 2<3   -> concordant
    C = 4, D = 0. n0 = 6. x has one tie group of 2, so n1 = 2*1/2 = 1.
    y has one tie group of 2, so n2 = 1. tau_b = 4/sqrt(5*5) = 0.8.
    """
    x = np.asarray([1.0, 1.0, 2.0, 3.0])
    y = np.asarray([1.0, 2.0, 2.0, 3.0])
    assert kendall_tau_b(x, y) == pytest.approx(0.8, rel=1e-15, abs=0.0)


def test_kendall_tau_b_perfect_agreement_is_one() -> None:
    """Identical untied vectors give C = n0 and D = 0, so tau_b = 1."""
    x = np.asarray([3.0, 1.0, 4.0, 1.5, 9.0])
    assert kendall_tau_b(x, x.copy()) == pytest.approx(1.0, rel=1e-15, abs=0.0)


def test_kendall_tau_b_reversal_is_minus_one() -> None:
    """A reversed ordering gives D = n0, so tau_b = -1."""
    x = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    assert kendall_tau_b(x, -x) == pytest.approx(-1.0, rel=1e-15, abs=0.0)


def test_kendall_tau_b_fully_tied_vector_returns_zero() -> None:
    """With one vector entirely tied the denominator of (1) is zero.

    This implementation returns 0.0 and ``scipy.stats.kendalltau`` returns NaN.
    The difference is declared in the module docstring of
    ``flakerank.disagreement`` and asserted here, and it is the ONLY place the
    two implementations are allowed to differ.
    """
    x = np.asarray([1.0, 1.0, 1.0, 1.0])
    y = np.asarray([1.0, 2.0, 3.0, 4.0])
    assert kendall_tau_b(x, y) == 0.0
    assert np.isnan(stats.kendalltau(x, y, variant="b").statistic)


def test_spearman_rho_with_ties_hand_calculated() -> None:
    """x = [1,1,2,3], y = [1,2,2,3]: rho = 3.75/4.5 = 0.8333333333333334.

    Hand arithmetic, using midranks:
      midranks(x) = [1.5, 1.5, 3, 4],  mean 2.5, deviations [-1, -1, 0.5, 1.5]
      midranks(y) = [1, 2.5, 2.5, 4],  mean 2.5, deviations [-1.5, 0, 0, 1.5]
      sum dx*dy = (-1)(-1.5) + (-1)(0) + (0.5)(0) + (1.5)(1.5) = 1.5 + 2.25 = 3.75
      sum dx^2  = 1 + 1 + 0.25 + 2.25 = 4.5
      sum dy^2  = 2.25 + 0 + 0 + 2.25 = 4.5
      rho = 3.75 / sqrt(4.5 * 4.5) = 3.75/4.5 = 0.8333333333333334
    """
    x = np.asarray([1.0, 1.0, 2.0, 3.0])
    y = np.asarray([1.0, 2.0, 2.0, 3.0])
    assert spearman_rho(x, y) == pytest.approx(3.75 / 4.5, rel=1e-14, abs=0.0)


def test_spearman_rho_untied_shortcut_agrees() -> None:
    """With no ties the midrank form equals 1 - 6*sum(d^2)/(n^3 - n).

    Hand arithmetic for x = [1,2,3,4], y = [1,2,4,3]: rank differences are
    0, 0, -1, +1 so sum(d^2) = 2, n = 4, n^3 - n = 60, and the shortcut gives
    1 - 12/60 = 0.8.
    """
    x = np.asarray([1.0, 2.0, 3.0, 4.0])
    y = np.asarray([1.0, 2.0, 4.0, 3.0])
    assert spearman_rho(x, y) == pytest.approx(0.8, rel=1e-14, abs=0.0)


def test_spearman_rho_zero_variance_returns_zero() -> None:
    """An entirely tied vector has zero midrank variance; rho is defined as 0."""
    assert spearman_rho(np.asarray([2.0, 2.0, 2.0]), np.asarray([1.0, 2.0, 3.0])) == 0.0


# ---------------------------------------------------------------------------
# Cross-check against scipy, an independent implementation
# ---------------------------------------------------------------------------


def test_tau_b_matches_scipy_on_the_committed_history(summaries: tuple) -> None:
    """On the real 72-test vectors our tau-b agrees with scipy to 1e-13."""
    rate = np.asarray([item.failure_probability for item in summaries])
    cost = np.asarray([item.wasted_seconds for item in summaries])
    ours = kendall_tau_b(cost, rate)
    theirs = float(stats.kendalltau(cost, rate, variant="b").statistic)
    assert ours == pytest.approx(theirs, rel=1e-13, abs=1e-15)


def test_rho_matches_scipy_on_the_committed_history(summaries: tuple) -> None:
    """On the real 72-test vectors our rho agrees with scipy to 1e-13."""
    rate = np.asarray([item.failure_probability for item in summaries])
    cost = np.asarray([item.wasted_seconds for item in summaries])
    ours = spearman_rho(cost, rate)
    theirs = float(stats.spearmanr(cost, rate).statistic)
    assert ours == pytest.approx(theirs, rel=1e-13, abs=1e-15)


@given(
    st.lists(
        st.integers(min_value=0, max_value=6), min_size=4, max_size=40
    ),
    st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=150, deadline=None)
def test_tau_b_matches_scipy_under_heavy_ties(values: list[int], seed: int) -> None:
    """Agreement with scipy holds on deliberately tie-heavy integer vectors.

    Small integer ranges are used so ties are frequent, because the tie
    correction is where two implementations of tau-b can differ. Cases where
    either vector is entirely tied are excluded, since the two deliberately
    disagree there and that disagreement has its own test.
    """
    generator = np.random.default_rng(seed)
    x = np.asarray(values, dtype=np.float64)
    y = generator.permutation(x)
    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return
    ours = kendall_tau_b(x, y)
    theirs = float(stats.kendalltau(x, y, variant="b").statistic)
    assert ours == pytest.approx(theirs, rel=1e-12, abs=1e-14)


@given(
    st.lists(st.integers(min_value=0, max_value=6), min_size=4, max_size=40),
    st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=150, deadline=None)
def test_rho_matches_scipy_under_heavy_ties(values: list[int], seed: int) -> None:
    """The midrank correction agrees with scipy's on tie-heavy vectors."""
    generator = np.random.default_rng(seed)
    x = np.asarray(values, dtype=np.float64)
    y = generator.permutation(x)
    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return
    ours = spearman_rho(x, y)
    theirs = float(stats.spearmanr(x, y).statistic)
    assert ours == pytest.approx(theirs, rel=1e-12, abs=1e-14)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


@given(
    st.lists(
        st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=30,
    ),
    st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=200, deadline=None)
def test_tau_b_lies_in_minus_one_to_one(values: list[float], seed: int) -> None:
    """tau-b is a correlation and cannot leave [-1, 1]."""
    generator = np.random.default_rng(seed)
    x = np.asarray(values, dtype=np.float64)
    y = generator.permutation(x)
    assert -1.0 - 1e-12 <= kendall_tau_b(x, y) <= 1.0 + 1e-12


@given(
    st.lists(
        st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=30,
    ),
    st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=200, deadline=None)
def test_rho_lies_in_minus_one_to_one(values: list[float], seed: int) -> None:
    """rho is a correlation and cannot leave [-1, 1]."""
    generator = np.random.default_rng(seed)
    x = np.asarray(values, dtype=np.float64)
    y = generator.permutation(x)
    assert -1.0 - 1e-12 <= spearman_rho(x, y) <= 1.0 + 1e-12


@given(
    st.lists(
        st.floats(min_value=1e-3, max_value=1e4, allow_nan=False, allow_infinity=False),
        min_size=3,
        max_size=25,
        unique=True,
    ),
    st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=200, deadline=None)
def test_both_statistics_are_invariant_to_a_positive_rescaling(
    values: list[float], scale: float
) -> None:
    """A rank statistic depends only on the order, so scaling one side is a no-op.

    Exact in exact arithmetic: a positive scaling preserves every pairwise
    comparison. Compared to tolerance rather than exactly, because the scaled
    values are rounded and a tie could be created or broken by that rounding.
    """
    x = np.asarray(values, dtype=np.float64)
    y = np.asarray(sorted(values), dtype=np.float64)
    assert kendall_tau_b(x, y) == pytest.approx(
        kendall_tau_b(x * scale, y), rel=1e-9, abs=1e-12
    )
    assert spearman_rho(x, y) == pytest.approx(
        spearman_rho(x * scale, y), rel=1e-9, abs=1e-12
    )


@given(
    st.lists(
        st.floats(min_value=1e-3, max_value=1e4, allow_nan=False, allow_infinity=False),
        min_size=3,
        max_size=25,
        unique=True,
    )
)
@settings(max_examples=150, deadline=None)
def test_both_statistics_are_symmetric(values: list[float]) -> None:
    """tau-b and rho do not depend on which vector is called first."""
    x = np.asarray(values, dtype=np.float64)
    y = np.asarray(sorted(values, reverse=True), dtype=np.float64)
    assert kendall_tau_b(x, y) == pytest.approx(kendall_tau_b(y, x), rel=1e-12, abs=1e-15)
    assert spearman_rho(x, y) == pytest.approx(spearman_rho(y, x), rel=1e-12, abs=1e-15)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_unequal_lengths_rejected() -> None:
    """A rank correlation compares two orderings of the SAME items."""
    with pytest.raises(ValueError, match="must have equal length"):
        kendall_tau_b(np.asarray([1.0, 2.0]), np.asarray([1.0, 2.0, 3.0]))


def test_single_item_rejected() -> None:
    """With one item there are no pairs to be concordant or discordant."""
    with pytest.raises(ValueError, match="at least 2 items"):
        spearman_rho(np.asarray([1.0]), np.asarray([2.0]))


def test_nan_rejected() -> None:
    """NaN has no rank and is rejected rather than sorted to an end."""
    with pytest.raises(ValueError, match="NaN"):
        kendall_tau_b(np.asarray([1.0, float("nan")]), np.asarray([1.0, 2.0]))
    with pytest.raises(ValueError, match="NaN"):
        midranks(np.asarray([1.0, float("nan")]))


def test_midranks_empty_rejected() -> None:
    """An empty vector has no ranks."""
    with pytest.raises(ValueError, match="non-empty"):
        midranks(np.asarray([]))


def test_two_dimensional_input_rejected() -> None:
    """A score vector is one-dimensional."""
    with pytest.raises(ValueError, match="one-dimensional"):
        midranks(np.zeros((2, 2)))


def test_rank_disagreement_validates_names_length() -> None:
    """A names/scores mismatch means the rankings came from different sets."""
    with pytest.raises(ValueError, match="names has"):
        rank_disagreement(
            ("a", "b", "c"),
            np.asarray([1.0, 2.0]),
            np.asarray([2.0, 1.0]),
            label_x="x",
            label_y="y",
            resamples=10,
        )


@pytest.mark.parametrize("field,value", [("top_k", 0), ("top_movers", -1)])
def test_rank_disagreement_validates_counts(field: str, value: int) -> None:
    """``top_k`` is at least 1 and ``top_movers`` is non-negative."""
    kwargs = {"top_k": 5, "top_movers": 5}
    kwargs[field] = value
    with pytest.raises(ValueError, match=field):
        rank_disagreement(
            ("a", "b"),
            np.asarray([1.0, 2.0]),
            np.asarray([2.0, 1.0]),
            label_x="x",
            label_y="y",
            resamples=10,
            **kwargs,  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# The assembled disagreement object
# ---------------------------------------------------------------------------


def test_disagreement_declares_its_resampling_unit() -> None:
    """Both intervals resample TESTS, and the declaration is on the object."""
    names = tuple(f"t{index}" for index in range(12))
    x = np.arange(12.0)
    y = np.arange(12.0)[::-1].copy()
    result = rank_disagreement(
        names, x, y, label_x="x", label_y="y", resamples=40, seed=7
    )
    assert result.tau_b_interval.resampling_unit == "test"
    assert result.rho_interval.resampling_unit == "test"
    result.tau_b_interval.require_resampling_unit("test")
    with pytest.raises(ValueError, match="resampling"):
        result.tau_b_interval.require_resampling_unit("session")


def test_disagreement_on_a_reversal_reports_full_displacement() -> None:
    """A reversed ranking of 12 items moves the ends by 11 positions.

    Hand arithmetic: with 12 items, position 1 becomes position 12, so the
    displacement is 1 - 12 = -11 and the maximum absolute displacement is 11.
    The top-10 overlap of a ranking against its own reversal is 8: positions
    1..10 of one are positions 3..12 of the other, and the sets {1..10} and
    {3..12} share 8 members.
    """
    names = tuple(f"t{index:02d}" for index in range(12))
    x = np.arange(12.0)
    y = np.arange(12.0)[::-1].copy()
    result = rank_disagreement(
        names, x, y, label_x="x", label_y="y", resamples=40, seed=7, top_k=10
    )
    assert result.tau_b == pytest.approx(-1.0, rel=1e-15, abs=0.0)
    assert result.max_displacement == 11
    assert result.top_k_overlap == 8


def test_disagreement_movers_are_deterministic() -> None:
    """Equal displacements are ordered by name, not by history order."""
    names = ("b_test", "a_test", "c_test")
    x = np.asarray([1.0, 2.0, 3.0])
    y = np.asarray([3.0, 2.0, 1.0])
    first = rank_disagreement(
        names, x, y, label_x="x", label_y="y", resamples=20, seed=3
    )
    second = rank_disagreement(
        names, x, y, label_x="x", label_y="y", resamples=20, seed=3
    )
    assert [mover.name for mover in first.movers] == [mover.name for mover in second.movers]
