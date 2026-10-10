"""Hypothesis property tests for the algebraic identities.

The identities tested here are the ones the package claims in its docstrings.
The five-term Brier identity is the load-bearing one: it is algebra, so it
must hold for *every* input, not only for the samples that happen to be in the
fixtures.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from calibaudit.binning import assign_bins, bin_edges
from calibaudit.decomposition import binned_decomposition, murphy_decomposition
from calibaudit.ece import calibration_gaps, expected_calibration_error
from calibaudit.scores import brier_score, log_score

_probs = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
_outcomes = st.sampled_from([0.0, 1.0])

_pairs = st.lists(st.tuples(_probs, _outcomes), min_size=1, max_size=120)
_pairs_multi = st.lists(st.tuples(_probs, _outcomes), min_size=2, max_size=120)


def _split(pairs):
    f = np.array([p[0] for p in pairs], dtype=float)
    o = np.array([p[1] for p in pairs], dtype=float)
    return f, o


@settings(max_examples=250, deadline=None)
@given(_pairs, st.integers(min_value=1, max_value=40))
def test_five_term_identity_holds_for_every_input(pairs, n_bins):
    f, o = _split(pairs)
    d = binned_decomposition(f, o, n_bins=n_bins)
    assert abs(d.identity_residual) <= 1e-13


@settings(max_examples=250, deadline=None)
@given(_pairs, st.integers(min_value=1, max_value=40))
def test_five_term_identity_holds_for_equal_mass_bins(pairs, n_bins):
    f, o = _split(pairs)
    d = binned_decomposition(f, o, n_bins=n_bins, strategy="equal_mass")
    assert abs(d.identity_residual) <= 1e-13


@settings(max_examples=250, deadline=None)
@given(_pairs)
def test_three_term_identity_is_exact_for_the_exact_decomposition(pairs):
    f, o = _split(pairs)
    d = murphy_decomposition(f, o)
    assert abs(d.three_term_residual) <= 1e-14
    assert d.within_bin_variance == 0.0
    assert d.within_bin_covariance == 0.0


@settings(max_examples=200, deadline=None)
@given(_pairs, st.integers(min_value=1, max_value=30))
def test_decomposition_terms_are_in_range(pairs, n_bins):
    f, o = _split(pairs)
    d = binned_decomposition(f, o, n_bins=n_bins)
    assert d.reliability >= -1e-18
    assert d.resolution >= -1e-18
    assert -1e-18 <= d.uncertainty <= 0.25 + 1e-18
    assert d.within_bin_variance >= -1e-18
    assert 0.0 <= d.brier <= 1.0 + 1e-15


@settings(max_examples=200, deadline=None)
@given(_pairs, st.integers(min_value=1, max_value=30))
def test_resolution_never_exceeds_uncertainty(pairs, n_bins):
    # Law of total variance: RES is the between-bin part of Var(o).
    f, o = _split(pairs)
    d = binned_decomposition(f, o, n_bins=n_bins)
    assert d.resolution <= d.uncertainty + 1e-14


@settings(max_examples=200, deadline=None)
@given(_pairs)
def test_brier_equals_mean_squared_error(pairs):
    f, o = _split(pairs)
    assert brier_score(f, o) == np.mean((f - o) ** 2)


@settings(max_examples=200, deadline=None)
@given(_pairs)
def test_brier_is_bounded_by_zero_and_one(pairs):
    f, o = _split(pairs)
    assert 0.0 <= brier_score(f, o) <= 1.0


@settings(max_examples=200, deadline=None)
@given(_pairs)
def test_log_score_is_non_negative_and_finite(pairs):
    f, o = _split(pairs)
    value = log_score(f, o)
    assert value >= 0.0
    assert np.isfinite(value)


@settings(max_examples=200, deadline=None)
@given(_pairs, st.integers(min_value=1, max_value=40))
def test_ece_is_bounded_and_matches_the_weighted_gap_sum(pairs, n_bins):
    f, o = _split(pairs)
    ece = expected_calibration_error(f, o, n_bins=n_bins)
    weights, gaps = calibration_gaps(f, o, n_bins=n_bins)
    assert 0.0 <= ece <= 1.0 + 1e-15
    assert ece == float(np.sum(weights * np.abs(gaps)))
    assert abs(float(weights.sum()) - 1.0) <= 1e-13


@settings(max_examples=200, deadline=None)
@given(_pairs, st.integers(min_value=1, max_value=40))
def test_ece_never_exceeds_mce(pairs, n_bins):
    from calibaudit.ece import maximum_calibration_error

    f, o = _split(pairs)
    assert (
        expected_calibration_error(f, o, n_bins=n_bins)
        <= maximum_calibration_error(f, o, n_bins=n_bins) + 1e-15
    )


@settings(max_examples=200, deadline=None)
@given(_pairs)
def test_ece_with_one_bin_equals_the_absolute_overall_gap(pairs):
    # Equal to within one ulp, not bit-identical: the bin accumulation uses
    # numpy.bincount's sequential summation while ndarray.mean uses pairwise
    # summation, and the two differ in the last bit on some inputs.
    f, o = _split(pairs)
    assert expected_calibration_error(f, o, n_bins=1) == pytest.approx(
        abs(float(f.mean() - o.mean())), abs=1e-15
    )


@settings(max_examples=200, deadline=None)
@given(
    st.lists(_probs, min_size=1, max_size=120), st.integers(min_value=1, max_value=40)
)
def test_equal_width_edges_are_independent_of_the_sample(forecasts, n_bins):
    a = bin_edges(forecasts, n_bins=n_bins, strategy="equal_width")
    b = bin_edges(np.zeros(1), n_bins=n_bins, strategy="equal_width")
    assert np.array_equal(a, b)


@settings(max_examples=250, deadline=None)
@given(
    st.lists(_probs, min_size=1, max_size=120),
    st.integers(min_value=1, max_value=40),
    st.sampled_from(["equal_width", "equal_mass"]),
)
def test_bin_edges_are_monotone_and_cover_the_unit_interval(forecasts, n_bins, strategy):
    edges = bin_edges(forecasts, n_bins=n_bins, strategy=strategy)
    assert edges.size == n_bins + 1
    assert np.all(np.diff(edges) >= 0.0)
    assert edges[0] <= 0.0
    assert edges[-1] >= 1.0


@settings(max_examples=250, deadline=None)
@given(
    st.lists(_probs, min_size=1, max_size=120),
    st.integers(min_value=1, max_value=40),
    st.sampled_from(["equal_width", "equal_mass"]),
)
def test_every_forecast_gets_a_label_in_range(forecasts, n_bins, strategy):
    labels, edges = assign_bins(forecasts, n_bins=n_bins, strategy=strategy)
    assert labels.shape == (len(forecasts),)
    assert int(labels.min()) >= 0
    assert int(labels.max()) <= n_bins - 1


@settings(max_examples=150, deadline=None)
@given(_pairs_multi)
def test_adding_a_constant_shift_to_forecasts_cannot_lower_a_perfect_score(pairs):
    # A perfectly calibrated deterministic forecast scores 0; perturbing it
    # cannot help. Checks the sign convention is negatively oriented.
    _, o = _split(pairs)
    assert brier_score(o, o) == 0.0
    perturbed = np.clip(o * 0.9 + 0.05, 0.0, 1.0)
    assert brier_score(perturbed, o) >= 0.0


@settings(max_examples=150, deadline=None)
@given(_pairs, st.integers(min_value=2, max_value=30))
def test_equal_mass_bins_are_balanced_up_to_ties(pairs, n_bins):
    f, _ = _split(pairs)
    labels, _ = assign_bins(f, n_bins=n_bins, strategy="equal_mass")
    counts = np.bincount(labels, minlength=n_bins)
    distinct = np.unique(f).size
    if distinct >= n_bins:
        # With enough distinct values no bin should hold more than a few times
        # the balanced share; quantile edges cannot do better under ties.
        share = f.size / n_bins
        assert counts.max() <= max(share * 4.0, share + distinct)
