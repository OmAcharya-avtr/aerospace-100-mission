"""Timing-error detectors: algebra, sign convention, phase invariance, validation."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from slotsync.detectors import (
    detector_by_name,
    early_late,
    early_late_dd,
    gardner,
    gardner_complex,
    mueller_muller,
    slice_antipodal,
)


def test_early_late_is_the_difference_hand_computed() -> None:
    # early = 0.8, late = 0.5 -> e = 0.3; squared form -> 0.64 - 0.25 = 0.39.
    early = np.array([0.8])
    late = np.array([0.5])
    assert early_late(early, late)[0] == pytest.approx(0.3)
    assert early_late(early, late, square=True)[0] == pytest.approx(0.39)


def test_early_late_decision_directed_removes_the_data_sign() -> None:
    # A '-1' symbol inverts the gates, so without the decision the sign flips.
    assert early_late_dd(np.array([-1.0]), np.array([-0.8]), np.array([-0.5]))[0] == pytest.approx(
        0.3
    )
    assert early_late_dd(np.array([1.0]), np.array([0.8]), np.array([0.5]))[0] == pytest.approx(0.3)


def test_gardner_is_mid_times_the_strobe_difference_hand_computed() -> None:
    # mid = 0.5, strobe = 1.0, previous = -1.0 -> e = 0.5 * (1.0 - (-1.0)) = 1.0.
    value = gardner(np.array([-1.0]), np.array([0.5]), np.array([1.0]))[0]
    assert value == pytest.approx(1.0)


def test_gardner_sign_convention_late_is_positive_hand_computed() -> None:
    # Triangular pulse of half-width one symbol, data (a[k-1], a[k]) = (-1, +1),
    # sampling late by eps. Then
    #   x[k]     = p(eps) * (+1)              = 1 - eps
    #   x[k-1]   = -p(eps) + p(eps - 1)       = -(1 - eps) + eps = -1 + 2 eps
    #   x[k-1/2] = -p(eps + 1/2) + p(eps - 1/2) = -(1/2 - eps) + (1/2 + eps) = 2 eps
    # so e = 2 eps * ((1 - eps) - (-1 + 2 eps)) = 2 eps (2 - 3 eps) > 0 for small eps > 0.
    eps = 0.01
    strobe = 1.0 - eps
    previous = -1.0 + 2.0 * eps
    mid = 2.0 * eps
    expected = 2.0 * eps * (2.0 - 3.0 * eps)
    assert gardner(np.array([previous]), np.array([mid]), np.array([strobe]))[0] == pytest.approx(
        expected
    )
    assert expected > 0.0


def test_mueller_muller_is_the_cross_product_hand_computed() -> None:
    # Same triangular construction: a[k] = +1, a[k-1] = -1, x[k] = 1 - eps,
    # x[k-1] = -1 + 2 eps, so
    #   e = a[k] x[k-1] - a[k-1] x[k] = (-1 + 2 eps) + (1 - eps) = eps.
    eps = 0.01
    value = mueller_muller(
        np.array([-1.0]), np.array([1.0]), np.array([-1.0 + 2.0 * eps]), np.array([1.0 - eps])
    )[0]
    assert value == pytest.approx(eps)


@settings(max_examples=80, deadline=None)
@given(
    st.floats(min_value=-np.pi, max_value=np.pi),
    st.floats(min_value=-2.0, max_value=2.0),
    st.floats(min_value=-2.0, max_value=2.0),
    st.floats(min_value=-2.0, max_value=2.0),
    st.floats(min_value=-2.0, max_value=2.0),
    st.floats(min_value=-2.0, max_value=2.0),
    st.floats(min_value=-2.0, max_value=2.0),
)
def test_gardner_complex_is_invariant_to_carrier_phase(
    phase: float, pr: float, pi: float, mr: float, mi: float, sr: float, si: float
) -> None:
    """The property that makes Gardner's detector the one used before carrier lock."""
    previous = np.array([complex(pr, pi)])
    mid = np.array([complex(mr, mi)])
    strobe = np.array([complex(sr, si)])
    rotation = np.exp(1j * phase)
    base = gardner_complex(previous, mid, strobe)[0]
    rotated = gardner_complex(previous * rotation, mid * rotation, strobe * rotation)[0]
    assert rotated == pytest.approx(base, abs=1e-12)


def test_gardner_complex_reduces_to_the_real_detector() -> None:
    previous = np.array([-1.0, 0.3])
    mid = np.array([0.5, -0.2])
    strobe = np.array([1.0, 0.7])
    assert np.allclose(
        gardner_complex(previous.astype(complex), mid.astype(complex), strobe.astype(complex)),
        gardner(previous, mid, strobe),
    )


def test_gardner_delivers_nothing_without_a_transition() -> None:
    # x[k] == x[k-1] on a run of identical symbols: the detector output is zero
    # whatever the timing error is.
    assert gardner(np.array([0.9]), np.array([0.4]), np.array([0.9]))[0] == 0.0


def test_slice_antipodal_maps_zero_to_plus_one() -> None:
    assert np.array_equal(
        slice_antipodal(np.array([-3.0, -1e-30, 0.0, 0.2])), np.array([-1.0, -1.0, 1.0, 1.0])
    )


def test_shape_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="same shape"):
        early_late(np.zeros(3), np.zeros(4))
    with pytest.raises(ValueError, match="same shape"):
        gardner(np.zeros(2), np.zeros(2), np.zeros(3))
    with pytest.raises(ValueError, match="same shape"):
        gardner_complex(
            np.zeros(2, dtype=complex),
            np.zeros(2, dtype=complex),
            np.zeros(3, dtype=complex),
        )


def test_non_finite_samples_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        early_late(np.array([np.nan]), np.array([0.0]))
    with pytest.raises(ValueError, match="non-finite"):
        gardner_complex(
            np.array([complex(np.inf, 0.0)]), np.array([0j]), np.array([0j])
        )


def test_empty_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one sample"):
        early_late(np.array([]), np.array([]))


def test_detector_name_validation() -> None:
    assert detector_by_name("gardner") == "gardner"
    with pytest.raises(ValueError, match="unknown detector"):
        detector_by_name("zero-crossing")
