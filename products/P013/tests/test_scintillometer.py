"""Tests for turbscope.scintillometer."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from turbscope.scintillometer import (
    SATURATION_ASYMPTOTE,
    invert_cn2_all_roots,
    invert_cn2_weak,
    is_weak_regime,
    rytov_variance,
    saturation_peak,
    scintillation_index_full,
    wave_number,
)

LAM = 500e-9


# --------------------------------------------------------------------- KAT
def test_wave_number_known_value():
    # k = 2*pi / 500e-9 = 12566370.614359174 rad/m (hand computed)
    assert wave_number(LAM) == pytest.approx(12566370.614359174, rel=1e-12)


def test_rytov_variance_plane_known_value():
    # sigma_R^2 = 1.23 * 1e-14 * k^(7/6) * 1000^(11/6)
    #   k^(7/6) with k=12566370.614359174 -> 1.23*1e-14*k^(7/6)*1000^(11/6)
    #   computed by hand (numpy) = 0.7452751136947144 (see test derivation script)
    v = rytov_variance(1e-14, 1000.0, LAM, "plane")
    assert float(v) == pytest.approx(0.7452751136947144, rel=1e-9)


def test_rytov_variance_spherical_known_value():
    # spherical coefficient 0.50 is exactly 1.23/0.50 = 2.46x smaller than plane
    v_sph = rytov_variance(1e-14, 1000.0, LAM, "spherical")
    v_pl = rytov_variance(1e-14, 1000.0, LAM, "plane")
    assert float(v_sph) == pytest.approx(0.30295736329053435, rel=1e-9)
    assert float(v_pl / v_sph) == pytest.approx(1.23 / 0.50, rel=1e-12)


def test_invert_cn2_weak_known_value():
    # weak inversion is the exact algebraic inverse of rytov_variance
    sigma = 0.7452751136947144
    cn2 = invert_cn2_weak(sigma, 1000.0, LAM, "plane")
    assert cn2 == pytest.approx(1e-14, rel=1e-9)


def test_scintillation_index_full_weak_limit_matches_rytov():
    # by construction sigma_I^2(x) -> x as x -> 0; at x=1e-6 the two agree to
    # better than 1e-5 relative (verified numerically, not asserted a priori)
    x = 1e-6
    assert float(scintillation_index_full(x)) == pytest.approx(x, rel=2e-5)


def test_scintillation_index_full_zero_is_zero():
    assert float(scintillation_index_full(0.0)) == pytest.approx(0.0, abs=1e-15)


# --------------------------------------------------------------- validation
@pytest.mark.parametrize(
    "cn2,length,lam,wave_type,exc",
    [
        (-1e-14, 1000.0, LAM, "plane", ValueError),
        (float("nan"), 1000.0, LAM, "plane", ValueError),
        (1e-14, 0.0, LAM, "plane", ValueError),
        (1e-14, -100.0, LAM, "plane", ValueError),
        (1e-14, 1000.0, -500e-9, "plane", ValueError),
        (1e-14, 1000.0, 1.0, "plane", ValueError),  # wavelength out of range
        (1e-14, 1000.0, LAM, "circular", ValueError),  # unknown wave type
    ],
)
def test_rytov_variance_rejects_invalid_input(cn2, length, lam, wave_type, exc):
    with pytest.raises(exc):
        rytov_variance(cn2, length, lam, wave_type)


def test_invert_cn2_weak_rejects_negative_measurement():
    with pytest.raises(ValueError):
        invert_cn2_weak(-0.1, 1000.0, LAM)


def test_scintillation_index_full_rejects_negative():
    with pytest.raises(ValueError):
        scintillation_index_full(-1.0)


def test_is_weak_regime_rejects_negative():
    with pytest.raises(ValueError):
        is_weak_regime(-0.5)


def test_invert_cn2_all_roots_rejects_bad_bracket():
    with pytest.raises(ValueError):
        invert_cn2_all_roots(1.0, 1000.0, LAM, rytov_search_max=-1.0)


def test_invert_cn2_all_roots_rejects_negative_measurement():
    with pytest.raises(ValueError):
        invert_cn2_all_roots(-0.1, 1000.0, LAM)


# -------------------------------------------------------------------- edge
def test_rytov_variance_zero_cn2_gives_zero():
    assert float(rytov_variance(0.0, 1000.0, LAM)) == 0.0


def test_rytov_variance_vectorised_matches_scalar():
    cn2 = np.array([1e-16, 1e-14, 1e-12])
    vec = rytov_variance(cn2, 500.0, LAM, "plane")
    scalars = [float(rytov_variance(c, 500.0, LAM, "plane")) for c in cn2]
    np.testing.assert_allclose(vec, scalars, rtol=1e-12)


def test_wave_number_boundary_wavelengths_accepted():
    assert wave_number(1.0e-7) > 0.0
    assert wave_number(1.0e-4) > 0.0


def test_wave_number_rejects_zero_and_negative():
    with pytest.raises(ValueError):
        wave_number(0.0)
    with pytest.raises(ValueError):
        wave_number(-1e-6)


# ---------------------------------------------------------- saturation shape
def test_saturation_peak_exceeds_asymptote():
    x_peak, val_peak = saturation_peak()
    assert x_peak > 0.0
    assert val_peak > SATURATION_ASYMPTOTE


def test_saturation_curve_approaches_asymptote_at_large_x():
    far = float(scintillation_index_full(500.0))
    assert far == pytest.approx(SATURATION_ASYMPTOTE, abs=0.05)


def test_saturation_curve_is_non_monotonic_past_the_peak():
    # value falls after the peak before re-approaching the asymptote
    x_peak, val_peak = saturation_peak()
    just_after = float(scintillation_index_full(x_peak * 3.0))
    assert just_after < val_peak


def test_invert_cn2_all_roots_finds_two_roots_between_asymptote_and_peak():
    _, val_peak = saturation_peak()
    target = 0.5 * (SATURATION_ASYMPTOTE + val_peak)  # strictly between the two
    result = invert_cn2_all_roots(target, 1000.0, LAM)
    assert result.is_multivalued
    assert len(result.rytov_roots) >= 2
    for r in result.rytov_roots:
        assert float(scintillation_index_full(r)) == pytest.approx(target, abs=1e-6)


def test_invert_cn2_all_roots_single_root_in_weak_regime():
    result = invert_cn2_all_roots(1e-4, 1000.0, LAM)
    assert not result.is_multivalued
    assert len(result.rytov_roots) == 1


def test_invert_cn2_all_roots_cn2_ordering_matches_rytov_ordering():
    # Cn2 = sigma_R^2 / const, const > 0, so root ordering is preserved
    _, val_peak = saturation_peak()
    target = 0.5 * (SATURATION_ASYMPTOTE + val_peak)
    result = invert_cn2_all_roots(target, 1000.0, LAM)
    assert list(result.rytov_roots) == sorted(result.rytov_roots)
    assert list(result.cn2_roots) == sorted(result.cn2_roots)


# --------------------------------------------------------------- properties
@given(st.floats(min_value=1e-20, max_value=1e-10), st.floats(min_value=10.0, max_value=5000.0))
def test_weak_round_trip_is_exact_algebraic_identity(cn2, length):
    """invert_cn2_weak is the exact algebraic inverse of rytov_variance (identity,
    not an approximation -- both are the same linear formula run forwards and
    backwards), for any positive Cn2 and path length."""
    sigma = float(rytov_variance(cn2, length, LAM, "plane"))
    back = invert_cn2_weak(sigma, length, LAM, "plane")
    assert back == pytest.approx(cn2, rel=1e-9)


@given(st.floats(min_value=1e-6, max_value=5e-2))
def test_scintillation_index_full_weak_tail_matches_identity(x):
    """For small sigma_R^2 the full curve is close to the identity, with error
    shrinking as x shrinks (both terms of the bridging function have zero
    derivative curvature mismatch at x=0 by construction)."""
    val = float(scintillation_index_full(x))
    assert val == pytest.approx(x, rel=0.05, abs=1e-6)


@given(st.floats(min_value=0.0, max_value=1e3))
def test_scintillation_index_full_is_never_negative(x):
    assert float(scintillation_index_full(x)) >= 0.0


@given(st.floats(min_value=1e-18, max_value=1e-10), st.floats(min_value=50.0, max_value=3000.0))
def test_rytov_variance_scales_linearly_with_cn2(cn2, length):
    """sigma_R^2 is exactly proportional to Cn2 at fixed L, lambda, wave_type."""
    base = float(rytov_variance(cn2, length, LAM, "plane"))
    doubled = float(rytov_variance(2.0 * cn2, length, LAM, "plane"))
    assert doubled == pytest.approx(2.0 * base, rel=1e-9)
