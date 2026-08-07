"""Known-answer, property and validation tests for the integrated seeing quantities."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cncast.baselines import bufton_wind, hv57
from cncast.seeing import (
    fried_parameter,
    greenwood_frequency,
    isoplanatic_angle,
    seeing_fwhm_arcsec,
    turbulence_moment,
)

FINE = np.linspace(0.0, 20_000.0, 20_001)


# ---------------------------------------------------------------- known answers


def test_r0_constant_slab_hand_computation() -> None:
    """Hand check: constant Cn^2 = 1e-15 over 0-10 km at lambda = 1.55 um.

    mu_0 = 1e-15 * 1e4                    = 1.0e-11 m^(1/3)
    k    = 2 pi / 1.55e-6                 = 4.0536671e6 rad/m
    k^2                                   = 1.6432217e13
    0.423 k^2 mu_0 = 0.423*1.6432217e13*1e-11 = 69.508278
    r0   = 69.508278^(-3/5):
           ln 69.508278 = 4.2414455;  -0.6 * that = -2.5448673
           e^-2.5448673 = 0.07848343 m = 7.848343 cm
    """
    h = np.linspace(0.0, 10_000.0, 10_001)
    cn2 = np.full_like(h, 1e-15)
    assert fried_parameter(h, cn2, 1.55e-6, 0.0) == pytest.approx(0.07848343, rel=1e-6)


def test_mu_53_constant_slab_closed_form() -> None:
    """int_0^H Cn^2 h^(5/3) dh = Cn^2 H^(8/3) / (8/3) for constant Cn^2.

    Cn^2 = 1e-15, H = 1e4 m: 1e-15 * (1e4)^(8/3) / (8/3)
                            = 1e-15 * 4.6415888e10 / 2.6666667
                            = 1.7405958e-5 m^2
    """
    h = np.linspace(0.0, 10_000.0, 100_001)
    cn2 = np.full_like(h, 1e-15)
    assert turbulence_moment(h, cn2, 5 / 3) == pytest.approx(1.7405958e-5, rel=1e-5)


def test_hv57_reproduces_its_five_seven_nickname() -> None:
    """HV 5/7 is named for r0 ~ 5 cm and theta0 ~ 7 urad at 500 nm, zenith.

    Computed here: r0 = 4.962 cm (-0.75 % from the nominal 5 cm) and
    theta0 = 7.011 urad (+0.16 % from the nominal 7 urad).  The tolerances below
    are the published rounding, not tuned tolerances.
    """
    cn2 = hv57(FINE)
    r0_cm = fried_parameter(FINE, cn2, 500e-9, 0.0) * 100.0
    th0_urad = isoplanatic_angle(FINE, cn2, 500e-9, 0.0) * 1e6
    assert r0_cm == pytest.approx(5.0, rel=0.02)
    assert th0_urad == pytest.approx(7.0, rel=0.02)


def test_greenwood_prefactor_identity() -> None:
    """2.31 lambda^(-6/5) == (0.102 k^2)^(3/5): the two published forms agree.

    (0.102 * (2 pi)^2)^(3/5) = (4.0275)^(3/5) = 2.3069, i.e. the coded 2.31
    prefactor to three significant figures.
    """
    lam = 500e-9
    k = 2 * np.pi / lam
    cn2 = hv57(FINE)
    wind = bufton_wind(FINE, 5.0)
    coded = greenwood_frequency(FINE, cn2, wind, lam, 0.0)
    integral = float(np.trapezoid(cn2 * wind ** (5 / 3), FINE))
    alternative = (0.102 * k**2 * integral) ** (3 / 5)
    assert coded == pytest.approx(alternative, rel=2e-3)


def test_seeing_fwhm_known_value() -> None:
    """FWHM = 0.98 lambda / r0.  For r0 = 10 cm at 500 nm:

    0.98 * 5e-7 / 0.1 = 4.9e-6 rad = 4.9e-6 * 206264.806 = 1.01070 arcsec.
    """
    assert seeing_fwhm_arcsec(0.10, 500e-9) == pytest.approx(1.01070, rel=1e-5)


# ---------------------------------------------------------------- scaling properties


@given(lam_nm=st.floats(min_value=400.0, max_value=2000.0))
@settings(max_examples=25, deadline=None)
def test_r0_scales_as_wavelength_to_the_six_fifths(lam_nm: float) -> None:
    """r0 ~ lambda^(6/5) exactly, since only k^2 carries the wavelength."""
    cn2 = hv57(FINE)
    base = fried_parameter(FINE, cn2, 500e-9, 0.0)
    got = fried_parameter(FINE, cn2, lam_nm * 1e-9, 0.0)
    assert got == pytest.approx(base * (lam_nm / 500.0) ** (6 / 5), rel=1e-9)


@given(zeta=st.floats(min_value=0.0, max_value=70.0))
@settings(max_examples=25, deadline=None)
def test_zenith_scaling_exponents(zeta: float) -> None:
    """r0 ~ cos(z)^(3/5) and theta0 ~ cos(z)^(8/5) from sec z and sec^(8/3) z."""
    cn2 = hv57(FINE)
    c = np.cos(np.radians(zeta))
    assert fried_parameter(FINE, cn2, 500e-9, zeta) == pytest.approx(
        fried_parameter(FINE, cn2, 500e-9, 0.0) * c ** (3 / 5), rel=1e-9
    )
    assert isoplanatic_angle(FINE, cn2, 500e-9, zeta) == pytest.approx(
        isoplanatic_angle(FINE, cn2, 500e-9, 0.0) * c ** (8 / 5), rel=1e-9
    )


@given(factor=st.floats(min_value=0.05, max_value=20.0))
@settings(max_examples=25, deadline=None)
def test_r0_scales_as_profile_amplitude_to_the_minus_three_fifths(factor: float) -> None:
    """Scaling the whole profile by c scales r0 by c^(-3/5) and theta0 likewise."""
    cn2 = hv57(FINE)
    r_base = fried_parameter(FINE, cn2, 500e-9, 0.0)
    t_base = isoplanatic_angle(FINE, cn2, 500e-9, 0.0)
    assert fried_parameter(FINE, factor * cn2, 500e-9, 0.0) == pytest.approx(
        r_base * factor ** (-3 / 5), rel=1e-9
    )
    assert isoplanatic_angle(FINE, factor * cn2, 500e-9, 0.0) == pytest.approx(
        t_base * factor ** (-3 / 5), rel=1e-9
    )


def test_stronger_turbulence_gives_smaller_r0() -> None:
    """Monotonicity: adding turbulence anywhere can only shrink r0."""
    cn2 = hv57(FINE)
    extra = cn2.copy()
    extra[5000:6000] += 1e-16
    assert fried_parameter(FINE, extra, 500e-9, 0.0) < fried_parameter(FINE, cn2, 500e-9, 0.0)


def test_high_layer_dominates_isoplanatic_angle() -> None:
    """theta0 uses an h^(5/3) weight, so a high layer hurts it far more than r0.

    Two profiles carry the same integrated turbulence, one concentrated at 500 m
    and one at 10 km.  r0 is identical; theta0 must be much smaller for the high
    layer.
    """
    h = np.linspace(0.0, 20_000.0, 20_001)
    low = np.zeros_like(h)
    high = np.zeros_like(h)
    low[400:600] = 1e-15
    high[9900:10100] = 1e-15
    assert fried_parameter(h, low, 500e-9, 0.0) == pytest.approx(
        fried_parameter(h, high, 500e-9, 0.0), rel=1e-9
    )
    assert isoplanatic_angle(h, high, 500e-9, 0.0) < 0.1 * isoplanatic_angle(h, low, 500e-9, 0.0)


# ---------------------------------------------------------------- input validation


def test_mismatched_shapes_rejected() -> None:
    with pytest.raises(ValueError, match="same shape"):
        fried_parameter(np.linspace(0, 10, 5), np.ones(4))


def test_non_increasing_altitudes_rejected() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        fried_parameter(np.array([0.0, 100.0, 50.0]), np.full(3, 1e-15))


def test_negative_cn2_rejected() -> None:
    with pytest.raises(ValueError, match="Cn\\^2"):
        turbulence_moment(np.array([0.0, 100.0]), np.array([1e-15, -1e-15]))


def test_single_point_profile_rejected() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        turbulence_moment(np.array([0.0]), np.array([1e-15]))


@pytest.mark.parametrize("lam", [0.0, -1.0, 1e-9, 1e-2])
def test_out_of_range_wavelength_rejected(lam: float) -> None:
    with pytest.raises(ValueError, match="wavelength_m"):
        fried_parameter(np.array([0.0, 100.0]), np.full(2, 1e-15), lam)


@pytest.mark.parametrize("zeta", [-1.0, 90.0, 120.0, np.inf])
def test_out_of_range_zenith_rejected(zeta: float) -> None:
    with pytest.raises(ValueError, match="zenith_angle_deg"):
        fried_parameter(np.array([0.0, 100.0]), np.full(2, 1e-15), 500e-9, zeta)


def test_zero_turbulence_rejected_for_r0() -> None:
    with pytest.raises(ValueError, match="undefined"):
        fried_parameter(np.array([0.0, 100.0]), np.zeros(2))


def test_negative_moment_order_rejected() -> None:
    with pytest.raises(ValueError, match="order"):
        turbulence_moment(np.array([0.0, 100.0]), np.full(2, 1e-15), -1.0)


def test_wind_shape_mismatch_rejected() -> None:
    h = np.array([0.0, 100.0, 200.0])
    with pytest.raises(ValueError, match="wind_m_s"):
        greenwood_frequency(h, np.full(3, 1e-15), np.array([5.0, 5.0]))


def test_negative_wind_rejected_for_greenwood() -> None:
    h = np.array([0.0, 100.0, 200.0])
    with pytest.raises(ValueError, match="wind_m_s"):
        greenwood_frequency(h, np.full(3, 1e-15), np.array([5.0, -5.0, 5.0]))


def test_invalid_r0_rejected_for_seeing() -> None:
    with pytest.raises(ValueError, match="r0_m"):
        seeing_fwhm_arcsec(0.0)
