"""Tests for turbscope.dimm."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from turbscope.dimm import (
    DIMM_LONG_SEP_COEFF,
    DIMM_PREFACTOR,
    DIMM_TRANS_SEP_COEFF,
    cn2_path_from_fried_parameter,
    differential_variance,
    fried_parameter_from_cn2_path,
    invert_cn2_from_variance,
)

LAM = 500e-9
D = 0.14
SEP = 0.20


# --------------------------------------------------------------------- KAT
def test_fried_parameter_known_value():
    # r0 = [0.423 k^2 sec(0) * 2e-14 * 200]^(-3/5), k = 2*pi/500e-9
    # hand computed = 0.03498710013249161 m (see derivation script)
    r0 = fried_parameter_from_cn2_path(2e-14, 200.0, LAM, zenith_deg=0.0)
    assert float(r0) == pytest.approx(0.03498710013249161, rel=1e-9)


def test_differential_variance_known_values():
    # r0 = 0.10 m, D=0.14, d=0.20, lambda=500e-9
    # bracket_l = 1 - 0.541*(0.20/0.14)^(-1/3) = 0.519643935057253
    # bracket_t = 1 - 0.798*(0.20/0.14)^(-1/3) = 0.29145260660940453
    # var_l = 0.358*(lam/D)^2*(D/r0)^(5/3)*bracket_l = 4.157378377325569e-12
    # var_t = 0.358*(lam/D)^2*(D/r0)^(5/3)*bracket_t = 2.331748112483241e-12
    # r0 = 0.10 corresponds to a specific (cn2, L); derive cn2 for that r0 first.
    r0_target = 0.10
    cn2 = cn2_path_from_fried_parameter(r0_target, 200.0, LAM, zenith_deg=0.0)
    var_l = differential_variance(cn2, 200.0, LAM, D, SEP, "longitudinal")
    var_t = differential_variance(cn2, 200.0, LAM, D, SEP, "transverse")
    assert float(var_l) == pytest.approx(4.157378377325569e-12, rel=1e-6)
    assert float(var_t) == pytest.approx(2.331748112483241e-12, rel=1e-6)


def test_invert_cn2_from_variance_recovers_known_cn2():
    r0_target = 0.10
    cn2_true = cn2_path_from_fried_parameter(r0_target, 200.0, LAM, zenith_deg=0.0)
    var_l = differential_variance(cn2_true, 200.0, LAM, D, SEP, "longitudinal")
    cn2_back = invert_cn2_from_variance(var_l, 200.0, LAM, D, SEP, "longitudinal")
    assert cn2_back == pytest.approx(cn2_true, rel=1e-9)


def test_bracket_coefficients_are_the_documented_constants():
    assert DIMM_PREFACTOR == pytest.approx(0.358)
    assert DIMM_LONG_SEP_COEFF == pytest.approx(0.541)
    assert DIMM_TRANS_SEP_COEFF == pytest.approx(0.798)


# --------------------------------------------------------------- validation
@pytest.mark.parametrize(
    "cn2,length,lam,exc",
    [
        (-1e-14, 200.0, LAM, ValueError),
        (0.0, 200.0, LAM, ValueError),  # r0 undefined for Cn2=0
        (float("nan"), 200.0, LAM, ValueError),
        (1e-14, 0.0, LAM, ValueError),
        (1e-14, 200.0, 1.0, ValueError),
    ],
)
def test_fried_parameter_rejects_invalid_input(cn2, length, lam, exc):
    with pytest.raises(exc):
        fried_parameter_from_cn2_path(cn2, length, lam)


def test_fried_parameter_rejects_bad_zenith():
    with pytest.raises(ValueError):
        fried_parameter_from_cn2_path(1e-14, 200.0, LAM, zenith_deg=90.0)
    with pytest.raises(ValueError):
        fried_parameter_from_cn2_path(1e-14, 200.0, LAM, zenith_deg=-1.0)


def test_differential_variance_rejects_overlapping_apertures():
    with pytest.raises(ValueError):
        differential_variance(1e-14, 200.0, LAM, 0.20, 0.14, "longitudinal")


def test_differential_variance_rejects_unknown_component():
    with pytest.raises(ValueError):
        differential_variance(1e-14, 200.0, LAM, D, SEP, "diagonal")


def test_differential_variance_rejects_non_positive_geometry():
    with pytest.raises(ValueError):
        differential_variance(1e-14, 200.0, LAM, 0.0, SEP, "longitudinal")
    with pytest.raises(ValueError):
        differential_variance(1e-14, 200.0, LAM, D, 0.0, "longitudinal")


def test_invert_cn2_from_variance_rejects_non_positive_variance():
    with pytest.raises(ValueError):
        invert_cn2_from_variance(0.0, 200.0, LAM, D, SEP, "longitudinal")
    with pytest.raises(ValueError):
        invert_cn2_from_variance(-1e-12, 200.0, LAM, D, SEP, "longitudinal")


def test_cn2_path_from_fried_parameter_rejects_non_positive_r0():
    with pytest.raises(ValueError):
        cn2_path_from_fried_parameter(0.0, 200.0, LAM)


# -------------------------------------------------------------------- edge
def test_longitudinal_variance_exceeds_transverse_for_same_geometry():
    # bracket_l > bracket_t always, since 0.541 < 0.798 -> [1 - 0.541 x] > [1 - 0.798 x]
    cn2 = 1e-14
    var_l = differential_variance(cn2, 200.0, LAM, D, SEP, "longitudinal")
    var_t = differential_variance(cn2, 200.0, LAM, D, SEP, "transverse")
    assert float(var_l) > float(var_t)


def test_differential_variance_increases_with_separation_toward_no_diffraction_limit():
    cn2 = 1e-14
    var_close = differential_variance(cn2, 200.0, LAM, D, 0.15, "longitudinal")
    var_far = differential_variance(cn2, 200.0, LAM, D, 2.0, "longitudinal")
    # both approach 0.358 (lam/D)^2 (D/r0)^(5/3) as d/D -> infinity
    assert float(var_far) > float(var_close)


def test_larger_zenith_angle_needs_less_cn2_for_the_same_r0():
    # Cn2 = r0^(-5/3) / (0.423 k^2 sec(zeta) L); sec(zeta) grows with zeta, so
    # the same r0 is reached with a smaller path-averaged Cn2 off zenith.
    cn2_zenith = cn2_path_from_fried_parameter(0.10, 200.0, LAM, zenith_deg=0.0)
    cn2_slant = cn2_path_from_fried_parameter(0.10, 200.0, LAM, zenith_deg=45.0)
    assert cn2_slant < cn2_zenith
    assert cn2_slant > 0.0


def test_differential_variance_vectorised_matches_scalar():
    cn2 = np.array([1e-16, 1e-14, 1e-12])
    vec = differential_variance(cn2, 200.0, LAM, D, SEP, "longitudinal")
    scalars = [float(differential_variance(c, 200.0, LAM, D, SEP, "longitudinal")) for c in cn2]
    np.testing.assert_allclose(vec, scalars, rtol=1e-12)


# --------------------------------------------------------------- properties
@given(
    st.floats(min_value=1e-18, max_value=1e-10),
    st.floats(min_value=20.0, max_value=3000.0),
    st.sampled_from(["longitudinal", "transverse"]),
)
def test_dimm_round_trip_is_exact_algebraic_identity(cn2, length, component):
    """invert_cn2_from_variance is the exact algebraic inverse of
    differential_variance for any positive Cn2, path length and component."""
    var = float(differential_variance(cn2, length, LAM, D, SEP, component))
    back = invert_cn2_from_variance(var, length, LAM, D, SEP, component)
    assert back == pytest.approx(cn2, rel=1e-7)


@given(st.floats(min_value=1e-18, max_value=1e-10), st.floats(min_value=20.0, max_value=3000.0))
def test_dimm_variance_scales_linearly_with_cn2(cn2, length):
    """Differential variance is exactly proportional to Cn2 at fixed geometry."""
    base = float(differential_variance(cn2, length, LAM, D, SEP, "longitudinal"))
    doubled = float(differential_variance(2.0 * cn2, length, LAM, D, SEP, "longitudinal"))
    assert doubled == pytest.approx(2.0 * base, rel=1e-9)


@given(st.floats(min_value=1e-18, max_value=1e-10), st.floats(min_value=20.0, max_value=3000.0))
def test_fried_and_cn2_conversions_are_mutual_inverses(cn2, length):
    r0 = float(fried_parameter_from_cn2_path(cn2, length, LAM))
    back = cn2_path_from_fried_parameter(r0, length, LAM)
    assert back == pytest.approx(cn2, rel=1e-9)
