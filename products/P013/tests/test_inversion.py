"""Tests for turbscope.inversion."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from turbscope.dimm import differential_variance
from turbscope.inversion import (
    PointEstimate,
    fuse_inverse_variance,
    invert_dimm_with_uncertainty,
    invert_scintillometer_weak_with_uncertainty,
    multi_sensor_closed_form_estimate,
)
from turbscope.scintillometer import rytov_variance, scintillation_index_full

LAM_S = 880e-9
LAM_D = 500e-9
D = 0.14
SEP = 0.20


# --------------------------------------------------------------------- KAT
def test_scintillometer_uncertainty_is_exactly_linear():
    # Cn2 = sigma_I2 / const is linear, so a 10% relative measurement error
    # gives exactly a 10% relative Cn2 error (hand-derivable: d(Cn2)/Cn2 =
    # d(sigma)/sigma for y = a*x).
    est = invert_scintillometer_weak_with_uncertainty(0.05, 0.10, 500.0, LAM_S, "spherical")
    assert est.cn2_std / est.cn2_path == pytest.approx(0.10, rel=1e-9)


def test_dimm_uncertainty_is_exactly_linear():
    est = invert_dimm_with_uncertainty(1e-12, 0.15, 500.0, LAM_D, D, SEP, "longitudinal")
    assert est.cn2_std / est.cn2_path == pytest.approx(0.15, rel=1e-9)


def test_fuse_inverse_variance_known_case():
    # Two estimates, 1.0 +/- 1.0 and 3.0 +/- 1.0: inverse-variance mean is
    # (1/1 + 3/1) / (1/1 + 1/1) = 2.0 ; combined std = (1/1 + 1/1)^(-1/2) = 1/sqrt(2)
    a = PointEstimate(cn2_path=1.0, cn2_std=1.0, source="a")
    b = PointEstimate(cn2_path=3.0, cn2_std=1.0, source="b")
    fused = fuse_inverse_variance([a, b])
    assert fused.cn2_path == pytest.approx(2.0, rel=1e-9)
    assert fused.cn2_std == pytest.approx(1.0 / 2**0.5, rel=1e-9)


def test_fuse_inverse_variance_weights_toward_lower_uncertainty():
    # A precise estimate (small std) should pull the fused value toward itself.
    a = PointEstimate(cn2_path=1.0, cn2_std=0.01, source="precise")
    b = PointEstimate(cn2_path=10.0, cn2_std=100.0, source="imprecise")
    fused = fuse_inverse_variance([a, b])
    assert abs(fused.cn2_path - 1.0) < abs(fused.cn2_path - 10.0)


# --------------------------------------------------------------- validation
def test_invert_scintillometer_rejects_negative_relative_std():
    with pytest.raises(ValueError):
        invert_scintillometer_weak_with_uncertainty(0.05, -0.1, 500.0, LAM_S)


def test_invert_dimm_rejects_negative_relative_std():
    with pytest.raises(ValueError):
        invert_dimm_with_uncertainty(1e-12, -0.1, 500.0, LAM_D, D, SEP, "longitudinal")


def test_fuse_inverse_variance_rejects_empty_list():
    with pytest.raises(ValueError):
        fuse_inverse_variance([])


def test_fuse_inverse_variance_rejects_non_positive_std():
    a = PointEstimate(cn2_path=1.0, cn2_std=0.0, source="bad")
    with pytest.raises(ValueError):
        fuse_inverse_variance([a])


# -------------------------------------------------------------------- edge
def test_fuse_single_estimate_returns_it_unchanged():
    a = PointEstimate(cn2_path=5.0, cn2_std=0.5, source="only")
    fused = fuse_inverse_variance([a])
    assert fused.cn2_path == pytest.approx(5.0)
    assert fused.cn2_std == pytest.approx(0.5)


def test_multi_sensor_closed_form_estimate_flags_weak_regime_correctly():
    # Choose a genuinely weak-regime case (small Cn2, short path).
    cn2_true = 1e-16
    length = 100.0
    r_var = float(rytov_variance(cn2_true, length, LAM_S, "spherical"))
    sigma_i2 = float(scintillation_index_full(r_var))
    var_l = float(differential_variance(cn2_true, length, LAM_D, D, SEP, "longitudinal"))
    var_t = float(differential_variance(cn2_true, length, LAM_D, D, SEP, "transverse"))
    result = multi_sensor_closed_form_estimate(
        sigma_i2, var_l, var_t, length, LAM_S, "spherical", LAM_D, D, SEP, 0.08, 0.10
    )
    assert result.weak_regime_scint is True
    assert len(result.individual) == 3
    assert result.fused.cn2_path > 0.0


def test_multi_sensor_closed_form_estimate_flags_saturated_regime():
    # Choose a deeply saturated case (large Cn2, long path).
    cn2_true = 1e-12
    length = 2000.0
    r_var = float(rytov_variance(cn2_true, length, LAM_S, "spherical"))
    sigma_i2 = float(scintillation_index_full(r_var))
    var_l = float(differential_variance(cn2_true, length, LAM_D, D, SEP, "longitudinal"))
    var_t = float(differential_variance(cn2_true, length, LAM_D, D, SEP, "transverse"))
    result = multi_sensor_closed_form_estimate(
        sigma_i2, var_l, var_t, length, LAM_S, "spherical", LAM_D, D, SEP, 0.08, 0.10
    )
    assert result.weak_regime_scint is False


# --------------------------------------------------------------- properties
@given(
    st.floats(min_value=1e-6, max_value=1.0),
    st.floats(min_value=0.0, max_value=0.5),
    st.floats(min_value=50.0, max_value=3000.0),
)
def test_scintillometer_relative_uncertainty_always_preserved(sigma, rel_std, length):
    est = invert_scintillometer_weak_with_uncertainty(sigma, rel_std, length, LAM_S, "spherical")
    if est.cn2_path > 0.0:
        assert est.cn2_std / est.cn2_path == pytest.approx(rel_std, rel=1e-6, abs=1e-12)


@given(
    st.floats(min_value=1.0, max_value=10.0),
    st.floats(min_value=0.01, max_value=100.0),
    st.floats(min_value=1.0, max_value=10.0),
    st.floats(min_value=0.01, max_value=100.0),
)
def test_fused_estimate_lies_between_the_two_inputs(v1, s1, v2, s2):
    a = PointEstimate(cn2_path=v1, cn2_std=s1, source="a")
    b = PointEstimate(cn2_path=v2, cn2_std=s2, source="b")
    fused = fuse_inverse_variance([a, b])
    lo, hi = sorted([v1, v2])
    assert lo - 1e-9 <= fused.cn2_path <= hi + 1e-9
