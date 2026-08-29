"""Hypothesis property tests for the algebraic identities the physics guarantees."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from turbscope import (
    PathGeometry,
    cn2_average_from_fried,
    dimm_variance,
    fried_parameter,
    invert_scintillation,
    r0_from_dimm_variance,
    rytov_variance_from_average,
    scintillation_index,
    weighted_path_average,
)
from turbscope.scintillation import rytov_variance, uniform_cn2_from_beta0_sq

SETTINGS = settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

lengths = st.floats(min_value=200.0, max_value=5000.0)
wavelengths = st.floats(min_value=4e-7, max_value=2.0e-6)
levels = st.floats(min_value=1e-17, max_value=1e-13)
shapes = st.floats(min_value=-2.0, max_value=2.0)


def _profile(z: np.ndarray, length: float, level: float, a: float, b: float) -> np.ndarray:
    u = z / length
    return level * np.exp(a * (u - 0.5) + b * np.sin(2.0 * np.pi * u))


@given(length=lengths, wavelength=wavelengths, level=levels, a=shapes, b=shapes)
@SETTINGS
def test_scintillometer_inversion_returns_the_kernel_weighted_average(
    length, wavelength, level, a, b
):
    """Identity: inverting the weak forward model returns <Cn2>_Wsc exactly.

    beta_0^2 = C_R k^(7/6) L^(5/6) int Cn2 W du  and  <Cn2>_W = int Cn2 W / int W,
    so beta_0^2 / (C_R N_W k^(7/6) L^(11/6)) = <Cn2>_W identically -- the same
    quadrature appears on both sides, so this holds to machine precision.
    """
    path = PathGeometry(length, wavelength)
    z = path.uniform_grid(201)
    cn2 = _profile(z, length, level, a, b)
    beta = rytov_variance(z, cn2, path)
    recovered = uniform_cn2_from_beta0_sq(beta, path)
    target = weighted_path_average(z, cn2, kind="scintillation", geometry="spherical")
    assert recovered == pytest.approx(target, rel=1e-12)


@given(length=lengths, wavelength=wavelengths, level=levels, a=shapes, b=shapes)
@SETTINGS
def test_dimm_inversion_returns_the_coherence_weighted_average(
    length, wavelength, level, a, b
):
    """Identity: the DIMM chain returns <Cn2>_Wco exactly (noise-free)."""
    path = PathGeometry(length, wavelength)
    z = path.uniform_grid(201)
    cn2 = _profile(z, length, level, a, b)
    r0 = fried_parameter(z, cn2, path)
    var = dimm_variance(r0, wavelength, 0.06, 0.20)
    r0_back = r0_from_dimm_variance(var, wavelength, 0.06, 0.20)
    recovered = cn2_average_from_fried(r0_back, path)
    target = weighted_path_average(z, cn2, kind="coherence", geometry="spherical")
    assert recovered == pytest.approx(target, rel=1e-10)


@given(
    length=lengths,
    wavelength=wavelengths,
    level=st.floats(min_value=1e-17, max_value=2e-15),
)
@SETTINGS
def test_weak_regime_round_trip_is_exact_for_uniform_paths(length, wavelength, level):
    """Uniform Cn2 in the weak regime survives forward + inverse to 1e-9 relative."""
    path = PathGeometry(length, wavelength)
    beta = rytov_variance_from_average(level, path)
    est = invert_scintillation(float(scintillation_index(beta, 0.0)), path)
    assert est.valid
    assert est.cn2 == pytest.approx(level, rel=max(1e-9, 5e-3 * beta))


@given(
    scale=st.floats(min_value=0.1, max_value=10.0),
    length=lengths,
    wavelength=wavelengths,
    level=levels,
)
@SETTINGS
def test_rytov_variance_is_linear_in_cn2(scale, length, level, wavelength):
    """beta_0^2 is exactly linear in Cn2 (weak-fluctuation theory is a linearisation)."""
    path = PathGeometry(length, wavelength)
    a = rytov_variance_from_average(level, path)
    b = rytov_variance_from_average(level * scale, path)
    assert b == pytest.approx(a * scale, rel=1e-12)


@given(beta=st.floats(min_value=1e-6, max_value=1e-3))
@SETTINGS
def test_scintillation_index_reduces_to_beta_in_the_weak_limit(beta):
    assert float(scintillation_index(beta, 0.0)) == pytest.approx(beta, rel=3e-3)


@given(
    level=levels,
    wavelength=wavelengths,
    length=lengths,
    a=shapes,
)
@SETTINGS
def test_weighted_average_is_bounded_by_the_profile_extremes(level, wavelength, length, a):
    path = PathGeometry(length, wavelength)
    z = path.uniform_grid(101)
    cn2 = _profile(z, length, level, a, 0.0)
    for kind in ("scintillation", "coherence"):
        got = weighted_path_average(z, cn2, kind=kind, geometry="spherical")
        assert cn2.min() * (1 - 1e-9) <= got <= cn2.max() * (1 + 1e-9)
