"""Unit and known-answer tests for the path geometry and weighting kernels."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate, special

from turbscope import (
    PathGeometry,
    coherence_weight,
    scintillation_weight,
    wavenumber,
    weight_normalisation,
    weighted_path_average,
)


def test_wavenumber_known_answer():
    # k = 2 pi / lambda; lambda = 1.55e-6 m -> k = 6.283185307/1.55e-6 = 4.05366794e6 rad/m
    assert wavenumber(1.55e-6) == pytest.approx(4.0536679e6, rel=1e-6)


def test_path_geometry_fresnel_scale():
    # sqrt(L/k) with L = 1000 m and k = 4.05366794e6 -> sqrt(2.46691e-4) = 0.0157064 m
    p = PathGeometry(1000.0, 1.55e-6)
    assert p.fresnel_scale_m() == pytest.approx(0.0157064, rel=1e-5)


def test_weight_normalisations_are_the_published_closed_forms():
    # int_0^1 u^(5/6)(1-u)^(5/6) du = B(11/6, 11/6) = 0.2205356554
    assert weight_normalisation("scintillation", "spherical") == pytest.approx(
        float(special.beta(11 / 6, 11 / 6)), rel=1e-12
    )
    assert weight_normalisation("scintillation", "spherical") == pytest.approx(0.2205356554, 1e-9)
    # int_0^1 (1-u)^(5/6) du = 6/11
    assert weight_normalisation("scintillation", "plane") == pytest.approx(6 / 11, rel=1e-12)
    # int_0^1 u^(5/3) du = 3/8
    assert weight_normalisation("coherence", "spherical") == pytest.approx(0.375, rel=1e-12)
    assert weight_normalisation("coherence", "plane") == pytest.approx(1.0, rel=1e-12)


def test_kernel_quadrature_matches_normalisation():
    u = np.linspace(0.0, 1.0, 4001)
    got = integrate.simpson(scintillation_weight(u, "spherical"), x=u)
    assert got == pytest.approx(weight_normalisation("scintillation", "spherical"), rel=1e-6)
    got = integrate.simpson(coherence_weight(u, "spherical"), x=u)
    assert got == pytest.approx(0.375, rel=1e-8)


def test_scintillation_kernel_is_symmetric_for_spherical_waves():
    u = np.linspace(0.0, 1.0, 101)
    w = scintillation_weight(u, "spherical")
    assert np.allclose(w, w[::-1], atol=1e-15)
    # and it vanishes at both endpoints: a spherical-wave scintillometer is blind
    # to turbulence at the source and at the receiver.
    assert w[0] == 0.0 and w[-1] == 0.0


def test_coherence_kernel_endpoints():
    u = np.array([0.0, 1.0])
    assert np.allclose(coherence_weight(u, "spherical"), [0.0, 1.0])
    assert np.allclose(coherence_weight(u, "plane"), [1.0, 1.0])


def test_weighted_average_of_uniform_profile_is_the_value(path):
    z = path.uniform_grid(201)
    cn2 = np.full_like(z, 3.7e-15)
    for kind in ("scintillation", "coherence"):
        got = weighted_path_average(z, cn2, kind=kind, geometry="spherical")
        assert got == pytest.approx(3.7e-15, rel=1e-12)


def test_weighted_average_known_answer_linear_ramp(path):
    # Cn2(z) = C0 * u with u = z/L.
    # Coherence kernel: int u^(5/3) * u du / int u^(5/3) du = (3/11)/(3/8) = 8/11 = 0.7272727
    # Scintillation kernel: symmetric, so the mean of a symmetric-weighted ramp is 1/2.
    z = path.uniform_grid(2001)
    c0 = 1e-14
    cn2 = c0 * z / path.length_m
    coh = weighted_path_average(z, cn2, kind="coherence", geometry="spherical")
    assert coh / c0 == pytest.approx(8 / 11, rel=1e-6)
    sc = weighted_path_average(z, cn2, kind="scintillation", geometry="spherical")
    assert sc / c0 == pytest.approx(0.5, rel=1e-6)


def test_uniform_grid_rejects_tiny_n(path):
    with pytest.raises(ValueError, match="n must be"):
        path.uniform_grid(2)
