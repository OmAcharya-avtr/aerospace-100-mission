"""Tests for the closed-form inversions, their intervals and the saturation regime."""

from __future__ import annotations

import numpy as np
import pytest

from turbscope import (
    PathGeometry,
    invert_dimm,
    invert_scintillation,
    saturation_peak,
    saturation_report,
    scintillation_branches,
    scintillation_index,
    weighted_path_average,
)
from turbscope.dimm import dimm_variance, fried_parameter
from turbscope.inversion import scintillation_index_relative_sigma
from turbscope.scintillation import rytov_variance, rytov_variance_from_average


def test_weak_round_trip_on_a_uniform_path(path, uniform_profile):
    """Noise-free weak-regime round trip: Cn2 -> sigma_I^2 -> Cn2."""
    z, cn2 = uniform_profile
    beta = rytov_variance(z, cn2, path)
    assert beta < 0.3, "this fixture must sit in the weak regime"
    sigma = float(scintillation_index(beta, 0.0))
    est = invert_scintillation(sigma, path, n_samples=1000, method="saturation")
    assert est.valid
    assert est.cn2 == pytest.approx(1e-15, rel=1e-5)


def test_weak_method_underestimates_outside_its_validity_range(path):
    """The linear inversion is biased low once beta_0^2 leaves the weak regime."""
    beta_true = 2.0
    sigma = float(scintillation_index(beta_true, 0.0))
    weak = invert_scintillation(sigma, path, method="weak")
    exact = invert_scintillation(sigma, path, method="saturation")
    assert weak.cn2 < exact.cn2
    assert any("weak-fluctuation limit" in n for n in weak.notes)


def test_inversion_recovers_the_scintillation_weighted_average_not_the_plain_one(path):
    """The estimator returns <Cn2>_W exactly; the plain path average it does not."""
    z = path.uniform_grid(1001)
    cn2 = 1e-15 * (1.0 + 4.0 * np.exp(-0.5 * ((z / path.length_m - 0.04) / 0.04) ** 2))
    beta = rytov_variance(z, cn2, path)
    est = invert_scintillation(float(scintillation_index(beta, 0.0)), path)
    target = weighted_path_average(z, cn2, kind="scintillation", geometry="spherical")
    assert est.cn2 == pytest.approx(target, rel=1e-6)
    plain = float(np.trapezoid(cn2, z) / path.length_m)
    assert abs(est.cn2 / plain - 1.0) > 0.05, "the weighting mismatch must be visible"


def test_saturation_report_fields():
    rep = saturation_report(0.0)
    assert rep.beta0_sq_peak == pytest.approx(7.2966, rel=1e-3)
    assert rep.sigma_i2_peak == pytest.approx(1.6921, rel=1e-3)
    lo, hi = rep.ambiguous_sigma_i2_range
    assert lo < hi <= rep.sigma_i2_peak


def test_multi_valued_reading_yields_two_branches(path):
    branches = scintillation_branches(1.5, 0.0)
    assert len(branches) == 2
    assert branches[0] == pytest.approx(2.8936, rel=1e-3)
    assert branches[1] == pytest.approx(31.542, rel=1e-3)
    est = invert_scintillation(1.5, path)
    assert est.ambiguous
    assert len(est.branches) == 2
    # the ambiguity is an order-of-magnitude problem, not a rounding problem
    assert est.branches[1] / est.branches[0] > 5.0


def test_reading_above_the_peak_is_reported_invalid(path):
    peak = saturation_peak(0.0)[1]
    est = invert_scintillation(peak * 1.05, path)
    assert not est.valid
    assert np.isnan(est.cn2)
    assert est.branches == ()
    assert any("exceeds the maximum attainable" in n for n in est.notes)


def test_relative_sigma_grows_as_the_peak_is_approached(path):
    """Sensitivity d(sigma_I^2)/d(beta_0^2) -> 0 at the peak, so the interval blows up."""
    b_peak = saturation_peak(0.0)[0]
    rel = []
    for frac in (0.05, 0.3, 0.8, 0.98):
        sigma = float(scintillation_index(b_peak * frac, 0.0))
        rel.append(invert_scintillation(sigma, path, n_samples=1000).relative_sigma)
    assert rel[0] < rel[1] < rel[2] < rel[3]
    assert rel[3] > 10.0 * rel[0]


def test_interval_brackets_the_point_estimate(path):
    est = invert_scintillation(0.05, path, n_samples=1000, coverage=0.9)
    assert est.cn2_lower < est.cn2 < est.cn2_upper
    wide = invert_scintillation(0.05, path, n_samples=1000, coverage=0.99)
    assert wide.cn2_upper > est.cn2_upper


def test_more_samples_narrow_the_interval(path):
    a = invert_scintillation(0.05, path, n_samples=100)
    b = invert_scintillation(0.05, path, n_samples=10_000)
    assert b.relative_sigma < a.relative_sigma
    # the estimator's standard error scales as 1/sqrt(N)
    assert a.relative_sigma / b.relative_sigma == pytest.approx(10.0, rel=0.02)


def test_relative_sigma_scaling_with_n_samples():
    a = scintillation_index_relative_sigma(0.1, 0.0, 500)
    b = scintillation_index_relative_sigma(0.1, 0.0, 2000)
    assert a / b == pytest.approx(2.0, rel=1e-9)


def test_dimm_round_trip(path):
    z = path.uniform_grid(401)
    cn2 = np.full_like(z, 2e-15)
    r0 = fried_parameter(z, cn2, path)
    var = dimm_variance(r0, path.wavelength_m, 0.06, 0.20)
    est = invert_dimm(var, path, subaperture_m=0.06, baseline_m=0.20, n_frames=500)
    assert est.valid
    assert est.cn2 == pytest.approx(2e-15, rel=1e-6)
    assert est.cn2_lower < est.cn2 < est.cn2_upper


def test_dimm_below_the_noise_floor_is_invalid(path):
    est = invert_dimm(
        1e-14, path, subaperture_m=0.06, baseline_m=0.20, noise_variance_rad2=2e-14
    )
    assert not est.valid
    assert any("noise floor" in n for n in est.notes)


def test_dimm_chi2_interval_width_scales_with_frames(path):
    a = invert_dimm(1e-12, path, subaperture_m=0.06, baseline_m=0.20, n_frames=50)
    b = invert_dimm(1e-12, path, subaperture_m=0.06, baseline_m=0.20, n_frames=5000)
    assert (b.cn2_upper - b.cn2_lower) < (a.cn2_upper - a.cn2_lower)


def test_rytov_variance_from_average_agrees_with_the_uniform_grid(path):
    z = path.uniform_grid(801)
    cn2 = np.full_like(z, 7e-16)
    assert rytov_variance(z, cn2, path) == pytest.approx(
        rytov_variance_from_average(7e-16, path), rel=5e-6
    )


def test_estimate_kernel_labels(path):
    s = invert_scintillation(0.02, path)
    d = invert_dimm(1e-12, path, subaperture_m=0.06, baseline_m=0.20)
    assert s.kernel == "scintillation"
    assert d.kernel == "coherence"


def test_path_geometry_mismatch_is_rejected():
    p = PathGeometry(1000.0, 1.55e-6)
    z = np.linspace(0.0, 500.0, 101)
    with pytest.raises(ValueError, match="span the geometry length"):
        rytov_variance(z, np.full_like(z, 1e-15), p)
