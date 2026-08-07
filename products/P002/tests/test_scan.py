"""Unit and property tests for trackbench.scan."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from trackbench.scan import (
    GaussianUncertainty,
    coverage_fraction,
    expected_acquisition_time_spiral,
    raster_scan,
    simulate_acquisition,
    spiral_scan,
    track_spacing,
)


# --------------------------------------------------------------------------
# GaussianUncertainty
# --------------------------------------------------------------------------
def test_uncertainty_rejects_nonpositive_sigma():
    with pytest.raises(ValueError, match="sigma"):
        GaussianUncertainty(0.0)
    with pytest.raises(ValueError, match="sigma"):
        GaussianUncertainty(-1e-4)


def test_uncertainty_rejects_nan_sigma():
    with pytest.raises(ValueError):
        GaussianUncertainty(float("nan"))


def test_rayleigh_one_sigma_known_answer():
    """Hand check: P(r <= sigma) = 1 - exp(-1/2) = 0.3934693402873666."""
    u = GaussianUncertainty(1e-4)
    assert u.prob_within(1e-4) == pytest.approx(1.0 - math.exp(-0.5), rel=1e-12)


def test_rayleigh_three_sigma_known_answer():
    """Hand check: P(r <= 3 sigma) = 1 - exp(-9/2) = 0.9888910034...."""
    u = GaussianUncertainty(2.5e-4)
    assert u.prob_within(3 * 2.5e-4) == pytest.approx(0.98889100346, rel=1e-9)


def test_containment_radius_roundtrip():
    u = GaussianUncertainty(3e-4)
    for p in (0.5, 0.9, 0.995):
        assert u.prob_within(u.containment_radius(p)) == pytest.approx(p, rel=1e-12)


def test_containment_radius_domain():
    u = GaussianUncertainty(1e-4)
    with pytest.raises(ValueError):
        u.containment_radius(0.0)
    with pytest.raises(ValueError):
        u.containment_radius(1.0)


def test_prob_within_negative_radius_raises():
    with pytest.raises(ValueError):
        GaussianUncertainty(1e-4).prob_within(-1.0)


def test_sample_shape_and_statistics():
    u = GaussianUncertainty(2e-4)
    s = u.sample(20000, np.random.default_rng(3))
    assert s.shape == (20000, 2)
    # sample std within 3 % of sigma for n = 20000 (SE of std ~ sigma/sqrt(2n))
    assert np.std(s) == pytest.approx(2e-4, rel=0.03)


def test_sample_rejects_zero_n():
    with pytest.raises(ValueError):
        GaussianUncertainty(1e-4).sample(0, np.random.default_rng(0))


# --------------------------------------------------------------------------
# track spacing
# --------------------------------------------------------------------------
def test_track_spacing_known_answer():
    """Hand check: 2 * 20 urad * (1 - 0.25) = 30 urad."""
    assert track_spacing(20e-6, 0.25) == pytest.approx(30e-6, rel=1e-12)


def test_track_spacing_zero_overlap_equals_diameter():
    assert track_spacing(1e-5, 0.0) == pytest.approx(2e-5)


@pytest.mark.parametrize("overlap", [-0.1, 1.0, 1.5])
def test_track_spacing_invalid_overlap(overlap):
    with pytest.raises(ValueError, match="overlap"):
        track_spacing(1e-5, overlap)


# --------------------------------------------------------------------------
# spiral / raster geometry
# --------------------------------------------------------------------------
def test_spiral_radial_pitch_equals_track_spacing():
    """Archimedean property: r(phi + 2 pi) - r(phi) = 2 pi a = s."""
    u = GaussianUncertainty(3e-4)
    p = spiral_scan(u, 2e-5, overlap=0.25)
    r = np.linalg.norm(p.points, axis=1)
    ang = np.unwrap(np.arctan2(p.points[:, 1], p.points[:, 0]))
    # radius after one extra turn
    i0 = len(r) // 2
    target = ang[i0] + 2 * math.pi
    j = int(np.searchsorted(ang, target))
    assert r[j] - r[i0] == pytest.approx(p.track_spacing, rel=0.02)


def test_spiral_reaches_containment_radius():
    u = GaussianUncertainty(3e-4)
    p = spiral_scan(u, 2e-5, containment=0.995)
    assert np.max(np.linalg.norm(p.points, axis=1)) >= 0.98 * p.max_radius


def test_spiral_step_length_matches_step_fraction():
    u = GaussianUncertainty(3e-4)
    p = spiral_scan(u, 2e-5, step_fraction=0.5)
    steps = np.linalg.norm(np.diff(p.points, axis=0), axis=1)
    # arc-length stepping: chord <= arc, and >= 90 % of it for these curvatures
    assert np.median(steps) == pytest.approx(0.5 * 2e-5, rel=0.1)


def test_spiral_center_offset_applied():
    u = GaussianUncertainty(3e-4)
    p = spiral_scan(u, 2e-5, center=(1e-3, -2e-3))
    assert p.points[0] == pytest.approx(np.array([1e-3, -2e-3]))


def test_raster_rows_spaced_by_track_spacing():
    u = GaussianUncertainty(3e-4)
    p = raster_scan(u, 2e-5, overlap=0.25)
    ys = np.unique(np.round(p.points[:, 1], 12))
    assert np.allclose(np.diff(ys), p.track_spacing, rtol=1e-9)


def test_raster_is_serpentine():
    u = GaussianUncertainty(2e-4)
    p = raster_scan(u, 3e-5)
    ys = np.unique(np.round(p.points[:, 1], 12))
    row0 = p.points[np.isclose(p.points[:, 1], ys[0])][:, 0]
    row1 = p.points[np.isclose(p.points[:, 1], ys[1])][:, 0]
    assert row0[0] < row0[-1]
    assert row1[0] > row1[-1]


def test_scan_time_and_speed_properties():
    u = GaussianUncertainty(3e-4)
    p = spiral_scan(u, 2e-5, dwell_time=1e-3)
    assert p.scan_time == pytest.approx(p.n_points * 1e-3)
    assert p.scan_speed > 0
    assert p.times[-1] == pytest.approx(p.scan_time)


def test_raster_covers_more_points_than_spiral():
    """The raster sweeps the bounding square, the spiral only the disc."""
    u = GaussianUncertainty(3e-4)
    sp = spiral_scan(u, 2e-5)
    ra = raster_scan(u, 2e-5)
    assert ra.n_points > sp.n_points


@pytest.mark.parametrize("kwargs", [
    {"beam_radius": 0.0},
    {"beam_radius": -1e-5},
    {"dwell_time": 0.0},
    {"step_fraction": 0.0},
    {"step_fraction": 2.0},
    {"containment": 1.0},
    {"containment": 0.0},
])
def test_spiral_input_validation(kwargs):
    u = GaussianUncertainty(3e-4)
    base = {"beam_radius": 2e-5}
    base.update(kwargs)
    with pytest.raises(ValueError):
        spiral_scan(u, **base)


def test_raster_input_validation():
    u = GaussianUncertainty(3e-4)
    with pytest.raises(ValueError):
        raster_scan(u, 2e-5, containment=1.5)


# --------------------------------------------------------------------------
# coverage
# --------------------------------------------------------------------------
def test_spiral_coverage_meets_containment():
    """Track spacing <= 2 R_beam must cover essentially all of the disc."""
    u = GaussianUncertainty(3e-4)
    p = spiral_scan(u, 2e-5, overlap=0.25, containment=0.995)
    cov = coverage_fraction(p, u, n_samples=20000, rng=np.random.default_rng(11))
    assert cov > 0.98


def test_coverage_drops_when_tracks_are_too_far_apart():
    """overlap -> negative is disallowed, so widen the beam design instead."""
    u = GaussianUncertainty(3e-4)
    good = spiral_scan(u, 2e-5, overlap=0.25)
    # a pattern designed for a large beam but flown with a small one leaves gaps
    good.beam_radius = 0.4 * 2e-5
    cov = coverage_fraction(good, u, n_samples=20000, rng=np.random.default_rng(5))
    assert cov < 0.9


def test_raster_coverage_high():
    u = GaussianUncertainty(2e-4)
    p = raster_scan(u, 2e-5, overlap=0.25)
    cov = coverage_fraction(p, u, n_samples=10000, rng=np.random.default_rng(2))
    assert cov > 0.98


# --------------------------------------------------------------------------
# acquisition simulation
# --------------------------------------------------------------------------
def test_acquisition_target_at_origin_detected_on_first_dwell():
    u = GaussianUncertainty(3e-4)
    p = spiral_scan(u, 2e-5, dwell_time=1e-3)
    t = simulate_acquisition(p, np.zeros(2), p_dwell=1.0)
    assert t == pytest.approx(1e-3)


def test_acquisition_returns_none_for_unreachable_target():
    u = GaussianUncertainty(1e-4)
    p = spiral_scan(u, 2e-5)
    assert simulate_acquisition(p, np.array([1.0, 1.0])) is None


def test_acquisition_time_increases_with_radius():
    u = GaussianUncertainty(3e-4)
    p = spiral_scan(u, 2e-5, dwell_time=1e-3)
    near = simulate_acquisition(p, np.array([5e-5, 0.0]), p_dwell=1.0)
    far = simulate_acquisition(p, np.array([5e-4, 0.0]), p_dwell=1.0)
    assert near is not None and far is not None
    assert far > near


def test_acquisition_probabilistic_detection_delays_acquisition():
    u = GaussianUncertainty(3e-4)
    p = spiral_scan(u, 2e-5, dwell_time=1e-3)
    tgt = np.array([2e-4, 0.0])
    certain = simulate_acquisition(p, tgt, p_dwell=1.0, rng=np.random.default_rng(0))
    noisy = [
        simulate_acquisition(p, tgt, p_dwell=0.2, rng=np.random.default_rng(s))
        for s in range(20)
    ]
    noisy = [t for t in noisy if t is not None]
    assert np.mean(noisy) > certain


@pytest.mark.parametrize("p_dwell", [0.0, -0.1, 1.1])
def test_acquisition_invalid_p_dwell(p_dwell):
    u = GaussianUncertainty(3e-4)
    p = spiral_scan(u, 2e-5)
    with pytest.raises(ValueError, match="p_dwell"):
        simulate_acquisition(p, np.zeros(2), p_dwell=p_dwell)


def test_acquisition_invalid_target_shape():
    u = GaussianUncertainty(3e-4)
    p = spiral_scan(u, 2e-5)
    with pytest.raises(ValueError, match="shape"):
        simulate_acquisition(p, np.zeros(3))


# --------------------------------------------------------------------------
# analytic expected acquisition time
# --------------------------------------------------------------------------
def test_expected_time_infinite_containment_limit():
    """E[T] -> 2 pi sigma^2 / (s v) as containment -> 1 (hand-derived limit)."""
    u = GaussianUncertainty(3e-4)
    s = track_spacing(2e-5, 0.25)
    v = 0.01
    got = expected_acquisition_time_spiral(u, 2e-5, 0.25, v, containment=0.999999)
    want = 2 * math.pi * u.sigma**2 / (s * v)
    assert got == pytest.approx(want, rel=0.02)


def test_expected_time_scales_inverse_with_speed():
    u = GaussianUncertainty(3e-4)
    a = expected_acquisition_time_spiral(u, 2e-5, 0.25, 0.01)
    b = expected_acquisition_time_spiral(u, 2e-5, 0.25, 0.02)
    assert a / b == pytest.approx(2.0, rel=1e-9)


def test_expected_time_grows_when_detection_is_unreliable():
    u = GaussianUncertainty(3e-4)
    full = expected_acquisition_time_spiral(u, 2e-5, 0.25, 0.01, p_pass=1.0)
    half = expected_acquisition_time_spiral(u, 2e-5, 0.25, 0.01, p_pass=0.5)
    assert half > full


@pytest.mark.parametrize("bad", [{"scan_speed": 0.0}, {"p_pass": 0.0}, {"p_pass": 1.5}])
def test_expected_time_validation(bad):
    u = GaussianUncertainty(3e-4)
    kw = {"beam_radius": 2e-5, "overlap": 0.25, "scan_speed": 0.01}
    kw.update(bad)
    with pytest.raises(ValueError):
        expected_acquisition_time_spiral(u, **kw)


# --------------------------------------------------------------------------
# property-based
# --------------------------------------------------------------------------
@given(
    sigma=st.floats(min_value=1e-5, max_value=1e-3),
    p=st.floats(min_value=0.01, max_value=0.99),
)
@settings(max_examples=40, deadline=None)
def test_property_containment_is_inverse_of_prob_within(sigma, p):
    u = GaussianUncertainty(sigma)
    assert u.prob_within(u.containment_radius(p)) == pytest.approx(p, rel=1e-9)


@given(
    beam=st.floats(min_value=1e-6, max_value=1e-4),
    overlap=st.floats(min_value=0.0, max_value=0.9),
)
@settings(max_examples=40, deadline=None)
def test_property_track_spacing_never_exceeds_beam_diameter(beam, overlap):
    assert 0 < track_spacing(beam, overlap) <= 2 * beam + 1e-18


@given(sigma=st.floats(min_value=5e-5, max_value=5e-4))
@settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_property_spiral_max_radius_matches_quantile(sigma):
    u = GaussianUncertainty(sigma)
    p = spiral_scan(u, 3e-5, containment=0.99)
    assert p.max_radius == pytest.approx(u.containment_radius(0.99), rel=1e-12)
