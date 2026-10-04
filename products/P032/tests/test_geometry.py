"""Line-of-sight geometry: closed forms, limits and validation."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from constellink.frames import WGS84_A_KM
from constellink.geometry import (
    DEFAULT_GRAZING_ALTITUDE_KM,
    central_angle,
    ground_max_central_angle,
    isl_clear,
    max_isl_central_angle,
    segment_min_radius,
    slant_range_at_elevation,
)

R_ORBIT = WGS84_A_KM + 550.0


def test_segment_min_radius_perpendicular_case():
    # Hand calculation: the segment from (0,-1,0) to (0,1,0) passes through the
    # origin, so the minimum distance is 0.
    assert segment_min_radius(np.array([0.0, -1.0, 0.0]),
                              np.array([0.0, 1.0, 0.0]))[0] == pytest.approx(0.0)


def test_segment_min_radius_foot_outside_segment():
    # Both endpoints on the +x side: the closest point is the nearer endpoint,
    # distance 3, not the perpendicular foot at the origin.
    d = segment_min_radius(np.array([3.0, 0.0, 0.0]), np.array([5.0, 0.0, 0.0]))
    assert d[0] == pytest.approx(3.0)


def test_segment_min_radius_equal_radius_chord_closed_form():
    # For equal radii r and central angle gamma, d_min = r cos(gamma/2).
    r = 7000.0
    for gamma_deg in (10.0, 45.0, 90.0, 150.0):
        g = np.deg2rad(gamma_deg)
        a = np.array([r, 0.0, 0.0])
        b = r * np.array([np.cos(g), np.sin(g), 0.0])
        assert segment_min_radius(a, b)[0] == pytest.approx(
            r * np.cos(g / 2.0), rel=1e-12)


def test_segment_min_radius_coincident_endpoints():
    a = np.array([7000.0, 0.0, 0.0])
    assert segment_min_radius(a, a)[0] == pytest.approx(7000.0)


def test_isl_clear_flips_at_gamma_max():
    gamma_max = max_isl_central_angle(R_ORBIT)
    for frac, expected in ((0.99, True), (1.01, False)):
        g = gamma_max * frac
        a = np.array([R_ORBIT, 0.0, 0.0])
        b = R_ORBIT * np.array([np.cos(g), np.sin(g), 0.0])
        assert bool(isl_clear(a, b)[0]) is expected


def test_max_isl_central_angle_closed_form():
    r_block = WGS84_A_KM + DEFAULT_GRAZING_ALTITUDE_KM
    expected = 2.0 * np.arccos(r_block / R_ORBIT)
    assert max_isl_central_angle(R_ORBIT) == pytest.approx(expected, rel=1e-14)


def test_max_isl_central_angle_rejects_orbit_inside_blocking_sphere():
    with pytest.raises(ValueError):
        max_isl_central_angle(WGS84_A_KM + 50.0)


@pytest.mark.parametrize("bad", [{"grazing_altitude_km": -1.0},
                                 {"earth_radius_km": 0.0}])
def test_isl_clear_rejects_bad_parameters(bad):
    a = np.array([R_ORBIT, 0.0, 0.0])
    b = np.array([0.0, R_ORBIT, 0.0])
    with pytest.raises(ValueError):
        isl_clear(a, b, **bad)


def test_central_angle_known_values():
    x = np.array([1.0, 0.0, 0.0])
    y = np.array([0.0, 1.0, 0.0])
    assert central_angle(x, y)[0] == pytest.approx(np.pi / 2.0, rel=1e-14)
    assert central_angle(x, x)[0] == pytest.approx(0.0, abs=1e-14)
    assert central_angle(x, -x)[0] == pytest.approx(np.pi, rel=1e-14)


def test_ground_max_central_angle_zero_elevation_is_the_horizon():
    # At eps = 0 the closed form reduces to arccos(R_e / r), the geometric
    # horizon half-angle.
    assert ground_max_central_angle(R_ORBIT, 0.0) == pytest.approx(
        np.arccos(WGS84_A_KM / R_ORBIT), rel=1e-14)


def test_ground_max_central_angle_hand_value():
    # Hand calculation for r = 6928.137 km, eps = 10 deg:
    #   arccos((6378.137/6928.137) * cos 10 deg) - 10 deg
    #   = arccos(0.9066454...) - 0.1745329 rad
    #   = 0.4356082 - 0.1745329 = 0.2610753 rad = 14.95664 deg
    lam = ground_max_central_angle(6928.137, 10.0)
    assert np.rad2deg(lam) == pytest.approx(14.95664, abs=1e-4)


def test_ground_max_central_angle_decreases_with_elevation():
    angles = [ground_max_central_angle(R_ORBIT, e) for e in (0.0, 5.0, 10.0, 30.0)]
    assert all(angles[i] > angles[i + 1] for i in range(len(angles) - 1))


@pytest.mark.parametrize("r,eps", [(WGS84_A_KM, 10.0), (R_ORBIT, 90.0),
                                   (R_ORBIT, -1.0)])
def test_ground_max_central_angle_rejects_bad_input(r, eps):
    with pytest.raises(ValueError):
        ground_max_central_angle(r, eps)


def test_slant_range_at_zenith_is_the_altitude():
    assert slant_range_at_elevation(R_ORBIT, 90.0) == pytest.approx(
        R_ORBIT - WGS84_A_KM, rel=1e-12)


def test_slant_range_at_horizon_closed_form():
    # At eps = 0 the slant range is sqrt(r^2 - R_e^2) (right triangle).
    expected = np.sqrt(R_ORBIT ** 2 - WGS84_A_KM ** 2)
    assert slant_range_at_elevation(R_ORBIT, 0.0) == pytest.approx(expected,
                                                                   rel=1e-12)


@given(eps=st.floats(0.0, 89.0))
@settings(max_examples=40, deadline=None)
def test_slant_range_decreases_with_elevation(eps):
    # Algebraic identity: slant range is strictly decreasing in elevation.
    assert slant_range_at_elevation(R_ORBIT, eps) > slant_range_at_elevation(
        R_ORBIT, eps + 1.0)


@given(gamma_deg=st.floats(0.1, 179.9))
@settings(max_examples=50, deadline=None)
def test_central_angle_matches_the_chord_relation(gamma_deg):
    # Algebraic identity: |b - a|^2 = r^2 + r^2 - 2 r^2 cos(gamma).
    r = 7000.0
    g = np.deg2rad(gamma_deg)
    a = np.array([r, 0.0, 0.0])
    b = r * np.array([np.cos(g), np.sin(g), 0.0])
    chord = float(np.linalg.norm(b - a))
    assert chord ** 2 == pytest.approx(2.0 * r ** 2 * (1.0 - np.cos(g)), rel=1e-9)
    assert central_angle(a, b)[0] == pytest.approx(g, rel=1e-9)
