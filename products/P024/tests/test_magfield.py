"""Tilted centred-dipole field model."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from detumblesim.constants import (
    B0_NT,
    IGRF14_2025_G10_NT,
    IGRF14_2025_G11_NT,
    IGRF14_2025_H11_NT,
    NT_TO_T,
    R_EARTH_M,
)
from detumblesim.magfield import (
    B0_T,
    DIPOLE_G_NT,
    dipole_field_ecef,
    dipole_field_eci,
    dipole_tilt_deg,
    eci_to_ecef_angle,
    field_magnitude_nt,
    geomagnetic_north_pole_deg,
    rot_z,
    spherical_position_ecef,
)


class TestCoefficients:
    def test_b0_is_the_norm_of_the_degree_one_terms(self):
        # sqrt(29350.0^2 + 1410.3^2 + 4545.5^2) = 29733.365... nT, computed by
        # hand from the IGRF-14 epoch-2025.0 coefficients.
        expected = np.sqrt(
            IGRF14_2025_G10_NT**2 + IGRF14_2025_G11_NT**2 + IGRF14_2025_H11_NT**2
        )
        assert np.isclose(B0_NT, expected, rtol=0.0, atol=1e-9)
        assert np.isclose(B0_NT, 29733.3654, atol=1e-3)
        assert np.isclose(B0_T, B0_NT * NT_TO_T)

    def test_g_vector_ordering(self):
        assert np.allclose(
            DIPOLE_G_NT,
            [IGRF14_2025_G11_NT, IGRF14_2025_H11_NT, IGRF14_2025_G10_NT],
        )


class TestGeomagneticPole:
    def test_known_answer_against_published_pole(self):
        # WDC Kyoto publishes the IGRF-14 geomagnetic (dipole) north pole for
        # epoch 2025 as 80.8 N, 72.8 W.  The value here is derived from the
        # same degree-1 coefficients, so it must agree to the published
        # rounding.
        lat, lon = geomagnetic_north_pole_deg()
        assert abs(lat - 80.8) < 0.05
        assert abs(lon - (-72.8)) < 0.05

    def test_tilt_is_pole_colatitude(self):
        lat, _ = geomagnetic_north_pole_deg()
        assert np.isclose(dipole_tilt_deg(), 90.0 - lat)
        assert 9.0 < dipole_tilt_deg() < 9.5

    def test_field_is_vertical_and_downward_at_the_pole(self):
        lat, lon = geomagnetic_north_pole_deg()
        pos = spherical_position_ecef(lat, lon, 1.0)
        b = dipole_field_ecef(pos)
        rhat = pos / np.linalg.norm(pos)
        # B is antiparallel to r_hat there: cos of the angle must be -1.
        assert np.isclose(float(b @ rhat) / float(np.linalg.norm(b)), -1.0, atol=1e-9)


class TestDipoleField:
    def test_known_answer_magnitude_law(self):
        # |B| = (a/r)^3 B0 sqrt(1 + 3 cos^2(theta_m)), theta_m the geomagnetic
        # colatitude.  At the geomagnetic equator (cos = 0) the surface field
        # is exactly B0; at the pole (cos = 1) it is exactly 2 B0.
        lat, lon = geomagnetic_north_pole_deg()
        pole = spherical_position_ecef(lat, lon, 0.0) * (1 + 1e-13)
        assert np.isclose(np.linalg.norm(dipole_field_ecef(pole)) / NT_TO_T, 2 * B0_NT,
                          rtol=1e-9)
        # A point 90 deg from the pole is on the geomagnetic equator.
        axis = -DIPOLE_G_NT / np.linalg.norm(DIPOLE_G_NT)
        perp = np.cross(axis, [0.0, 0.0, 1.0])
        perp = perp / np.linalg.norm(perp)
        assert np.isclose(
            np.linalg.norm(dipole_field_ecef(perp * R_EARTH_M * (1 + 1e-13))) / NT_TO_T,
            B0_NT, rtol=1e-9,
        )

    def test_inverse_cube_falloff(self):
        p = np.array([R_EARTH_M * 1.5, 0.0, 0.0])
        b1 = np.linalg.norm(dipole_field_ecef(p))
        b2 = np.linalg.norm(dipole_field_ecef(2.0 * p))
        assert np.isclose(b1 / b2, 8.0, rtol=1e-12)

    def test_rejects_positions_inside_the_earth(self):
        with pytest.raises(ValueError, match="outside the Earth"):
            dipole_field_ecef([0.0, 0.0, 0.5 * R_EARTH_M])

    def test_rejects_bad_shape(self):
        with pytest.raises(ValueError, match="shape"):
            dipole_field_ecef([1.0, 2.0])

    def test_vectorised_matches_scalar(self):
        pts = np.array(
            [[7e6, 0.0, 0.0], [0.0, 7.2e6, 1e5], [1e6, -2e6, 7e6]], dtype=float
        )
        many = dipole_field_ecef(pts)
        one = np.array([dipole_field_ecef(p) for p in pts])
        assert np.allclose(many, one)

    def test_eci_equals_ecef_at_zero_rotation_angle(self):
        p = np.array([7e6, 1e6, 2e6])
        assert np.allclose(dipole_field_eci(p, 0.0, 0.0), dipole_field_ecef(p))

    def test_eci_rotation_preserves_magnitude(self):
        p = np.array([7e6, 1e6, 2e6])
        a = np.linalg.norm(dipole_field_eci(p, 0.0, 0.0))
        b = np.linalg.norm(dipole_field_eci(p, 3600.0, 0.4))
        # Same |r| but a different Earth-fixed point, so magnitudes differ;
        # what must hold is that the rotation itself is an isometry.
        rot = rot_z(eci_to_ecef_angle(3600.0, 0.4))
        assert np.isclose(
            b, float(np.linalg.norm(dipole_field_ecef(rot @ p))), rtol=1e-12
        )
        assert a > 0.0

    @given(
        x=st.floats(-3.0, 3.0), y=st.floats(-3.0, 3.0), z=st.floats(-3.0, 3.0)
    )
    @settings(max_examples=60, deadline=None)
    def test_magnitude_law_holds_everywhere(self, x, y, z):
        v = np.array([x, y, z])
        n = float(np.linalg.norm(v))
        if n < 0.2:
            return
        r = 1.3 * R_EARTH_M * v / n
        b = dipole_field_ecef(r)
        cos_m = float((r / np.linalg.norm(r)) @ (DIPOLE_G_NT / B0_NT))
        expected = (
            (R_EARTH_M / np.linalg.norm(r)) ** 3 * B0_T * np.sqrt(1.0 + 3.0 * cos_m**2)
        )
        assert np.isclose(float(np.linalg.norm(b)), expected, rtol=1e-10)


class TestSphericalPosition:
    def test_known_answer(self):
        p = spherical_position_ecef(0.0, 0.0, 0.0)
        assert np.allclose(p, [R_EARTH_M, 0.0, 0.0])
        p = spherical_position_ecef(90.0, 0.0, 100.0)
        assert np.allclose(p, [0.0, 0.0, R_EARTH_M + 1e5], atol=1e-6)

    @pytest.mark.parametrize("lat", [-91.0, 91.0])
    def test_rejects_bad_latitude(self, lat):
        with pytest.raises(ValueError, match="lat_deg"):
            spherical_position_ecef(lat, 0.0, 500.0)

    def test_rejects_negative_altitude(self):
        with pytest.raises(ValueError, match="alt_km"):
            spherical_position_ecef(0.0, 0.0, -1.0)

    def test_field_magnitude_at_surface_is_finite(self):
        assert 20000.0 < field_magnitude_nt(0.0, 0.0, 0.0) < 70000.0

    def test_longitude_symmetry_at_the_equator(self):
        # |B| depends only on (g . r_hat)^2, so lon and lon+180 give the same
        # magnitude at the geographic equator.
        assert np.isclose(
            field_magnitude_nt(0.0, 0.0, 500.0), field_magnitude_nt(0.0, 180.0, 500.0)
        )
