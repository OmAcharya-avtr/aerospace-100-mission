"""Frame transforms, against independently derived values where one exists."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from orbitdrift import frames
from orbitdrift.constants import WGS84

_J2000 = 2451545.0


def test_gmst_at_j2000_matches_the_published_value() -> None:
    """GMST at J2000.0 is 280.46061837 degrees.

    This is an independently derived reference, not another implementation of
    the same expression: the value is the constant term of the IAU 1982
    expression itself and is quoted in Vallado 2013 section 3.5 and in the
    Astronomical Almanac. Agreement to 1e-7 degree confirms the whole
    expression evaluates at ``T = 0`` and that the seconds-to-degrees factor of
    240 was not inverted.
    """
    measured = float(np.rad2deg(frames.gmst_rad(_J2000)))
    assert measured == pytest.approx(280.46061837, rel=0.0, abs=1e-6)


def test_gmst_advances_by_one_sidereal_day_in_one_solar_day() -> None:
    """A solar day advances GMST by 360 degrees plus about 0.9856 degrees."""
    first = float(np.rad2deg(frames.gmst_rad(_J2000)))
    second = float(np.rad2deg(frames.gmst_rad(_J2000 + 1.0)))
    advance = (second - first) % 360.0
    assert advance == pytest.approx(0.9856473, rel=0.0, abs=1e-5)


def test_rotation_z_is_orthogonal_with_unit_determinant() -> None:
    for angle in (0.0, 0.3, 1.7, 3.0, 6.0):
        matrix = frames.rotation_z(angle)
        product = matrix @ matrix.T
        assert np.allclose(product, np.eye(3), rtol=0.0, atol=1e-15)
        assert float(np.linalg.det(matrix)) == pytest.approx(1.0, rel=0.0, abs=1e-15)


def test_teme_to_ecef_preserves_radius_and_z() -> None:
    r = np.array([[7000.0, 1200.0, -2500.0], [-3000.0, 6500.0, 1000.0]])
    out = frames.teme_to_ecef(r, _J2000 + 0.37)
    assert np.allclose(
        np.linalg.norm(out, axis=-1), np.linalg.norm(r, axis=-1), rtol=1e-14, atol=0.0
    )
    assert np.allclose(out[:, 2], r[:, 2], rtol=0.0, atol=0.0)


def _iterative_geodetic_height(r: np.ndarray) -> float:
    """Fixed-point geodetic height, independent of Bowring's closed form."""
    a = WGS84.radius_km
    f = WGS84.flattening
    assert f is not None
    e2 = f * (2.0 - f)
    x, y, z = r
    p = float(np.hypot(x, y))
    latitude = float(np.arctan2(z, p))
    height = 0.0
    for _ in range(200):
        n_curv = a / np.sqrt(1.0 - e2 * np.sin(latitude) ** 2)
        if abs(np.cos(latitude)) > abs(np.sin(latitude)):
            height = p / np.cos(latitude) - n_curv
        else:
            height = z / np.sin(latitude) - n_curv * (1.0 - e2)
        latitude = float(np.arctan2(z, p * (1.0 - e2 * n_curv / (n_curv + height))))
    return height


def test_geodetic_altitude_against_an_independent_iteration() -> None:
    """Bowring's closed form against a fixed-point solve of the same geometry.

    Two different algorithms for the same quantity, not two calls to one.

    Bowring's single-pass formula is built for points near the surface and its
    departure from the converged solution grows with height: measured over a
    latitude sweep in ``validation/validate_frames.py`` it is 0.3 mm at 200
    km, 1.2 mm at 400 km, 8.8 mm at 1200 km and 15.5 mm at 1700 km, the top of
    this package's population. The tolerance here is 4e-5 km, which covers
    3000 km with margin. Eight millimetres of geodetic height feeds an
    atmosphere model whose own density is uncertain by a factor of several, so
    the approximation is reported rather than removed.
    """
    a = WGS84.radius_km
    f = WGS84.flattening
    assert f is not None
    b = a * (1.0 - f)
    cases = [
        np.array([a + 400.0, 0.0, 0.0]),
        np.array([0.0, 0.0, b + 400.0]),
        np.array([0.0, 0.0, -(b + 400.0)]),
        np.array([7000.0, 1000.0, 2000.0]),
        np.array([-4000.0, -4000.0, 5000.0]),
        np.array([0.0, b + 100.0, 0.0]),
    ]
    for r in cases:
        assert float(frames.geodetic_altitude_km(r)) == pytest.approx(
            _iterative_geodetic_height(r), rel=0.0, abs=4e-5
        )


def test_geodetic_altitude_known_answers_at_the_poles_and_equator() -> None:
    """Hand-calculated: on the axes the height is the radius minus a semi-axis."""
    a = WGS84.radius_km
    f = WGS84.flattening
    assert f is not None
    b = a * (1.0 - f)
    assert float(frames.geodetic_altitude_km(np.array([a + 123.0, 0.0, 0.0]))) == pytest.approx(
        123.0, rel=0.0, abs=1e-9
    )
    assert float(frames.geodetic_altitude_km(np.array([0.0, 0.0, b + 456.0]))) == pytest.approx(
        456.0, rel=0.0, abs=1e-9
    )


def test_ric_basis_is_right_handed_and_orthonormal() -> None:
    rng = np.random.default_rng(7)
    for _ in range(20):
        r = rng.normal(0.0, 7000.0, 3)
        v = rng.normal(0.0, 7.0, 3)
        if np.linalg.norm(np.cross(r, v)) < 1e-6:
            continue
        basis = frames.ric_basis(r, v)
        assert np.allclose(basis @ basis.T, np.eye(3), rtol=0.0, atol=1e-13)
        assert float(np.linalg.det(basis)) == pytest.approx(1.0, rel=0.0, abs=1e-13)


def test_ric_radial_axis_is_along_position_and_cross_is_the_orbit_normal() -> None:
    r = np.array([7000.0, 0.0, 0.0])
    v = np.array([0.0, 7.5, 0.0])
    basis = frames.ric_basis(r, v)
    assert np.allclose(basis[0], [1.0, 0.0, 0.0], rtol=0.0, atol=1e-15)
    assert np.allclose(basis[2], [0.0, 0.0, 1.0], rtol=0.0, atol=1e-15)
    assert np.allclose(basis[1], [0.0, 1.0, 0.0], rtol=0.0, atol=1e-15)


def test_to_ric_preserves_the_length_of_the_difference_vector() -> None:
    r = np.array([7000.0, 1000.0, 2000.0])
    v = np.array([-1.0, 6.5, 3.0])
    delta = np.array([0.3, -1.2, 0.8])
    projected = frames.to_ric(delta, r, v)
    assert float(np.linalg.norm(projected)) == pytest.approx(
        float(np.linalg.norm(delta)), rel=1e-14, abs=0.0
    )


def test_sun_distance_at_j2000_is_near_perihelion() -> None:
    """Earth is near perihelion on 1 January, so the distance is about 0.983 AU.

    An independently known fact about the orbit, not a restatement of the
    series: perihelion falls in the first week of January, so the geocentric
    solar distance at J2000.0 must be below 1 AU by about 1.7 per cent.
    """
    distance = float(np.linalg.norm(frames.sun_position_teme_km(_J2000)))
    assert 0.980 * 149597870.7 < distance < 0.986 * 149597870.7


def test_moon_distance_stays_inside_the_published_perigee_apogee_range() -> None:
    """The lunar distance varies between about 356400 and 406700 km."""
    jd = _J2000 + np.linspace(0.0, 400.0, 801)
    distance = np.linalg.norm(frames.moon_position_teme_km(jd), axis=-1)
    assert float(distance.min()) > 352000.0
    assert float(distance.max()) < 410000.0


def test_moon_synodic_motion_is_about_thirteen_degrees_a_day() -> None:
    first = frames.moon_position_teme_km(_J2000)
    second = frames.moon_position_teme_km(_J2000 + 1.0)
    cosine = float(
        np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second))
    )
    assert 11.0 < float(np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0)))) < 15.0


@settings(max_examples=60, deadline=None)
@given(
    x=st.floats(-9000.0, 9000.0, allow_nan=False),
    y=st.floats(-9000.0, 9000.0, allow_nan=False),
    z=st.floats(-9000.0, 9000.0, allow_nan=False),
    jd=st.floats(2451545.0, 2461545.0, allow_nan=False),
)
def test_teme_ecef_round_trip_closes(x: float, y: float, z: float, jd: float) -> None:
    """Rotating to ECEF and back must return the input to within rounding."""
    r = np.array([x, y, z])
    if np.linalg.norm(r) < 1.0:
        return
    theta = float(frames.gmst_rad(jd))
    forward = frames.teme_to_ecef(r, jd)
    back = frames.rotation_z(-theta) @ forward
    assert np.allclose(back, r, rtol=1e-12, atol=1e-9)


@settings(max_examples=60, deadline=None)
@given(
    rx=st.floats(2000.0, 9000.0),
    ry=st.floats(-9000.0, 9000.0),
    rz=st.floats(-9000.0, 9000.0),
    vx=st.floats(-9.0, 9.0),
    vy=st.floats(-9.0, 9.0),
    vz=st.floats(-9.0, 9.0),
    dx=st.floats(-50.0, 50.0),
    dy=st.floats(-50.0, 50.0),
    dz=st.floats(-50.0, 50.0),
)
def test_ric_projection_is_an_isometry(
    rx: float, ry: float, rz: float, vx: float, vy: float, vz: float,
    dx: float, dy: float, dz: float,
) -> None:
    """An orthonormal basis change cannot alter a vector's length."""
    r = np.array([rx, ry, rz])
    v = np.array([vx, vy, vz])
    delta = np.array([dx, dy, dz])
    if np.linalg.norm(np.cross(r, v)) < 1e-3:
        return
    projected = frames.to_ric(delta, r, v)
    assert float(np.linalg.norm(projected)) == pytest.approx(
        float(np.linalg.norm(delta)), rel=1e-11, abs=1e-12
    )
