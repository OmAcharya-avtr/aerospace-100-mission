"""Property-based tests of the algebraic identities each torque expression must satisfy.

These are the identities that would survive any change of coefficient, so they catch a
sign error or a transposed cross product that a single known-answer test can miss.
"""

from __future__ import annotations

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from disturbtorque import (
    MU_EARTH,
    aerodynamic_torque,
    body_from_lvlh,
    circular_orbit_state,
    dipole_field_eci,
    gravity_gradient_torque,
    lvlh_from_eci,
    magnetic_torque,
    orbital_period,
    solar_radiation_torque,
    sun_unit_vector_eci,
)

finite = dict(allow_nan=False, allow_infinity=False)
angle = st.floats(-np.pi, np.pi, **finite)
small = st.floats(-1.0, 1.0, **finite)
pos = st.floats(0.1, 100.0, **finite)


def _unit(x, y, z):
    v = np.array([x, y, z], dtype=float)
    n = float(np.linalg.norm(v))
    assume(n > 1e-3)
    return v / n


@given(small, small, small, pos, pos, pos, st.floats(6.6e6, 4.3e7, **finite))
@settings(max_examples=200, deadline=None)
def test_gravity_gradient_is_perpendicular_to_nadir(x, y, z, ia, ib, ic, radius):
    # T = (3 mu / R^3) u x (I u) is a cross product with u, so T . u = 0 identically,
    # and it is also perpendicular to I u.
    u = _unit(x, y, z)
    moments = np.sort([ia, ib, ic])
    assume(moments[0] + moments[1] > moments[2] * 1.001)
    inertia = np.diag(moments)
    t = gravity_gradient_torque(inertia, u, radius)
    scale = float(np.max(moments)) * 3.0 * MU_EARTH / radius**3
    assert abs(float(t @ u)) <= 1e-12 * scale
    assert abs(float(t @ (inertia @ u))) <= 1e-11 * scale * np.max(moments)


@given(small, small, small, pos, pos, pos, st.floats(6.6e6, 2.0e7, **finite),
       st.floats(1.1, 4.0, **finite))
@settings(max_examples=100, deadline=None)
def test_gravity_gradient_scales_as_inverse_cube_of_radius(x, y, z, ia, ib, ic, radius, factor):
    u = _unit(x, y, z)
    moments = np.sort([ia, ib, ic])
    assume(moments[0] + moments[1] > moments[2] * 1.001)
    inertia = np.diag(moments)
    t1 = gravity_gradient_torque(inertia, u, radius)
    t2 = gravity_gradient_torque(inertia, u, radius * factor)
    assume(float(np.linalg.norm(t1)) > 1e-30)
    assert np.allclose(t2, t1 / factor**3, rtol=1e-10)


@given(
    st.floats(1e-16, 1e-8, **finite),
    st.floats(1000.0, 9000.0, **finite),
    st.floats(1.5, 3.0, **finite),
    st.floats(0.01, 20.0, **finite),
    st.floats(1.1, 10.0, **finite),
)
@settings(max_examples=200, deadline=None)
def test_aerodynamic_scales_linearly_in_density_and_area_and_quadratically_in_speed(
    rho, speed, cd, area, factor
):
    offset = np.array([0.01, -0.02, 0.05])
    base = aerodynamic_torque(rho, [speed, 0.3 * speed, 0.0], cd, area, offset)
    assume(float(np.linalg.norm(base)) > 1e-30)
    assert np.allclose(
        aerodynamic_torque(rho * factor, [speed, 0.3 * speed, 0.0], cd, area, offset),
        base * factor,
        rtol=1e-12,
    )
    assert np.allclose(
        aerodynamic_torque(rho, [speed, 0.3 * speed, 0.0], cd, area * factor, offset),
        base * factor,
        rtol=1e-12,
    )
    assert np.allclose(
        aerodynamic_torque(rho, [speed * factor, 0.3 * speed * factor, 0.0], cd, area, offset),
        base * factor**2,
        rtol=1e-12,
    )


@given(small, small, small, small, small, small)
@settings(max_examples=200, deadline=None)
def test_aerodynamic_torque_is_perpendicular_to_the_offset_and_the_flow(ox, oy, oz, vx, vy, vz):
    offset = np.array([ox, oy, oz])
    v = np.array([vx, vy, vz]) * 7000.0
    assume(np.linalg.norm(offset) > 1e-3 and np.linalg.norm(v) > 1.0)
    assume(all(abs(c) < 1e3 for c in offset))
    t = aerodynamic_torque(1e-12, v, 2.2, 1.0, offset)
    # the tolerance is set by the torque *scale* |offset| * |F|, not by |t|, which can be
    # arbitrarily small when the offset is nearly parallel to the flow
    force = 0.5 * 1e-12 * 2.2 * 1.0 * float(np.linalg.norm(v)) ** 2
    scale = float(np.linalg.norm(offset)) * force
    assert abs(float(t @ offset)) <= 1e-12 * scale * np.linalg.norm(offset)
    assert abs(float(t @ v)) <= 1e-12 * scale * np.linalg.norm(v)


@given(small, small, small, small, small, small, st.floats(0.0, 1.0, **finite))
@settings(max_examples=200, deadline=None)
def test_solar_torque_is_perpendicular_to_the_sun_line_and_the_offset(ox, oy, oz, sx, sy, sz, q):
    offset = np.array([ox, oy, oz])
    assume(np.linalg.norm(offset) > 1e-3)
    s = _unit(sx, sy, sz)
    t = solar_radiation_torque(s, 1.5, q, offset)
    force = 4.5398073356e-06 * 1.5 * (1.0 + q)
    scale = float(np.linalg.norm(offset)) * force
    assert abs(float(t @ s)) <= 1e-12 * scale
    assert abs(float(t @ offset)) <= 1e-12 * scale * np.linalg.norm(offset)


@given(small, small, small, small, small, small)
@settings(max_examples=200, deadline=None)
def test_magnetic_torque_is_perpendicular_to_m_and_b_and_antisymmetric(mx, my, mz, bx, by, bz):
    m = np.array([mx, my, mz])
    b = np.array([bx, by, bz]) * 1e-5
    # exclude physically meaningless magnitudes, where the tolerance scale itself
    # underflows to zero in float64
    assume(np.linalg.norm(m) > 1e-4 and np.linalg.norm(b) > 1e-12)
    t = magnetic_torque(m, b)
    scale = float(np.linalg.norm(m)) * float(np.linalg.norm(b))
    assert abs(float(t @ m)) <= 1e-12 * scale * (np.linalg.norm(m) + 1.0)
    assert abs(float(t @ b)) <= 1e-12 * scale * (np.linalg.norm(b) + 1.0)
    assert np.allclose(magnetic_torque(b, m), -t, rtol=0.0, atol=1e-30)


@given(st.floats(6.6e6, 4.2e7, **finite), st.floats(-1.5, 1.5, **finite), angle, angle)
@settings(max_examples=150, deadline=None)
def test_lvlh_frame_is_orthonormal_right_handed_and_nadir_aligned(radius, inc, raan, u):
    r, v = circular_orbit_state(radius, inc, raan, u)
    assume(abs(np.sin(inc)) > 1e-6 or True)
    c = lvlh_from_eci(r, v)
    assert np.allclose(c @ c.T, np.eye(3), atol=1e-12)
    assert float(np.linalg.det(c)) == __import__("pytest").approx(1.0, abs=1e-12)
    assert np.allclose(c @ (r / np.linalg.norm(r)), [0.0, 0.0, -1.0], atol=1e-12)
    assert np.allclose(c @ (v / np.linalg.norm(v)), [1.0, 0.0, 0.0], atol=1e-12)


@given(angle, angle, angle)
@settings(max_examples=150, deadline=None)
def test_body_from_lvlh_is_a_proper_rotation(yaw, pitch, roll):
    c = body_from_lvlh(yaw, pitch, roll)
    assert np.allclose(c @ c.T, np.eye(3), atol=1e-13)
    assert float(np.linalg.det(c)) > 0.0


@given(st.floats(6.6e6, 4.2e7, **finite), st.floats(-1.5, 1.5, **finite), angle, angle)
@settings(max_examples=150, deadline=None)
def test_dipole_magnitude_lies_between_the_equatorial_and_polar_values(radius, inc, raan, u):
    r, _ = circular_orbit_state(radius, inc, raan, u)
    b = float(np.linalg.norm(dipole_field_eci(r)))
    k_over_r3 = 7.96e15 / radius**3
    assert k_over_r3 * (1.0 - 1e-12) <= b <= 2.0 * k_over_r3 * (1.0 + 1e-12)


@given(st.floats(6.6e6, 4.2e7, **finite), st.floats(1.1, 5.0, **finite))
@settings(max_examples=100, deadline=None)
def test_kepler_third_law(radius, factor):
    t1 = orbital_period(radius)
    t2 = orbital_period(radius * factor)
    assert t2 / t1 == __import__("pytest").approx(factor**1.5, rel=1e-12)


@given(st.floats(2440000.0, 2470000.0, **finite))
@settings(max_examples=200, deadline=None)
def test_sun_vector_is_a_unit_vector(jd):
    assert abs(float(np.linalg.norm(sun_unit_vector_eci(jd))) - 1.0) < 1e-14
