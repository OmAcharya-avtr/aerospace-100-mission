"""Property-based tests for the algebraic identities the package relies on."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from momentummgr import (
    MU_EARTH,
    aerodynamic_torque,
    averaged_controllability,
    body_dcm_from_lvlh,
    circular_state,
    dipole_field_eci,
    gravity_gradient_torque,
    lvlh_dcm,
    magnetic_dump_command,
    node_axes,
    orbital_period,
    pyramid_four,
    residual_dipole_torque,
    srp_torque,
    tetrahedral_four,
    thruster_dump,
    uncontrollable_fraction,
)

SETTINGS = settings(max_examples=60, deadline=None,
                    suppress_health_check=[HealthCheck.too_slow])
angles = st.floats(-np.pi, np.pi, allow_nan=False, allow_infinity=False)
small = st.floats(-1.0, 1.0, allow_nan=False, allow_infinity=False)
positive = st.floats(0.1, 10.0, allow_nan=False, allow_infinity=False)


def _unit(x: float, y: float, z: float) -> np.ndarray | None:
    v = np.array([x, y, z])
    n = float(np.linalg.norm(v))
    return None if n < 1e-6 else v / n


@SETTINGS
@given(angles, angles.filter(lambda a: True))
def test_node_axes_are_an_orthonormal_right_handed_triad(inc: float, raan: float) -> None:
    p, q, h = node_axes(inc, raan)
    m = np.vstack([p, q, h])
    assert np.allclose(m @ m.T, np.eye(3), atol=1e-13)
    assert float(np.linalg.det(m)) == pytest.approx(1.0, abs=1e-13)


@SETTINGS
@given(angles, angles, angles)
def test_body_dcm_is_orthonormal(yaw: float, pitch: float, roll: float) -> None:
    c = body_dcm_from_lvlh(yaw, pitch, roll)
    assert np.allclose(c @ c.T, np.eye(3), atol=1e-13)
    assert float(np.linalg.det(c)) == pytest.approx(1.0, abs=1e-13)


@SETTINGS
@given(angles, angles, st.floats(0.0, 2 * np.pi))
def test_lvlh_axes_match_the_analytic_triad(inc: float, raan: float, u: float) -> None:
    radius = 6.878137e6
    r, v = circular_state(radius, inc, raan, u)
    c = lvlh_dcm(r, v)
    assert np.allclose(c @ c.T, np.eye(3), atol=1e-12)
    # z along nadir, x along velocity for a circular orbit
    assert np.allclose(c[2], -r / np.linalg.norm(r), atol=1e-12)
    assert np.allclose(c[0], v / np.linalg.norm(v), atol=1e-12)


@SETTINGS
@given(small, small, small, positive, positive, positive)
def test_gravity_gradient_is_perpendicular_to_nadir(
    x: float, y: float, z: float, a: float, b: float, cc: float
) -> None:
    u = _unit(x, y, z)
    if u is None:
        return
    moments = np.sort(np.array([a, b, cc]))
    if moments[0] + moments[1] < moments[2]:
        moments[2] = moments[0] + moments[1]
    t = gravity_gradient_torque(np.diag(moments), u, 7.0e6)
    # T = 3 n^2 (u x I u) is orthogonal to u exactly; the residual is roundoff on the
    # natural scale 3 n^2 |I u|, not on |T|, which is itself zero for an isotropic inertia.
    scale = 3.0 * MU_EARTH / 7.0e6**3 * float(np.linalg.norm(np.diag(moments) @ u))
    assert abs(float(t @ u)) <= 1e-12 * scale


@SETTINGS
@given(small, small, small)
def test_gravity_gradient_is_invariant_under_nadir_sign_flip(
    x: float, y: float, z: float
) -> None:
    u = _unit(x, y, z)
    if u is None:
        return
    inertia = np.diag([4.0, 8.0, 10.0])
    assert np.allclose(
        gravity_gradient_torque(inertia, u, 7.0e6),
        gravity_gradient_torque(inertia, -u, 7.0e6),
        atol=1e-20,
    )


@SETTINGS
@given(small, small, small, st.floats(1.0, 5.0))
def test_aerodynamic_torque_is_quadratic_in_speed(
    x: float, y: float, z: float, scale: float
) -> None:
    v = _unit(x, y, z)
    if v is None:
        return
    args = (1e-12, 2.2, 0.6, [0.02, 0.01, 0.05])
    t1 = aerodynamic_torque(args[0], v * 7600.0, *args[1:])
    t2 = aerodynamic_torque(args[0], v * 7600.0 * scale, *args[1:])
    assert np.allclose(t2, t1 * scale**2, rtol=1e-12)


@SETTINGS
@given(small, small, small, st.floats(0.0, 1.0))
def test_srp_torque_is_linear_in_one_plus_q(x: float, y: float, z: float, q: float) -> None:
    s = _unit(x, y, z)
    if s is None:
        return
    t0 = srp_torque(s, 1.2, 0.0, [0.02, 0.0, 0.03])
    tq = srp_torque(s, 1.2, q, [0.02, 0.0, 0.03])
    assert np.allclose(tq, t0 * (1.0 + q), rtol=1e-12)


@SETTINGS
@given(small, small, small, small, small, small)
def test_magnetic_torque_is_perpendicular_to_both_arguments(
    mx: float, my: float, mz: float, bx: float, by: float, bz: float
) -> None:
    m = np.array([mx, my, mz])
    b = np.array([bx, by, bz]) * 1e-5
    t = residual_dipole_torque(m, b)
    scale = float(np.linalg.norm(m)) * float(np.linalg.norm(b))
    if scale == 0.0:
        return
    # m x B is orthogonal to both factors exactly; roundoff scales with |m| |B|.
    assert abs(float(t @ m)) <= 1e-12 * scale * float(np.linalg.norm(m))
    assert abs(float(t @ b)) <= 1e-12 * scale * float(np.linalg.norm(b))


@SETTINGS
@given(small, small, small, small, small, small)
def test_dump_command_torque_is_always_perpendicular_to_the_field(
    hx: float, hy: float, hz: float, bx: float, by: float, bz: float
) -> None:
    h = np.array([hx, hy, hz]) * 0.02
    b = np.array([bx, by, bz]) * 1e-5
    if np.linalg.norm(b) < 1e-9 or np.linalg.norm(h) < 1e-6:
        return
    cmd = magnetic_dump_command(h, b, gain=1.0, max_dipole_am2=5.0)
    scale = float(np.linalg.norm(cmd.dipole_am2)) * float(np.linalg.norm(b)) ** 2
    assert abs(float(cmd.torque_nm @ b)) <= 1e-12 * max(scale, 1e-300)
    assert 0.0 <= cmd.uncontrollable_fraction <= 1.0 + 1e-12


@SETTINGS
@given(small, small, small, small, small, small)
def test_uncontrollable_fraction_is_a_direction_cosine(
    hx: float, hy: float, hz: float, bx: float, by: float, bz: float
) -> None:
    h = np.array([[hx, hy, hz]])
    b = np.array([[bx, by, bz]]) * 1e-5
    if np.linalg.norm(b) < 1e-9:
        return
    f = uncontrollable_fraction(h, b)
    assert np.all(f >= -1e-15) and np.all(f <= 1.0 + 1e-12)
    # scaling either vector cannot change a direction cosine
    assert np.allclose(f, uncontrollable_fraction(h * 3.0, b * 7.0), atol=1e-12)


@SETTINGS
@given(small, small, small, st.sampled_from([0.3, 0.6, 1.0]))
def test_wheel_allocation_preserves_the_body_request(
    x: float, y: float, z: float, frac: float
) -> None:
    for array in (pyramid_four(), tetrahedral_four()):
        d = _unit(x, y, z)
        if d is None:
            return
        h_body = d * array.guaranteed_body_envelope_nms * 0.5
        alloc = array.allocate(h_body, avoid_zero_speed=True, envelope_fraction=frac)
        assert np.allclose(array.body_momentum(alloc.wheel_momentum_nms), h_body, atol=1e-15)


@SETTINGS
@given(small, small, small)
def test_biasing_never_reduces_the_zero_speed_margin(x: float, y: float, z: float) -> None:
    w = pyramid_four()
    d = _unit(x, y, z)
    if d is None:
        return
    h_body = d * w.guaranteed_body_envelope_nms * 0.5
    plain = w.allocate(h_body, avoid_zero_speed=False)
    biased = w.allocate(h_body, avoid_zero_speed=True)
    assert biased.min_abs_momentum_nms >= plain.min_abs_momentum_nms - 1e-15


@SETTINGS
@given(st.floats(6.6e6, 4.2e7))
def test_period_follows_keplers_third_law(radius: float) -> None:
    t1 = orbital_period(radius)
    t2 = orbital_period(radius * 4.0)
    assert t2 / t1 == pytest.approx(8.0, rel=1e-12)


@SETTINGS
@given(st.floats(0.001, 1.0), st.floats(0.05, 2.0), st.floats(50.0, 320.0))
def test_thruster_propellant_is_linear_in_momentum(dh: float, arm: float, isp: float) -> None:
    a = thruster_dump(dh, arm, isp)
    b = thruster_dump(2.0 * dh, arm, isp)
    assert b.propellant_kg == pytest.approx(2.0 * a.propellant_kg, rel=1e-12)


@SETTINGS
@given(angles, angles)
def test_controllability_gramian_has_trace_two_and_bounded_eigenvalues(
    inc: float, raan: float
) -> None:
    u = np.linspace(0.0, 2.0 * np.pi, 241)
    r, _ = circular_state(6.878137e6, inc, raan, u)
    b = dipole_field_eci(r)
    gram, eig, _ = averaged_controllability(b)
    assert float(gram.trace()) == pytest.approx(2.0, abs=1e-12)
    assert np.all(eig >= -1e-12) and np.all(eig <= 1.0 + 1e-12)
    assert np.allclose(gram, gram.T, atol=1e-14)
