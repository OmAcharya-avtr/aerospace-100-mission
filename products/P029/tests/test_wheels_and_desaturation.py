"""Unit tests for the wheel array and the desaturation models, including edge cases."""

from __future__ import annotations

import numpy as np
import pytest

from momentummgr import (
    averaged_controllability,
    circular_state,
    count_zero_crossings,
    dipole_cost,
    dipole_field_eci,
    magnetic_dump_command,
    orthogonal_three,
    pyramid_four,
    reference_orbit,
    tetrahedral_four,
    thruster_dump,
)


def test_null_space_dimensions() -> None:
    assert pyramid_four().null_basis.shape == (4, 1)
    assert tetrahedral_four().null_basis.shape == (4, 1)
    assert orthogonal_three().null_basis.shape == (3, 0)


def test_null_vector_is_annihilated_by_the_distribution_matrix() -> None:
    w = pyramid_four()
    n = w.null_basis[:, 0]
    assert np.allclose(w.distribution_matrix @ n, np.zeros(3), atol=1e-15)
    # the isotropic pyramid's null direction is (1, -1, 1, -1)/2 up to sign
    assert np.allclose(np.abs(n), 0.5, atol=1e-12)


def test_biasing_is_a_no_op_without_redundancy() -> None:
    w = orthogonal_three()
    h = np.array([0.01, -0.02, 0.005])
    a = w.allocate(h, avoid_zero_speed=True)
    b = w.allocate(h, avoid_zero_speed=False)
    assert np.array_equal(a.wheel_momentum_nms, b.wheel_momentum_nms)
    assert a.null_coefficients.size == 0
    assert np.allclose(a.wheel_momentum_nms, h, atol=1e-15)


def test_biasing_lifts_a_wheel_off_zero() -> None:
    # Minimum norm gives h_i = (3/4) a_i . h_body, so any request perpendicular to a wheel
    # axis puts that wheel exactly at zero. Wheel 0's axis is (sin b, 0, cos b), so
    # h_body along (cos b, 0, -sin b) is perpendicular to it and wheel 0 gets nothing.
    w = pyramid_four()
    b = np.arctan(np.sqrt(2.0))
    h_body = np.array([np.cos(b), 0.0, -np.sin(b)]) * 0.02
    plain = w.allocate(h_body, avoid_zero_speed=False)
    biased = w.allocate(h_body, avoid_zero_speed=True)
    assert plain.min_abs_momentum_nms < 1e-12
    assert biased.min_abs_momentum_nms > 0.2 * w.max_momentum_nms
    assert np.allclose(w.body_momentum(biased.wheel_momentum_nms), h_body, atol=1e-15)


def test_allocation_reports_infeasibility_rather_than_silently_clipping() -> None:
    w = pyramid_four(max_momentum_nms=0.05)
    too_big = np.array([0.0, 0.0, 1.0]) * 10.0
    alloc = w.allocate(too_big, avoid_zero_speed=True)
    assert alloc.feasible is False
    assert np.allclose(w.body_momentum(alloc.wheel_momentum_nms), too_big, atol=1e-12)


def test_speeds_follow_from_momenta() -> None:
    w = pyramid_four(wheel_inertia_kg_m2=2.0e-3, max_momentum_nms=0.05)
    speeds = w.speeds_rad_s(np.array([0.01, -0.02, 0.0, 0.005]))
    assert np.allclose(speeds, np.array([5.0, -10.0, 0.0, 2.5]), atol=1e-14)


def test_saturation_fraction() -> None:
    w = pyramid_four(max_momentum_nms=0.05)
    assert w.saturation_fraction([0.01, -0.025, 0.0, 0.005]) == pytest.approx(0.5)


def test_zero_crossing_counter() -> None:
    hist = np.array([[1.0], [0.5], [-0.5], [-1.0], [0.5]])
    assert count_zero_crossings(hist).tolist() == [2]
    # with a deadband of 0.6 the +-0.5 samples are ignored, leaving 1.0 -> -1.0 -> nothing
    assert count_zero_crossings(hist, deadband_nms=0.6).tolist() == [1]
    assert count_zero_crossings(np.zeros((5, 2))).tolist() == [0, 0]


def test_dipole_command_saturation_preserves_direction() -> None:
    h = np.array([0.05, -0.02, 0.01])
    b = np.array([1.0, 2.0, -3.0]) * 1e-5
    free = magnetic_dump_command(h, b, gain=1.0)
    limited = magnetic_dump_command(h, b, gain=1.0, max_dipole_am2=0.5)
    assert limited.saturated is True
    assert float(np.linalg.norm(limited.dipole_am2)) == pytest.approx(0.5, rel=1e-12)
    cos = float(free.dipole_am2 @ limited.dipole_am2) / (
        np.linalg.norm(free.dipole_am2) * np.linalg.norm(limited.dipole_am2)
    )
    assert cos == pytest.approx(1.0, abs=1e-12)


def test_uncontrollable_component_plus_torque_direction_span_the_space() -> None:
    h = np.array([0.03, 0.01, -0.02])
    b = np.array([2.0, -1.0, 0.5]) * 1e-5
    cmd = magnetic_dump_command(h, b, gain=1.0)
    reconstructed = cmd.uncontrollable_nms - cmd.torque_nm
    assert np.allclose(reconstructed, h, atol=1e-14)


def test_zero_momentum_gives_zero_command_and_zero_fraction() -> None:
    cmd = magnetic_dump_command([0.0, 0.0, 0.0], [0.0, 0.0, 3e-5], gain=1.0)
    assert np.allclose(cmd.dipole_am2, np.zeros(3), atol=1e-30)
    assert cmd.uncontrollable_fraction == 0.0


def test_equatorial_orbit_has_a_blocked_axis() -> None:
    orbit = reference_orbit(500.0)
    u = np.linspace(0.0, 2.0 * np.pi, 361)
    r, _ = circular_state(orbit.radius_m, 0.0, 0.0, u)
    _, eig, _ = averaged_controllability(dipole_field_eci(r))
    assert float(eig[0]) == pytest.approx(0.0, abs=1e-12)
    assert float(eig[1]) == pytest.approx(1.0, abs=1e-12)


def test_inclined_orbit_is_controllable_on_average() -> None:
    orbit = reference_orbit(500.0)
    u = np.linspace(0.0, 2.0 * np.pi, 361)
    r, _ = circular_state(orbit.radius_m, np.radians(51.6), 0.0, u)
    _, eig, _ = averaged_controllability(dipole_field_eci(r))
    assert float(eig[0]) > 0.5


def test_dipole_cost_of_a_constant_command() -> None:
    t = np.linspace(0.0, 100.0, 11)
    m = np.tile(np.array([3.0, 4.0, 0.0]), (11, 1))
    assert dipole_cost(m, t) == pytest.approx(500.0, rel=1e-14)


def test_thruster_efficiency_scales_propellant() -> None:
    full = thruster_dump(0.05, 0.5, 220.0, efficiency=1.0)
    half = thruster_dump(0.05, 0.5, 220.0, efficiency=0.5)
    assert half.propellant_kg == pytest.approx(2.0 * full.propellant_kg, rel=1e-14)


def test_thruster_accepts_a_vector_or_a_scalar() -> None:
    a = thruster_dump(np.array([0.03, 0.04, 0.0]), 0.5, 220.0)
    b = thruster_dump(0.05, 0.5, 220.0)
    assert a.propellant_kg == pytest.approx(b.propellant_kg, rel=1e-14)
