"""Known-answer tests. Every target is worked out by hand in the comment above it."""

from __future__ import annotations

import numpy as np
import pytest

from momentummgr import (
    EARTH_REDUCED_DIPOLE,
    MU_EARTH,
    SRP_PRESSURE_1AU,
    STANDARD_GRAVITY,
    aerodynamic_torque,
    density,
    dipole_field_eci,
    eclipse_fraction,
    gravity_gradient_torque,
    gravity_gradient_worst_case,
    magnetic_dump_command,
    momentum_per_orbit_eci,
    node_axes,
    orbital_period,
    pyramid_four,
    reference_orbit,
    reference_smallsat,
    residual_dipole_torque,
    srp_torque,
    sun_direction_for_beta,
    thruster_dump,
)


def test_solar_pressure_constant() -> None:
    # P = Phi / c = 1361 / 299792458 = 4.53980733...e-06 N m^-2
    assert SRP_PRESSURE_1AU == pytest.approx(4.5398073356e-06, rel=1e-10)


def test_orbital_period_at_500_km() -> None:
    # R = 6378137 + 500000 = 6878137 m; R^3 = 3.2539619167e20 m^3
    # T = 2 pi sqrt(R^3 / 3.986004418e14) = 2 pi * 903.52380... = 5676.9780... s
    assert orbital_period(6878137.0) == pytest.approx(5676.9780, rel=1e-7)


def test_gravity_gradient_planar_case() -> None:
    # I = diag(4, 8, 10), u = (0, 0.5, sqrt(3)/2) i.e. 30 deg off body z in the y-z plane.
    # I u = (0, 4, 5 sqrt 3);  u x (I u) = (0.5 * 5 sqrt 3 - (sqrt 3 / 2) * 4, 0, 0)
    #                                    = (sqrt(3)/2, 0, 0) = (0.8660254, 0, 0)
    # 3 mu / R^3 = 3 * 3.986004418e14 / 3.2539619167e20 = 3.6749087912e-06 s^-2
    # T_x = 3.6749087912e-06 * 0.8660254038 = 3.1825643698e-06 N m
    r = 6878137.0
    u = np.array([0.0, 0.5, np.sqrt(3.0) / 2.0])
    t = gravity_gradient_torque(np.diag([4.0, 8.0, 10.0]), u, r)
    assert t[0] == pytest.approx(3.1825643698e-06, rel=1e-10)
    assert t[1] == pytest.approx(0.0, abs=1e-22)
    assert t[2] == pytest.approx(0.0, abs=1e-22)


def test_gravity_gradient_vanishes_along_a_principal_axis() -> None:
    t = gravity_gradient_torque(np.diag([4.0, 8.0, 10.0]), [0.0, 0.0, 1.0], 6878137.0)
    assert np.linalg.norm(t) == pytest.approx(0.0, abs=1e-30)


def test_gravity_gradient_worst_case_matches_the_45_degree_value() -> None:
    # T_max = (3 mu / 2 R^3) |Izz - Iyy| = 3.6749087912e-06 / 2 * 2 = 3.6749087912e-06 N m
    r = 6878137.0
    worst = gravity_gradient_worst_case(8.0, 10.0, r)
    assert worst == pytest.approx(3.6749087912e-06, rel=1e-10)
    theta = np.radians(45.0)
    u = np.array([0.0, np.sin(theta), np.cos(theta)])
    t = gravity_gradient_torque(np.diag([4.0, 8.0, 10.0]), u, r)
    assert abs(float(t[0])) == pytest.approx(worst, rel=1e-12)


def test_density_table_base_points() -> None:
    # 500 km is a table base altitude, so rho is the tabulated 6.967e-13 exactly.
    # 550 km: 6.967e-13 * exp(-50 / 63.822) = 6.967e-13 * 0.45665... = 3.1815e-13
    assert float(density(500e3)) == pytest.approx(6.967e-13, rel=1e-15)
    assert float(density(550e3)) == pytest.approx(6.967e-13 * np.exp(-50.0 / 63.822), rel=1e-15)


def test_aerodynamic_torque_hand_case() -> None:
    # 0.5 * 6.967e-13 * 2.2 * 0.6 = 4.5982200e-13; |F| = that * 7600^2 = 2.6559319e-05 N
    # T = (0, 0, 0.05) x (-2.6559319e-05, 0, 0) = (0, -1.3279659e-06, 0) N m
    t = aerodynamic_torque(6.967e-13, [7600.0, 0.0, 0.0], 2.2, 0.6, [0.0, 0.0, 0.05])
    assert t[1] == pytest.approx(-1.3279659e-06, rel=1e-8)
    assert t[0] == pytest.approx(0.0, abs=1e-24)
    assert t[2] == pytest.approx(0.0, abs=1e-24)


def test_srp_torque_hand_case_and_eclipse() -> None:
    # |F| = 4.5398073e-06 * 1.2 * 1.6 = 8.7164301e-06 N along -z
    # T = (0.02, 0, 0) x (0, 0, -8.7164301e-06) = (0, 1.7432860e-07, 0) N m
    t = srp_torque([0.0, 0.0, 1.0], 1.2, 0.6, [0.02, 0.0, 0.0])
    assert t[1] == pytest.approx(1.7432860e-07, rel=1e-8)
    dark = srp_torque([0.0, 0.0, 1.0], 1.2, 0.6, [0.02, 0.0, 0.0], illuminated=False)
    assert np.linalg.norm(dark) == 0.0


def test_residual_dipole_torque_hand_case() -> None:
    # m x B with m = (0.05, 0.05, 0.10), B = (0, 0, 3e-05):
    # (0.05*3e-05 - 0, 0 - 0.05*3e-05, 0) = (1.5e-06, -1.5e-06, 0) N m
    t = residual_dipole_torque([0.05, 0.05, 0.10], [0.0, 0.0, 3.0e-5])
    assert t[0] == pytest.approx(1.5e-06, rel=1e-14)
    assert t[1] == pytest.approx(-1.5e-06, rel=1e-14)
    assert t[2] == pytest.approx(0.0, abs=1e-25)


def test_dipole_field_equator_and_pole() -> None:
    # k / r^3 = 7.96e15 / (7e6)^3 = 7.96e15 / 3.43e20 = 2.3206997e-05 T
    k_over_r3 = EARTH_REDUCED_DIPOLE / 7.0e6**3
    b_eq = dipole_field_eci([7.0e6, 0.0, 0.0])
    b_pole = dipole_field_eci([0.0, 0.0, 7.0e6])
    assert b_eq[2] == pytest.approx(k_over_r3, rel=1e-14)
    assert b_eq[2] == pytest.approx(2.3206997e-05, rel=1e-7)
    assert b_pole[2] == pytest.approx(-2.0 * k_over_r3, rel=1e-14)
    assert np.linalg.norm(b_pole) == pytest.approx(2.0 * np.linalg.norm(b_eq), rel=1e-14)


def test_untilted_dipole_is_the_default() -> None:
    r = np.array([[6.9e6, 1.0e6, 2.0e6]])
    assert np.allclose(dipole_field_eci(r), dipole_field_eci(r, tilt_rad=0.0, rotation_angle_rad=3.1))


def test_node_axes_are_orthonormal_and_right_handed() -> None:
    p, q, h = node_axes(np.radians(51.6), np.radians(37.0))
    for v in (p, q, h):
        assert np.linalg.norm(v) == pytest.approx(1.0, rel=1e-15)
    assert float(p @ q) == pytest.approx(0.0, abs=1e-15)
    assert np.allclose(np.cross(p, q), h, atol=1e-15)


def test_eclipse_fraction_closed_form() -> None:
    # P027 publishes 0.3695911346 for this geometry; derived here from
    # f = (pi - arccos(-sqrt(1-(Re/R)^2)/cos beta)) / pi, no sampling.
    orbit = reference_orbit(500.0)
    sun = sun_direction_for_beta(orbit.inclination_rad, orbit.raan_rad, np.radians(20.0))
    f = eclipse_fraction(orbit.radius_m, orbit.inclination_rad, orbit.raan_rad, sun)
    assert f == pytest.approx(0.3695911346, rel=1e-10)


def test_full_sun_orbit_has_no_eclipse() -> None:
    orbit = reference_orbit(500.0)
    sun = sun_direction_for_beta(orbit.inclination_rad, orbit.raan_rad, np.radians(85.0))
    assert eclipse_fraction(orbit.radius_m, orbit.inclination_rad, orbit.raan_rad, sun) == 0.0


def test_momentum_per_orbit_matches_the_p027_published_reference() -> None:
    # P027 disturbtorque publishes, for this exact environment, an ECI momentum over one
    # orbit of [-2.4484285744e-03, 2.8372528957e-03, -2.2167544874e-03] N m s from an
    # independent implementation. See validation/p027_cross_check.py.
    sc = reference_smallsat()
    orbit = reference_orbit(500.0)
    sun = sun_direction_for_beta(orbit.inclination_rad, orbit.raan_rad, np.radians(20.0))
    dh = momentum_per_orbit_eci(sc, orbit, sun, "total")
    expected = np.array([-2.4484285744e-03, 2.8372528957e-03, -2.2167544874e-03])
    assert np.linalg.norm(dh - expected) / np.linalg.norm(expected) < 5e-10


def test_gravity_gradient_momentum_matches_p027() -> None:
    sc = reference_smallsat()
    orbit = reference_orbit(500.0)
    sun = sun_direction_for_beta(orbit.inclination_rad, orbit.raan_rad, np.radians(20.0))
    dh = momentum_per_orbit_eci(sc, orbit, sun, "gravity_gradient")
    assert float(np.linalg.norm(dh)) == pytest.approx(1.084062e-02, rel=2e-6)


def test_pyramid_geometry() -> None:
    # sin b = sqrt(2/3), cos b = 1/sqrt(3); A A^T = (4/3) I; A^+ = (3/4) A^T
    w = pyramid_four()
    aat = w.distribution_matrix @ w.distribution_matrix.T
    assert np.allclose(aat, (4.0 / 3.0) * np.eye(3), atol=1e-14)
    assert w.guaranteed_body_envelope_nms == pytest.approx(4.0 / 3.0 * 0.05, rel=1e-14)
    alloc = w.allocate([0.0, 0.0, 0.02], avoid_zero_speed=False)
    # each wheel gets (3/4) cos b h = 0.75 / sqrt(3) * 0.02 = 8.660254e-03 N m s
    assert np.allclose(alloc.wheel_momentum_nms, 0.75 / np.sqrt(3.0) * 0.02, atol=1e-15)


def test_magnetic_dump_hand_case() -> None:
    # B = (0, 0, 3e-05) T, h = (0.01, 0, 0) N m s, k = 1 s^-1.
    # B x h = (0, 3e-07, 0); m = -(1 / 9e-10)(0, 3e-07, 0) = (0, -333.333..., 0) A m^2
    # T = m x B = (-0.01, 0, 0) N m = -k h, because h is perpendicular to B.
    cmd = magnetic_dump_command([0.01, 0.0, 0.0], [0.0, 0.0, 3.0e-5], gain=1.0)
    assert cmd.dipole_am2[1] == pytest.approx(-0.01 / 3.0e-5, rel=1e-14)
    assert cmd.torque_nm[0] == pytest.approx(-0.01, rel=1e-14)
    assert cmd.uncontrollable_fraction == pytest.approx(0.0, abs=1e-15)


def test_momentum_along_the_field_is_uncontrollable() -> None:
    cmd = magnetic_dump_command([0.0, 0.0, 0.01], [0.0, 0.0, 3.0e-5], gain=1.0)
    assert np.linalg.norm(cmd.dipole_am2) == pytest.approx(0.0, abs=1e-18)
    assert cmd.uncontrollable_fraction == pytest.approx(1.0, rel=1e-15)


def test_thruster_dump_hand_case() -> None:
    # I = 2 * 0.05 / 0.5 = 0.2 N s; m_p = 0.2 / (220 * 9.80665) = 9.2701474e-05 kg
    d = thruster_dump(0.05, 0.5, 220.0)
    assert d.impulse_ns == pytest.approx(0.2, rel=1e-15)
    assert d.propellant_kg == pytest.approx(0.2 / (220.0 * STANDARD_GRAVITY), rel=1e-15)
    assert d.propellant_kg == pytest.approx(9.2701474e-05, rel=1e-7)


def test_mu_earth_is_the_wgs84_value() -> None:
    assert MU_EARTH == 3.986004418e14
