"""Known-answer tests. Every expected value below is hand arithmetic, written out in the
comment above the assertion, and matches ``validation/hand_calculations.py``."""

from __future__ import annotations

import numpy as np
import pytest

from disturbtorque import (
    EARTH_DIPOLE_MOMENT,
    MU_EARTH,
    R_EARTH_EQUATORIAL,
    R_EARTH_MEAN,
    SOLAR_IRRADIANCE_1AU,
    SPEED_OF_LIGHT,
    SRP_PRESSURE_1AU,
    aerodynamic_force,
    aerodynamic_torque,
    density,
    dipole_field_eci,
    dipole_field_magnitude,
    eclipse_fraction_cylindrical,
    gravity_gradient_max_magnitude,
    gravity_gradient_planar,
    gravity_gradient_torque,
    julian_date,
    magnetic_torque,
    orbital_period,
    solar_radiation_force,
    solar_radiation_torque,
    sun_direction_for_beta,
    sun_distance_au,
    sun_unit_vector_eci,
)

R7 = 7.0e6


def test_gravity_gradient_45_deg_hand_value():
    # I = diag(10, 20, 30) kg m^2, R = 7e6 m, nadir at 45 deg in the y-z plane.
    #   R^3 = 3.43e20 m^3
    #   3 mu / R^3 = 3 * 3.986004418e14 / 3.43e20 = 1.1958013254e15 / 3.43e20
    #              = 3.4863012402e-06 s^-2
    #   I u = (0, 20*0.70710678, 30*0.70710678) = (0, 14.14213562, 21.21320344)
    #   u x I u = (0.70710678 * (21.21320344 - 14.14213562), 0, 0) = (5.0, 0, 0)
    #   T = 3.4863012402e-06 * 5.0 = 1.7431506201e-05 N m about +x
    t = gravity_gradient_torque(np.diag([10.0, 20.0, 30.0]), [0.0, np.sqrt(0.5), np.sqrt(0.5)], R7)
    assert t[0] == pytest.approx(1.7431506201e-05, rel=1e-9)
    assert t[1] == 0.0
    assert t[2] == 0.0


def test_gravity_gradient_analytic_maximum_at_45_deg():
    # T(theta) = 3 mu / (2 R^3) (Izz - Iyy) sin 2 theta, maximum at theta = 45 deg where
    # sin 2 theta = 1, giving 1.7431506201e-06 * 10 = 1.7431506201e-05 N m.
    t_max = gravity_gradient_max_magnitude(20.0, 30.0, R7)
    assert t_max == pytest.approx(1.7431506201e-05, rel=1e-9)
    theta = np.linspace(0.0, np.pi / 2, 18001)
    sweep = gravity_gradient_planar(20.0, 30.0, theta, R7)
    assert np.degrees(theta[int(np.argmax(np.abs(sweep)))]) == pytest.approx(45.0, abs=1e-6)
    assert float(np.max(np.abs(sweep))) == pytest.approx(t_max, rel=1e-12)


def test_gravity_gradient_tensor_and_planar_forms_agree():
    theta = np.linspace(0.0, np.pi, 401)
    inertia = np.diag([10.0, 20.0, 30.0])
    tensor = np.array(
        [gravity_gradient_torque(inertia, [0.0, np.sin(a), np.cos(a)], R7)[0] for a in theta]
    )
    assert np.allclose(tensor, gravity_gradient_planar(20.0, 30.0, theta, R7), rtol=0.0, atol=1e-19)


def test_gravity_gradient_vanishes_on_a_principal_axis():
    inertia = np.diag([10.0, 20.0, 30.0])
    for axis in ([1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]):
        assert np.linalg.norm(gravity_gradient_torque(inertia, axis, R7)) == 0.0


def test_aerodynamic_hand_value():
    # rho = 1e-12, v = (7500, 0, 0), Cd = 2.2, A = 1.5, offset = (0, 0, 0.1)
    #   0.5 * 2.2 * 1.5 * 1e-12 = 1.65e-12 ; |v|^2 = 5.625e7
    #   |F| = 1.65e-12 * 5.625e7 = 9.28125e-05 N along -x
    #   T = (0,0,0.1) x (-9.28125e-05,0,0) = (0, -9.28125e-06, 0) N m
    f = aerodynamic_force(1e-12, [7500.0, 0.0, 0.0], 2.2, 1.5)
    assert f[0] == pytest.approx(-9.28125e-05, rel=1e-12)
    t = aerodynamic_torque(1e-12, [7500.0, 0.0, 0.0], 2.2, 1.5, [0.0, 0.0, 0.1])
    assert t[1] == pytest.approx(-9.28125e-06, rel=1e-12)
    assert t[0] == 0.0 and t[2] == 0.0


def test_solar_radiation_hand_value():
    # P = 1361 / 299792458 = 4.5398073356e-06 N m^-2
    # |F| = P * 2.0 * (1 + 0.6) = P * 3.2 = 1.4527383474e-05 N along -z
    # T = (0.3,0,0) x (0,0,-|F|) = (0, 0.3*|F|, 0) = (0, 4.3582150422e-06, 0) N m
    p_hand = 1361.0 / 299792458.0
    assert SRP_PRESSURE_1AU == pytest.approx(p_hand, rel=1e-15)
    assert SOLAR_IRRADIANCE_1AU / SPEED_OF_LIGHT == pytest.approx(p_hand, rel=1e-15)
    f = solar_radiation_force([0.0, 0.0, 1.0], 2.0, 0.6)
    assert f[2] == pytest.approx(-p_hand * 3.2, rel=1e-12)
    t = solar_radiation_torque([0.0, 0.0, 1.0], 2.0, 0.6, [0.3, 0.0, 0.0])
    assert t[1] == pytest.approx(0.3 * p_hand * 3.2, rel=1e-12)


def test_solar_radiation_inverse_square_and_eclipse():
    base = solar_radiation_torque([0.0, 0.0, 1.0], 2.0, 0.6, [0.3, 0.0, 0.0])
    far = solar_radiation_torque([0.0, 0.0, 1.0], 2.0, 0.6, [0.3, 0.0, 0.0], distance_au=2.0)
    assert far[1] == pytest.approx(0.25 * base[1], rel=1e-12)
    dark = solar_radiation_torque([0.0, 0.0, 1.0], 2.0, 0.6, [0.3, 0.0, 0.0], illuminated=False)
    assert np.linalg.norm(dark) == 0.0


def test_magnetic_hand_value():
    # m = (0.1, 0, 0) A m^2 crossed with B = (0, 3e-5, 0) T gives (0, 0, 3e-6) N m,
    # since 1 A m^2 * 1 T = 1 N m exactly.
    t = magnetic_torque([0.1, 0.0, 0.0], [0.0, 3e-5, 0.0])
    assert t[2] == pytest.approx(3e-06, rel=1e-14)
    assert t[0] == 0.0 and t[1] == 0.0
    assert np.linalg.norm(magnetic_torque([0.1, 0, 0], [3e-5, 0, 0])) == 0.0


def test_dipole_field_at_the_surface():
    # k / Re^3 = 7.96e15 / 6371200^3 = 3.0778634807e-05 T at the equator, northward;
    # twice that at the pole, directed into the Earth.
    k_over_re3 = EARTH_DIPOLE_MOMENT / R_EARTH_MEAN**3
    assert k_over_re3 == pytest.approx(3.0778634807e-05, rel=1e-9)
    b_eq = dipole_field_eci([R_EARTH_MEAN, 0.0, 0.0])
    assert b_eq[2] == pytest.approx(k_over_re3, rel=1e-14)
    assert np.linalg.norm(b_eq) == pytest.approx(k_over_re3, rel=1e-14)
    b_pole = dipole_field_eci([0.0, 0.0, R_EARTH_MEAN])
    assert b_pole[2] == pytest.approx(-2.0 * k_over_re3, rel=1e-14)


def test_dipole_vector_matches_closed_form_magnitude():
    dec = np.linspace(-np.pi / 2, np.pi / 2, 181)
    r = R_EARTH_MEAN * np.stack([np.cos(dec), np.zeros_like(dec), np.sin(dec)], axis=1)
    assert np.allclose(
        np.linalg.norm(dipole_field_eci(r), axis=1),
        dipole_field_magnitude(R_EARTH_MEAN, dec),
        rtol=1e-14,
    )


def test_orbit_period_and_speed_at_500_km():
    # R = 6378137 + 500000 = 6878137 m; v = sqrt(mu/R) = 7612.608173 m s^-1;
    # T = 2 pi R / v = 5676.9780 s.
    r = R_EARTH_EQUATORIAL + 500_000.0
    v = np.sqrt(MU_EARTH / r)
    assert v == pytest.approx(7612.608173, rel=1e-9)
    assert orbital_period(r) == pytest.approx(5676.9780285, rel=1e-9)
    assert orbital_period(r) == pytest.approx(2 * np.pi * r / v, rel=1e-14)


def test_density_table_spot_values():
    # Band base values are exact at the base altitude by construction.
    assert float(density(500_000.0)) == pytest.approx(6.967e-13, rel=1e-15)
    assert float(density(400_000.0)) == pytest.approx(3.725e-12, rel=1e-15)
    # Half a scale height above 500 km: 6.967e-13 * exp(-0.5) = 4.22562e-13
    assert float(density(500_000.0 + 0.5 * 63822.0)) == pytest.approx(
        6.967e-13 * np.exp(-0.5), rel=1e-12
    )


def test_julian_date_j2000():
    assert julian_date(2000, 1, 1, 12, 0, 0.0) == 2451545.0
    assert julian_date(2000, 1, 1, 0, 0, 0.0) == 2451544.5


def test_sun_model_annual_extremes():
    jds = np.array([julian_date(2026, 1, 1) + d for d in range(366)])
    dists = np.array([sun_distance_au(j) for j in jds])
    decs = np.degrees([np.arcsin(sun_unit_vector_eci(j)[2]) for j in jds])
    assert dists.min() == pytest.approx(0.9833, abs=5e-4)
    assert dists.max() == pytest.approx(1.0167, abs=5e-4)
    assert decs.max() == pytest.approx(23.44, abs=0.05)
    assert decs.min() == pytest.approx(-23.44, abs=0.05)


def test_eclipse_fraction_matches_closed_form():
    # f_ecl(beta) = (1/pi) arccos( sqrt(R^2 - Re^2) / (R cos beta) ).
    r = R_EARTH_EQUATORIAL + 500_000.0
    inc = np.radians(51.6)
    for beta_deg in (0.0, 30.0, 60.0):
        s = sun_direction_for_beta(inc, 0.0, np.radians(beta_deg))
        arg = np.sqrt(r**2 - R_EARTH_EQUATORIAL**2) / (r * np.cos(np.radians(beta_deg)))
        expected = float(np.arccos(np.clip(arg, -1.0, 1.0)) / np.pi)
        got = eclipse_fraction_cylindrical(r, inc, 0.0, s, n_samples=100_000)
        assert got == pytest.approx(expected, abs=5e-5)
    # Above the critical beta the orbit never enters the shadow cylinder.
    beta_crit = np.arccos(np.sqrt(1.0 - (R_EARTH_EQUATORIAL / r) ** 2))
    s_high = sun_direction_for_beta(inc, 0.0, beta_crit + np.radians(1.0))
    assert eclipse_fraction_cylindrical(r, inc, 0.0, s_high, n_samples=20_000) == 0.0
