"""Input validation: every public entry point must reject bad input with an actionable
message rather than returning a plausible wrong number."""

from __future__ import annotations

import numpy as np
import pytest

from disturbtorque import (
    Orbit,
    Spacecraft,
    aerodynamic_torque,
    compute_profile,
    density,
    dipole_field_eci,
    eclipse_fraction_cylindrical,
    gravity_gradient_max_magnitude,
    gravity_gradient_torque,
    julian_date,
    magnetic_torque,
    orbital_period,
    reference_orbit,
    reference_smallsat,
    solar_radiation_torque,
    sun_direction_for_beta,
)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"inertia": np.diag([-1.0, 2.0, 3.0])},          # not positive definite
        {"inertia": np.zeros((3, 3))},                    # degenerate
        {"inertia": np.diag([1.0, 2.0, 10.0])},           # violates the triangle inequality
        {"inertia": np.ones((3, 4))},                     # wrong shape
        {"inertia": np.diag([1.0, 2.0, 2.5]), "drag_area_m2": -1.0},
        {"inertia": np.diag([1.0, 2.0, 2.5]), "srp_reflectance": 1.5},
        {"inertia": np.diag([1.0, 2.0, 2.5]), "srp_reflectance": -0.1},
        {"inertia": np.diag([1.0, 2.0, 2.5]), "residual_dipole_am2": [1.0, 2.0]},
        {"inertia": np.diag([1.0, 2.0, 2.5]), "cp_aero_offset_m": [np.nan, 0.0, 0.0]},
        {"inertia": np.diag([1.0, 2.0, 2.5]), "mass_kg": 0.0},
    ],
)
def test_spacecraft_rejects_bad_properties(kwargs):
    with pytest.raises((ValueError, TypeError)):
        Spacecraft(**kwargs)


def test_spacecraft_rejects_asymmetric_inertia():
    bad = np.array([[10.0, 1.0, 0.0], [0.0, 12.0, 0.0], [0.0, 0.0, 14.0]])
    with pytest.raises(ValueError, match="symmetric"):
        Spacecraft(inertia=bad)


def test_spacecraft_accepts_principal_moments_as_a_length_3_sequence():
    sc = Spacecraft(inertia=[4.0, 8.0, 10.0])
    assert sc.inertia.shape == (3, 3)
    assert np.allclose(sc.principal_moments, [4.0, 8.0, 10.0])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"altitude_m": -1.0},
        {"altitude_m": 0.0},
        {"altitude_m": np.inf},
        {"altitude_m": 5e5, "inclination_rad": 4.0},
        {"altitude_m": 5e5, "mu": -1.0},
        {"altitude_m": 5e5, "raan_rad": np.nan},
    ],
)
def test_orbit_rejects_bad_parameters(kwargs):
    with pytest.raises((ValueError, TypeError)):
        Orbit(**kwargs)


def test_gravity_gradient_rejects_non_unit_nadir():
    with pytest.raises(ValueError, match="unit vector"):
        gravity_gradient_torque(np.diag([1.0, 2.0, 2.5]), [0.0, 0.0, 2.0], 7e6)
    with pytest.raises(ValueError, match="non-zero"):
        gravity_gradient_torque(np.diag([1.0, 2.0, 2.5]), [0.0, 0.0, 0.0], 7e6)


def test_gravity_gradient_rejects_bad_radius_and_mu():
    inertia = np.diag([1.0, 2.0, 2.5])
    with pytest.raises(ValueError, match="radius_m"):
        gravity_gradient_torque(inertia, [0, 0, 1.0], -7e6)
    with pytest.raises(ValueError, match="mu"):
        gravity_gradient_torque(inertia, [0, 0, 1.0], 7e6, mu=0.0)
    with pytest.raises(ValueError):
        gravity_gradient_max_magnitude(-1.0, 2.0, 7e6)


def test_aerodynamic_rejects_bad_inputs():
    with pytest.raises(ValueError, match="density_kg_m3"):
        aerodynamic_torque(-1e-12, [7500.0, 0, 0], 2.2, 1.0, [0, 0, 0.1])
    with pytest.raises(ValueError, match="drag_coefficient"):
        aerodynamic_torque(1e-12, [7500.0, 0, 0], -2.2, 1.0, [0, 0, 0.1])
    with pytest.raises(ValueError, match="trailing dimension 3"):
        aerodynamic_torque(1e-12, [7500.0, 0.0], 2.2, 1.0, [0, 0, 0.1])
    with pytest.raises(ValueError, match="finite"):
        aerodynamic_torque(1e-12, [np.inf, 0, 0], 2.2, 1.0, [0, 0, 0.1])


def test_solar_rejects_bad_inputs():
    with pytest.raises(ValueError, match="reflectance"):
        solar_radiation_torque([0, 0, 1.0], 1.0, 1.4, [0.1, 0, 0])
    with pytest.raises(ValueError, match="distance_au"):
        solar_radiation_torque([0, 0, 1.0], 1.0, 0.5, [0.1, 0, 0], distance_au=0.0)
    with pytest.raises(ValueError, match="unit vector"):
        solar_radiation_torque([0, 0, 3.0], 1.0, 0.5, [0.1, 0, 0])


def test_magnetic_rejects_bad_inputs():
    with pytest.raises(ValueError, match="shape"):
        magnetic_torque([0.1, 0.0], [0, 3e-5, 0])
    with pytest.raises(ValueError, match="finite"):
        magnetic_torque([0.1, 0, 0], [np.nan, 0, 0])
    with pytest.raises(ValueError, match="dipole_moment"):
        dipole_field_eci([7e6, 0, 0], dipole_moment=-1.0)
    with pytest.raises(ValueError, match="non-zero"):
        dipole_field_eci([0.0, 0.0, 0.0])


def test_density_range_and_extrapolation():
    with pytest.raises(ValueError, match=">= 0"):
        density(-1.0)
    with pytest.raises(ValueError, match="allow_extrapolation"):
        density(1_000_001.0)
    assert float(density(1_200_000.0, allow_extrapolation=True)) > 0.0
    with pytest.raises(ValueError, match="finite"):
        density(np.nan)


def test_period_and_julian_date_ranges():
    with pytest.raises(ValueError, match="radius_m"):
        orbital_period(0.0)
    with pytest.raises(ValueError, match=r"\[1900, 2100\]"):
        julian_date(1899, 1, 1)
    with pytest.raises(ValueError, match=r"\[1, 12\]"):
        julian_date(2026, 13, 1)


def test_compute_profile_rejects_bad_arguments():
    sc, orb = reference_smallsat(), reference_orbit()
    sun = sun_direction_for_beta(orb.inclination_rad, orb.raan_rad, 0.0)
    with pytest.raises(TypeError, match="Spacecraft"):
        compute_profile(orb, orb, sun)
    with pytest.raises(TypeError, match="Orbit"):
        compute_profile(sc, sc, sun)
    with pytest.raises(ValueError, match="n_samples"):
        compute_profile(sc, orb, sun, n_samples=4)
    with pytest.raises(ValueError, match="unit vector"):
        compute_profile(sc, orb, [0.0, 0.0, 2.0])
    with pytest.raises(ValueError, match="distance_au"):
        compute_profile(sc, orb, sun, distance_au=-1.0)


def test_profile_rejects_unknown_source_and_frame():
    sc, orb = reference_smallsat(), reference_orbit()
    sun = sun_direction_for_beta(orb.inclination_rad, orb.raan_rad, 0.0)
    prof = compute_profile(sc, orb, sun, n_samples=17)
    with pytest.raises(ValueError, match="source must be"):
        prof.torque("thermal")
    with pytest.raises(ValueError, match="frame must be"):
        prof.torque("total", "lvlh")


def test_eclipse_fraction_rejects_tiny_sample_counts():
    with pytest.raises(ValueError, match="n_samples"):
        eclipse_fraction_cylindrical(7e6, 0.0, 0.0, [1.0, 0.0, 0.0], n_samples=3)


def test_beta_out_of_range_rejected():
    with pytest.raises(ValueError, match="beta_rad"):
        sun_direction_for_beta(0.5, 0.0, 2.0)
