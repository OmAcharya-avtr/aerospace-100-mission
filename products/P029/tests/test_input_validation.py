"""Invalid input must fail at the call site with an actionable message."""

from __future__ import annotations

import numpy as np
import pytest

from momentummgr import (
    CircularOrbit,
    SpacecraftProperties,
    WheelArray,
    aerodynamic_torque,
    averaged_controllability,
    density,
    dipole_cost,
    dipole_field_eci,
    episode_cost,
    eclipse_boundaries,
    gravity_gradient_torque,
    lvlh_dcm,
    magnetic_dump_command,
    momentum_per_orbit_eci,
    pyramid_four,
    reference_orbit,
    reference_smallsat,
    residual_dipole_torque,
    srp_torque,
    sun_direction_for_beta,
    sweep_orbit,
    thruster_dump,
    tune_fixed_threshold,
    count_zero_crossings,
)
from momentummgr.policies import FixedThresholdScheduler


@pytest.mark.parametrize(
    ("inertia", "match"),
    [
        (np.array([[1.0, 0.5], [0.5, 1.0]]), "shape"),
        (np.diag([-1.0, 2.0, 3.0]), "positive definite"),
        (np.array([[1.0, 0.9, 0.0], [0.1, 1.0, 0.0], [0.0, 0.0, 1.0]]), "symmetric"),
        (np.diag([1.0, 1.0, 5.0]), "triangle inequality"),
        (np.diag([np.nan, 1.0, 1.0]), "finite"),
    ],
)
def test_inertia_rejects_bad_tensors(inertia: np.ndarray, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        gravity_gradient_torque(inertia, [0.0, 0.0, 1.0], 7.0e6)


def test_nadir_must_be_a_unit_vector() -> None:
    with pytest.raises(ValueError, match="unit vector"):
        gravity_gradient_torque(np.eye(3), [0.0, 0.0, 2.0], 7.0e6)


def test_radius_must_be_positive() -> None:
    with pytest.raises(ValueError, match="radius_m must be > 0"):
        gravity_gradient_torque(np.eye(3), [0.0, 0.0, 1.0], -1.0)


def test_negative_density_and_area_rejected() -> None:
    with pytest.raises(ValueError, match="density_kg_m3 must be >= 0"):
        aerodynamic_torque(-1.0, [7600.0, 0.0, 0.0], 2.2, 0.6, [0.0, 0.0, 0.05])
    with pytest.raises(ValueError, match="area_m2 must be >= 0"):
        aerodynamic_torque(1e-12, [7600.0, 0.0, 0.0], 2.2, -0.6, [0.0, 0.0, 0.05])


def test_reflectance_outside_zero_one_rejected() -> None:
    with pytest.raises(ValueError, match=r"reflectance must lie in \[0.0, 1.0\]"):
        srp_torque([0.0, 0.0, 1.0], 1.2, 1.4, [0.02, 0.0, 0.0])


def test_density_range() -> None:
    with pytest.raises(ValueError, match="must be >= 0 m"):
        density(-1.0)
    with pytest.raises(ValueError, match="allow_extrapolation"):
        density(1_200_000.0)
    assert float(density(1_200_000.0, allow_extrapolation=True)) > 0.0


def test_dipole_field_rejects_bad_positions_and_rotations() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        dipole_field_eci([0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="trailing dimension 3"):
        dipole_field_eci(np.zeros((4, 2)))
    with pytest.raises(ValueError, match="scalar or have length"):
        dipole_field_eci(np.ones((3, 3)) * 7e6, rotation_angle_rad=[0.0, 1.0])


def test_residual_dipole_field_shape() -> None:
    with pytest.raises(ValueError, match="trailing dimension 3"):
        residual_dipole_torque([0.1, 0.0, 0.0], np.zeros((2, 2)))


def test_orbit_rejects_non_positive_altitude() -> None:
    with pytest.raises(ValueError, match="altitude_m must be > 0"):
        CircularOrbit(altitude_m=0.0)
    with pytest.raises(ValueError, match="inclination_rad must lie"):
        CircularOrbit(altitude_m=5e5, inclination_rad=4.0)


def test_spacecraft_rejects_bad_offsets() -> None:
    with pytest.raises(ValueError, match=r"cp_aero_offset_m must have shape \(3,\)"):
        SpacecraftProperties(inertia=np.eye(3), cp_aero_offset_m=np.zeros(2))
    with pytest.raises(ValueError, match="mass_kg must be > 0"):
        SpacecraftProperties(inertia=np.eye(3), mass_kg=-5.0)


def test_lvlh_rejects_degenerate_states() -> None:
    with pytest.raises(ValueError, match="must be non-zero"):
        lvlh_dcm([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="parallel"):
        lvlh_dcm([7.0e6, 0.0, 0.0], [1.0, 0.0, 0.0])


def test_eclipse_boundaries_reject_a_body_bigger_than_the_orbit() -> None:
    sun = np.array([1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="must be smaller than radius_m"):
        eclipse_boundaries(6.0e6, 0.0, 0.0, sun, body_radius_m=7.0e6)


def test_sweep_rejects_bad_arguments() -> None:
    sc = reference_smallsat()
    orbit = reference_orbit(500.0)
    sun = sun_direction_for_beta(orbit.inclination_rad, orbit.raan_rad, 0.3)
    with pytest.raises(TypeError, match="must be a SpacecraftProperties"):
        sweep_orbit(object(), orbit, sun)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must be a CircularOrbit"):
        sweep_orbit(sc, object(), sun)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_samples must be >= 9"):
        sweep_orbit(sc, orbit, sun, n_samples=4)
    with pytest.raises(ValueError, match="unit vector"):
        sweep_orbit(sc, orbit, [1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="source must be"):
        momentum_per_orbit_eci(sc, orbit, sun, "gremlin")


def test_sweep_torque_frame_and_source_validation() -> None:
    sc = reference_smallsat()
    orbit = reference_orbit(500.0)
    sun = sun_direction_for_beta(orbit.inclination_rad, orbit.raan_rad, 0.3)
    sweep = sweep_orbit(sc, orbit, sun, n_samples=17)
    with pytest.raises(ValueError, match="frame must be"):
        sweep.torque("total", "lvlh")
    with pytest.raises(ValueError, match="source must be"):
        sweep.torque("gremlin", "body")


def test_wheel_array_validation() -> None:
    with pytest.raises(ValueError, match=r"axes must have shape \(n, 3\)"):
        WheelArray(axes=np.zeros((4, 2)), wheel_inertia_kg_m2=1e-3, max_momentum_nms=0.05)
    with pytest.raises(ValueError, match="at least 3 wheels"):
        WheelArray(axes=np.eye(3)[:2], wheel_inertia_kg_m2=1e-3, max_momentum_nms=0.05)
    with pytest.raises(ValueError, match="unit vector"):
        WheelArray(axes=np.eye(3) * 2.0, wheel_inertia_kg_m2=1e-3, max_momentum_nms=0.05)
    coplanar = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0] / np.sqrt(2.0)])
    with pytest.raises(ValueError, match="rank 3"):
        WheelArray(axes=coplanar, wheel_inertia_kg_m2=1e-3, max_momentum_nms=0.05)
    with pytest.raises(ValueError, match="wheel_inertia_kg_m2 must be finite and > 0"):
        WheelArray(axes=np.eye(3), wheel_inertia_kg_m2=-1e-3, max_momentum_nms=0.05)
    with pytest.raises(ValueError, match="max_momentum_nms must be > 0"):
        WheelArray(axes=np.eye(3), wheel_inertia_kg_m2=1e-3, max_momentum_nms=0.0)


def test_wheel_momentum_shape_validation() -> None:
    w = pyramid_four()
    with pytest.raises(ValueError, match="trailing dimension 4"):
        w.body_momentum([1.0, 2.0])
    with pytest.raises(ValueError, match="trailing dimension 4"):
        w.speeds_rad_s([1.0, 2.0])
    with pytest.raises(ValueError, match="envelope_fraction must lie"):
        w.allocate([0.0, 0.0, 0.01], envelope_fraction=1.5)
    with pytest.raises(ValueError, match=r"must have shape \(N, n\)"):
        count_zero_crossings(np.zeros(5))


def test_magnetic_command_validation() -> None:
    with pytest.raises(ValueError, match="must be non-zero"):
        magnetic_dump_command([0.01, 0.0, 0.0], [0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="max_dipole_am2 must be > 0"):
        magnetic_dump_command([0.01, 0.0, 0.0], [0.0, 0.0, 3e-5], max_dipole_am2=0.0)


def test_controllability_validation() -> None:
    with pytest.raises(ValueError, match=r"must have shape \(N, 3\)"):
        averaged_controllability(np.zeros(3))
    with pytest.raises(ValueError, match="at least two field samples"):
        averaged_controllability(np.ones((1, 3)))
    with pytest.raises(ValueError, match="non-zero at every sample"):
        averaged_controllability(np.zeros((5, 3)))
    with pytest.raises(ValueError, match="must be increasing"):
        averaged_controllability(np.ones((5, 3)), np.zeros(5))


def test_dipole_cost_validation() -> None:
    with pytest.raises(ValueError, match=r"must have shape \(N, 3\)"):
        dipole_cost(np.zeros(4), np.zeros(4))
    with pytest.raises(ValueError, match=r"time_s must have shape \(5,\)"):
        dipole_cost(np.zeros((5, 3)), np.zeros(4))


def test_thruster_validation() -> None:
    with pytest.raises(ValueError, match="moment_arm_m must be > 0"):
        thruster_dump(0.05, 0.0, 220.0)
    with pytest.raises(ValueError, match="specific_impulse_s must be > 0"):
        thruster_dump(0.05, 0.5, -1.0)
    with pytest.raises(ValueError, match="efficiency must lie"):
        thruster_dump(0.05, 0.5, 220.0, efficiency=2.0)


def test_scheduler_validation() -> None:
    with pytest.raises(ValueError, match="must not exceed on_fraction"):
        FixedThresholdScheduler(on_fraction=0.3, off_fraction=0.6)
    with pytest.raises(ValueError, match="episodes must be non-empty"):
        tune_fixed_threshold([])
    with pytest.raises(TypeError, match="decider"):
        from momentummgr import evaluate_policy

        evaluate_policy(object(), [])


def test_episode_cost_is_monotone_in_each_term() -> None:
    base = episode_cost(0.1, 0.1, 0.5)
    assert episode_cost(0.2, 0.1, 0.5) > base
    assert episode_cost(0.1, 0.2, 0.5) > base
    assert episode_cost(0.1, 0.1, 1.5) > base
