"""Unit tests for navbench.truth — attitude, orbit and airborne truth generators."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from navbench import (
    MU_EARTH,
    R_EARTH,
    AttitudeTruth,
    Trajectory,
    airborne_trajectory,
    attitude_trajectory,
    circular_orbit_state,
    dcm_from_quat,
    orbit_trajectory,
    quat_conjugate,
    quat_from_axis_angle,
    quat_from_euler_zyx,
    quat_identity,
    quat_multiply,
    small_angle_from_quat,
)


class TestAttitudeTrajectory:
    def test_shapes(self, inertia):
        tr = attitude_trajectory(
            inertia=inertia, quat0=quat_identity(), omega0=np.zeros(3), dt=0.1, n_steps=50
        )
        assert tr.t.shape == (51,)
        assert tr.quat.shape == (51, 4)
        assert tr.omega.shape == (51, 3)
        assert tr.torque.shape == (51, 3)
        assert tr.n_steps == 50

    def test_time_grid(self, inertia):
        tr = attitude_trajectory(
            inertia=inertia, quat0=quat_identity(), omega0=np.zeros(3), dt=0.25, n_steps=8
        )
        assert np.allclose(tr.t, np.arange(9) * 0.25)

    def test_zero_rate_zero_torque_is_static(self, inertia):
        tr = attitude_trajectory(
            inertia=inertia, quat0=quat_identity(), omega0=np.zeros(3), dt=1.0, n_steps=100
        )
        assert np.allclose(tr.quat, quat_identity())
        assert np.allclose(tr.omega, 0.0)

    def test_principal_axis_spin_matches_analytic(self, inertia):
        tr = attitude_trajectory(
            inertia=inertia, quat0=quat_identity(), omega0=np.array([0.0, 0.0, 0.05]),
            dt=0.5, n_steps=200,
        )
        exact = quat_from_axis_angle([0, 0, 1], 0.05 * 100.0)
        assert abs(abs(float(tr.quat[-1] @ exact)) - 1.0) < 1e-14

    def test_quaternions_stay_unit_norm(self, inertia):
        tr = attitude_trajectory(
            inertia=inertia, quat0=quat_from_euler_zyx(0.3, 0.2, -0.4),
            omega0=np.array([0.5, -0.8, 0.3]), dt=0.05, n_steps=400,
        )
        assert np.max(np.abs(np.linalg.norm(tr.quat, axis=1) - 1.0)) < 1e-14

    def test_torque_free_conserves_energy(self, inertia):
        tr = attitude_trajectory(
            inertia=inertia, quat0=quat_from_euler_zyx(0.3, 0.2, -0.4),
            omega0=np.array([0.05, -0.08, 0.03]), dt=0.1, n_steps=1000,
        )
        e = tr.kinetic_energy()
        assert np.max(np.abs(e - e[0])) / e[0] < 1e-12

    def test_torque_free_conserves_inertial_angular_momentum(self, inertia):
        tr = attitude_trajectory(
            inertia=inertia, quat0=quat_from_euler_zyx(0.3, 0.2, -0.4),
            omega0=np.array([0.05, -0.08, 0.03]), dt=0.1, n_steps=1000,
        )
        h = tr.angular_momentum()
        assert np.max(np.linalg.norm(h - h[0], axis=1)) / np.linalg.norm(h[0]) < 1e-9

    def test_constant_torque_about_principal_axis_gives_linear_rate(self):
        j = np.diag([4.0, 4.0, 4.0])
        tr = attitude_trajectory(
            inertia=j, quat0=quat_identity(), omega0=np.zeros(3), dt=0.1, n_steps=100,
            torque_fn=lambda t, q, w: np.array([0.0, 0.0, 0.8]),
        )
        assert np.allclose(tr.omega[:, 2], 0.2 * tr.t, atol=1e-13)

    def test_torque_recorded(self, inertia):
        tr = attitude_trajectory(
            inertia=inertia, quat0=quat_identity(), omega0=np.zeros(3), dt=1.0, n_steps=5,
            torque_fn=lambda t, q, w: np.array([t, 0.0, 0.0]),
        )
        assert np.allclose(tr.torque[:, 0], tr.t)

    def test_interval_rate_reproduces_attitude_change(self, short_attitude_truth):
        tr = short_attitude_truth
        w = tr.interval_rate()
        assert w.shape == (tr.n_steps, 3)
        dt = float(tr.t[1] - tr.t[0])
        for k in (0, 50, 199):
            dq = quat_multiply(quat_conjugate(tr.quat[k]), tr.quat[k + 1])
            assert np.allclose(small_angle_from_quat(dq) / dt, w[k], atol=1e-15)

    def test_interval_rate_close_to_but_not_equal_endpoint_rate(self, short_attitude_truth):
        tr = short_attitude_truth
        w = tr.interval_rate()
        d = np.max(np.abs(tr.omega[1:] - w))
        assert 0.0 < d < 1e-3

    @pytest.mark.parametrize("dt", [0.0, -1.0, np.nan, np.inf])
    def test_bad_dt_raises(self, inertia, dt):
        with pytest.raises(ValueError, match="dt"):
            attitude_trajectory(
                inertia=inertia, quat0=quat_identity(), omega0=np.zeros(3), dt=dt, n_steps=5
            )

    @pytest.mark.parametrize("n", [0, -3])
    def test_bad_n_steps_raises(self, inertia, n):
        with pytest.raises(ValueError, match="n_steps"):
            attitude_trajectory(
                inertia=inertia, quat0=quat_identity(), omega0=np.zeros(3), dt=1.0, n_steps=n
            )

    def test_nonfinite_omega0_raises(self, inertia):
        with pytest.raises(ValueError, match="finite"):
            attitude_trajectory(
                inertia=inertia, quat0=quat_identity(),
                omega0=np.array([np.nan, 0.0, 0.0]), dt=1.0, n_steps=5,
            )


class TestOrbitTrajectory:
    def test_circular_orbit_state_speed(self):
        r, v = circular_orbit_state(500e3)
        assert np.linalg.norm(r) == pytest.approx(R_EARTH + 500e3)
        assert np.linalg.norm(v) == pytest.approx(np.sqrt(MU_EARTH / (R_EARTH + 500e3)))

    def test_circular_orbit_state_inclination(self):
        _, v = circular_orbit_state(500e3, np.pi / 6)
        assert v[2] / np.linalg.norm(v) == pytest.approx(np.sin(np.pi / 6))

    @pytest.mark.parametrize("alt", [0.0, -100.0, np.nan])
    def test_bad_altitude_raises(self, alt):
        with pytest.raises(ValueError, match="altitude"):
            circular_orbit_state(alt)

    def test_bad_mu_raises(self):
        with pytest.raises(ValueError, match="mu"):
            circular_orbit_state(500e3, 0.0, mu=-1.0)

    def test_energy_conserved_over_one_revolution(self):
        r0, v0 = circular_orbit_state(500e3)
        a = float(np.linalg.norm(r0))
        period = 2 * np.pi * np.sqrt(a**3 / MU_EARTH)
        tr = orbit_trajectory(position0=r0, velocity0=v0, dt=period / 500, n_steps=500)
        e = tr.specific_energy()
        assert np.max(np.abs(e - e[0])) / abs(e[0]) < 1e-10

    def test_specific_energy_matches_vis_viva(self):
        r0, v0 = circular_orbit_state(800e3)
        a = float(np.linalg.norm(r0))
        tr = orbit_trajectory(position0=r0, velocity0=v0, dt=10.0, n_steps=5)
        assert tr.specific_energy()[0] == pytest.approx(-MU_EARTH / (2 * a), rel=1e-14)

    def test_angular_momentum_conserved(self):
        r0, v0 = circular_orbit_state(500e3, 0.5)
        tr = orbit_trajectory(position0=r0, velocity0=v0, dt=10.0, n_steps=500)
        h = tr.angular_momentum()
        assert np.max(np.linalg.norm(h - h[0], axis=1)) / np.linalg.norm(h[0]) < 1e-10

    def test_closes_after_one_period(self):
        r0, v0 = circular_orbit_state(500e3)
        a = float(np.linalg.norm(r0))
        period = 2 * np.pi * np.sqrt(a**3 / MU_EARTH)
        tr = orbit_trajectory(position0=r0, velocity0=v0, dt=period / 1000, n_steps=1000)
        assert np.linalg.norm(tr.position[-1] - r0) / a < 1e-7

    def test_radius_constant_for_circular_orbit(self):
        r0, v0 = circular_orbit_state(500e3)
        tr = orbit_trajectory(position0=r0, velocity0=v0, dt=5.0, n_steps=400)
        radii = np.linalg.norm(tr.position, axis=1)
        assert np.ptp(radii) / radii[0] < 1e-10

    def test_singular_position_raises(self):
        with pytest.raises(ValueError, match="singularity"):
            orbit_trajectory(position0=np.zeros(3), velocity0=np.ones(3), dt=1.0, n_steps=2)

    def test_nonfinite_state_raises(self):
        with pytest.raises(ValueError, match="finite"):
            orbit_trajectory(
                position0=np.array([7e6, np.nan, 0.0]), velocity0=np.zeros(3),
                dt=1.0, n_steps=2,
            )

    def test_propagator_bad_mu_raises(self):
        r0, v0 = circular_orbit_state(500e3)
        with pytest.raises(ValueError, match="mu"):
            orbit_trajectory(position0=r0, velocity0=v0, dt=1.0, n_steps=2, mu=0.0)


class TestAirborneTrajectory:
    def test_straight_line(self):
        tr = airborne_trajectory(
            position0=[0, 0, 1000], velocity0=[100, 0, 0], dt=1.0, n_steps=10
        )
        assert np.allclose(tr.position[:, 0], np.arange(11) * 100.0)
        assert np.allclose(tr.velocity, [100, 0, 0])
        assert np.allclose(tr.acceleration, 0.0)

    def test_climb(self):
        tr = airborne_trajectory(
            position0=[0, 0, 0], velocity0=[0, 0, 0], dt=1.0, n_steps=4, climb_rate_m_s2=2.0
        )
        assert np.allclose(tr.position[:, 2], 0.5 * 2.0 * tr.t**2)
        assert np.allclose(tr.velocity[:, 2], 2.0 * tr.t)

    def test_coordinated_turn_radius(self):
        speed, omega = 200.0, 0.02
        n = int(round(2 * np.pi / omega))
        tr = airborne_trajectory(
            position0=[0, 0, 0], velocity0=[speed, 0, 0], dt=1.0, n_steps=n,
            turn_rate_rad_s=omega,
        )
        centre = np.array([0.0, speed / omega])
        radii = np.linalg.norm(tr.position[:, :2] - centre, axis=1)
        assert np.max(np.abs(radii - speed / omega)) / (speed / omega) < 1e-12

    def test_coordinated_turn_preserves_speed(self):
        tr = airborne_trajectory(
            position0=[0, 0, 0], velocity0=[150, 0, 0], dt=0.5, n_steps=200,
            turn_rate_rad_s=0.05,
        )
        assert np.allclose(np.linalg.norm(tr.velocity[:, :2], axis=1), 150.0)

    def test_acceleration_is_centripetal(self):
        tr = airborne_trajectory(
            position0=[0, 0, 0], velocity0=[100, 0, 0], dt=1.0, n_steps=5,
            turn_rate_rad_s=0.1,
        )
        for k in range(6):
            assert abs(float(tr.acceleration[k, :2] @ tr.velocity[k, :2])) < 1e-10

    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_bad_turn_rate_raises(self, bad):
        with pytest.raises(ValueError, match="turn_rate"):
            airborne_trajectory(
                position0=[0, 0, 0], velocity0=[1, 0, 0], dt=1.0, n_steps=2,
                turn_rate_rad_s=bad,
            )

    def test_bad_climb_raises(self):
        with pytest.raises(ValueError, match="climb"):
            airborne_trajectory(
                position0=[0, 0, 0], velocity0=[1, 0, 0], dt=1.0, n_steps=2,
                climb_rate_m_s2=np.nan,
            )

    def test_nonfinite_state_raises(self):
        with pytest.raises(ValueError, match="finite"):
            airborne_trajectory(
                position0=[0, np.nan, 0], velocity0=[1, 0, 0], dt=1.0, n_steps=2
            )


class TestTrajectory:
    def test_valid_combination(self, short_attitude_truth):
        n = short_attitude_truth.t.size
        traj = Trajectory(
            attitude=short_attitude_truth,
            position=np.zeros((n, 3)),
            velocity=np.zeros((n, 3)),
            acceleration=np.zeros((n, 3)),
        )
        assert traj.t.size == n
        assert traj.dt == pytest.approx(0.5)

    def test_mismatched_grid_raises(self, short_attitude_truth):
        with pytest.raises(ValueError, match="shape"):
            Trajectory(
                attitude=short_attitude_truth,
                position=np.zeros((5, 3)),
                velocity=np.zeros((5, 3)),
                acceleration=np.zeros((5, 3)),
            )

    def test_attitude_truth_is_frozen(self, short_attitude_truth):
        assert isinstance(short_attitude_truth, AttitudeTruth)
        with pytest.raises(dataclasses.FrozenInstanceError):
            short_attitude_truth.t = np.zeros(3)  # type: ignore[misc]

    def test_angular_momentum_uses_inertial_frame(self, inertia):
        tr = attitude_trajectory(
            inertia=inertia, quat0=quat_from_axis_angle([0, 0, 1], 0.7),
            omega0=np.array([0.0, 0.0, 0.1]), dt=1.0, n_steps=3,
        )
        h_body = inertia @ np.array([0.0, 0.0, 0.1])
        expected = dcm_from_quat(tr.quat[0]) @ h_body
        assert np.allclose(tr.angular_momentum()[0], expected)
