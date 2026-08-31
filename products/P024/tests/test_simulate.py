"""Simulator configuration, integration and bookkeeping."""

from __future__ import annotations

import numpy as np
import pytest

from detumblesim.control import BDotController
from detumblesim.magfield import dipole_field_eci
from detumblesim.orbit import CircularOrbit
from detumblesim.simulate import (
    DetumbleConfig,
    crossing_time,
    field_history_eci,
    simulate_detumble,
)
from detumblesim.spacecraft import Magnetorquer, inertia_from_diagonal

J = inertia_from_diagonal(0.05, 0.06, 0.04)


def base_config(**kw) -> DetumbleConfig:
    args = {
        "inertia": J,
        "orbit": CircularOrbit(altitude_km=500.0, inclination_deg=97.4),
        "magnetorquer": Magnetorquer.isotropic(0.2),
        "omega0_rad_s": np.radians([8.0, -6.0, 5.0]),
        "duration_s": 2000.0,
        "control_dt_s": 2.0,
        "substeps": 2,
        "target_rate_rad_s": np.radians(1.0),
    }
    args.update(kw)
    return DetumbleConfig(**args)


class TestConfigValidation:
    @pytest.mark.parametrize(
        "kw,msg",
        [
            ({"duration_s": 0.0}, "duration_s"),
            ({"control_dt_s": -1.0}, "control_dt_s"),
            ({"control_dt_s": 1e6}, "must not exceed"),
            ({"substeps": 0}, "substeps"),
            ({"target_rate_rad_s": -1.0}, "target_rate_rad_s"),
            ({"mag_noise_t": -1e-9}, "mag_noise_t"),
            ({"omega0_rad_s": np.array([1.0, 2.0])}, "omega0_rad_s"),
            ({"omega0_rad_s": np.array([1.0, np.nan, 0.0])}, "non-finite"),
        ],
    )
    def test_rejects_bad_inputs(self, kw, msg):
        with pytest.raises(ValueError, match=msg):
            base_config(**kw)

    def test_normalises_the_initial_quaternion(self):
        cfg = base_config(q0=np.array([2.0, 0.0, 0.0, 0.0]))
        assert np.allclose(cfg.q0, [1.0, 0.0, 0.0, 0.0])

    def test_rejects_bad_inertia(self):
        with pytest.raises(ValueError, match="positive definite"):
            base_config(inertia=np.diag([1.0, 1.0, -1.0]))


class TestFieldHistory:
    def test_matches_the_scalar_field_function(self):
        o = CircularOrbit(altitude_km=600.0, inclination_deg=51.6, gmst0_rad=1.2)
        t = np.array([0.0, 137.0, 3000.0, 9000.0])
        many = field_history_eci(o, t)
        one = np.array([dipole_field_eci(o.position_eci(ti), ti, o.gmst0_rad) for ti in t])
        assert np.allclose(many, one, rtol=1e-12, atol=1e-20)

    def test_magnitudes_are_physical_in_leo(self):
        o = CircularOrbit(altitude_km=500.0, inclination_deg=97.4)
        b = np.linalg.norm(field_history_eci(o, np.linspace(0, o.period_s, 200)), axis=1)
        assert np.all(b > 1e-5) and np.all(b < 6e-5)


class TestCrossingTime:
    def test_known_answer_interpolation(self):
        # Between t = 0 (value 2) and t = 1 (value 0), the value 1 is crossed
        # at t = 0.5 exactly.
        assert np.isclose(crossing_time(np.array([0.0, 1.0]), np.array([2.0, 0.0]), 1.0), 0.5)

    def test_returns_nan_when_never_crossed(self):
        assert np.isnan(crossing_time(np.array([0.0, 1.0]), np.array([2.0, 3.0]), 1.0))

    def test_handles_first_sample_already_below(self):
        assert crossing_time(np.array([5.0, 6.0]), np.array([0.5, 0.4]), 1.0) == 5.0

    def test_handles_flat_segment(self):
        assert crossing_time(np.array([0.0, 1.0]), np.array([1.0, 1.0]), 1.0) == 0.0


class TestSimulation:
    def test_shapes_and_bookkeeping(self):
        cfg = base_config(duration_s=200.0)
        r = simulate_detumble(cfg, BDotController(gain=1e5))
        n = r.t_s.size
        assert n == int(200.0 / 2.0) + 1
        assert r.omega_rad_s.shape == (n, 3)
        assert r.quat.shape == (n, 4)
        assert r.b_body_t.shape == (n, 3)
        assert r.dipole_am2.shape == (n, 3)
        assert r.torque_nm.shape == (n, 3)
        assert r.rate_norm_rad_s.shape == (n,)
        assert np.allclose(r.t_s, np.arange(n) * 2.0)

    def test_first_step_commands_no_dipole(self):
        r = simulate_detumble(base_config(duration_s=100.0), BDotController(gain=1e6))
        assert np.allclose(r.dipole_am2[0], 0.0)

    def test_quaternion_stays_normalised(self):
        r = simulate_detumble(base_config(duration_s=4000.0), BDotController(gain=1e5))
        assert np.allclose(np.linalg.norm(r.quat, axis=1), 1.0, atol=1e-12)
        assert r.max_quat_norm_error < 1e-5

    def test_torque_is_perpendicular_to_the_field_every_step(self):
        r = simulate_detumble(base_config(duration_s=3000.0), BDotController(gain=3e5))
        dots = np.sum(r.torque_nm * r.b_body_t, axis=1)
        scale = np.linalg.norm(r.torque_nm, axis=1) * np.linalg.norm(r.b_body_t, axis=1)
        assert np.all(np.abs(dots) <= 1e-10 * (scale + 1e-30))

    def test_dipole_never_exceeds_the_limit(self):
        r = simulate_detumble(base_config(duration_s=3000.0), BDotController(gain=1e7))
        assert np.all(np.abs(r.dipole_am2) <= 0.2 + 1e-15)
        assert r.saturated_fraction > 0.0

    def test_zero_gain_is_rejected_by_the_controller(self):
        with pytest.raises(ValueError, match="gain"):
            BDotController(gain=0.0)

    def test_no_control_means_constant_momentum(self):
        # With a vanishing dipole limit the torque is ~0, so the inertial
        # angular momentum magnitude is conserved by the rigid-body dynamics.
        cfg = base_config(
            duration_s=3000.0, magnetorquer=Magnetorquer.isotropic(1e-12)
        )
        r = simulate_detumble(cfg, BDotController(gain=1e5))
        h = r.h_norm_nms
        assert np.allclose(h, h[0], rtol=1e-9)

    def test_energy_falls_under_bdot(self):
        r = simulate_detumble(base_config(duration_s=6000.0), BDotController(gain=1e5))
        assert r.energy_j[-1] < 0.5 * r.energy_j[0]

    def test_stop_when_detumbled_truncates(self):
        long_cfg = base_config(duration_s=23000.0, stop_when_detumbled=True)
        short = simulate_detumble(long_cfg, BDotController(gain=2e5))
        assert short.detumbled
        assert short.t_s[-1] < 23000.0
        full = simulate_detumble(
            base_config(duration_s=23000.0, stop_when_detumbled=False),
            BDotController(gain=2e5),
        )
        assert np.isclose(short.detumble_time_s, full.detumble_time_s, rtol=1e-9)

    def test_seeded_noise_is_reproducible(self):
        cfg = {"duration_s": 2000.0, "mag_noise_t": 2e-7, "seed": 7}
        a = simulate_detumble(base_config(**cfg), BDotController(gain=1e5))
        b = simulate_detumble(base_config(**cfg), BDotController(gain=1e5))
        assert np.array_equal(a.omega_rad_s, b.omega_rad_s)
        c = simulate_detumble(
            base_config(**{**cfg, "seed": 8}), BDotController(gain=1e5)
        )
        assert not np.array_equal(a.omega_rad_s, c.omega_rad_s)

    def test_substep_convergence(self):
        # Halving the internal step must change the detumble time by far less
        # than any conclusion in this package depends on.
        cfg = {"duration_s": 23000.0, "stop_when_detumbled": True}
        t1 = simulate_detumble(base_config(substeps=1, **cfg), BDotController(gain=1e5))
        t4 = simulate_detumble(base_config(substeps=4, **cfg), BDotController(gain=1e5))
        assert abs(t1.detumble_time_s - t4.detumble_time_s) / t4.detumble_time_s < 1e-3
