"""Closed-loop simulation: consistency, determinism and the two filter paths."""

from __future__ import annotations

import numpy as np
import pytest

from fdiscope.faults import FaultSpec, FaultType
from fdiscope.plant import ControllerGains, PlantConfig, loop_matrices
from fdiscope.residuals import nis_consistency, whiteness
from fdiscope.simulate import LoopConfig, build_filter, simulate_loop


class TestLoopConfig:
    @pytest.mark.parametrize("kwargs", [{"n_steps": 1}, {"ref_period_s": 0.0}, {"x0": (0.0,)}])
    def test_rejects_bad_parameters(self, kwargs):
        with pytest.raises(ValueError):
            LoopConfig(**kwargs)

    def test_rejects_non_finite_amplitude(self):
        with pytest.raises(ValueError, match="ref_amplitude_rad"):
            LoopConfig(ref_amplitude_rad=float("nan"))


class TestSimulationBasics:
    def test_rejects_wrong_config_type(self):
        with pytest.raises(TypeError, match="LoopConfig"):
            simulate_loop({"n_steps": 10})

    def test_rejects_wrong_fault_type(self):
        with pytest.raises(TypeError, match="FaultSpec"):
            simulate_loop(LoopConfig(n_steps=10), fault="sensor_bias")

    def test_history_shapes(self):
        run = simulate_loop(LoopConfig(n_steps=137, seed=1))
        assert run.t_s.shape == (137,)
        for arr in (run.x_true, run.x_est, run.innovation, run.residual):
            assert arr.shape == (137, 2)
        assert run.nis.shape == (137,)
        assert run.u_cmd_nm.shape == (137,)

    def test_onset_step_is_minus_one_for_a_healthy_run(self):
        assert simulate_loop(LoopConfig(n_steps=50)).onset_step == -1

    def test_onset_step_is_reported_for_a_faulted_run(self):
        run = simulate_loop(LoopConfig(n_steps=50), FaultSpec(FaultType.SENSOR_STUCK, 10, 0.0, 0))
        assert run.onset_step == 10

    def test_nis_matches_the_residual(self):
        run = simulate_loop(LoopConfig(n_steps=400, seed=2))
        assert np.allclose(run.nis, np.sum(run.residual**2, axis=1), rtol=1e-10)

    def test_residual_matches_the_innovation_and_covariance(self):
        run = simulate_loop(LoopConfig(n_steps=300, seed=4))
        chol = np.linalg.cholesky(run.innovation_cov)
        assert np.allclose(run.residual, np.linalg.solve(chol, run.innovation.T).T, atol=1e-12)

    def test_seeded_runs_are_bit_identical(self):
        a = simulate_loop(LoopConfig(n_steps=200, seed=17))
        b = simulate_loop(LoopConfig(n_steps=200, seed=17))
        assert np.array_equal(a.residual, b.residual)

    def test_different_seeds_differ(self):
        a = simulate_loop(LoopConfig(n_steps=200, seed=17))
        b = simulate_loop(LoopConfig(n_steps=200, seed=18))
        assert not np.allclose(a.residual, b.residual)

    def test_noise_free_run_has_zero_residual(self):
        # With no noise and a perfect model the filter tracks exactly, so a
        # healthy run's innovation is identically zero.  This is what makes the
        # noise-free run usable as a fault signature.
        run = simulate_loop(LoopConfig(n_steps=300, noise=False))
        assert np.allclose(run.residual, 0.0, atol=1e-12)

    def test_torque_respects_the_wheel_limit(self):
        cfg = LoopConfig(n_steps=400, seed=6, ref_amplitude_rad=5.0, ref_period_s=2.0)
        run = simulate_loop(cfg)
        assert np.all(np.abs(run.u_cmd_nm) <= cfg.plant.max_torque_nm + 1e-15)
        assert np.all(np.abs(run.u_actual_nm) <= cfg.plant.max_torque_nm + 1e-15)

    def test_healthy_run_has_equal_commanded_and_delivered_torque(self):
        run = simulate_loop(LoopConfig(n_steps=200, seed=8))
        assert np.array_equal(run.u_cmd_nm, run.u_actual_nm)

    def test_actuator_fault_separates_commanded_from_delivered(self):
        run = simulate_loop(
            LoopConfig(n_steps=400, seed=8),
            FaultSpec(FaultType.ACTUATOR_LOSS_OF_EFFECT, 100, 0.8, 0),
        )
        assert np.array_equal(run.u_cmd_nm[:100], run.u_actual_nm[:100])
        assert not np.allclose(run.u_cmd_nm[100:], run.u_actual_nm[100:])


class TestFilterPathsAgree:
    def test_steady_state_gain_matches_the_full_recursion(self):
        # The filter is time-invariant and is started at the steady-state
        # covariance, so the fast path is algebraically identical to running
        # the full Joseph-form recursion every step.
        spec = FaultSpec(FaultType.SENSOR_DRIFT, 200, 1e-5, 1)
        cfg = LoopConfig(n_steps=600, seed=3)
        fast = simulate_loop(cfg, spec, steady_state_gain=True)
        slow = simulate_loop(cfg, spec, steady_state_gain=False)
        assert np.allclose(fast.residual, slow.residual, atol=1e-11)
        assert np.allclose(fast.nis, slow.nis, atol=1e-10)
        assert np.allclose(fast.u_actual_nm, slow.u_actual_nm, atol=1e-14)

    def test_agreement_holds_for_the_healthy_fast_path_too(self):
        cfg = LoopConfig(n_steps=400, seed=9)
        fast = simulate_loop(cfg, None, steady_state_gain=True)
        slow = simulate_loop(cfg, None, steady_state_gain=False)
        assert np.allclose(fast.residual, slow.residual, atol=1e-11)

    def test_explicit_none_fault_matches_no_fault(self):
        cfg = LoopConfig(n_steps=200, seed=12)
        assert np.array_equal(
            simulate_loop(cfg).residual, simulate_loop(cfg, FaultSpec(FaultType.NONE)).residual
        )


class TestFilterConsistency:
    def test_healthy_residual_is_unit_covariance_and_white(self):
        r = np.concatenate(
            [simulate_loop(LoopConfig(n_steps=3000, seed=100 + i)).residual[300:] for i in range(6)]
        )
        check = nis_consistency(r)
        assert check.consistent, f"mean NIS {check.mean_nis} outside [{check.low}, {check.high}]"
        # 4-sigma rather than the returned 5 % band: with six statistics a
        # single 2-sigma excursion is common and would make this flaky.
        rho, _ = whiteness(r, 3)
        limit = 4.0 / np.sqrt(r.shape[0])
        assert np.all(np.abs(rho) < limit), f"autocorrelation {rho} exceeds {limit}"

    def test_innovation_covariance_matches_the_riccati_solution(self):
        from fdiscope.kalman import steady_state_covariance

        cfg = LoopConfig(n_steps=50)
        run = simulate_loop(cfg)
        _, s = steady_state_covariance(build_filter(loop_matrices(cfg.plant)))
        assert np.allclose(run.innovation_cov, s)


class TestFaultResponse:
    @pytest.mark.parametrize(
        "spec",
        [
            FaultSpec(FaultType.SENSOR_BIAS, 400, 4.0 * np.sqrt(1.2185e-7), 1),
            FaultSpec(FaultType.SENSOR_STUCK, 400, 0.0, 0),
            FaultSpec(FaultType.SENSOR_DROPOUT, 400, 0.0, 0),
            FaultSpec(FaultType.ACTUATOR_LOSS_OF_EFFECT, 400, 0.9, 0),
            FaultSpec(FaultType.ACTUATOR_STUCK, 400, 0.0, 0),
            FaultSpec(FaultType.ACTUATOR_RUNAWAY, 400, 2e-4, 0),
        ],
    )
    def test_every_fault_raises_the_mean_nis(self, spec):
        run = simulate_loop(LoopConfig(n_steps=1200, seed=21), spec)
        assert np.mean(run.nis[600:]) > np.mean(run.nis[:400])

    def test_a_constant_angle_bias_leaves_no_steady_state_residual(self):
        # Structural: F has a unit eigenvalue in the angle direction and both
        # states are measured, so the estimate simply shifts by the bias.
        sigma = float(np.sqrt(PlantConfig().attitude_var_rad2))
        run = simulate_loop(
            LoopConfig(n_steps=2000, noise=False),
            FaultSpec(FaultType.SENSOR_BIAS, 400, 8.0 * sigma, 0),
        )
        assert np.max(np.abs(run.residual[420:520])) > 1.0
        assert np.max(np.abs(run.residual[1500:])) < 1e-3

    def test_a_constant_rate_bias_does_leave_one(self):
        sigma = float(np.sqrt(PlantConfig().gyro_var_rad2_s2))
        run = simulate_loop(
            LoopConfig(n_steps=2000, noise=False),
            FaultSpec(FaultType.SENSOR_BIAS, 400, 4.0 * sigma, 1),
        )
        assert np.linalg.norm(run.residual[-1]) > 3.9

    def test_gains_change_the_command(self):
        soft = simulate_loop(
            LoopConfig(n_steps=300, seed=5, gains=ControllerGains(natural_freq_rad_s=0.1))
        )
        stiff = simulate_loop(
            LoopConfig(n_steps=300, seed=5, gains=ControllerGains(natural_freq_rad_s=0.6))
        )
        assert np.std(stiff.u_cmd_nm) > np.std(soft.u_cmd_nm)
