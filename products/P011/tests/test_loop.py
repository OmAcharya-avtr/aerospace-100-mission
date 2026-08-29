"""Tests for waveforge.loop (configuration, assembly and closed-loop run)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from waveforge.control import noise_variance_gain
from waveforge.loop import AOConfig, AOSystem, LoopResult
from waveforge.predictor import PureDelayPredictor


class TestConfig:
    def test_frame_time(self):
        assert AOConfig(frame_rate_hz=500.0).frame_time_s == pytest.approx(2e-3)

    def test_d_over_r0(self):
        assert AOConfig(diameter_m=0.5, r0_m=0.1).d_over_r0 == pytest.approx(5.0)

    def test_is_frozen(self):
        with pytest.raises(AttributeError):
            AOConfig().gain = 0.9

    def test_bad_frame_rate(self):
        with pytest.raises(ValueError, match="frame_rate_hz"):
            AOConfig(frame_rate_hz=0.0)

    def test_bad_filtered_modes(self):
        with pytest.raises(ValueError, match="n_filtered_modes"):
            AOConfig(n_filtered_modes=-1)

    def test_bad_condition_threshold(self):
        with pytest.raises(ValueError, match="condition_threshold"):
            AOConfig(condition_threshold=1.0)


class TestAssembly:
    def test_components_built(self, tiny_system):
        assert tiny_system.pupil.n_pix == 32
        assert tiny_system.sensor.n_valid > 0
        assert tiny_system.mirror.n_actuators > 0

    def test_interaction_matrix_shape(self, tiny_system):
        assert tiny_system.interaction_matrix.shape == (
            tiny_system.sensor.n_slopes,
            tiny_system.mirror.n_actuators,
        )

    def test_reconstructor_shape(self, tiny_system):
        assert tiny_system.reconstructor.shape == (
            tiny_system.mirror.n_actuators,
            tiny_system.sensor.n_slopes,
        )

    def test_propagation_matrix_shape(self, tiny_system):
        assert tiny_system.propagation_matrix.shape == (
            tiny_system.pupil.n_valid,
            tiny_system.sensor.n_slopes,
        )

    def test_reconstructor_inverts_the_controlled_subspace(self, tiny_system):
        # R D is a projector onto the retained modes, so (R D)^2 = R D
        rd = tiny_system.reconstructor @ tiny_system.interaction_matrix
        assert np.allclose(rd @ rd, rd, atol=1e-8)

    def test_controlled_mode_count(self, tiny_system):
        assert 0 < tiny_system.n_controlled_modes <= tiny_system.sensor.n_slopes

    def test_filtering_reduces_rank(self, tiny_config):
        few = AOSystem(replace(tiny_config, n_filtered_modes=1)).n_controlled_modes
        many = AOSystem(replace(tiny_config, n_filtered_modes=4)).n_controlled_modes
        assert many < few

    def test_over_filtering_raises(self, tiny_config):
        with pytest.raises(ValueError, match="every controlled mode"):
            AOSystem(replace(tiny_config, n_filtered_modes=10_000))

    def test_open_loop_slopes_shape(self, tiny_system):
        slopes = tiny_system.open_loop_slopes(7)
        assert slopes.shape == (7, tiny_system.sensor.n_slopes)

    def test_open_loop_slopes_start_offset(self, tiny_system):
        a = tiny_system.open_loop_slopes(3, start_frame=0)
        b = tiny_system.open_loop_slopes(3, start_frame=5)
        assert not np.allclose(a, b)

    def test_open_loop_bad_arguments(self, tiny_system):
        with pytest.raises(ValueError, match="n_frames"):
            tiny_system.open_loop_slopes(0)
        with pytest.raises(ValueError, match="start_frame"):
            tiny_system.open_loop_slopes(3, start_frame=-1)


class TestErrorBudgetIntegration:
    def test_terms_are_finite(self, tiny_system):
        budget = tiny_system.error_budget()
        assert np.isfinite(budget.total)
        assert budget.fitting > 0.0
        assert budget.temporal > 0.0

    def test_noiseless_system_has_zero_noise_term(self, tiny_system):
        assert tiny_system.error_budget().noise == 0.0

    def test_noise_term_grows_as_flux_falls(self, tiny_config):
        bright = AOSystem(replace(tiny_config, photon_flux=1e4)).error_budget().noise
        faint = AOSystem(replace(tiny_config, photon_flux=1e2)).error_budget().noise
        assert 0.0 < bright < faint

    def test_noise_term_uses_the_closed_loop_gain(self, tiny_config):
        low = AOSystem(replace(tiny_config, photon_flux=1e3, gain=0.2)).error_budget().noise
        high = AOSystem(replace(tiny_config, photon_flux=1e3, gain=0.8)).error_budget().noise
        ratio = noise_variance_gain(0.8, tiny_config.delay_frames) / noise_variance_gain(
            0.2, tiny_config.delay_frames
        )
        assert high / low == pytest.approx(ratio, rel=1e-9)

    def test_unstable_gain_gives_zero_noise_term(self, tiny_config):
        # eta is infinite, so the analytic budget cannot be formed; the code
        # reports zero rather than nan and the loop test shows the divergence.
        system = AOSystem(replace(tiny_config, photon_flux=1e3, gain=1.9, delay_frames=2))
        assert system.error_budget().noise == 0.0

    def test_custom_fitting_coefficient(self, tiny_system):
        a = tiny_system.error_budget().fitting
        b = tiny_system.error_budget(fitting_coefficient=0.14).fitting
        assert b / a == pytest.approx(0.14 / 0.28, rel=1e-9)


class TestClosedLoop:
    @pytest.fixture(scope="class")
    @staticmethod
    def result(tiny_system):
        return tiny_system.run(120, warmup_frames=40)

    def test_result_type(self, result):
        assert isinstance(result, LoopResult)
        assert result.n_frames == 120

    def test_arrays_have_the_right_length(self, result):
        for array in (
            result.residual_variance,
            result.open_loop_variance,
            result.strehl,
            result.saturated_fraction,
        ):
            assert array.shape == (120,)

    def test_loop_reduces_variance(self, result):
        assert result.mean_residual_variance < result.mean_open_loop_variance
        assert result.rejection_db > 0.0

    def test_strehl_in_range(self, result):
        assert np.all((result.strehl > 0.0) & (result.strehl <= 1.0))

    def test_no_saturation_without_stroke_limit(self, result):
        assert result.max_saturated_fraction == 0.0

    def test_not_diverged(self, result):
        assert not result.diverged

    def test_prediction_sigma_absent_without_predictor(self, result):
        assert result.prediction_sigma is None

    def test_reproducible(self, tiny_system):
        a = tiny_system.run(60, warmup_frames=20, rng=5)
        b = tiny_system.run(60, warmup_frames=20, rng=5)
        assert np.allclose(a.residual_variance, b.residual_variance)

    def test_higher_gain_rejects_more_when_stable(self, tiny_config):
        low = AOSystem(replace(tiny_config, gain=0.1)).run(150, warmup_frames=60)
        high = AOSystem(replace(tiny_config, gain=0.5)).run(150, warmup_frames=60)
        assert high.mean_residual_variance < low.mean_residual_variance

    def test_longer_latency_is_worse(self, tiny_config):
        fast = AOSystem(replace(tiny_config, delay_frames=1, gain=0.3)).run(150, warmup_frames=60)
        slow = AOSystem(replace(tiny_config, delay_frames=4, gain=0.3)).run(150, warmup_frames=60)
        assert slow.mean_residual_variance > fast.mean_residual_variance

    def test_pure_delay_predictor_runs(self, tiny_system):
        result = tiny_system.run(120, warmup_frames=40, predictor=PureDelayPredictor())
        assert result.prediction_sigma is not None
        assert result.mean_residual_variance < result.mean_open_loop_variance

    def test_predictor_horizon_must_match_latency(self, tiny_system):
        class Wrong:
            n_history = 2
            horizon = 7

            def predict(self, history):  # pragma: no cover - never reached
                return history[-1], None

        with pytest.raises(ValueError, match="horizon"):
            tiny_system.run(10, warmup_frames=2, predictor=Wrong())

    @pytest.mark.parametrize("n_frames", [1, 0, 2.5])
    def test_bad_n_frames(self, tiny_system, n_frames):
        with pytest.raises(ValueError, match="n_frames"):
            tiny_system.run(n_frames)

    def test_bad_warmup(self, tiny_system):
        with pytest.raises(ValueError, match="warmup_frames"):
            tiny_system.run(10, warmup_frames=10)

    def test_bad_divergence_threshold(self, tiny_system):
        with pytest.raises(ValueError, match="divergence_threshold"):
            tiny_system.run(10, warmup_frames=2, divergence_threshold=0.0)

    def test_summary_needs_frames_after_warmup(self, tiny_system):
        result = tiny_system.run(10, warmup_frames=2)
        object.__setattr__(result, "warmup_frames", 50)
        with pytest.raises(ValueError, match="warmup_frames"):
            _ = result.mean_residual_variance


class TestRunOverrides:
    def test_gain_override_changes_the_result(self, tiny_system):
        low = tiny_system.run(150, warmup_frames=60, gain=0.1)
        high = tiny_system.run(150, warmup_frames=60, gain=0.5)
        assert high.mean_residual_variance < low.mean_residual_variance

    def test_gain_override_matches_a_rebuilt_system(self, tiny_config):
        from dataclasses import replace as _replace

        rebuilt = AOSystem(_replace(tiny_config, gain=0.25)).run(80, warmup_frames=20, rng=9)
        overridden = AOSystem(tiny_config).run(80, warmup_frames=20, rng=9, gain=0.25)
        assert np.allclose(rebuilt.residual_variance, overridden.residual_variance)

    def test_delay_override_matches_a_rebuilt_system(self, tiny_config):
        from dataclasses import replace as _replace

        rebuilt = AOSystem(_replace(tiny_config, delay_frames=3)).run(80, warmup_frames=20, rng=9)
        overridden = AOSystem(tiny_config).run(80, warmup_frames=20, rng=9, delay_frames=3)
        assert np.allclose(rebuilt.residual_variance, overridden.residual_variance)

    def test_delay_override_is_checked_against_the_predictor(self, tiny_system):
        from waveforge.predictor import LinearSlopePredictor

        model = LinearSlopePredictor(n_history=2, horizon=2)
        with pytest.raises(ValueError, match="horizon"):
            tiny_system.run(20, warmup_frames=5, predictor=model, delay_frames=3)

    @pytest.mark.parametrize("gain", [0.0, -0.5])
    def test_bad_gain_override(self, tiny_system, gain):
        with pytest.raises(ValueError, match="gain"):
            tiny_system.run(20, warmup_frames=5, gain=gain)

    def test_bad_delay_override(self, tiny_system):
        with pytest.raises(ValueError, match="delay_frames"):
            tiny_system.run(20, warmup_frames=5, delay_frames=0)
