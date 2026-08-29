"""Failure-mode tests (Level 3 requirement).

Three ways an adaptive-optics loop fails in service are exercised here:
actuator saturation, Shack-Hartmann subaperture dropout, and loop instability
at excessive gain.  Each test asserts the *documented* behaviour, including the
cases where performance legitimately gets worse.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from waveforge.control import Integrator, stability_limit_gain
from waveforge.dm import DeformableMirror
from waveforge.loop import AOConfig, AOSystem
from waveforge.pupil import PupilGrid


class TestActuatorSaturation:
    def test_stroke_limit_is_reported(self, tiny_config):
        system = AOSystem(replace(tiny_config, stroke_rad=0.2, gain=0.5))
        result = system.run(150, warmup_frames=50)
        assert result.max_saturated_fraction > 0.0

    def test_saturation_degrades_the_residual(self, tiny_config):
        free = AOSystem(replace(tiny_config, gain=0.5)).run(150, warmup_frames=50)
        clipped = AOSystem(replace(tiny_config, gain=0.5, stroke_rad=0.15)).run(
            150, warmup_frames=50
        )
        assert clipped.mean_residual_variance > free.mean_residual_variance

    def test_generous_stroke_never_saturates(self, tiny_config):
        result = AOSystem(replace(tiny_config, stroke_rad=1e6, gain=0.5)).run(
            120, warmup_frames=40
        )
        assert result.max_saturated_fraction == 0.0

    def test_saturated_loop_stays_bounded(self, tiny_config):
        # Clipping is a hard nonlinearity; it must not make the loop blow up.
        result = AOSystem(replace(tiny_config, stroke_rad=0.05, gain=0.5)).run(
            200, warmup_frames=60
        )
        assert not result.diverged
        assert np.all(np.isfinite(result.residual_variance))

    def test_mirror_clip_is_the_mechanism(self):
        mirror = DeformableMirror(PupilGrid(32, 0.5), n_act=5, stroke_rad=0.5)
        clipped, fraction = mirror.clip(np.full(mirror.n_actuators, 10.0))
        assert np.all(clipped == 0.5)
        assert fraction == pytest.approx(1.0)

    def test_integrator_command_limit(self):
        it = Integrator(4, gain=1.0, delay_frames=1, command_limit=0.25)
        out = it.step(np.full(4, 5.0))
        assert np.all(out == 0.25)
        assert it.last_saturated_fraction == pytest.approx(1.0)


class TestSensorDropout:
    def test_dropout_degrades_the_residual(self, tiny_config):
        clean = AOSystem(replace(tiny_config, gain=0.4)).run(150, warmup_frames=50, rng=3)
        dropped = AOSystem(replace(tiny_config, gain=0.4, dropout_probability=0.3)).run(
            150, warmup_frames=50, rng=3
        )
        assert dropped.mean_residual_variance > clean.mean_residual_variance

    def test_loop_survives_heavy_dropout(self, tiny_config):
        result = AOSystem(replace(tiny_config, gain=0.3, dropout_probability=0.5)).run(
            200, warmup_frames=60, rng=4
        )
        assert np.all(np.isfinite(result.residual_variance))

    def test_total_dropout_leaves_the_loop_open(self, tiny_config):
        # 99 % dropout carries almost no information; the residual must be
        # comparable with the open-loop input rather than mysteriously small.
        result = AOSystem(replace(tiny_config, gain=0.3, dropout_probability=0.99)).run(
            150, warmup_frames=50, rng=5
        )
        assert result.mean_residual_variance > 0.3 * result.mean_open_loop_variance

    def test_dropout_flags_are_exposed(self, tiny_system):
        sensor = tiny_system.sensor
        sensor_with_dropout = replace(
            AOConfig(), dropout_probability=0.4
        )  # config-level plumbing
        assert sensor_with_dropout.dropout_probability == 0.4
        measurement = sensor.measure(np.zeros((32, 32)), 0)
        assert measurement.valid.shape == (sensor.n_valid,)


class TestLoopInstability:
    def test_gain_above_the_limit_diverges(self, tiny_config):
        limit = stability_limit_gain(2)
        system = AOSystem(replace(tiny_config, gain=1.6 * limit, delay_frames=2))
        result = system.run(200, warmup_frames=50, divergence_threshold=1e6)
        assert result.diverged

    def test_gain_below_the_limit_is_stable(self, tiny_config):
        limit = stability_limit_gain(2)
        result = AOSystem(replace(tiny_config, gain=0.6 * limit, delay_frames=2)).run(
            200, warmup_frames=50
        )
        assert not result.diverged

    def test_long_latency_makes_a_previously_safe_gain_unstable(self, tiny_config):
        # g = 0.9 is stable at d = 1 (limit 2.0) and unstable at d = 4
        # (limit 0.445)
        safe = AOSystem(replace(tiny_config, gain=0.9, delay_frames=1)).run(
            200, warmup_frames=50
        )
        unsafe = AOSystem(replace(tiny_config, gain=0.9, delay_frames=4)).run(
            200, warmup_frames=50
        )
        assert not safe.diverged
        assert unsafe.diverged

    def test_divergence_stops_the_run_early(self, tiny_config):
        result = AOSystem(replace(tiny_config, gain=1.9, delay_frames=2)).run(
            300, warmup_frames=50, divergence_threshold=1e4
        )
        assert result.diverged
        # after divergence the trace is held at its last value
        assert result.residual_variance[-1] == result.residual_variance[-2]

    def test_integrator_reports_instability(self):
        assert not Integrator(1, gain=1.2, delay_frames=2).is_stable
        assert Integrator(1, gain=0.8, delay_frames=2).is_stable

    def test_noise_gain_is_infinite_beyond_the_limit(self):
        from waveforge.control import noise_variance_gain

        assert np.isinf(noise_variance_gain(1.1, 2))
        assert np.isfinite(noise_variance_gain(0.9, 2))
