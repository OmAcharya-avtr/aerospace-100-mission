"""Seeded scenario generation: balance, determinism and range compliance."""

from __future__ import annotations

import numpy as np
import pytest

from fdiscope.faults import FAULT_CLASSES, FaultType
from fdiscope.plant import PlantConfig
from fdiscope.scenarios import (
    DEFAULT_N_STEPS,
    DEFAULT_RANGES,
    MagnitudeRanges,
    ScenarioSet,
    sample_scenario,
    sample_scenarios,
)


class TestSampleScenario:
    def test_is_deterministic(self):
        a = sample_scenario(1234, index=3)
        b = sample_scenario(1234, index=3)
        assert a.fault == b.fault
        assert a.seed == b.seed and a.n_steps == b.n_steps

    def test_index_selects_the_class(self):
        for i, fault in enumerate(FAULT_CLASSES):
            assert sample_scenario(500 + i, index=i).label is fault

    def test_fault_class_override_wins(self):
        sc = sample_scenario(7, index=3, fault_class=FaultType.ACTUATOR_STUCK)
        assert sc.label is FaultType.ACTUATOR_STUCK

    def test_healthy_scenario_has_zero_magnitude(self):
        sc = sample_scenario(11, fault_class=FaultType.NONE)
        assert sc.fault.magnitude == 0.0
        assert sc.label is FaultType.NONE

    def test_onset_lies_in_the_documented_window(self):
        for seed in range(50):
            assert 600 <= sample_scenario(seed, index=seed).onset_step <= 1300

    def test_onset_leaves_room_for_the_delay_horizon(self):
        for seed in range(50):
            sc = sample_scenario(seed, index=seed)
            assert sc.onset_step + 600 <= sc.n_steps

    def test_default_step_count(self):
        assert sample_scenario(0).n_steps == DEFAULT_N_STEPS

    def test_config_carries_the_seed_and_step_count(self):
        sc = sample_scenario(99, index=1, n_steps=321)
        cfg = sc.config()
        assert cfg.seed == 99 and cfg.n_steps == 321 and cfg.noise

    def test_config_can_disable_noise(self):
        assert not sample_scenario(1).config(noise=False).noise


class TestMagnitudeRanges:
    def test_bias_magnitudes_stay_inside_the_sigma_range(self):
        p = PlantConfig()
        sigmas = (np.sqrt(p.attitude_var_rad2), np.sqrt(p.gyro_var_rad2_s2))
        lo, hi = DEFAULT_RANGES.bias_sigma
        for seed in range(200):
            sc = sample_scenario(seed, fault_class=FaultType.SENSOR_BIAS)
            in_sigmas = abs(sc.fault.magnitude) / sigmas[sc.fault.channel]
            assert lo - 1e-9 <= in_sigmas <= hi + 1e-9

    def test_loss_of_effectiveness_stays_inside_its_range(self):
        lo, hi = DEFAULT_RANGES.loss_fraction
        for seed in range(200):
            sc = sample_scenario(seed, fault_class=FaultType.ACTUATOR_LOSS_OF_EFFECT)
            assert lo - 1e-9 <= sc.fault.magnitude <= hi + 1e-9

    def test_runaway_stays_inside_its_range(self):
        lo, hi = DEFAULT_RANGES.runaway_nm_per_s
        for seed in range(200):
            sc = sample_scenario(seed, fault_class=FaultType.ACTUATOR_RUNAWAY)
            assert lo - 1e-12 <= abs(sc.fault.magnitude) <= hi + 1e-12

    def test_drift_uses_the_channel_specific_range(self):
        p = PlantConfig()
        for seed in range(200):
            sc = sample_scenario(seed, fault_class=FaultType.SENSOR_DRIFT)
            sigma = np.sqrt(p.attitude_var_rad2 if sc.fault.channel == 0 else p.gyro_var_rad2_s2)
            span = (
                DEFAULT_RANGES.drift_angle_sigma_per_s
                if sc.fault.channel == 0
                else DEFAULT_RANGES.drift_rate_sigma_per_s
            )
            rate = abs(sc.fault.magnitude) / sigma
            assert span[0] - 1e-9 <= rate <= span[1] + 1e-9

    def test_custom_ranges_are_honoured(self):
        ranges = MagnitudeRanges(bias_sigma=(20.0, 20.0))
        p = PlantConfig()
        sc = sample_scenario(3, fault_class=FaultType.SENSOR_BIAS, ranges=ranges)
        sigma = np.sqrt(p.attitude_var_rad2 if sc.fault.channel == 0 else p.gyro_var_rad2_s2)
        assert np.isclose(abs(sc.fault.magnitude) / sigma, 20.0)

    def test_both_signs_are_drawn(self):
        signs = {
            np.sign(sample_scenario(s, fault_class=FaultType.SENSOR_BIAS).fault.magnitude)
            for s in range(40)
        }
        assert signs == {-1.0, 1.0}

    def test_both_channels_are_drawn(self):
        channels = {
            sample_scenario(s, fault_class=FaultType.SENSOR_BIAS).fault.channel for s in range(40)
        }
        assert channels == {0, 1}


class TestSampleScenarios:
    def test_class_balance_is_exact_for_a_multiple_of_eight(self):
        counts = {}
        for sc in sample_scenarios(80, 3000):
            counts[sc.label] = counts.get(sc.label, 0) + 1
        assert set(counts) == set(FAULT_CLASSES)
        assert set(counts.values()) == {10}

    def test_seeds_are_consecutive(self):
        seeds = [sc.seed for sc in sample_scenarios(10, 4242)]
        assert seeds == list(range(4242, 4252))

    def test_disjoint_seed_blocks_give_disjoint_scenarios(self):
        train = {sc.seed for sc in sample_scenarios(50, 1000)}
        test = {sc.seed for sc in sample_scenarios(50, 5000)}
        assert train.isdisjoint(test)

    def test_rejects_zero_count(self):
        with pytest.raises(ValueError, match="n must be >= 1"):
            sample_scenarios(0, 1)


class TestScenarioSet:
    def test_length_and_labels(self):
        scenarios = sample_scenarios(16, 900)
        s = ScenarioSet(name="demo", scenarios=scenarios)
        assert len(s) == 16
        assert s.labels() == [sc.label for sc in scenarios]
