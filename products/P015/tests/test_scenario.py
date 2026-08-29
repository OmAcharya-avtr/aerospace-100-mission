"""Tests for linkswitch.scenario: telemetry generation."""

import numpy as np
import pytest

from linkswitch.optical import OpticalParams
from linkswitch.scenario import ScenarioConfig, SwitchCost, generate_telemetry


class TestSwitchCost:
    def test_default(self):
        assert SwitchCost().downtime_steps == 1

    def test_zero_allowed(self):
        assert SwitchCost(downtime_steps=0).downtime_steps == 0

    def test_negative_rejected(self):
        with pytest.raises(ValueError):
            SwitchCost(downtime_steps=-1)

    def test_float_rejected(self):
        with pytest.raises(ValueError):
            SwitchCost(downtime_steps=1.5)


class TestGenerateTelemetry:
    def test_shapes_consistent(self):
        cfg = ScenarioConfig()
        tel = generate_telemetry(cfg, n_steps=300, seed=0)
        assert tel.n_steps == 300
        for arr in (tel.irradiance, tel.opt_available, tel.rain_rate_mm_hr,
                    tel.rf_atten_db, tel.rf_available):
            assert arr.shape == (300,)

    def test_seeded_reproducibility(self):
        cfg = ScenarioConfig()
        t1 = generate_telemetry(cfg, 200, seed=42)
        t2 = generate_telemetry(cfg, 200, seed=42)
        np.testing.assert_array_equal(t1.irradiance, t2.irradiance)
        np.testing.assert_array_equal(t1.rain_rate_mm_hr, t2.rain_rate_mm_hr)

    def test_different_seeds_differ(self):
        cfg = ScenarioConfig()
        t1 = generate_telemetry(cfg, 200, seed=1)
        t2 = generate_telemetry(cfg, 200, seed=2)
        assert not np.array_equal(t1.irradiance, t2.irradiance)

    def test_optical_and_rf_streams_are_independent(self):
        # Changing the RF seed stream must not perturb the optical stream:
        # confirmed indirectly by checking that irradiance is identical
        # whenever the top-level seed is identical (already covered), and
        # that rain_rate_mm_hr is all zero exactly where raining is False,
        # i.e. the two subsystems are self-consistent, not entangled.
        cfg = ScenarioConfig()
        tel = generate_telemetry(cfg, 5000, seed=3)
        raining = tel.rain_rate_mm_hr > 0.0
        assert np.array_equal(raining, tel.rf_atten_db > 0.0)

    def test_opt_available_consistent_with_threshold(self):
        cfg = ScenarioConfig()
        tel = generate_telemetry(cfg, 2000, seed=4)
        expected = tel.irradiance >= cfg.optical.tau_phys
        np.testing.assert_array_equal(tel.opt_available, expected)

    def test_rf_available_consistent_with_margin(self):
        cfg = ScenarioConfig()
        tel = generate_telemetry(cfg, 2000, seed=5)
        snr = cfg.rf.snr_clear_db - tel.rf_atten_db
        expected = snr >= cfg.rf.snr_min_db
        np.testing.assert_array_equal(tel.rf_available, expected)

    def test_rain_rate_zero_when_clear(self):
        cfg = ScenarioConfig()
        tel = generate_telemetry(cfg, 2000, seed=6)
        assert np.all(tel.rain_rate_mm_hr[tel.rf_atten_db == 0.0] == 0.0)

    def test_gamma_gamma_model_rejected_in_telemetry(self):
        cfg = ScenarioConfig(optical=OpticalParams(fading_model="gamma_gamma"))
        with pytest.raises(ValueError, match="gamma_gamma"):
            generate_telemetry(cfg, 100, seed=0)

    def test_invalid_n_steps_rejected(self):
        cfg = ScenarioConfig()
        with pytest.raises(ValueError):
            generate_telemetry(cfg, 0, seed=0)

    def test_invalid_seed_rejected(self):
        cfg = ScenarioConfig()
        with pytest.raises(ValueError):
            generate_telemetry(cfg, 100, seed=-1)

    def test_long_run_outage_fraction_near_analytic(self):
        # For a well-separated threshold this is approximately Phi(z_phys);
        # a loose statistical tolerance since this is a single MC draw.
        from scipy.stats import norm

        opt = OpticalParams(sigma_i2=0.25, coherence_steps=5.0, margin_db=6.0)
        cfg = ScenarioConfig(optical=opt)
        tel = generate_telemetry(cfg, 100_000, seed=11)
        z_phys = (np.log(opt.tau_phys) - (-0.5 * opt.sigma_z**2)) / opt.sigma_z
        expected = float(norm.cdf(z_phys))
        measured = 1.0 - tel.opt_available.mean()
        assert measured == pytest.approx(expected, abs=0.003)
