"""Benchmark / regression test: pinned seeded output.

If any of these numbers change, something in the RNG usage, the AR(1)
recursion, the rain Markov chain, the feature/label construction, or the
RandomForest wiring has changed behaviour -- intentional or not. Regenerate
the pinned values only after confirming the change is intentional (never to
silence a real regression).
"""

import numpy as np
import pytest

from linkswitch.learn import train_outage_predictor
from linkswitch.optical import OpticalParams
from linkswitch.policies import FixedThresholdPolicy, HysteresisPolicy, LearnedPolicy
from linkswitch.rf import RFParams
from linkswitch.scenario import ScenarioConfig, SwitchCost, generate_telemetry
from linkswitch.simulate import simulate_policy


@pytest.fixture(scope="module")
def pinned_config():
    return ScenarioConfig(
        optical=OpticalParams(sigma_i2=0.25, coherence_steps=5.0, margin_db=6.0,
                              rate_mbps=1000.0),
        rf=RFParams(rate_mbps=150.0),
        switch_cost=SwitchCost(downtime_steps=1),
    )


class TestPinnedTelemetry:
    def test_irradiance_first_five_values(self, pinned_config):
        tel = generate_telemetry(pinned_config, n_steps=50, seed=2026)
        # Pinned against this exact implementation on numpy's default_rng /
        # SeedSequence; see module docstring.
        expected = np.array([
            0.9424476337090381, 0.9619058536091016, 1.0020585721391000,
            0.9317384292426188, 0.8745195028411990,
        ])
        np.testing.assert_allclose(tel.irradiance[:5], expected, rtol=1e-9)

    def test_outage_fraction_50000_steps_seed_2026(self, pinned_config):
        tel = generate_telemetry(pinned_config, n_steps=50_000, seed=2026)
        outage_fraction = float(1.0 - tel.opt_available.mean())
        assert outage_fraction == pytest.approx(0.00362, abs=2e-4)


class TestPinnedFixedThresholdRun:
    def test_metrics_at_seed_7(self, pinned_config):
        tel = generate_telemetry(pinned_config, n_steps=5000, seed=7)
        policy = FixedThresholdPolicy(tau=pinned_config.optical.tau_phys)
        select = policy.select_channels(tel)
        m = simulate_policy(tel, select, pinned_config)
        assert m.throughput_mbps == pytest.approx(993.44, abs=0.5)
        assert m.switch_count == 26
        assert m.outage_steps == 26


class TestPinnedHysteresisRun:
    def test_metrics_at_seed_7(self, pinned_config):
        tel = generate_telemetry(pinned_config, n_steps=5000, seed=7)
        tau = pinned_config.optical.tau_phys
        policy = HysteresisPolicy(tau_low=tau * 0.8, tau_high=tau * 1.2)
        select = policy.select_channels(tel)
        m = simulate_policy(tel, select, pinned_config)
        assert m.switch_count == 10
        assert m.outage_steps == 21


class TestPinnedLearnedPipeline:
    def test_deterministic_switch_count_at_fixed_seeds(self, pinned_config):
        tau = pinned_config.optical.tau_phys
        train_tels = [generate_telemetry(pinned_config, 500, seed=90_000 + i)
                     for i in range(10)]
        model = train_outage_predictor(train_tels, tau_phys=tau, horizon=5, window=8,
                                        random_state=0)
        test_tel = generate_telemetry(pinned_config, 5000, seed=7)
        policy = LearnedPolicy(model, tau_phys=tau, confidence_threshold=0.5, window=8)
        select = policy.select_channels(test_tel)
        m = simulate_policy(test_tel, select, pinned_config)
        # Pinned end-to-end regression value (RandomForest + full pipeline).
        assert m.switch_count == 6
        assert m.throughput_mbps == pytest.approx(995.26, abs=0.5)
