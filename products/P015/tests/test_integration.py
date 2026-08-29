"""End-to-end integration tests: telemetry -> train -> policy -> compare."""

import numpy as np
import pytest

from linkswitch.learn import train_outage_predictor
from linkswitch.optical import OpticalParams
from linkswitch.policies import FixedThresholdPolicy, HysteresisPolicy, LearnedPolicy
from linkswitch.rf import RFParams
from linkswitch.scenario import ScenarioConfig, SwitchCost, generate_telemetry
from linkswitch.simulate import simulate_policy


class TestFullPipeline:
    def test_generate_train_evaluate_all_three_policies(self):
        opt = OpticalParams(sigma_i2=0.4, coherence_steps=3.0, margin_db=3.0, rate_mbps=800.0)
        rf = RFParams(rate_mbps=120.0)
        cfg = ScenarioConfig(optical=opt, rf=rf, switch_cost=SwitchCost(downtime_steps=1))
        tau = opt.tau_phys

        train_tels = [generate_telemetry(cfg, 400, seed=100 + i) for i in range(8)]
        model = train_outage_predictor(train_tels, tau_phys=tau, horizon=5, window=6,
                                        random_state=0)
        assert model.is_fitted

        test_tel = generate_telemetry(cfg, 1000, seed=999)

        fixed = FixedThresholdPolicy(tau=tau)
        hyst = HysteresisPolicy(tau_low=tau * 0.85, tau_high=tau * 1.15)
        learned = LearnedPolicy(model, tau_phys=tau, confidence_threshold=0.5, window=6)

        results = {}
        for name, policy in (("fixed", fixed), ("hysteresis", hyst), ("learned", learned)):
            select = policy.select_channels(test_tel)
            assert select.shape == (1000,)
            assert select.dtype == bool
            m = simulate_policy(test_tel, select, cfg)
            results[name] = m
            # Every policy must produce physically sane metrics.
            assert 0.0 <= m.outage_fraction <= 1.0
            assert m.switch_count >= 0
            assert 0.0 <= m.throughput_mbps <= opt.rate_mbps

        # Sanity: none of the three policies should be pathologically worse
        # than doing nothing (staying on RF always, which floors throughput
        # near rf.rate_mbps since RF is highly available in this scenario).
        rf_only = simulate_policy(test_tel, np.zeros(1000, dtype=bool), cfg)
        for name, m in results.items():
            assert m.throughput_mbps >= rf_only.throughput_mbps * 0.5, name

    def test_deterministic_full_pipeline(self):
        # The whole pipeline is deterministic given fixed seeds throughout.
        def run():
            opt = OpticalParams(sigma_i2=0.3, coherence_steps=4.0, margin_db=4.0)
            cfg = ScenarioConfig(optical=opt)
            tels = [generate_telemetry(cfg, 300, seed=i) for i in range(5)]
            model = train_outage_predictor(tels, tau_phys=opt.tau_phys, horizon=4, window=5,
                                            random_state=1)
            test_tel = generate_telemetry(cfg, 500, seed=42)
            policy = LearnedPolicy(model, tau_phys=opt.tau_phys, confidence_threshold=0.5,
                                    window=5)
            select = policy.select_channels(test_tel)
            return simulate_policy(test_tel, select, cfg)

        m1 = run()
        m2 = run()
        assert m1 == m2

    def test_zero_switch_cost_fixed_threshold_at_tau_phys_is_reactive_optimal(self):
        # With zero switch cost, a fixed threshold set exactly at tau_phys
        # should deliver essentially the theoretical maximum optical uptime
        # (it reacts instantly and correctly to every physical outage).
        opt = OpticalParams(sigma_i2=0.3, coherence_steps=4.0, margin_db=5.0, rate_mbps=1000.0)
        rf = RFParams(rate_mbps=100.0)
        cfg = ScenarioConfig(optical=opt, rf=rf, switch_cost=SwitchCost(downtime_steps=0))
        tel = generate_telemetry(cfg, 20_000, seed=5)
        policy = FixedThresholdPolicy(tau=opt.tau_phys)
        select = policy.select_channels(tel)
        m = simulate_policy(tel, select, cfg)
        expected_floor = opt.rate_mbps * tel.opt_available.mean()
        assert m.throughput_mbps == pytest.approx(expected_floor, rel=0.02)
