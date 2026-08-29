"""Tests for linkswitch.metrics: confidence-interval aggregation."""

import numpy as np
import pytest

from linkswitch.metrics import aggregate_runs, compare_policies, mean_ci
from linkswitch.optical import OpticalParams
from linkswitch.policies import FixedThresholdPolicy, HysteresisPolicy
from linkswitch.rf import RFParams
from linkswitch.scenario import ScenarioConfig
from linkswitch.simulate import RunMetrics


class TestMeanCI:
    def test_known_answer_symmetric_data(self):
        # values = [1, 2, 3, 4, 5]; mean=3, std(ddof=1)=sqrt(2.5)=1.5811...
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        agg = mean_ci(values, ci_level=0.95)
        assert agg.mean == pytest.approx(3.0)
        # 95% CI half-width = t_{0.975, 4} * std / sqrt(5) = 2.776445 * 1.581139 / 2.236068
        from scipy import stats
        t_crit = stats.t.ppf(0.975, df=4)
        expected_hw = t_crit * np.std(values, ddof=1) / np.sqrt(5)
        assert agg.ci_high - agg.mean == pytest.approx(expected_hw, rel=1e-9)
        assert agg.mean - agg.ci_low == pytest.approx(expected_hw, rel=1e-9)

    def test_ci_contains_mean(self):
        agg = mean_ci(np.array([1.0, 5.0, 3.0, 9.0, 2.0]))
        assert agg.ci_low <= agg.mean <= agg.ci_high

    def test_n_equals_1_collapses_to_point(self):
        agg = mean_ci(np.array([7.0]))
        assert agg.mean == agg.ci_low == agg.ci_high == 7.0

    def test_constant_values_zero_width(self):
        agg = mean_ci(np.full(10, 4.0))
        assert agg.ci_low == agg.ci_high == pytest.approx(4.0)

    def test_wider_ci_level_gives_wider_interval(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
        agg95 = mean_ci(values, ci_level=0.95)
        agg99 = mean_ci(values, ci_level=0.99)
        assert (agg99.ci_high - agg99.ci_low) > (agg95.ci_high - agg95.ci_low)

    def test_more_samples_narrows_ci(self):
        rng = np.random.default_rng(0)
        small = mean_ci(rng.normal(size=5))
        large = mean_ci(rng.normal(size=500))
        assert (large.ci_high - large.ci_low) < (small.ci_high - small.ci_low)

    def test_invalid_ci_level_rejected(self):
        with pytest.raises(ValueError):
            mean_ci(np.array([1.0, 2.0]), ci_level=1.5)
        with pytest.raises(ValueError):
            mean_ci(np.array([1.0, 2.0]), ci_level=0.0)

    def test_empty_rejected(self):
        with pytest.raises(ValueError):
            mean_ci(np.array([]))


class TestAggregateRuns:
    def test_aggregates_three_metrics(self):
        runs = [
            RunMetrics(throughput_mbps=t, outage_steps=o, outage_fraction=o / 100,
                       switch_count=s, n_steps=100)
            for t, o, s in [(900.0, 5, 2), (950.0, 3, 1), (920.0, 4, 3)]
        ]
        agg = aggregate_runs(runs)
        assert set(agg.keys()) == {"throughput_mbps", "outage_fraction", "switch_count"}
        assert agg["throughput_mbps"].mean == pytest.approx((900 + 950 + 920) / 3)
        assert agg["switch_count"].mean == pytest.approx((2 + 1 + 3) / 3)

    def test_empty_rejected(self):
        with pytest.raises(ValueError):
            aggregate_runs([])


class TestComparePolicies:
    def test_paired_same_telemetry_used(self):
        # With switch_cost = 0 and identical thresholds, fixed-threshold and
        # a degenerate hysteresis (tau_low == tau_high) must produce
        # IDENTICAL per-rep metrics, because they see the same telemetry and
        # make the same decisions -- this validates the pairing.
        from linkswitch.scenario import SwitchCost

        cfg = ScenarioConfig(switch_cost=SwitchCost(downtime_steps=0))
        tau = cfg.optical.tau_phys
        factories = {
            "fixed": lambda: FixedThresholdPolicy(tau=tau),
            "hyst_degenerate": lambda: HysteresisPolicy(tau_low=tau, tau_high=tau),
        }
        results = compare_policies(cfg, factories, n_steps=300, n_reps=5, seed0=0)
        assert results["fixed"]["throughput_mbps"].mean == pytest.approx(
            results["hyst_degenerate"]["throughput_mbps"].mean
        )
        assert results["fixed"]["switch_count"].mean == pytest.approx(
            results["hyst_degenerate"]["switch_count"].mean
        )

    def test_empty_factories_rejected(self):
        cfg = ScenarioConfig()
        with pytest.raises(ValueError):
            compare_policies(cfg, {}, n_steps=100, n_reps=5, seed0=0)

    def test_invalid_n_reps_rejected(self):
        cfg = ScenarioConfig()
        with pytest.raises(ValueError):
            compare_policies(cfg, {"fixed": lambda: FixedThresholdPolicy(tau=0.5)},
                              n_steps=100, n_reps=0, seed0=0)

    def test_all_requested_policies_present(self):
        cfg = ScenarioConfig(optical=OpticalParams(margin_db=4.0), rf=RFParams())
        tau = cfg.optical.tau_phys
        factories = {
            "a": lambda: FixedThresholdPolicy(tau=tau),
            "b": lambda: HysteresisPolicy(tau_low=tau * 0.9, tau_high=tau * 1.1),
        }
        results = compare_policies(cfg, factories, n_steps=200, n_reps=10, seed0=1)
        assert set(results.keys()) == {"a", "b"}
        for name, metrics in results.items():
            for m in metrics.values():
                assert m.n == 10
