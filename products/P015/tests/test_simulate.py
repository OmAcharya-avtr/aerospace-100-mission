"""Tests for linkswitch.simulate: policy scoring and Monte Carlo driving loop."""

import numpy as np
import pytest

from linkswitch.optical import OpticalParams
from linkswitch.policies import FixedThresholdPolicy
from linkswitch.rf import RFParams
from linkswitch.scenario import ScenarioConfig, SwitchCost, Telemetry, generate_telemetry
from linkswitch.simulate import run_monte_carlo, simulate_policy


def _make_config(downtime_steps=1):
    return ScenarioConfig(
        optical=OpticalParams(rate_mbps=1000.0),
        rf=RFParams(rate_mbps=100.0),
        switch_cost=SwitchCost(downtime_steps=downtime_steps),
    )


def _telemetry_from(opt_available, rf_available):
    n = len(opt_available)
    return Telemetry(
        irradiance=np.ones(n),
        opt_available=np.array(opt_available, dtype=bool),
        rain_rate_mm_hr=np.zeros(n),
        rf_atten_db=np.zeros(n),
        rf_available=np.array(rf_available, dtype=bool),
    )


class TestSimulatePolicyKnownAnswers:
    def test_always_optical_always_available_no_switches(self):
        cfg = _make_config(downtime_steps=1)
        tel = _telemetry_from([True] * 5, [True] * 5)
        select = np.array([True] * 5)
        m = simulate_policy(tel, select, cfg)
        # Hand calc: 5 steps, always optical, always available, no switches.
        # throughput = 5 * 1000 / 5 = 1000. outage_steps = 0. switch_count = 0.
        assert m.throughput_mbps == pytest.approx(1000.0)
        assert m.outage_steps == 0
        assert m.switch_count == 0
        assert m.n_steps == 5

    def test_one_switch_incurs_downtime(self):
        cfg = _make_config(downtime_steps=2)
        # Select optical for 2 steps, then RF for 3 steps; both channels
        # always available.
        select = np.array([True, True, False, False, False])
        tel = _telemetry_from([True] * 5, [True] * 5)
        m = simulate_policy(tel, select, cfg)
        # Hand calc: t0=opt(1000), t1=opt(1000), t2=switch->downtime(0),
        # t3=downtime(0), t4=rf(100). Sum = 1000+1000+0+0+100 = 2100.
        # throughput = 2100/5 = 420. outage_steps = 2 (the two downtime steps).
        assert m.throughput_mbps == pytest.approx(420.0)
        assert m.outage_steps == 2
        assert m.switch_count == 1

    def test_selected_channel_unavailable_gives_zero_rate(self):
        cfg = _make_config(downtime_steps=0)
        select = np.array([True, True, True])
        tel = _telemetry_from([True, False, True], [True, True, True])
        m = simulate_policy(tel, select, cfg)
        # Hand calc: t0=1000 (opt avail), t1=0 (opt selected but down),
        # t2=1000. throughput = 2000/3 = 666.666...
        assert m.throughput_mbps == pytest.approx(2000.0 / 3.0)
        assert m.outage_steps == 1
        assert m.switch_count == 0

    def test_both_channels_down_gives_zero(self):
        cfg = _make_config(downtime_steps=0)
        select = np.array([True, False])
        tel = _telemetry_from([False, True], [True, False])
        m = simulate_policy(tel, select, cfg)
        assert m.throughput_mbps == pytest.approx(0.0)
        assert m.outage_steps == 2

    def test_multiple_switches_counted(self):
        cfg = _make_config(downtime_steps=0)
        select = np.array([True, False, True, False, True])
        tel = _telemetry_from([True] * 5, [True] * 5)
        m = simulate_policy(tel, select, cfg)
        assert m.switch_count == 4

    def test_shape_mismatch_rejected(self):
        cfg = _make_config()
        tel = _telemetry_from([True] * 5, [True] * 5)
        with pytest.raises(ValueError):
            simulate_policy(tel, np.array([True, False]), cfg)

    def test_outage_fraction_consistent_with_outage_steps(self):
        cfg = _make_config(downtime_steps=1)
        tel = _telemetry_from([True, False, True, True], [True, True, True, True])
        select = np.array([True, True, True, True])
        m = simulate_policy(tel, select, cfg)
        assert m.outage_fraction == pytest.approx(m.outage_steps / m.n_steps)


class TestRunMonteCarlo:
    def test_number_of_reps(self):
        cfg = _make_config()
        runs = run_monte_carlo(cfg, lambda: FixedThresholdPolicy(tau=cfg.optical.tau_phys),
                                n_steps=100, n_reps=7, seed0=0)
        assert len(runs) == 7

    def test_different_seeds_give_different_telemetry(self):
        # A higher scintillation index and tighter margin than the module
        # default so outages are frequent enough within 200 steps that
        # per-rep throughput varies (the default scenario's outages are too
        # rare over 200 steps to guarantee variation across only 5 reps).
        cfg = ScenarioConfig(optical=OpticalParams(sigma_i2=0.5, margin_db=2.0))
        runs = run_monte_carlo(cfg, lambda: FixedThresholdPolicy(tau=cfg.optical.tau_phys),
                                n_steps=200, n_reps=5, seed0=0)
        throughputs = {r.throughput_mbps for r in runs}
        assert len(throughputs) > 1  # not all identical

    def test_seeded_reproducibility(self):
        cfg = _make_config()

        def factory():
            return FixedThresholdPolicy(tau=cfg.optical.tau_phys)

        runs1 = run_monte_carlo(cfg, factory, n_steps=200, n_reps=5, seed0=10)
        runs2 = run_monte_carlo(cfg, factory, n_steps=200, n_reps=5, seed0=10)
        assert [r.throughput_mbps for r in runs1] == [r.throughput_mbps for r in runs2]

    def test_invalid_n_reps_rejected(self):
        cfg = _make_config()
        with pytest.raises(ValueError):
            run_monte_carlo(cfg, lambda: FixedThresholdPolicy(tau=cfg.optical.tau_phys),
                            n_steps=100, n_reps=0, seed0=0)


class TestIntegrationRealTelemetry:
    def test_generate_then_simulate_end_to_end(self):
        cfg = _make_config(downtime_steps=1)
        tel = generate_telemetry(cfg, 1000, seed=0)
        policy = FixedThresholdPolicy(tau=cfg.optical.tau_phys)
        select = policy.select_channels(tel)
        m = simulate_policy(tel, select, cfg)
        assert 0.0 <= m.outage_fraction <= 1.0
        assert m.switch_count >= 0
        assert m.throughput_mbps > 0.0
        assert m.throughput_mbps <= cfg.optical.rate_mbps
