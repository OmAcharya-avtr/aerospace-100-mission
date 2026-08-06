"""Performance tests: runtime bounds for the documented compute budget.

Bounds are deliberately loose (well above the measured values recorded in
validation/v3_performance.txt) so they catch order-of-magnitude regressions
rather than machine-to-machine noise.
"""

from __future__ import annotations

import time

import pytest

from beamtwin.budget import LinkParams, compute_budget
from beamtwin.channel import ChannelParams, sample_received_power_dbm
from beamtwin.scenario import run_twin, scenario_from_dict

LINK = LinkParams(range_m=10_000.0, attenuation_db_per_km=2.5, rx_sensitivity_dbm=-30.0)
CHANNEL = ChannelParams(cn2=5e-16, pointing_jitter_rad=5e-6)


class TestRuntimeBounds:
    def test_million_sample_monte_carlo_under_10s(self):
        # Measured 0.080 s on the 2-core reference machine (V3).
        t0 = time.perf_counter()
        sample_received_power_dbm(LINK, CHANNEL, n_samples=1_000_000, seed=1)
        assert time.perf_counter() - t0 < 10.0

    def test_monte_carlo_throughput_above_100k_per_second(self):
        # Measured ~1.2e7 samples/s (V3); bound is 100x lower.
        n = 500_000
        t0 = time.perf_counter()
        sample_received_power_dbm(LINK, CHANNEL, n_samples=n, seed=2)
        elapsed = time.perf_counter() - t0
        assert n / elapsed > 100_000

    def test_scenario_report_under_5s(self):
        # R13 bound; measured 0.017 s (V3).
        s = scenario_from_dict(
            {
                "link": {"range_km": 10.0, "attenuation_db_per_km": 2.5},
                "channel": {"cn2": 5e-16, "pointing_jitter_urad": 5.0},
                "monte_carlo": {"n_samples": 200_000, "seed": 1},
            }
        )
        t0 = time.perf_counter()
        run_twin(s)
        assert time.perf_counter() - t0 < 5.0

    def test_deterministic_budget_is_fast(self):
        t0 = time.perf_counter()
        for _ in range(1000):
            compute_budget(LINK)
        assert time.perf_counter() - t0 < 2.0

    def test_memory_guard_rejects_oversized_run(self):
        # The guard must trip before allocating tens of GB.
        with pytest.raises(ValueError, match="budget"):
            sample_received_power_dbm(LINK, CHANNEL, n_samples=100_000_000, seed=1)
