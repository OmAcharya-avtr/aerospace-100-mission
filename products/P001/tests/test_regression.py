"""Regression tests with pinned seeded outputs.

Every number below was produced by this exact code with the stated seed and
recorded on 2026-08-06. A change to any of them means the numerical
behaviour of BeamTwin changed: investigate before updating the pin.
"""

from __future__ import annotations

import numpy as np
import pytest

from beamtwin.budget import LinkParams
from beamtwin.channel import ChannelParams, sample_received_power_dbm
from beamtwin.scenario import run_twin, scenario_from_dict

# --- Pinned scenario A: 10 km hazy terrestrial link, seed 1234, n = 50000 ---
SCENARIO_A = {
    "name": "regression-a",
    "link": {
        "range_km": 10.0,
        "attenuation_db_per_km": 2.5,
        "rx_sensitivity_dbm": -30.0,
        "pointing_bias_urad": 2.0,
    },
    "channel": {"cn2": 5.0e-16, "pointing_jitter_urad": 5.0},
    "monte_carlo": {"n_samples": 50_000, "seed": 1234},
}

PINNED_A = {
    "margin_db": 11.947252286960218,
    "fade_probability": 0.0011,
    "n_fades": 55,
    "analytic_baseline": 0.0002666137026738942,
    "rytov_variance": 0.6782107162167273,
    "margin_p01_db": 2.5740287177508163,
    "margin_mean_db": 10.100117004220763,
}

# --- Pinned raw sample stream: 1000 samples, seed 7 ---
PINNED_B_FIRST3 = [0.7546231854227785, 1.9481221743301074, -0.33197068937111485]
PINNED_B_MEAN = -0.1807236472616479


@pytest.fixture(scope="module")
def report_a():
    return run_twin(scenario_from_dict(SCENARIO_A))


class TestPinnedScenarioA:
    def test_margin_db(self, report_a):
        assert report_a["budget"]["margin_db"] == pytest.approx(PINNED_A["margin_db"], rel=1e-12)

    def test_fade_probability(self, report_a):
        assert report_a["monte_carlo"]["fade_probability"] == pytest.approx(
            PINNED_A["fade_probability"], rel=1e-12
        )

    def test_fade_count_exact(self, report_a):
        assert report_a["monte_carlo"]["n_fades"] == PINNED_A["n_fades"]

    def test_analytic_baseline(self, report_a):
        assert report_a["analytic_baseline"][
            "fade_probability_scintillation_only"
        ] == pytest.approx(PINNED_A["analytic_baseline"], rel=1e-12)

    def test_rytov_variance(self, report_a):
        assert report_a["channel"]["rytov_variance"] == pytest.approx(
            PINNED_A["rytov_variance"], rel=1e-12
        )

    def test_margin_first_percentile(self, report_a):
        assert report_a["monte_carlo"]["margin_percentiles_db"]["p01"] == pytest.approx(
            PINNED_A["margin_p01_db"], rel=1e-12
        )

    def test_margin_mean(self, report_a):
        assert report_a["monte_carlo"]["margin_moments"]["mean_db"] == pytest.approx(
            PINNED_A["margin_mean_db"], rel=1e-12
        )

    def test_confidence_interval_brackets_pinned_probability(self, report_a):
        mc = report_a["monte_carlo"]
        assert mc["fade_ci95_low"] <= mc["fade_probability"] <= mc["fade_ci95_high"]


class TestPinnedSampleStream:
    @pytest.fixture(scope="class")
    @staticmethod
    def samples():
        return sample_received_power_dbm(
            LinkParams(range_m=10_000.0),
            ChannelParams(cn2=1e-15, pointing_jitter_rad=5e-6),
            n_samples=1000,
            seed=7,
        ).samples_dbm

    def test_first_three_samples(self, samples):
        assert samples[:3].tolist() == pytest.approx(PINNED_B_FIRST3, rel=1e-12)

    def test_stream_mean(self, samples):
        assert float(samples.mean()) == pytest.approx(PINNED_B_MEAN, rel=1e-12)

    def test_stream_is_reproducible(self, samples):
        again = sample_received_power_dbm(
            LinkParams(range_m=10_000.0),
            ChannelParams(cn2=1e-15, pointing_jitter_rad=5e-6),
            n_samples=1000,
            seed=7,
        ).samples_dbm
        assert np.array_equal(samples, again)


class TestReproducibility:
    def test_identical_reports_for_identical_seed(self):
        from beamtwin.scenario import report_to_json

        a = report_to_json(run_twin(scenario_from_dict(SCENARIO_A)))
        b = report_to_json(run_twin(scenario_from_dict(SCENARIO_A)))
        assert a == b

    def test_prefix_property_of_sample_stream(self):
        # Drawing more samples must extend, not change, the existing stream:
        # confirms a single seeded generator sequence.
        link = LinkParams(range_m=8000.0)
        ch = ChannelParams(cn2=1e-15, pointing_jitter_rad=0.0)
        short = sample_received_power_dbm(link, ch, n_samples=100, seed=99).samples_dbm
        long = sample_received_power_dbm(link, ch, n_samples=500, seed=99).samples_dbm
        assert np.array_equal(short, long[:100])
