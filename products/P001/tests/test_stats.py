"""Unit tests for beamtwin.stats (fade probability, baseline, percentiles)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import norm

from beamtwin.budget import LinkParams, compute_budget
from beamtwin.channel import ChannelParams, build_channel_model, sample_received_power_dbm
from beamtwin.stats import (
    analytic_fade_probability_lognormal,
    fade_probability,
    margin_moments,
    margin_percentiles,
)


class TestFadeProbability:
    def test_counts_samples_below_threshold(self):
        s = np.array([-40.0, -35.0, -25.0, -20.0])
        est = fade_probability(s, -30.0)
        assert est.n_fades == 2
        assert est.probability == pytest.approx(0.5)

    def test_no_fades_gives_zero_with_positive_upper_bound(self):
        est = fade_probability(np.zeros(1000), -30.0)
        assert est.probability == 0.0
        # Wilson lower bound is analytically 0 at k=0; allow float rounding (~1e-19).
        assert est.ci_low == pytest.approx(0.0, abs=1e-12)
        assert est.ci_high > 0.0  # Wilson interval remains informative at k=0

    def test_all_fades_gives_one(self):
        est = fade_probability(np.full(500, -60.0), -30.0)
        assert est.probability == 1.0
        assert est.ci_high == 1.0

    def test_ci_brackets_estimate(self):
        rng = np.random.default_rng(0)
        s = rng.normal(-30.0, 3.0, size=20_000)
        est = fade_probability(s, -32.0)
        assert est.ci_low <= est.probability <= est.ci_high

    def test_ci_narrows_with_more_samples(self):
        rng = np.random.default_rng(1)
        widths = []
        for n in (1000, 10_000, 100_000):
            s = rng.normal(-30.0, 3.0, size=n)
            est = fade_probability(s, -33.0)
            widths.append(est.ci_high - est.ci_low)
        assert all(b < a for a, b in zip(widths, widths[1:]))

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            fade_probability(np.array([]), -30.0)

    def test_rejects_2d(self):
        with pytest.raises(ValueError):
            fade_probability(np.zeros((2, 3)), -30.0)

    def test_rejects_nan_threshold(self):
        with pytest.raises(ValueError):
            fade_probability(np.zeros(10), float("nan"))


class TestAnalyticBaseline:
    def test_zero_sigma_step_function(self):
        assert analytic_fade_probability_lognormal(-1.0, 0.0) == 1.0
        assert analytic_fade_probability_lognormal(1.0, 0.0) == 0.0

    def test_known_answer_from_formula(self):
        # P = Phi((ln(10^-M/10) + sigma^2/2)/sigma); M=10 dB, sigma=0.7195.
        m, s = 10.0, 0.7195
        expected = norm.cdf((-m * math.log(10) / 10 + 0.5 * s**2) / s)
        assert analytic_fade_probability_lognormal(m, s) == pytest.approx(expected)

    def test_zero_margin_gives_just_over_half(self):
        # At M = 0 the threshold is the mean irradiance; lognormal median is
        # below the mean, so P_fade > 0.5.
        p = analytic_fade_probability_lognormal(0.0, 0.7)
        assert 0.5 < p < 0.7

    def test_monotone_decreasing_in_margin(self):
        ps = [analytic_fade_probability_lognormal(m, 0.7) for m in (0.0, 5.0, 10.0, 15.0)]
        assert all(b < a for a, b in zip(ps, ps[1:]))

    def test_monotone_increasing_in_sigma_at_high_margin(self):
        ps = [analytic_fade_probability_lognormal(10.0, s) for s in (0.2, 0.4, 0.7, 1.0)]
        assert all(b > a for a, b in zip(ps, ps[1:]))

    def test_in_unit_interval(self):
        for m in (-20.0, 0.0, 20.0):
            for s in (0.1, 0.5, 1.2):
                assert 0.0 <= analytic_fade_probability_lognormal(m, s) <= 1.0

    def test_rejects_negative_sigma(self):
        with pytest.raises(ValueError):
            analytic_fade_probability_lognormal(10.0, -0.1)

    def test_rejects_nan_margin(self):
        with pytest.raises(ValueError):
            analytic_fade_probability_lognormal(float("nan"), 0.5)

    def test_matches_monte_carlo_in_scintillation_only_limit(self):
        # Validation V2a in miniature: analytic must land inside the MC CI.
        link = LinkParams(range_m=10_000.0, attenuation_db_per_km=2.5)
        p_rx = compute_budget(link).received_power_dbm
        import dataclasses

        link = dataclasses.replace(link, rx_sensitivity_dbm=p_rx - 6.0)
        ch = ChannelParams(cn2=5e-16, pointing_jitter_rad=0.0)
        res = sample_received_power_dbm(link, ch, n_samples=200_000, seed=2024)
        est = fade_probability(res.samples_dbm, link.rx_sensitivity_dbm)
        model = build_channel_model(link, ch)
        analytic = analytic_fade_probability_lognormal(
            compute_budget(link).margin_db, model.sigma_ln
        )
        assert est.ci_low <= analytic <= est.ci_high


class TestMarginStatistics:
    def test_percentiles_ordered(self):
        rng = np.random.default_rng(3)
        s = rng.normal(-20.0, 4.0, size=50_000)
        p = margin_percentiles(s, -30.0)
        vals = [p["p01"], p["p05"], p["p50"], p["p95"], p["p99"]]
        assert all(b >= a for a, b in zip(vals, vals[1:]))

    def test_percentiles_shift_with_sensitivity(self):
        s = np.full(1000, -20.0)
        p1 = margin_percentiles(s, -30.0)
        p2 = margin_percentiles(s, -40.0)
        assert p2["p50"] - p1["p50"] == pytest.approx(10.0)

    def test_percentile_keys(self):
        p = margin_percentiles(np.zeros(100), -30.0, percentiles=(10.0, 90.0))
        assert set(p) == {"p10", "p90"}

    def test_rejects_out_of_range_percentile(self):
        with pytest.raises(ValueError):
            margin_percentiles(np.zeros(10), -30.0, percentiles=(150.0,))

    def test_moments_known_answer(self):
        s = np.array([-30.0, -20.0, -10.0])  # margins 0, 10, 20 about -30
        m = margin_moments(s, -30.0)
        assert m["mean_db"] == pytest.approx(10.0)
        assert m["variance_db2"] == pytest.approx(200.0 / 3.0)
        assert m["std_db"] == pytest.approx(math.sqrt(200.0 / 3.0))

    def test_std_is_sqrt_variance(self):
        rng = np.random.default_rng(5)
        s = rng.normal(-25.0, 2.0, size=10_000)
        m = margin_moments(s, -30.0)
        assert m["std_db"] == pytest.approx(math.sqrt(m["variance_db2"]))

    def test_moments_reject_empty(self):
        with pytest.raises(ValueError):
            margin_moments(np.array([]), -30.0)
