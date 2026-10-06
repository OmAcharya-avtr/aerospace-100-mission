"""Tests for the correlated fading channel."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from acmpilot.channel import (
    ChannelConfig,
    correlation_time_s,
    fade_statistics,
    gamma_gamma_irradiance,
    gamma_gamma_shapes,
    gauss_markov_path,
    irradiance_path,
    lognormal_irradiance,
    rice_level_crossing_rate_hz,
    snr_db_path,
)


class TestGaussMarkovPath:
    def test_length_and_finiteness(self, rng):
        path = gauss_markov_path(500, dt_s=1e-3, tau_c_s=10e-3, rng=rng)
        assert path.shape == (500,)
        assert np.all(np.isfinite(path))

    def test_known_answer_rho_equals_one_over_e(self):
        # dt == tau_c, so rho = exp(-1) = 0.36787944117144233 by hand.
        cfg = ChannelConfig(slot_s=1e-3, tau_c_s=1e-3)
        assert cfg.rho == pytest.approx(0.36787944117144233, rel=1e-12)

    def test_stationary_variance_and_lag1(self):
        rng = np.random.default_rng(5)
        path = gauss_markov_path(200_000, dt_s=1e-3, tau_c_s=5e-3, rng=rng)
        centred = path - path.mean()
        lag1 = float(np.dot(centred[:-1], centred[1:]) / np.dot(centred, centred))
        assert path.var() == pytest.approx(1.0, abs=0.02)
        assert lag1 == pytest.approx(float(np.exp(-0.2)), abs=0.01)

    def test_reproducible_from_seed(self):
        a = gauss_markov_path(100, dt_s=1e-3, tau_c_s=1e-2, rng=np.random.default_rng(7))
        b = gauss_markov_path(100, dt_s=1e-3, tau_c_s=1e-2, rng=np.random.default_rng(7))
        np.testing.assert_array_equal(a, b)

    def test_long_tau_c_approaches_constant(self, rng):
        path = gauss_markov_path(2000, dt_s=1e-6, tau_c_s=1.0, rng=rng)
        assert float(np.max(np.abs(np.diff(path)))) < 0.1

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"n_samples": 0}, "n_samples"),
            ({"dt_s": 0.0}, "dt_s"),
            ({"dt_s": -1.0}, "dt_s"),
            ({"tau_c_s": 0.0}, "tau_c_s"),
            ({"tau_c_s": -1.0}, "tau_c_s"),
        ],
    )
    def test_rejects_invalid_input(self, rng, kwargs, message):
        args = {"n_samples": 10, "dt_s": 1e-3, "tau_c_s": 1e-2, "rng": rng}
        args.update(kwargs)
        with pytest.raises(ValueError, match=message):
            gauss_markov_path(**args)

    @given(
        n=st.integers(min_value=8, max_value=400),
        tau_ratio=st.floats(min_value=0.05, max_value=50.0),
    )
    @settings(max_examples=25, deadline=None)
    def test_property_finite_and_right_length(self, n, tau_ratio):
        path = gauss_markov_path(
            n, dt_s=1e-3, tau_c_s=1e-3 * tau_ratio, rng=np.random.default_rng(3)
        )
        assert path.size == n
        assert np.all(np.isfinite(path))


class TestLognormalIrradiance:
    def test_known_answer_sigma_chi_squared(self):
        # sigma_i2 = exp(4 sigma_chi^2) - 1; choosing sigma_i2 = 1 gives
        # sigma_chi^2 = ln(2)/4 = 0.1732867951399863 by hand. The irradiance at
        # driver value 0 is then exp(-2 * 0.1732868) = exp(-0.3465736) = 0.7071068.
        value = lognormal_irradiance(np.zeros(1), sigma_i2=1.0)
        assert float(value[0]) == pytest.approx(0.7071067811865476, rel=1e-12)

    def test_unit_mean(self):
        rng = np.random.default_rng(11)
        driver = gauss_markov_path(400_000, dt_s=1e-3, tau_c_s=2e-3, rng=rng)
        irradiance = lognormal_irradiance(driver, sigma_i2=0.5)
        assert float(irradiance.mean()) == pytest.approx(1.0, abs=0.02)

    def test_scintillation_index_recovered(self):
        rng = np.random.default_rng(13)
        driver = gauss_markov_path(400_000, dt_s=1e-3, tau_c_s=2e-3, rng=rng)
        irradiance = lognormal_irradiance(driver, sigma_i2=0.5)
        measured = float(irradiance.var() / irradiance.mean() ** 2)
        assert measured == pytest.approx(0.5, rel=0.05)

    def test_strictly_positive(self, rng):
        driver = gauss_markov_path(1000, dt_s=1e-3, tau_c_s=1e-2, rng=rng)
        assert np.all(lognormal_irradiance(driver, sigma_i2=2.0) > 0.0)

    @pytest.mark.parametrize("bad", [0.0, -0.1])
    def test_rejects_non_positive_sigma_i2(self, bad):
        with pytest.raises(ValueError, match="sigma_i2"):
            lognormal_irradiance(np.zeros(4), sigma_i2=bad)


class TestGammaGamma:
    def test_known_answer_equal_shapes(self):
        # ratio = 1 gives alpha = beta = b with sigma_i2 = 2/b + 1/b^2.
        # sigma_i2 = 3 gives u^2 + 2u - 3 = 0 with u = 1/b, so u = 1, b = 1.
        alpha, beta = gamma_gamma_shapes(3.0, ratio=1.0)
        assert alpha == pytest.approx(1.0, rel=1e-12)
        assert beta == pytest.approx(1.0, rel=1e-12)

    def test_known_answer_ratio_four(self):
        # ratio = 4, sigma_i2 = 1.5: 0.25 u^2 + 1.25 u - 1.5 = 0, u = 1, beta = 1,
        # alpha = 4, both by hand.
        alpha, beta = gamma_gamma_shapes(1.5, ratio=4.0)
        assert (alpha, beta) == pytest.approx((4.0, 1.0), rel=1e-12)

    @pytest.mark.parametrize("sigma_i2", [0.1, 0.5, 1.0, 2.0, 5.0])
    @pytest.mark.parametrize("ratio", [1.0, 2.0, 4.0, 10.0])
    def test_shapes_satisfy_equation_7(self, sigma_i2, ratio):
        alpha, beta = gamma_gamma_shapes(sigma_i2, ratio=ratio)
        reconstructed = 1.0 / alpha + 1.0 / beta + 1.0 / (alpha * beta)
        assert reconstructed == pytest.approx(sigma_i2, rel=1e-10)

    def test_alpha_greater_than_beta(self):
        alpha, beta = gamma_gamma_shapes(0.8, ratio=4.0)
        assert alpha > beta > 0.0

    def test_unit_mean_and_scintillation_index(self):
        rng = np.random.default_rng(17)
        big = gauss_markov_path(300_000, dt_s=1e-3, tau_c_s=2e-3, rng=rng)
        small = gauss_markov_path(300_000, dt_s=1e-3, tau_c_s=2e-3, rng=rng)
        irradiance = gamma_gamma_irradiance(big, small, sigma_i2=0.7)
        assert float(irradiance.mean()) == pytest.approx(1.0, abs=0.03)
        measured = float(irradiance.var() / irradiance.mean() ** 2)
        assert measured == pytest.approx(0.7, rel=0.08)

    def test_rejects_mismatched_driver_shapes(self):
        with pytest.raises(ValueError, match="driver shapes"):
            gamma_gamma_irradiance(np.zeros(5), np.zeros(6), sigma_i2=0.5)

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_rejects_non_positive_sigma_i2(self, bad):
        with pytest.raises(ValueError, match="sigma_i2"):
            gamma_gamma_shapes(bad)

    def test_rejects_ratio_below_one(self):
        with pytest.raises(ValueError, match="ratio"):
            gamma_gamma_shapes(0.5, ratio=0.5)

    @given(sigma_i2=st.floats(min_value=0.02, max_value=8.0))
    @settings(max_examples=30, deadline=None)
    def test_property_shape_solution_round_trips(self, sigma_i2):
        alpha, beta = gamma_gamma_shapes(sigma_i2, ratio=4.0)
        assert 1.0 / alpha + 1.0 / beta + 1.0 / (alpha * beta) == pytest.approx(
            sigma_i2, rel=1e-9
        )


class TestChannelConfig:
    def test_defaults(self):
        cfg = ChannelConfig()
        assert cfg.slot_s == 1e-3
        assert cfg.tau_c_s == 10e-3
        assert cfg.marginal == "lognormal"
        assert cfg.detector_exponent == 1.0

    def test_delay_slots_rounds(self):
        cfg = ChannelConfig(slot_s=1e-3)
        assert cfg.delay_slots(0.0) == 0
        assert cfg.delay_slots(10e-3) == 10
        assert cfg.delay_slots(10.4e-3) == 10
        assert cfg.delay_slots(10.6e-3) == 11

    def test_delay_slots_rejects_negative(self):
        with pytest.raises(ValueError, match="tau_s"):
            ChannelConfig().delay_slots(-1e-3)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"slot_s": 0.0}, "slot_s"),
            ({"tau_c_s": -1.0}, "tau_c_s"),
            ({"sigma_i2": 0.0}, "sigma_i2"),
            ({"marginal": "rayleigh"}, "marginal"),
            ({"detector_exponent": 0.0}, "detector_exponent"),
        ],
    )
    def test_rejects_invalid_config(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            ChannelConfig(**kwargs)

    def test_rho_matches_equation_2(self):
        cfg = ChannelConfig(slot_s=2e-3, tau_c_s=8e-3)
        assert cfg.rho == pytest.approx(float(np.exp(-0.25)), rel=1e-12)


class TestPaths:
    def test_irradiance_path_positive_both_marginals(self):
        for marginal in ("lognormal", "gamma-gamma"):
            cfg = ChannelConfig(marginal=marginal)
            assert np.all(irradiance_path(cfg, 2000, 3) > 0.0)

    def test_snr_path_reproducible(self, config):
        a = snr_db_path(config, 500, 42)
        b = snr_db_path(config, 500, 42)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self, config):
        assert not np.array_equal(
            snr_db_path(config, 500, 1), snr_db_path(config, 500, 2)
        )

    def test_detector_exponent_doubles_db_swing(self):
        one = snr_db_path(ChannelConfig(detector_exponent=1.0), 4000, 9)
        two = snr_db_path(ChannelConfig(detector_exponent=2.0), 4000, 9)
        deviation_one = one - 14.0
        deviation_two = two - 14.0
        np.testing.assert_allclose(deviation_two, 2.0 * deviation_one, rtol=1e-12)

    def test_gamma_gamma_correlation_time_shorter_than_driver(self):
        cfg = ChannelConfig(marginal="gamma-gamma", tau_c_s=10e-3, sigma_i2=1.0)
        irradiance = irradiance_path(cfg, 200_000, 21)
        measured = correlation_time_s(irradiance, dt_s=cfg.slot_s)
        assert measured < cfg.tau_c_s

    def test_lognormal_db_correlation_time_recovers_tau_c(self, config):
        snr = snr_db_path(config, 300_000, 23)
        measured = correlation_time_s(snr, dt_s=config.slot_s)
        assert measured == pytest.approx(config.tau_c_s, rel=0.06)


class TestCorrelationTime:
    def test_rejects_short_series(self):
        with pytest.raises(ValueError, match="8 samples"):
            correlation_time_s(np.zeros(4), dt_s=1e-3)

    def test_rejects_constant_series(self):
        with pytest.raises(ValueError, match="constant"):
            correlation_time_s(np.ones(32), dt_s=1e-3)

    def test_rejects_bad_dt(self):
        with pytest.raises(ValueError, match="dt_s"):
            correlation_time_s(np.arange(32.0), dt_s=0.0)

    def test_white_noise_has_sub_slot_correlation_time(self):
        white = np.random.default_rng(1).standard_normal(20_000)
        assert correlation_time_s(white, dt_s=1e-3) < 1.1e-3


class TestFadeStatistics:
    def test_known_answer_hand_built_series(self):
        # Series in dB: [10, 0, 0, 10, 0, 10] against threshold 5 dB, dt = 1 ms.
        # Below-threshold samples: indices 1, 2, 4 -> outage fraction 3/6 = 0.5.
        # Runs: [1,2] (length 2, complete, both ends interior) and [4] (length 1,
        # complete). Downward crossings: 0->1 and 3->4, so 2 crossings in 6 ms
        # -> 333.333... Hz. Mean complete fade duration = (2 + 1)/2 = 1.5 ms.
        series = np.array([10.0, 0.0, 0.0, 10.0, 0.0, 10.0])
        stats = fade_statistics(series, 5.0, dt_s=1e-3)
        assert stats["outage_fraction"] == pytest.approx(0.5)
        assert stats["n_fades"] == 2
        assert stats["mean_fade_duration_s"] == pytest.approx(1.5e-3)
        assert stats["max_fade_duration_s"] == pytest.approx(2e-3)
        assert stats["median_fade_duration_s"] == pytest.approx(1.5e-3)
        assert stats["level_crossing_rate_hz"] == pytest.approx(2.0 / 6e-3)

    def test_censored_runs_excluded_from_durations(self):
        # Both runs touch an end, so no complete fade exists.
        series = np.array([0.0, 0.0, 10.0, 10.0, 0.0, 0.0])
        stats = fade_statistics(series, 5.0, dt_s=1e-3)
        assert stats["n_fades"] == 0
        assert np.isnan(stats["mean_fade_duration_s"])
        assert stats["outage_fraction"] == pytest.approx(4.0 / 6.0)

    def test_never_below_threshold(self):
        stats = fade_statistics(np.full(20, 10.0), 0.0, dt_s=1e-3)
        assert stats["outage_fraction"] == 0.0
        assert stats["level_crossing_rate_hz"] == 0.0
        assert np.isnan(stats["max_fade_duration_s"])

    def test_always_below_threshold(self):
        stats = fade_statistics(np.zeros(20), 10.0, dt_s=1e-3)
        assert stats["outage_fraction"] == 1.0
        assert stats["n_fades"] == 0

    def test_rejects_short_series(self):
        with pytest.raises(ValueError, match="2 samples"):
            fade_statistics(np.zeros(1), 0.0, dt_s=1e-3)

    def test_rejects_bad_dt(self):
        with pytest.raises(ValueError, match="dt_s"):
            fade_statistics(np.zeros(10), 0.0, dt_s=-1.0)

    def test_duration_times_rate_equals_outage(self, config):
        snr = snr_db_path(config, 200_000, 31)
        threshold = float(np.quantile(snr, 0.1))
        stats = fade_statistics(snr, threshold, dt_s=config.slot_s)
        product = stats["mean_fade_duration_s"] * stats["level_crossing_rate_hz"]
        assert product == pytest.approx(stats["outage_fraction"], rel=1e-6)

    def test_mean_fade_duration_grows_with_tau_c(self):
        previous = 0.0
        for tau_c_ms in (2.5, 5.0, 10.0, 20.0):
            cfg = ChannelConfig(tau_c_s=tau_c_ms * 1e-3, sigma_i2=0.5)
            snr = snr_db_path(cfg, 200_000, 37)
            threshold = float(np.quantile(snr, 0.1))
            value = fade_statistics(snr, threshold, dt_s=cfg.slot_s)[
                "mean_fade_duration_s"
            ]
            assert value > previous
            previous = value


class TestRiceCrossingRate:
    def test_matches_measured_within_three_percent(self, config):
        snr = snr_db_path(config, 400_000, 41)
        for threshold in (8.0, 10.0, 12.0):
            measured = fade_statistics(snr, threshold, dt_s=config.slot_s)[
                "level_crossing_rate_hz"
            ]
            analytic = rice_level_crossing_rate_hz(
                sigma_i2=config.sigma_i2,
                tau_c_s=config.tau_c_s,
                threshold_db=threshold,
                mean_snr_db=config.mean_snr_db,
                slot_s=config.slot_s,
            )
            assert measured == pytest.approx(analytic, rel=0.03)

    def test_rate_falls_as_threshold_falls(self, config):
        rates = [
            rice_level_crossing_rate_hz(
                sigma_i2=config.sigma_i2, tau_c_s=config.tau_c_s, threshold_db=t,
                mean_snr_db=config.mean_snr_db, slot_s=config.slot_s,
            )
            for t in (4.0, 6.0, 8.0, 10.0)
        ]
        assert rates == sorted(rates)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"sigma_i2": 0.0}, "sigma_i2"),
            ({"tau_c_s": 0.0}, "tau_c_s"),
            ({"slot_s": 0.0}, "slot_s"),
        ],
    )
    def test_rejects_invalid_input(self, kwargs, message):
        args = {
            "sigma_i2": 0.5, "tau_c_s": 1e-2, "threshold_db": 8.0,
            "mean_snr_db": 14.0, "slot_s": 1e-3,
        }
        args.update(kwargs)
        with pytest.raises(ValueError, match=message):
            rice_level_crossing_rate_hz(**args)
