"""Channel model: marginal, correlation, parameter relations, input validation.

Exercises REQ-01, REQ-02, REQ-03, REQ-04, REQ-05 of docs/REQUIREMENTS.md.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats

from codedfade.channel import (
    ChannelConfig,
    autocorrelation,
    correlated_gaussian,
    gamma_gamma_parameters_from_si,
    gamma_gamma_scintillation_index,
    generate_amplitude,
    generate_irradiance,
    measured_correlation_time,
    rytov_to_gamma_gamma,
)


class TestConfigValidation:
    def test_rejects_non_positive_scintillation_index(self) -> None:
        with pytest.raises(ValueError, match="scintillation_index"):
            ChannelConfig(0.0, 1e-3, 1e6)

    def test_rejects_non_positive_correlation_time(self) -> None:
        with pytest.raises(ValueError, match="correlation_time_s"):
            ChannelConfig(0.5, -1e-3, 1e6)

    def test_rejects_non_positive_sample_rate(self) -> None:
        with pytest.raises(ValueError, match="sample_rate_hz"):
            ChannelConfig(0.5, 1e-3, 0.0)

    def test_rejects_unknown_marginal(self) -> None:
        with pytest.raises(ValueError, match="marginal"):
            ChannelConfig(0.5, 1e-3, 1e6, marginal="rician")  # type: ignore[arg-type]

    def test_rejects_unknown_kernel(self) -> None:
        with pytest.raises(ValueError, match="kernel"):
            ChannelConfig(0.5, 1e-3, 1e6, kernel="brownian")  # type: ignore[arg-type]

    def test_rejects_undersampled_correlation(self) -> None:
        with pytest.raises(ValueError, match="resolve the correlation"):
            ChannelConfig(0.5, 1e-6, 1e6)

    def test_log_irradiance_variance_is_log1p_of_si(self) -> None:
        cfg = ChannelConfig(0.6, 2e-4, 1e6)
        assert cfg.log_irradiance_variance == pytest.approx(np.log(1.6))

    def test_samples_per_correlation_time(self) -> None:
        cfg = ChannelConfig(0.6, 2e-4, 1e6)
        assert cfg.samples_per_correlation_time == pytest.approx(200.0)


class TestCorrelatedGaussian:
    @pytest.mark.parametrize("kernel", ["exp", "gauss"])
    def test_unit_variance_and_zero_mean(self, kernel: str) -> None:
        g = correlated_gaussian(200_000, 50.0, np.random.default_rng(1), kernel)  # type: ignore[arg-type]
        assert abs(float(g.mean())) < 0.05
        assert float(g.std()) == pytest.approx(1.0, abs=0.05)

    def test_ar1_autocorrelation_matches_exp_kernel(self) -> None:
        """AR(1) with rho = exp(-1/Lc) reproduces R(k) = rho**k exactly in the mean."""
        lc = 20.0
        g = correlated_gaussian(400_000, lc, np.random.default_rng(2), "exp")
        acf = autocorrelation(g, 40)
        expected = np.exp(-np.arange(41) / lc)
        assert np.max(np.abs(acf - expected)) < 0.02

    def test_gauss_kernel_autocorrelation_is_gaussian(self) -> None:
        lc = 20.0
        g = correlated_gaussian(400_000, lc, np.random.default_rng(3), "gauss")
        acf = autocorrelation(g, 40)
        expected = np.exp(-((np.arange(41) / lc) ** 2))
        assert np.max(np.abs(acf - expected)) < 0.03

    def test_one_over_e_point_is_the_correlation_time(self) -> None:
        for kernel in ("exp", "gauss"):
            lc = 25.0
            g = correlated_gaussian(400_000, lc, np.random.default_rng(4), kernel)  # type: ignore[arg-type]
            acf = autocorrelation(g, 80)
            assert acf[int(lc)] == pytest.approx(np.exp(-1.0), abs=0.03)

    def test_rejects_bad_inputs(self) -> None:
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="n_samples"):
            correlated_gaussian(0, 10.0, rng)
        with pytest.raises(ValueError, match="samples_per_correlation_time"):
            correlated_gaussian(10, 0.0, rng)
        with pytest.raises(ValueError, match="kernel"):
            correlated_gaussian(10, 10.0, rng, "sinc")  # type: ignore[arg-type]

    def test_seed_determinism(self) -> None:
        a = correlated_gaussian(1000, 10.0, np.random.default_rng(7), "exp")
        b = correlated_gaussian(1000, 10.0, np.random.default_rng(7), "exp")
        assert np.array_equal(a, b)


class TestLognormalMarginal:
    def test_unit_mean_irradiance(self, config: ChannelConfig) -> None:
        i = generate_irradiance(config, 400_000)
        assert float(i.mean()) == pytest.approx(1.0, rel=0.03)

    def test_sample_scintillation_index_matches_target(self, config: ChannelConfig) -> None:
        i = generate_irradiance(config, 400_000)
        si = float(np.var(i) / np.mean(i) ** 2)
        assert si == pytest.approx(config.scintillation_index, rel=0.10)

    def test_log_irradiance_passes_ks_test_against_normal(
        self, config: ChannelConfig
    ) -> None:
        """Known-answer: ln I must be N(-s^2/2, s^2) with s^2 = ln(1+SI).

        The series is standardised with the model's own mean and variance and then
        tested against the standard normal, so the test checks the distribution
        shape and the model's claimed parameters at once.
        """
        i = generate_irradiance(config, 50_000)
        s2 = config.log_irradiance_variance
        # thin to roughly independent samples before the KS test
        thinned = np.log(i)[:: int(4 * config.samples_per_correlation_time)]
        z = (thinned + 0.5 * s2) / np.sqrt(s2)
        ks = stats.kstest(z, "norm")
        assert ks.pvalue > 0.01, f"KS p = {ks.pvalue}"

    def test_amplitude_is_sqrt_irradiance(self, config: ChannelConfig) -> None:
        a = generate_amplitude(config, 1000)
        i = generate_irradiance(config, 1000)
        assert np.allclose(a * a, i)

    def test_measured_correlation_time_recovers_the_target(
        self, config: ChannelConfig
    ) -> None:
        i = generate_irradiance(config, 400_000)
        tau = measured_correlation_time(i, config.sample_rate_hz)
        assert tau == pytest.approx(config.correlation_time_s, rel=0.15)


class TestGammaGamma:
    def test_scintillation_index_identity(self) -> None:
        """(6) with alpha,beta from (9) must equal exp(sx2+sy2)-1."""
        for sr2 in (0.05, 0.3, 1.0, 4.0, 16.0):
            alpha, beta = rytov_to_gamma_gamma(sr2)
            sx2 = 0.49 * sr2 / (1.0 + 1.11 * sr2 ** (6 / 5)) ** (7 / 6)
            sy2 = 0.51 * sr2 / (1.0 + 0.69 * sr2 ** (6 / 5)) ** (5 / 6)
            assert gamma_gamma_scintillation_index(alpha, beta) == pytest.approx(
                np.expm1(sx2 + sy2), rel=1e-12
            )

    def test_parameter_inversion_round_trip(self) -> None:
        for si in (0.1, 0.5, 1.0, 1.2):
            alpha, beta = gamma_gamma_parameters_from_si(si)
            assert gamma_gamma_scintillation_index(alpha, beta) == pytest.approx(
                si, rel=1e-6
            )

    def test_inversion_rejects_unreachable_target(self) -> None:
        """The plane-wave scintillation index saturates, so a target above the peak
        has no solution and must be refused rather than solved on the wrong branch."""
        with pytest.raises(ValueError, match="outside the reachable plane-wave"):
            gamma_gamma_parameters_from_si(1e-9)
        with pytest.raises(ValueError, match="saturates"):
            gamma_gamma_parameters_from_si(5.0)
        with pytest.raises(ValueError, match="scintillation_index must be > 0"):
            gamma_gamma_parameters_from_si(0.0)

    def test_the_peak_is_located_on_the_increasing_branch(self) -> None:
        from codedfade.channel import GAMMA_GAMMA_SI_PEAK

        sr2, si = GAMMA_GAMMA_SI_PEAK
        assert 1.0 < sr2 < 100.0
        assert si == pytest.approx(
            gamma_gamma_scintillation_index(*rytov_to_gamma_gamma(sr2))
        )
        for other in (sr2 * 0.5, sr2 * 2.0):
            assert gamma_gamma_scintillation_index(*rytov_to_gamma_gamma(other)) < si

    def test_rejects_non_positive_rytov(self) -> None:
        with pytest.raises(ValueError, match="rytov_variance"):
            rytov_to_gamma_gamma(0.0)

    def test_rejects_non_positive_shape(self) -> None:
        with pytest.raises(ValueError, match="alpha and beta"):
            gamma_gamma_scintillation_index(0.0, 1.0)

    def test_sample_marginal_matches_target_si(self) -> None:
        cfg = ChannelConfig(0.8, 2e-4, 1e6, marginal="gammagamma", seed=5)
        i = generate_irradiance(cfg, 300_000)
        assert float(i.mean()) == pytest.approx(1.0, rel=0.05)
        si = float(np.var(i) / np.mean(i) ** 2)
        assert si == pytest.approx(0.8, rel=0.15)


class TestAutocorrelation:
    def test_lag_zero_is_one(self) -> None:
        x = np.random.default_rng(0).standard_normal(1000)
        assert autocorrelation(x, 10)[0] == pytest.approx(1.0)

    def test_rejects_bad_lag(self) -> None:
        x = np.zeros(10)
        with pytest.raises(ValueError, match="max_lag"):
            autocorrelation(x, 0)
        with pytest.raises(ValueError, match="max_lag"):
            autocorrelation(x, 10)

    def test_rejects_non_1d(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            autocorrelation(np.zeros((4, 4)), 2)

    def test_measured_correlation_time_rejects_non_positive(self) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            measured_correlation_time(np.array([1.0, -1.0, 2.0]), 1.0)

    def test_measured_correlation_time_returns_nan_when_not_reached(self) -> None:
        x = np.ones(100) + 1e-9 * np.arange(100)
        assert np.isnan(measured_correlation_time(x, 1.0, max_lag=5))


@settings(max_examples=25, deadline=None)
@given(
    si=st.floats(min_value=0.05, max_value=2.0),
    lc=st.floats(min_value=4.0, max_value=200.0),
    seed=st.integers(min_value=0, max_value=1000),
)
def test_lognormal_irradiance_is_always_positive(si: float, lc: float, seed: int) -> None:
    cfg = ChannelConfig(si, lc / 1e6, 1e6, seed=seed)
    i = generate_irradiance(cfg, 2000)
    assert np.all(i > 0)
    assert np.all(np.isfinite(i))
