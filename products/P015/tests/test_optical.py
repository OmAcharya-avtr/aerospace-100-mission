"""Unit / known-answer / property tests for linkswitch.optical."""

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from linkswitch.optical import (
    OpticalParams,
    gamma_gamma_sigma_i2,
    irradiance_threshold_from_margin_db,
    lognormal_sigma_z,
    sample_gamma_gamma_irradiance,
    sample_lognormal_irradiance,
    simulate_ar1_log_irradiance,
    validate_gamma_gamma_params,
    validate_sigma_i2,
)


class TestLognormalSigmaZ:
    def test_known_answer_sigma_i2_point_25(self):
        # sigma_z^2 = ln(1 + 0.25) = ln(1.25) = 0.22314355131420976...
        # sigma_z = sqrt(0.22314355131420976) = 0.47238072707743883...
        assert lognormal_sigma_z(0.25) == pytest.approx(0.47238072707743883, rel=1e-12)

    def test_known_answer_sigma_i2_point_1(self):
        # sigma_z^2 = ln(1.1) = 0.09531017980432486
        assert lognormal_sigma_z(0.1) == pytest.approx(math.sqrt(math.log(1.1)), rel=1e-12)

    def test_zero_sigma_i2_rejected(self):
        with pytest.raises(ValueError):
            lognormal_sigma_z(0.0)

    def test_negative_sigma_i2_rejected(self):
        with pytest.raises(ValueError):
            lognormal_sigma_z(-0.1)

    def test_nan_rejected(self):
        with pytest.raises(ValueError):
            lognormal_sigma_z(float("nan"))

    def test_above_weak_limit_warns(self):
        with pytest.warns(UserWarning, match="weak-fluctuation"):
            lognormal_sigma_z(1.5)

    def test_at_weak_limit_no_warning(self):
        with _no_warning_ctx():
            validate_sigma_i2(1.0)


def _no_warning_ctx():
    import warnings

    class _Ctx:
        def __enter__(self):
            warnings.simplefilter("error")
            return self

        def __exit__(self, *exc):
            warnings.resetwarnings()
            return False

    return _Ctx()


class TestThresholdFromMargin:
    def test_known_answer_zero_margin(self):
        # 0 dB margin -> tau = 10^0 = 1 (outage only exactly at the mean)
        assert irradiance_threshold_from_margin_db(0.0) == pytest.approx(1.0)

    def test_known_answer_10db(self):
        # 10 dB -> tau = 10^(-1) = 0.1 exactly (by definition of dB)
        assert irradiance_threshold_from_margin_db(10.0) == pytest.approx(0.1, rel=1e-12)

    def test_known_answer_6db(self):
        # 6 dB -> tau = 10^(-0.6) = 0.251188643150958...
        assert irradiance_threshold_from_margin_db(6.0) == pytest.approx(
            0.251188643150958, rel=1e-9
        )

    def test_negative_margin_rejected(self):
        with pytest.raises(ValueError):
            irradiance_threshold_from_margin_db(-1.0)

    def test_monotone_decreasing_in_margin(self):
        margins = np.array([0.0, 3.0, 6.0, 9.0, 12.0])
        taus = np.array([irradiance_threshold_from_margin_db(m) for m in margins])
        assert np.all(np.diff(taus) < 0.0)


class TestSampleLognormalIrradiance:
    def test_mean_is_one(self):
        rng = np.random.default_rng(0)
        samples = sample_lognormal_irradiance(rng, 500_000, sigma_i2=0.3)
        assert samples.mean() == pytest.approx(1.0, abs=0.01)

    def test_variance_matches_sigma_i2(self):
        rng = np.random.default_rng(1)
        sigma_i2 = 0.3
        samples = sample_lognormal_irradiance(rng, 1_000_000, sigma_i2=sigma_i2)
        # Var[I] = E[I]^2 * sigma_I^2 = sigma_i2 since E[I]=1
        assert samples.var() == pytest.approx(sigma_i2, rel=0.03)

    def test_all_positive(self):
        rng = np.random.default_rng(2)
        samples = sample_lognormal_irradiance(rng, 1000, sigma_i2=0.2)
        assert np.all(samples > 0.0)

    def test_seeded_reproducibility(self):
        s1 = sample_lognormal_irradiance(np.random.default_rng(42), 100, 0.2)
        s2 = sample_lognormal_irradiance(np.random.default_rng(42), 100, 0.2)
        np.testing.assert_array_equal(s1, s2)


class TestAR1LogIrradiance:
    def test_length(self):
        rng = np.random.default_rng(0)
        out = simulate_ar1_log_irradiance(rng, 100, sigma_i2=0.2, coherence_steps=5.0)
        assert out.shape == (100,)

    def test_all_positive(self):
        rng = np.random.default_rng(0)
        out = simulate_ar1_log_irradiance(rng, 500, sigma_i2=0.2, coherence_steps=5.0)
        assert np.all(out > 0.0)

    def test_marginal_mean_matches_lognormal(self):
        # A long AR(1) run should have the same marginal mean as i.i.d. draws
        # (stationary distribution is the same lognormal by construction).
        rng = np.random.default_rng(7)
        out = simulate_ar1_log_irradiance(rng, 200_000, sigma_i2=0.25, coherence_steps=4.0)
        assert out.mean() == pytest.approx(1.0, abs=0.02)

    def test_lag1_autocorrelation_matches_rho(self):
        # rho = exp(-1/coherence_steps); measure the lag-1 correlation of the
        # standardised log-series and compare to the design value.
        coherence_steps = 6.0
        rho_expected = math.exp(-1.0 / coherence_steps)
        rng = np.random.default_rng(3)
        out = simulate_ar1_log_irradiance(rng, 300_000, sigma_i2=0.25, coherence_steps=coherence_steps)
        z = np.log(out)
        z = (z - z.mean()) / z.std()
        rho_hat = np.corrcoef(z[:-1], z[1:])[0, 1]
        assert rho_hat == pytest.approx(rho_expected, abs=0.02)

    def test_high_coherence_gives_slowly_varying_series(self):
        rng = np.random.default_rng(0)
        slow = simulate_ar1_log_irradiance(rng, 200, sigma_i2=0.2, coherence_steps=1000.0)
        rng2 = np.random.default_rng(0)
        fast = simulate_ar1_log_irradiance(rng2, 200, sigma_i2=0.2, coherence_steps=0.01)
        # Mean absolute step-to-step change should be far smaller for the
        # slowly-decorrelating (high coherence) series.
        assert np.mean(np.abs(np.diff(slow))) < np.mean(np.abs(np.diff(fast)))

    def test_seeded_reproducibility(self):
        a = simulate_ar1_log_irradiance(np.random.default_rng(5), 50, 0.2, 4.0)
        b = simulate_ar1_log_irradiance(np.random.default_rng(5), 50, 0.2, 4.0)
        np.testing.assert_array_equal(a, b)

    def test_n_steps_must_be_positive(self):
        with pytest.raises(ValueError):
            simulate_ar1_log_irradiance(np.random.default_rng(0), 0, 0.2, 4.0)

    def test_coherence_steps_must_be_positive(self):
        with pytest.raises(ValueError):
            simulate_ar1_log_irradiance(np.random.default_rng(0), 10, 0.2, 0.0)

    def test_n_steps_must_be_int_not_float(self):
        with pytest.raises(ValueError):
            simulate_ar1_log_irradiance(np.random.default_rng(0), 10.5, 0.2, 4.0)


class TestGammaGamma:
    def test_known_answer_sigma_i2(self):
        # alpha=4, beta=2: 1/4 + 1/2 + 1/8 = 0.875
        assert gamma_gamma_sigma_i2(4.0, 2.0) == pytest.approx(0.875, rel=1e-12)

    def test_symmetric_in_alpha_beta(self):
        assert gamma_gamma_sigma_i2(3.0, 7.0) == pytest.approx(gamma_gamma_sigma_i2(7.0, 3.0))

    def test_validate_rejects_nonpositive(self):
        with pytest.raises(ValueError):
            validate_gamma_gamma_params(0.0, 2.0)
        with pytest.raises(ValueError):
            validate_gamma_gamma_params(2.0, -1.0)

    def test_sample_mean_is_one(self):
        rng = np.random.default_rng(0)
        samples = sample_gamma_gamma_irradiance(rng, 500_000, alpha=4.0, beta=6.0)
        assert samples.mean() == pytest.approx(1.0, rel=0.02)

    def test_sample_variance_matches_formula(self):
        rng = np.random.default_rng(1)
        alpha, beta = 5.0, 8.0
        samples = sample_gamma_gamma_irradiance(rng, 1_000_000, alpha=alpha, beta=beta)
        expected_var = gamma_gamma_sigma_i2(alpha, beta)  # E[I]=1 so Var = sigma_I^2
        assert samples.var() == pytest.approx(expected_var, rel=0.05)

    def test_sample_all_positive(self):
        rng = np.random.default_rng(2)
        samples = sample_gamma_gamma_irradiance(rng, 1000, alpha=2.0, beta=3.0)
        assert np.all(samples > 0.0)

    @given(alpha=st.floats(0.5, 20.0), beta=st.floats(0.5, 20.0))
    @settings(max_examples=50)
    def test_property_sigma_i2_always_positive(self, alpha, beta):
        assert gamma_gamma_sigma_i2(alpha, beta) > 0.0


class TestOpticalParams:
    def test_defaults_construct(self):
        p = OpticalParams()
        assert p.sigma_i2 > 0
        assert p.tau_phys < 1.0  # margin_db > 0 => threshold below the mean

    def test_tau_phys_consistent_with_helper(self):
        p = OpticalParams(margin_db=4.0)
        assert p.tau_phys == pytest.approx(irradiance_threshold_from_margin_db(4.0))

    def test_sigma_z_consistent_with_helper(self):
        p = OpticalParams(sigma_i2=0.4)
        assert p.sigma_z == pytest.approx(lognormal_sigma_z(0.4))

    def test_invalid_fading_model_rejected(self):
        with pytest.raises(ValueError):
            OpticalParams(fading_model="rician")

    def test_invalid_rate_rejected(self):
        with pytest.raises(ValueError):
            OpticalParams(rate_mbps=0.0)

    def test_invalid_coherence_rejected(self):
        with pytest.raises(ValueError):
            OpticalParams(coherence_steps=-1.0)

    def test_frozen_dataclass_is_immutable(self):
        p = OpticalParams()
        with pytest.raises(Exception):
            p.sigma_i2 = 0.9
