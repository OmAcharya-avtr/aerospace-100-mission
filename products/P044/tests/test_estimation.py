"""Tests for the channel-state estimation error model."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from aperturediv.channel import lognormal_sigma_log
from aperturediv.estimation import (
    DB_PER_NEPER,
    estimate_from_log_error,
    estimate_stale_noisy,
    log_error_sigma,
    log_error_sigma_db,
    measurement_sigma_for_target,
)


class TestLogErrorSigma:
    def test_fresh_and_clean_is_zero(self):
        assert log_error_sigma(0.6, 1.0, 0.0) == pytest.approx(0.0, abs=1e-15)

    def test_pure_measurement_noise(self):
        assert log_error_sigma(0.6, 1.0, 0.7) == pytest.approx(0.7, rel=1e-14)

    def test_pure_staleness_known_answer(self):
        # si = 1 -> s^2 = ln 2. rho_t = 0.5 -> sigma_e = sqrt(2 ln2 * 0.5) = sqrt(ln 2).
        assert log_error_sigma(1.0, 0.5, 0.0) == pytest.approx(np.sqrt(np.log(2.0)), rel=1e-12)

    def test_fully_decorrelated_known_answer(self):
        # rho_t = 0 -> sigma_e = sqrt(2) s.
        s = lognormal_sigma_log(0.6)
        assert log_error_sigma(0.6, 0.0, 0.0) == pytest.approx(np.sqrt(2.0) * s, rel=1e-12)

    def test_increases_as_estimate_goes_stale(self):
        vals = [log_error_sigma(0.6, r, 0.2) for r in (1.0, 0.9, 0.5, 0.0)]
        assert all(y > x for x, y in zip(vals, vals[1:], strict=False))

    @given(st.floats(0.01, 3.0), st.floats(-1.0, 1.0), st.floats(0.0, 2.0))
    def test_quadrature_sum(self, si, rho, sm):
        s2 = lognormal_sigma_log(si) ** 2
        want = np.sqrt(2.0 * s2 * (1.0 - rho) + sm * sm)
        assert log_error_sigma(si, rho, sm) == pytest.approx(want, rel=1e-12)

    @pytest.mark.parametrize("rho", [-1.1, 1.1])
    def test_rejects_bad_correlation(self, rho):
        with pytest.raises(ValueError, match="temporal_correlation"):
            log_error_sigma(0.6, rho, 0.0)

    @pytest.mark.parametrize("sm", [-0.1, float("nan")])
    def test_rejects_bad_measurement_sigma(self, sm):
        with pytest.raises(ValueError, match="measurement_sigma"):
            log_error_sigma(0.6, 1.0, sm)


class TestDbConversion:
    def test_constant_value(self):
        # 10 / ln 10 = 4.342944819032518 dB per natural log unit.
        assert DB_PER_NEPER == pytest.approx(4.342944819032518, rel=1e-14)

    def test_round_number(self):
        assert log_error_sigma_db(1.0) == pytest.approx(4.342944819032518, rel=1e-14)

    def test_linear(self):
        assert log_error_sigma_db(2.0) == pytest.approx(2.0 * log_error_sigma_db(1.0), rel=1e-14)


class TestMeasurementSigmaForTarget:
    def test_inverts_log_error_sigma(self):
        for rho in (1.0, 0.95, 0.8):
            sm = measurement_sigma_for_target(0.9, rho, 1.0)
            assert log_error_sigma(0.9, rho, sm) == pytest.approx(1.0, rel=1e-12)

    def test_fresh_estimate_needs_the_whole_budget(self):
        assert measurement_sigma_for_target(0.9, 1.0, 0.8) == pytest.approx(0.8, rel=1e-14)

    def test_raises_when_staleness_exceeds_target(self):
        with pytest.raises(ValueError, match="staleness alone"):
            measurement_sigma_for_target(0.9, 0.0, 0.05)

    def test_rejects_negative_target(self):
        with pytest.raises(ValueError, match="target must be"):
            measurement_sigma_for_target(0.9, 1.0, -1.0)


class TestEstimateFromLogError:
    def test_zero_error_reproduces_truth(self, rng):
        truth = np.array([[1.0, 2.0], [0.5, 0.25]])
        est = estimate_from_log_error(truth, 0.0, rng)
        assert np.allclose(est.irradiance_estimated, truth, rtol=1e-14)
        assert np.allclose(est.sigma_e, 0.0)

    def test_conditionally_unbiased(self, rng):
        truth = np.full((400_000, 1), 2.0)
        est = estimate_from_log_error(truth, 0.8, rng)
        assert est.irradiance_estimated.mean() == pytest.approx(2.0, rel=0.01)

    def test_log_error_variance_matches_sigma_e(self, rng):
        truth = np.full((200_000, 1), 1.0)
        est = estimate_from_log_error(truth, 0.7, rng)
        log_err = np.log(est.irradiance_estimated) - np.log(truth)
        assert log_err.std() == pytest.approx(0.7, rel=0.02)

    def test_per_row_sigma_e(self, rng):
        truth = np.ones((6, 2))
        sig = np.linspace(0.0, 1.0, 6)
        est = estimate_from_log_error(truth, sig, rng)
        assert np.allclose(est.sigma_e, sig)
        assert np.allclose(est.irradiance_estimated[0], 1.0, rtol=1e-14)

    def test_amplitude_and_db_properties(self, rng):
        est = estimate_from_log_error(np.full((3, 2), 4.0), 0.0, rng)
        assert np.allclose(est.amplitude_true, 2.0)
        assert np.allclose(est.sigma_e_db, 0.0)

    def test_one_dimensional_promoted(self, rng):
        est = estimate_from_log_error(np.array([1.0, 2.0]), 0.0, rng)
        assert est.irradiance_estimated.shape == (2, 1)

    def test_rejects_nonpositive_truth(self, rng):
        with pytest.raises(ValueError, match="strictly positive"):
            estimate_from_log_error(np.array([[0.0]]), 0.1, rng)

    def test_rejects_wrong_sigma_shape(self, rng):
        with pytest.raises(ValueError, match="scalar or have shape"):
            estimate_from_log_error(np.ones((4, 2)), np.ones(3), rng)

    def test_rejects_negative_sigma(self, rng):
        with pytest.raises(ValueError, match="finite and >= 0"):
            estimate_from_log_error(np.ones((2, 2)), -0.1, rng)

    def test_rejects_three_dimensional(self, rng):
        with pytest.raises(ValueError, match=r"shape \(n, L\)"):
            estimate_from_log_error(np.ones((2, 2, 2)), 0.1, rng)


class TestEstimateStaleNoisy:
    def test_marginals_of_the_truth(self, rng, line_array_correlation):
        ce = estimate_stale_noisy(100_000, 0.6, line_array_correlation, 0.9, 0.3, rng)
        assert ce.irradiance_true.shape == (100_000, 4)
        assert np.allclose(ce.irradiance_true.mean(axis=0), 1.0, atol=0.03)

    def test_reported_sigma_e_matches_the_formula(self, rng, line_array_correlation):
        ce = estimate_stale_noisy(20_000, 0.6, line_array_correlation, 0.85, 0.4, rng)
        assert np.allclose(ce.sigma_e, log_error_sigma(0.6, 0.85, 0.4))

    def test_realised_log_error_matches_sigma_e(self, rng, line_array_correlation):
        ce = estimate_stale_noisy(200_000, 0.6, line_array_correlation, 0.9, 0.3, rng)
        err = np.log(ce.irradiance_estimated) - np.log(ce.irradiance_true)
        assert err.std() == pytest.approx(float(ce.sigma_e[0]), rel=0.03)

    def test_fresh_and_clean_reproduces_truth(self, rng, line_array_correlation):
        ce = estimate_stale_noisy(5_000, 0.6, line_array_correlation, 1.0, 0.0, rng)
        assert np.allclose(ce.irradiance_estimated, ce.irradiance_true, rtol=1e-12)

    @pytest.mark.parametrize("rho,sm", [(1.5, 0.1), (-1.5, 0.1), (0.9, -0.1)])
    def test_rejects_bad_input(self, rho, sm, rng, line_array_correlation):
        with pytest.raises(ValueError):
            estimate_stale_noisy(100, 0.6, line_array_correlation, rho, sm, rng)
