"""Tests for inter-aperture correlation and the correlated samplers."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from aperturediv.channel import (
    gamma_gamma_params_from_rytov,
    gamma_gamma_scintillation_index,
    lognormal_sigma_log,
)
from aperturediv.correlation import (
    correlation_matrix,
    equispaced_positions,
    irradiance_correlation_matrix,
    log_to_irradiance_correlation,
    nearest_psd,
    sample_correlated_gamma_gamma,
    sample_correlated_lognormal,
)


class TestPositions:
    def test_known_answer(self):
        assert np.allclose(equispaced_positions(4, 0.05), [0.0, 0.05, 0.10, 0.15])

    def test_single_aperture(self):
        assert np.allclose(equispaced_positions(1, 0.05), [0.0])

    @pytest.mark.parametrize("n,pitch", [(0, 0.05), (-1, 0.05), (3, -0.05), (3, float("nan"))])
    def test_rejects_bad_input(self, n, pitch):
        with pytest.raises(ValueError):
            equispaced_positions(n, pitch)


class TestCorrelationMatrix:
    def test_shape_symmetry_diagonal(self):
        r = correlation_matrix(equispaced_positions(4, 0.05), 0.10)
        assert r.shape == (4, 4)
        assert np.allclose(r, r.T)
        assert np.allclose(np.diag(r), 1.0)

    def test_known_answer_adjacent(self):
        # d = 0.05, rho_c = 0.10: exp(-(0.5)^2) = exp(-0.25) = 0.7788008.
        r = correlation_matrix(equispaced_positions(2, 0.05), 0.10)
        assert r[0, 1] == pytest.approx(np.exp(-0.25), rel=1e-12)

    def test_decreasing_with_separation(self):
        r = correlation_matrix(equispaced_positions(5, 0.03), 0.08)
        assert np.all(np.diff(r[0]) < 0.0)

    def test_two_dimensional_positions(self):
        pos = np.array([[0.0, 0.0], [0.0, 0.1], [0.1, 0.0]])
        r = correlation_matrix(pos, 0.1)
        assert r[0, 1] == pytest.approx(r[0, 2], rel=1e-12)
        assert r[1, 2] == pytest.approx(np.exp(-2.0), rel=1e-12)

    def test_positive_semidefinite_gaussian(self):
        r = correlation_matrix(equispaced_positions(6, 0.02), 0.07)
        assert np.linalg.eigvalsh(r).min() > -1e-10

    def test_rejects_bad_shape(self):
        with pytest.raises(ValueError):
            correlation_matrix(np.zeros((2, 2, 2)), 0.1)

    def test_rejects_non_finite(self):
        with pytest.raises(ValueError):
            correlation_matrix(np.array([0.0, np.nan]), 0.1)


class TestLogToIrradianceCorrelation:
    def test_endpoints_exact(self):
        assert float(log_to_irradiance_correlation(0.0, 0.6)) == pytest.approx(0.0, abs=1e-15)
        assert float(log_to_irradiance_correlation(1.0, 0.6)) == pytest.approx(1.0, rel=1e-14)

    def test_known_answer(self):
        # si = 1 -> s^2 = ln 2. At R = 0.5: (exp(0.5 ln2) - 1)/(exp(ln2) - 1)
        # = (sqrt 2 - 1)/1 = 0.41421356.
        assert float(log_to_irradiance_correlation(0.5, 1.0)) == pytest.approx(
            np.sqrt(2.0) - 1.0, rel=1e-12
        )

    @given(st.floats(0.001, 0.999), st.floats(0.01, 4.0))
    def test_never_exceeds_log_correlation(self, r, si):
        assert float(log_to_irradiance_correlation(r, si)) < r + 1e-12

    def test_gap_grows_with_si(self):
        gaps = [0.5 - float(log_to_irradiance_correlation(0.5, si)) for si in (0.1, 0.5, 1.0, 2.0)]
        assert all(y > x for x, y in zip(gaps, gaps[1:], strict=False))

    @pytest.mark.parametrize("bad", [-1.5, 1.5])
    def test_rejects_out_of_range(self, bad):
        with pytest.raises(ValueError):
            log_to_irradiance_correlation(bad, 0.6)

    def test_matrix_form_has_unit_diagonal(self):
        m = irradiance_correlation_matrix(equispaced_positions(3, 0.05), 0.1, 0.6)
        assert np.allclose(np.diag(m), 1.0)


class TestNearestPsd:
    def test_identity_on_valid_matrix(self, line_array_correlation):
        out = nearest_psd(line_array_correlation)
        assert np.max(np.abs(out - line_array_correlation)) < 1e-10

    def test_repairs_invalid_matrix(self):
        bad = np.array([[1.0, 0.9, 0.9], [0.9, 1.0, -0.9], [0.9, -0.9, 1.0]])
        assert np.linalg.eigvalsh(bad).min() < 0.0
        fixed = nearest_psd(bad, floor=1e-12)
        assert np.linalg.eigvalsh(fixed).min() > -1e-10
        assert np.allclose(np.diag(fixed), 1.0)

    def test_rejects_non_square(self):
        with pytest.raises(ValueError):
            nearest_psd(np.zeros((2, 3)))


class TestCorrelatedLognormalSampler:
    def test_marginals(self, rng, line_array_correlation):
        x = sample_correlated_lognormal(200_000, 0.6, line_array_correlation, rng)
        assert x.shape == (200_000, 4)
        assert np.allclose(x.mean(axis=0), 1.0, atol=0.02)
        assert np.allclose(x.var(axis=0) / x.mean(axis=0) ** 2, 0.6, atol=0.03)

    def test_log_correlation_reproduced(self, rng, line_array_correlation):
        x = sample_correlated_lognormal(200_000, 0.6, line_array_correlation, rng)
        s = lognormal_sigma_log(0.6)
        z = (np.log(x) + 0.5 * s * s) / s
        assert np.max(np.abs(np.corrcoef(z.T) - line_array_correlation)) < 0.01

    def test_irradiance_correlation_reproduced(self, rng):
        pos = equispaced_positions(3, 0.05)
        r = correlation_matrix(pos, 0.10)
        x = sample_correlated_lognormal(200_000, 0.6, r, rng)
        want = irradiance_correlation_matrix(pos, 0.10, 0.6)
        assert np.max(np.abs(np.corrcoef(x.T) - want)) < 0.01

    def test_identity_matrix_gives_independent_columns(self, rng):
        x = sample_correlated_lognormal(100_000, 0.6, np.eye(3), rng)
        off = np.corrcoef(x.T) - np.eye(3)
        assert np.max(np.abs(off)) < 0.02

    def test_fully_correlated_gives_identical_columns(self, rng):
        # An all-ones matrix is singular, so the sampler falls back to the
        # eigenvalue-clipped repair; the columns then agree only to the
        # rounding of that repair, not bit for bit.
        x = sample_correlated_lognormal(1000, 0.6, np.ones((3, 3)), rng)
        assert np.allclose(x[:, 0], x[:, 1], rtol=1e-5)
        assert np.allclose(x[:, 1], x[:, 2], rtol=1e-5)

    def test_reproducible(self, line_array_correlation):
        a = sample_correlated_lognormal(500, 0.5, line_array_correlation,
                                        np.random.default_rng(3))
        b = sample_correlated_lognormal(500, 0.5, line_array_correlation,
                                        np.random.default_rng(3))
        assert np.array_equal(a, b)

    @pytest.mark.parametrize("n", [0, -5])
    def test_rejects_bad_n(self, n, rng, line_array_correlation):
        with pytest.raises(ValueError):
            sample_correlated_lognormal(n, 0.6, line_array_correlation, rng)

    def test_rejects_non_unit_diagonal(self, rng):
        with pytest.raises(ValueError, match="unit diagonal"):
            sample_correlated_lognormal(10, 0.6, np.array([[2.0, 0.0], [0.0, 1.0]]), rng)

    def test_rejects_non_square(self, rng):
        with pytest.raises(ValueError, match="square"):
            sample_correlated_lognormal(10, 0.6, np.ones((2, 3)), rng)


class TestCorrelatedGammaGammaSampler:
    def test_marginals(self, rng, line_array_correlation):
        a, b = gamma_gamma_params_from_rytov(1.0)
        x = sample_correlated_gamma_gamma(150_000, a, b, line_array_correlation, rng)
        assert x.shape == (150_000, 4)
        assert np.allclose(x.mean(axis=0), 1.0, atol=0.03)
        assert np.allclose(
            x.var(axis=0) / x.mean(axis=0) ** 2,
            gamma_gamma_scintillation_index(a, b),
            atol=0.06,
        )

    def test_independent_small_scale_dilutes_correlation(self, rng, line_array_correlation):
        a, b = gamma_gamma_params_from_rytov(1.0)
        x = sample_correlated_gamma_gamma(150_000, a, b, line_array_correlation, rng)
        observed = np.corrcoef(x.T)[0, 1]
        assert 0.0 < observed < line_array_correlation[0, 1]

    def test_both_factors_correlated_raises_correlation(self, rng, line_array_correlation):
        a, b = gamma_gamma_params_from_rytov(1.0)
        both = sample_correlated_gamma_gamma(
            100_000, a, b, line_array_correlation, rng,
            small_scale_correlation=line_array_correlation,
        )
        only_large = sample_correlated_gamma_gamma(
            100_000, a, b, line_array_correlation, rng
        )
        assert np.corrcoef(both.T)[0, 1] > np.corrcoef(only_large.T)[0, 1]

    def test_identity_gives_independent_columns(self, rng):
        a, b = gamma_gamma_params_from_rytov(1.0)
        x = sample_correlated_gamma_gamma(100_000, a, b, np.eye(3), rng)
        assert np.max(np.abs(np.corrcoef(x.T) - np.eye(3))) < 0.02

    @pytest.mark.parametrize("a,b", [(0.0, 1.0), (1.0, -1.0)])
    def test_rejects_bad_shapes(self, a, b, rng):
        with pytest.raises(ValueError):
            sample_correlated_gamma_gamma(10, a, b, np.eye(2), rng)

    def test_rejects_mismatched_small_scale(self, rng):
        with pytest.raises(ValueError, match="match"):
            sample_correlated_gamma_gamma(
                10, 4.0, 2.0, np.eye(3), rng, small_scale_correlation=np.eye(2)
            )

    def test_rejects_bad_n(self, rng):
        with pytest.raises(ValueError):
            sample_correlated_gamma_gamma(0, 4.0, 2.0, np.eye(2), rng)
