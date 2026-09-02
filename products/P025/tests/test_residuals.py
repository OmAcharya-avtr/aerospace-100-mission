"""Residual normalisation and filter-consistency checks."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fdiscope.residuals import (
    nis_consistency,
    nis_from_residual,
    normalise,
    whiteness,
)


class TestNormalise:
    def test_known_answer_diagonal(self):
        # S = diag(4, 9) -> L = diag(2, 3), so r = (y0/2, y1/3).
        r = normalise([[2.0, 3.0], [-4.0, 9.0]], np.diag([4.0, 9.0]))
        assert np.allclose(r, [[1.0, 1.0], [-2.0, 3.0]])

    def test_round_trip_reproduces_the_innovation(self):
        s = np.array([[4.0, 1.0], [1.0, 2.0]])
        y = np.array([[1.0, -2.0], [0.5, 0.25]])
        chol = np.linalg.cholesky(s)
        assert np.allclose((chol @ normalise(y, s).T).T, y)

    def test_normalised_residual_has_identity_covariance(self):
        rng = np.random.default_rng(0)
        s = np.array([[4.0, 1.5], [1.5, 2.0]])
        y = (np.linalg.cholesky(s) @ rng.standard_normal((2, 40000))).T
        cov = np.cov(normalise(y, s).T)
        assert np.allclose(cov, np.eye(2), atol=0.03)

    def test_rejects_mismatched_covariance(self):
        with pytest.raises(ValueError, match="innovation_cov must be"):
            normalise(np.zeros((5, 2)), np.eye(3))

    def test_rejects_non_positive_definite_covariance(self):
        with pytest.raises(np.linalg.LinAlgError):
            normalise(np.zeros((5, 2)), -np.eye(2))


class TestNisFromResidual:
    def test_known_answer(self):
        # (3, 4) -> 25, (1, 0) -> 1, (0, 0) -> 0
        rows = [[3.0, 4.0], [1.0, 0.0], [0.0, 0.0]]
        assert np.allclose(nis_from_residual(rows), [25.0, 1.0, 0.0])

    def test_rejects_three_dimensional_input(self):
        with pytest.raises(ValueError, match="must be"):
            nis_from_residual(np.zeros((2, 2, 2)))


class TestNisConsistency:
    def test_accepts_a_standard_normal_sample(self):
        r = np.random.default_rng(1).standard_normal((20000, 2))
        check = nis_consistency(r)
        assert check.consistent
        assert np.isclose(check.expected, 2.0)
        assert check.n_samples == 20000

    def test_rejects_an_inflated_sample(self):
        r = 1.3 * np.random.default_rng(2).standard_normal((20000, 2))
        assert not nis_consistency(r).consistent

    def test_bounds_bracket_the_expectation(self):
        check = nis_consistency(np.random.default_rng(3).standard_normal((5000, 2)))
        assert check.low < check.expected < check.high

    def test_bounds_tighten_with_more_samples(self):
        rng = np.random.default_rng(4)
        wide = nis_consistency(rng.standard_normal((200, 2)))
        tight = nis_consistency(rng.standard_normal((20000, 2)))
        assert (tight.high - tight.low) < (wide.high - wide.low)

    @pytest.mark.parametrize("bad_level", [0.0, 1.0, -0.5])
    def test_rejects_bad_level(self, bad_level):
        with pytest.raises(ValueError, match="level"):
            nis_consistency(np.zeros((10, 2)), level=bad_level)

    def test_rejects_too_few_samples(self):
        with pytest.raises(ValueError, match="N >= 2"):
            nis_consistency(np.zeros((1, 2)))


class TestWhiteness:
    def test_white_noise_is_inside_the_band(self):
        # The returned bound is the 5 % two-sided level, so with 8 statistics
        # a single exceedance happens about a third of the time by chance and
        # would make a strict test flaky.  A 4-sigma band is asserted instead;
        # the 5 % bound is still returned and reported.
        n = 20000
        rho, bound = whiteness(np.random.default_rng(5).standard_normal((n, 2)), 4)
        assert rho.shape == (4, 2)
        assert np.isclose(bound, 1.96 / np.sqrt(n))
        assert np.all(np.abs(rho) < 4.0 / np.sqrt(n))

    def test_a_random_walk_is_not(self):
        walk = np.cumsum(np.random.default_rng(6).standard_normal((5000, 2)), axis=0)
        rho, bound = whiteness(walk, 1)
        assert np.all(rho[0] > 10.0 * bound)

    def test_lag_one_autocorrelation_of_an_ar1_matches_its_coefficient(self):
        # x_k = 0.6 x_{k-1} + e_k has lag-1 autocorrelation 0.6.
        rng = np.random.default_rng(7)
        n = 60000
        e = rng.standard_normal(n)
        x = np.zeros(n)
        for k in range(1, n):
            x[k] = 0.6 * x[k - 1] + e[k]
        rho, _ = whiteness(np.stack([x, x], axis=1), 1)
        assert np.isclose(rho[0, 0], 0.6, atol=0.02)

    def test_rejects_bad_lag(self):
        with pytest.raises(ValueError, match="max_lag"):
            whiteness(np.zeros((10, 2)), 0)

    def test_rejects_too_short_a_sequence(self):
        with pytest.raises(ValueError, match="N > max_lag"):
            whiteness(np.zeros((3, 2)), 5)

    @settings(max_examples=20, deadline=None)
    @given(n=st.integers(100, 2000))
    def test_bound_matches_the_closed_form(self, n):
        _, bound = whiteness(np.random.default_rng(n).standard_normal((n, 2)), 2)
        assert np.isclose(bound, 1.96 / np.sqrt(n))
