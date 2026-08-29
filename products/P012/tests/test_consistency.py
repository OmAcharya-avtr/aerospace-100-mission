"""Unit tests for navbench.consistency — NEES, NIS, chi-squared bounds, whiteness."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from conftest import random_spd
from navbench import (
    KalmanFilter,
    chi2_bounds,
    consistency_test,
    constant_velocity_cwna,
    ensemble_consistency,
    innovation_whiteness,
    nees,
    nis,
    simulate_linear_system,
)


class TestChi2Bounds:
    def test_single_sample_matches_scipy(self):
        lo, hi = chi2_bounds(3, 1, 0.05)
        assert lo == pytest.approx(float(stats.chi2.ppf(0.025, 3)))
        assert hi == pytest.approx(float(stats.chi2.ppf(0.975, 3)))

    def test_ensemble_divides_by_m(self):
        lo, hi = chi2_bounds(2, 50, 0.05)
        assert lo == pytest.approx(float(stats.chi2.ppf(0.025, 100)) / 50.0)
        assert hi == pytest.approx(float(stats.chi2.ppf(0.975, 100)) / 50.0)

    def test_bounds_bracket_the_dof(self):
        for dof in (1, 2, 3, 6, 10):
            for m in (1, 10, 100):
                lo, hi = chi2_bounds(dof, m)
                assert lo < dof < hi

    def test_bounds_tighten_with_more_runs(self):
        w1 = np.subtract(*reversed(chi2_bounds(2, 10)))
        w2 = np.subtract(*reversed(chi2_bounds(2, 1000)))
        assert w2 < w1

    def test_smaller_alpha_widens(self):
        lo5, hi5 = chi2_bounds(2, 50, 0.05)
        lo1, hi1 = chi2_bounds(2, 50, 0.01)
        assert lo1 < lo5 and hi1 > hi5

    @pytest.mark.parametrize("dof", [0, -1])
    def test_bad_dof_raises(self, dof):
        with pytest.raises(ValueError, match="dof"):
            chi2_bounds(dof)

    def test_bad_n_runs_raises(self):
        with pytest.raises(ValueError, match="n_runs"):
            chi2_bounds(2, 0)

    @pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, np.nan])
    def test_bad_alpha_raises(self, alpha):
        with pytest.raises(ValueError, match="alpha"):
            chi2_bounds(2, 1, alpha)


class TestNees:
    def test_identity_covariance_gives_squared_norm(self):
        e = np.array([[3.0, 4.0]])
        assert nees(e, np.eye(2)[None, ...])[0] == pytest.approx(25.0)

    def test_hand_computed_diagonal(self):
        """e = [2, 3], P = diag(4, 9) -> 4/4 + 9/9 = 2."""
        assert nees([[2.0, 3.0]], np.diag([4.0, 9.0])[None, ...])[0] == pytest.approx(2.0)

    def test_zero_error_gives_zero(self):
        assert nees(np.zeros((5, 3)), np.repeat(np.eye(3)[None], 5, axis=0))[0] == 0.0

    def test_broadcast_single_covariance(self, rng):
        e = rng.standard_normal((10, 3))
        vals = nees(e, np.eye(3))
        assert vals.shape == (10,)
        assert np.allclose(vals, np.sum(e**2, axis=1))

    def test_1d_error_accepted(self):
        assert nees([3.0, 4.0], np.eye(2))[0] == pytest.approx(25.0)

    def test_invariant_under_linear_change_of_variables(self, rng):
        """NEES is invariant under x -> A x with P -> A P A^T."""
        e = rng.standard_normal(4)
        p = random_spd(4, rng)
        a = rng.standard_normal((4, 4)) + 4 * np.eye(4)
        v1 = nees(e[None, :], p[None, ...])[0]
        v2 = nees((a @ e)[None, :], (a @ p @ a.T)[None, ...])[0]
        assert v1 == pytest.approx(v2, rel=1e-9)

    def test_mean_matches_dof_for_gaussian_samples(self, rng):
        n = 4
        p = random_spd(n, rng)
        chol = np.linalg.cholesky(p)
        e = (chol @ rng.standard_normal((n, 40000))).T
        assert float(np.mean(nees(e, p))) == pytest.approx(n, rel=0.03)

    def test_indefinite_covariance_raises(self):
        with pytest.raises(ValueError, match="positive definite"):
            nees([[1.0, 1.0]], np.diag([1.0, -1.0])[None, ...])

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape"):
            nees(np.zeros((5, 3)), np.repeat(np.eye(2)[None], 5, axis=0))

    def test_nonfinite_error_raises(self):
        with pytest.raises(ValueError, match="finite"):
            nees([[np.nan, 1.0]], np.eye(2))

    def test_nonfinite_covariance_raises(self):
        p = np.eye(2)
        p[0, 0] = np.inf
        with pytest.raises(ValueError, match="finite"):
            nees([[1.0, 1.0]], p)


class TestNis:
    def test_scalar_case(self):
        assert nis([[2.0]], np.array([[[4.0]]]))[0] == pytest.approx(1.0)

    def test_nan_row_yields_nan(self):
        v = np.array([[1.0], [np.nan], [2.0]])
        s = np.repeat(np.eye(1)[None], 3, axis=0)
        out = nis(v, s)
        assert np.isfinite(out[0]) and np.isnan(out[1]) and np.isfinite(out[2])

    def test_broadcast_single_covariance(self, rng):
        v = rng.standard_normal((10, 2))
        out = nis(v, np.eye(2))
        assert np.allclose(out, np.sum(v**2, axis=1))

    def test_mean_matches_dof(self, rng):
        m = 3
        s = random_spd(m, rng)
        chol = np.linalg.cholesky(s)
        v = (chol @ rng.standard_normal((m, 40000))).T
        assert float(np.nanmean(nis(v, s))) == pytest.approx(m, rel=0.03)

    def test_indefinite_raises(self):
        with pytest.raises(ValueError, match="positive definite"):
            nis([[1.0, 1.0]], np.diag([1.0, -1.0])[None, ...])

    def test_nonfinite_covariance_raises(self):
        s = np.eye(2)[None, ...].copy()
        s[0, 0, 0] = np.nan
        with pytest.raises(ValueError, match="not finite"):
            nis([[1.0, 1.0]], s)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape"):
            nis(np.zeros((4, 2)), np.repeat(np.eye(3)[None], 4, axis=0))


class TestConsistencyTest:
    def test_consistent_verdict(self, rng):
        samples = stats.chi2.rvs(2, size=500, random_state=1)
        res = consistency_test(samples, 2)
        assert res.verdict == "consistent"
        assert res.passed
        assert res.dof == 2
        assert res.n_samples == 500

    def test_optimistic_verdict(self):
        res = consistency_test(np.full(200, 20.0), 2)
        assert res.verdict == "optimistic"
        assert not res.passed

    def test_pessimistic_verdict(self):
        res = consistency_test(np.full(200, 0.1), 2)
        assert res.verdict == "pessimistic"
        assert not res.passed

    def test_nan_samples_dropped(self):
        s = np.array([1.0, np.nan, 3.0, np.nan])
        assert consistency_test(s, 2).n_samples == 2

    def test_fraction_inside_reported(self):
        lo, hi = chi2_bounds(2, 1)
        s = np.array([lo * 0.5, (lo + hi) / 2.0, hi * 2.0, (lo + hi) / 2.0])
        assert consistency_test(s, 2).fraction_inside == pytest.approx(0.5)

    def test_summary_is_one_line(self):
        s = consistency_test(np.full(50, 2.0), 2).summary()
        assert "\n" not in s
        assert "NEES" in s

    def test_independent_flag_in_summary(self):
        assert "indicative" in consistency_test(
            np.full(50, 2.0), 2, independent=False
        ).summary()

    def test_all_nan_raises(self):
        with pytest.raises(ValueError, match="no finite"):
            consistency_test(np.full(5, np.nan), 2)

    def test_bad_statistic_raises(self):
        with pytest.raises(ValueError, match="statistic"):
            consistency_test(np.ones(10), 2, statistic="RMSE")


class TestEnsembleConsistency:
    def test_shapes(self, rng):
        a = rng.chisquare(2, size=(20, 50))
        avg, lo, hi = ensemble_consistency(a, 2)
        assert avg.shape == (50,)
        assert lo < 2.0 < hi

    def test_average_is_column_mean(self, rng):
        a = rng.chisquare(2, size=(7, 5))
        avg, _, _ = ensemble_consistency(a, 2)
        assert np.allclose(avg, np.mean(a, axis=0))

    def test_nan_ignored(self):
        a = np.array([[1.0, 2.0], [np.nan, 4.0]])
        avg, _, _ = ensemble_consistency(a, 1)
        assert avg[0] == pytest.approx(1.0)
        assert avg[1] == pytest.approx(3.0)

    def test_correct_filter_falls_inside(self):
        """The headline claim: a correctly specified filter is inside its bounds."""
        f, q = constant_velocity_cwna(1.0, 0.05)
        h = np.array([[1.0, 0.0]])
        r = np.array([[9.0]])
        p0 = np.diag([100.0, 10.0])
        m, n = 40, 120
        runs = np.zeros((m, n))
        for i in range(m):
            rng = np.random.default_rng(3000 + i)
            truth, meas = simulate_linear_system(
                f, h, q, r, np.array([0.0, 1.0]), n, rng
            )
            res = KalmanFilter(f, h, q, r, np.zeros(2), p0).run(meas)
            runs[i] = nees(truth - res.x_post, res.p_post)
        avg, lo, hi = ensemble_consistency(runs[:, 20:], 2)
        assert lo <= float(np.mean(avg)) <= hi

    def test_underspecified_q_leaves_the_bounds(self):
        f, q_true = constant_velocity_cwna(1.0, 0.05)
        _, q_bad = constant_velocity_cwna(1.0, 0.05 / 25.0)
        h = np.array([[1.0, 0.0]])
        r = np.array([[9.0]])
        p0 = np.diag([100.0, 10.0])
        m, n = 30, 120
        runs = np.zeros((m, n))
        for i in range(m):
            rng = np.random.default_rng(4000 + i)
            truth, meas = simulate_linear_system(
                f, h, q_true, r, np.array([0.0, 1.0]), n, rng
            )
            res = KalmanFilter(f, h, q_bad, r, np.zeros(2), p0).run(meas)
            runs[i] = nees(truth - res.x_post, res.p_post)
        avg, lo, hi = ensemble_consistency(runs[:, 20:], 2)
        assert float(np.mean(avg)) > hi

    def test_bad_statistic_raises(self):
        with pytest.raises(ValueError, match="statistic"):
            ensemble_consistency(np.ones((3, 4)), 2, statistic="oops")

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            ensemble_consistency(np.zeros((0, 5)), 2)


class TestWhiteness:
    def test_white_noise_passes(self, rng):
        w = innovation_whiteness(rng.standard_normal(2000), max_lag=10)
        assert w.passed
        assert w.n_samples == 2000

    def test_lag_zero_is_one(self, rng):
        w = innovation_whiteness(rng.standard_normal(500), max_lag=5)
        assert w.autocorrelation[0] == pytest.approx(1.0)

    def test_correlated_sequence_fails(self, rng):
        x = np.cumsum(rng.standard_normal(2000))
        assert not innovation_whiteness(x, max_lag=10).passed

    def test_band_shrinks_with_n(self, rng):
        a = innovation_whiteness(rng.standard_normal(400), max_lag=5).band
        b = innovation_whiteness(rng.standard_normal(4000), max_lag=5).band
        assert b < a

    def test_band_matches_normal_quantile(self, rng):
        w = innovation_whiteness(rng.standard_normal(1000), max_lag=5, alpha=0.05)
        assert w.band == pytest.approx(1.959963985 / np.sqrt(1000), rel=1e-6)

    def test_multidimensional_input(self, rng):
        w = innovation_whiteness(rng.standard_normal((1000, 3)), max_lag=8)
        assert w.autocorrelation.size == 9

    def test_nan_rows_dropped(self, rng):
        x = rng.standard_normal((500, 1))
        x[10] = np.nan
        assert innovation_whiteness(x, max_lag=5).n_samples == 499

    def test_summary_mentions_lag(self, rng):
        s = innovation_whiteness(rng.standard_normal(500), max_lag=5).summary()
        assert "lag" in s and "\n" not in s

    def test_bad_max_lag_raises(self, rng):
        with pytest.raises(ValueError, match="max_lag"):
            innovation_whiteness(rng.standard_normal(100), max_lag=0)

    def test_too_few_samples_raises(self, rng):
        with pytest.raises(ValueError, match="more than"):
            innovation_whiteness(rng.standard_normal(5), max_lag=10)

    def test_zero_variance_raises(self):
        with pytest.raises(ValueError, match="zero variance"):
            innovation_whiteness(np.ones(100), max_lag=5)

    def test_bad_alpha_raises(self, rng):
        with pytest.raises(ValueError, match="alpha"):
            innovation_whiteness(rng.standard_normal(200), max_lag=5, alpha=1.5)
