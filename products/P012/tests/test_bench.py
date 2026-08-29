"""Unit tests for navbench.bench — scoring and comparison."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from navbench import (
    DIVERGENCE_QUANTILE,
    KalmanFilter,
    compare_scores,
    constant_velocity_cwna,
    score_run,
    simulate_linear_system,
)


@pytest.fixture
def scored_run(cv_model):
    f, q, h, r, x0, p0 = cv_model
    rng = np.random.default_rng(1234)
    truth, meas = simulate_linear_system(f, h, q, r, np.array([0.0, 1.0]), 200, rng)
    res = KalmanFilter(f, h, q, r, x0, p0).run(meas)
    return truth, res


class TestScoreRun:
    def test_fields(self, scored_run):
        truth, res = scored_run
        s = score_run("KF", truth, res.x_post, res.p_post, res.innovation,
                      res.innovation_cov, burn_in=20)
        assert s.name == "KF"
        assert s.rmse.shape == (2,)
        assert s.rmse_total > 0.0
        assert s.n_steps == 180
        assert s.nis_result is not None

    def test_rmse_total_is_rms_over_all_entries(self, scored_run):
        truth, res = scored_run
        s = score_run("KF", truth, res.x_post, res.p_post, burn_in=0)
        err = truth - res.x_post
        assert s.rmse_total == pytest.approx(float(np.sqrt(np.mean(err**2))))

    def test_per_component_rmse(self, scored_run):
        truth, res = scored_run
        s = score_run("KF", truth, res.x_post, res.p_post, burn_in=0)
        err = truth - res.x_post
        assert np.allclose(s.rmse, np.sqrt(np.mean(err**2, axis=0)))

    def test_perfect_estimate_has_zero_rmse(self, scored_run):
        truth, res = scored_run
        s = score_run("perfect", truth, truth, res.p_post, burn_in=0)
        assert s.rmse_total == 0.0
        assert s.mean_nees == 0.0

    def test_nis_omitted_when_not_supplied(self, scored_run):
        truth, res = scored_run
        s = score_run("KF", truth, res.x_post, res.p_post)
        assert s.nis_result is None
        assert np.isnan(s.mean_nis)

    def test_burn_in_excludes_leading_steps(self, scored_run):
        truth, res = scored_run
        a = score_run("a", truth, res.x_post, res.p_post, burn_in=0)
        b = score_run("b", truth, res.x_post, res.p_post, burn_in=50)
        assert b.n_steps == a.n_steps - 50

    def test_correct_filter_mean_nees_near_dof(self, scored_run):
        """A single run's TIME average is not a valid chi-squared test — successive
        steps are correlated, so the tight M-sample band is routinely exceeded even
        by a correct filter. The bench labels it indicative for that reason. Here we
        only assert the mean is in the right neighbourhood of dof = 2."""
        truth, res = scored_run
        s = score_run("KF", truth, res.x_post, res.p_post, burn_in=20)
        assert 1.0 < s.mean_nees < 3.5
        assert not s.diverged

    def test_ensemble_of_correct_filters_is_consistent(self, cv_model):
        """The defensible form: pool independent runs, then test."""
        f, q, h, r, x0, p0 = cv_model
        pooled = []
        for i in range(40):
            rng = np.random.default_rng(8000 + i)
            truth, meas = simulate_linear_system(
                f, h, q, r, np.array([0.0, 1.0]), 60, rng
            )
            res = KalmanFilter(f, h, q, r, x0, p0).run(meas)
            pooled.append(
                score_run("KF", truth, res.x_post, res.p_post, burn_in=20).mean_nees
            )
        assert float(np.mean(pooled)) == pytest.approx(2.0, abs=0.35)

    def test_divergence_flag(self, scored_run):
        truth, res = scored_run
        bad = res.x_post.copy()
        bad[-1] += 1e6
        s = score_run("bad", truth, bad, res.p_post, burn_in=20)
        assert s.diverged

    def test_divergence_threshold_value(self):
        assert DIVERGENCE_QUANTILE == 0.9999
        assert float(stats.chi2.ppf(DIVERGENCE_QUANTILE, 2)) == pytest.approx(18.4207, abs=1e-3)

    def test_summary_multiline(self, scored_run):
        truth, res = scored_run
        s = score_run("KF", truth, res.x_post, res.p_post, res.innovation,
                      res.innovation_cov, burn_in=20)
        text = s.summary()
        assert text.count("\n") == 3
        assert "RMSE" in text and "NEES" in text and "NIS" in text

    def test_shape_mismatch_raises(self, scored_run):
        truth, res = scored_run
        with pytest.raises(ValueError, match="must match"):
            score_run("KF", truth[:-1], res.x_post, res.p_post)

    def test_bad_covariance_shape_raises(self, scored_run):
        truth, res = scored_run
        with pytest.raises(ValueError, match="shape"):
            score_run("KF", truth, res.x_post, res.p_post[:, :1, :1])

    @pytest.mark.parametrize("burn", [-1, 200, 500])
    def test_bad_burn_in_raises(self, scored_run, burn):
        truth, res = scored_run
        with pytest.raises(ValueError, match="burn_in"):
            score_run("KF", truth, res.x_post, res.p_post, burn_in=burn)


class TestCompareScores:
    def test_table_has_a_row_per_estimator(self, scored_run):
        truth, res = scored_run
        scores = [
            score_run(f"est{i}", truth, res.x_post, res.p_post, res.innovation,
                      res.innovation_cov, burn_in=10)
            for i in range(3)
        ]
        text = compare_scores(scores)
        assert len(text.splitlines()) == 5  # header, rule, 3 rows
        for i in range(3):
            assert f"est{i}" in text

    def test_shows_n_a_when_nis_missing(self, scored_run):
        truth, res = scored_run
        text = compare_scores([score_run("KF", truth, res.x_post, res.p_post)])
        assert "n/a" in text

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            compare_scores([])

    def test_verdict_column_present(self, scored_run):
        truth, res = scored_run
        text = compare_scores([score_run("KF", truth, res.x_post, res.p_post, burn_in=20)])
        assert "consistent" in text or "optimistic" in text or "pessimistic" in text

    def test_optimistic_filter_flagged(self):
        f, q_true = constant_velocity_cwna(1.0, 0.05)
        _, q_bad = constant_velocity_cwna(1.0, 0.05 / 50.0)
        h = np.array([[1.0, 0.0]])
        r = np.array([[9.0]])
        rng = np.random.default_rng(77)
        truth, meas = simulate_linear_system(f, h, q_true, r, np.array([0.0, 1.0]), 300, rng)
        res = KalmanFilter(f, h, q_bad, r, np.zeros(2), np.diag([100.0, 10.0])).run(meas)
        s = score_run("bad", truth, res.x_post, res.p_post, burn_in=30)
        assert s.nees_result.verdict == "optimistic"
