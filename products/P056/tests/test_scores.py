"""Scoring rules: known answers, agreement with scikit-learn, and properties."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import brier_score_loss, log_loss

from calibaudit.scores import (
    LOG_SCORE_CLIP,
    base_rate_forecast,
    brier_score,
    brier_skill_score,
    constant_forecast,
    log_score,
    log_skill_score,
    skill_score,
)


class TestBrierKnownAnswers:
    def test_perfect_deterministic_forecast_scores_zero(self):
        # f = o exactly, so every squared error is 0.
        assert brier_score([0.0, 1.0, 1.0, 0.0], [0, 1, 1, 0]) == 0.0

    def test_worst_deterministic_forecast_scores_one(self):
        # f = 1 - o, so every squared error is 1.
        assert brier_score([1.0, 0.0], [0, 1]) == 1.0

    def test_constant_half_scores_one_quarter(self):
        # (0.5 - o)^2 = 0.25 for o in {0, 1}; hand value exactly 1/4.
        assert brier_score([0.5] * 7, [0, 1, 1, 0, 1, 0, 0]) == pytest.approx(0.25, abs=0)

    def test_hand_computed_four_sample_case(self):
        # f = (0.1, 0.4, 0.6, 0.9), o = (0, 0, 1, 1)
        # errors: 0.01, 0.16, 0.16, 0.01 -> sum 0.34, mean 0.085
        assert brier_score([0.1, 0.4, 0.6, 0.9], [0, 0, 1, 1]) == pytest.approx(
            0.085, abs=1e-15
        )

    def test_base_rate_forecast_scores_the_uncertainty_term(self):
        # A constant forecast at the base rate p scores exactly p(1-p).
        o = np.array([1.0] * 3 + [0.0] * 7)
        p = o.mean()
        assert brier_score(base_rate_forecast(o), o) == pytest.approx(p * (1 - p), abs=1e-15)

    @pytest.mark.parametrize("n_pos,n_neg", [(1, 9), (5, 5), (9, 1), (2, 98)])
    def test_base_rate_matches_closed_form_for_several_rates(self, n_pos, n_neg):
        o = np.array([1.0] * n_pos + [0.0] * n_neg)
        p = n_pos / (n_pos + n_neg)
        assert brier_score(constant_forecast(p, o.size), o) == pytest.approx(
            p * (1 - p), abs=1e-14
        )


class TestLogScoreKnownAnswers:
    def test_half_forecast_scores_log_two(self):
        # -log(0.5) = log 2 = 0.6931471805599453
        assert log_score([0.5, 0.5], [0, 1]) == pytest.approx(np.log(2.0), abs=1e-15)

    def test_hand_computed_two_sample_case(self):
        # -(log 0.9 + log 0.9)/2 = -log 0.9 = 0.10536051565782628
        assert log_score([0.1, 0.9], [0, 1]) == pytest.approx(-np.log(0.9), abs=1e-15)

    def test_confident_mistake_is_clipped_not_infinite(self):
        value = log_score([0.0], [1])
        assert np.isfinite(value)
        assert value == pytest.approx(-np.log(LOG_SCORE_CLIP), rel=1e-12)

    def test_clip_outside_range_is_rejected(self):
        with pytest.raises(ValueError, match="clip must lie"):
            log_score([0.5], [1], clip=0.5)
        with pytest.raises(ValueError, match="clip must lie"):
            log_score([0.5], [1], clip=0.0)


class TestAgreementWithScikitLearn:
    def test_brier_matches_sklearn(self, overconfident_large):
        s = overconfident_large
        ours = brier_score(s.forecasts, s.outcomes)
        theirs = brier_score_loss(s.outcomes, s.forecasts)
        assert ours == pytest.approx(theirs, abs=1e-14)

    def test_log_score_matches_sklearn(self, overconfident_large):
        s = overconfident_large
        ours = log_score(s.forecasts, s.outcomes)
        theirs = log_loss(s.outcomes, s.forecasts, labels=[0, 1])
        assert ours == pytest.approx(theirs, abs=1e-12)

    def test_brier_matches_sklearn_on_rare_events(self, rare_large):
        s = rare_large
        assert brier_score(s.forecasts, s.outcomes) == pytest.approx(
            brier_score_loss(s.outcomes, s.forecasts), abs=1e-14
        )


class TestSkillScores:
    def test_identical_forecast_has_zero_skill(self, overconfident_small):
        s = overconfident_small
        assert brier_skill_score(s.forecasts, s.outcomes, s.forecasts) == 0.0

    def test_perfect_forecast_against_coin_flip_has_skill_one(self):
        o = np.array([0.0, 1.0, 1.0, 0.0])
        assert brier_skill_score(o, o, constant_forecast(0.5, 4)) == pytest.approx(1.0)

    def test_worse_than_reference_gives_negative_skill(self):
        o = np.array([0.0, 1.0])
        poor = np.array([0.9, 0.1])
        ref = constant_forecast(0.5, 2)
        assert brier_skill_score(poor, o, ref) < 0.0

    def test_log_skill_score_sign_matches_log_scores(self, overconfident_large):
        s = overconfident_large
        ref = base_rate_forecast(s.outcomes)
        skill = log_skill_score(s.forecasts, s.outcomes, ref)
        better = log_score(s.forecasts, s.outcomes) < log_score(ref, s.outcomes)
        assert (skill > 0) == better

    def test_zero_reference_score_is_rejected(self):
        with pytest.raises(ValueError, match="undefined"):
            skill_score(0.1, 0.0)

    def test_skill_score_formula(self):
        assert skill_score(0.08, 0.2) == pytest.approx(0.6)


class TestReferenceForecasts:
    def test_constant_forecast_shape_and_value(self):
        f = constant_forecast(0.3, 5)
        assert f.shape == (5,)
        assert np.all(f == 0.3)

    @pytest.mark.parametrize("bad", [-0.01, 1.01, 2.0])
    def test_constant_forecast_rejects_out_of_range(self, bad):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            constant_forecast(bad, 3)

    def test_constant_forecast_rejects_zero_length(self):
        with pytest.raises(ValueError, match="at least 1"):
            constant_forecast(0.5, 0)

    def test_base_rate_forecast_is_constant_at_the_mean(self):
        o = [1, 1, 0, 0, 0]
        f = base_rate_forecast(o)
        assert np.all(f == 0.4)
