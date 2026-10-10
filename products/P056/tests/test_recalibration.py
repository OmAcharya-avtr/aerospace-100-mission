"""Recalibrators, the held-out audit, and the cases where the baseline wins."""

from __future__ import annotations

import numpy as np
import pytest

from calibaudit.recalibration import (
    IsotonicCalibration,
    PlattScaling,
    RawForecast,
    get_method,
    recalibration_audit,
    sample_size_sweep,
)
from calibaudit.scores import brier_score
from calibaudit.synthetic import get_spec, sample_forecast


class TestRawBaseline:
    def test_raw_is_the_identity(self, overconfident_small):
        s = overconfident_small
        model = RawForecast().fit(s.forecasts, s.outcomes)
        assert np.array_equal(model.predict(s.forecasts), s.forecasts)

    def test_raw_is_marked_not_learned(self):
        assert RawForecast().is_learned is False
        assert PlattScaling().is_learned is True
        assert IsotonicCalibration().is_learned is True

    def test_get_method_returns_the_right_classes(self):
        assert isinstance(get_method("raw"), RawForecast)
        assert isinstance(get_method("platt"), PlattScaling)
        assert isinstance(get_method("isotonic"), IsotonicCalibration)


class TestPlattScaling:
    def test_identity_data_gives_near_identity_parameters(self):
        # Outcomes drawn with probability exactly equal to the forecast, so
        # the maximum-likelihood map is a = 1, b = 0 up to sampling noise.
        s = sample_forecast(get_spec("calibrated"), 20000, seed=56500)
        model = PlattScaling().fit(s.forecasts, s.outcomes)
        assert model.a_ == pytest.approx(1.0, abs=0.08)
        assert model.b_ == pytest.approx(0.0, abs=0.05)

    def test_recovers_the_known_temperature(self):
        # The overconfident spec is f = sigmoid(logit(p) / 0.6), so the exact
        # inverse map is sigmoid(0.6 * logit(f)): a = 0.6, b = 0.
        s = sample_forecast(get_spec("overconfident"), 40000, seed=56501)
        model = PlattScaling().fit(s.forecasts, s.outcomes)
        assert model.a_ == pytest.approx(0.6, abs=0.05)
        assert model.b_ == pytest.approx(0.0, abs=0.05)

    def test_recovers_the_known_logit_shift(self):
        # biased_high is f = sigmoid(logit(p) + 0.6), inverse a = 1, b = -0.6.
        s = sample_forecast(get_spec("biased_high"), 40000, seed=56502)
        model = PlattScaling().fit(s.forecasts, s.outcomes)
        assert model.a_ == pytest.approx(1.0, abs=0.08)
        assert model.b_ == pytest.approx(-0.6, abs=0.08)

    def test_predictions_stay_in_the_unit_interval(self, overconfident_small):
        s = overconfident_small
        p = PlattScaling().fit(s.forecasts, s.outcomes).predict([0.0, 1e-9, 0.5, 1.0])
        assert np.all((p >= 0.0) & (p <= 1.0))

    def test_boundary_forecasts_do_not_produce_nan(self, overconfident_small):
        s = overconfident_small
        p = PlattScaling().fit(s.forecasts, s.outcomes).predict([0.0, 1.0])
        assert np.all(np.isfinite(p))

    def test_map_is_monotone(self, overconfident_large):
        s = overconfident_large
        model = PlattScaling().fit(s.forecasts, s.outcomes)
        grid = np.linspace(0.001, 0.999, 200)
        out = model.predict(grid)
        assert np.all(np.diff(out) > 0) or np.all(np.diff(out) < 0)

    def test_target_smoothing_changes_the_fit(self, overconfident_small):
        s = overconfident_small
        plain = PlattScaling(target_smoothing=False).fit(s.forecasts, s.outcomes)
        smooth = PlattScaling(target_smoothing=True).fit(s.forecasts, s.outcomes)
        assert plain.a_ != smooth.a_ or plain.b_ != smooth.b_

    def test_target_smoothing_keeps_the_fit_finite_on_separable_data(self):
        # Separable in logit(f): plain MLE drives the slope to infinity.
        f = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        o = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        smooth = PlattScaling(target_smoothing=True).fit(f, o)
        assert np.isfinite(smooth.a_)
        assert np.isfinite(smooth.b_)

    def test_optimizer_message_is_recorded(self, overconfident_small):
        s = overconfident_small
        model = PlattScaling().fit(s.forecasts, s.outcomes)
        assert model.optimizer_message_ != "not fitted"

    def test_fit_reduces_log_score_in_sample(self, overconfident_large):
        from calibaudit.scores import log_score

        s = overconfident_large
        model = PlattScaling().fit(s.forecasts, s.outcomes)
        assert log_score(model.predict(s.forecasts), s.outcomes) <= log_score(
            s.forecasts, s.outcomes
        )


class TestIsotonicCalibration:
    def test_map_is_non_decreasing(self, overconfident_large):
        s = overconfident_large
        model = IsotonicCalibration().fit(s.forecasts, s.outcomes)
        grid = np.linspace(0.0, 1.0, 500)
        assert np.all(np.diff(model.predict(grid)) >= -1e-15)

    def test_predictions_stay_in_the_unit_interval(self, overconfident_small):
        s = overconfident_small
        p = IsotonicCalibration().fit(s.forecasts, s.outcomes).predict(
            np.linspace(0, 1, 50)
        )
        assert np.all((p >= 0.0) & (p <= 1.0))

    def test_out_of_range_forecasts_are_clipped_not_extrapolated(self):
        f = np.array([0.4, 0.45, 0.5, 0.55, 0.6])
        o = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
        model = IsotonicCalibration().fit(f, o)
        assert model.predict([0.0])[0] == pytest.approx(model.predict([0.4])[0])
        assert model.predict([1.0])[0] == pytest.approx(model.predict([0.6])[0])

    def test_step_count_is_small_relative_to_the_sample(self, overconfident_large):
        s = overconfident_large
        model = IsotonicCalibration().fit(s.forecasts, s.outcomes)
        assert 1 < model.n_steps < s.n_samples

    def test_isotonic_emits_exact_zero_or_one_and_so_can_blow_up_the_log_score(self):
        # Pool-adjacent-violators returns the bin average, which on a pure
        # block is exactly 0 or 1. A single held-out disagreement then costs
        # -log(clip). This is why isotonic's log score is reported separately.
        f = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        o = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        model = IsotonicCalibration().fit(f, o)
        out = model.predict(f)
        assert np.any(out == 0.0)
        assert np.any(out == 1.0)


class TestUncertaintyOutput:
    def test_interval_brackets_the_point_estimate(self, overconfident_large):
        s = overconfident_large
        model = PlattScaling(n_bootstrap=60, seed=1).fit(s.forecasts, s.outcomes)
        grid = np.linspace(0.05, 0.95, 19)
        point, lo, hi = model.predict_with_interval(grid, level=0.9)
        assert np.all(lo <= hi)
        # The point estimate is the full-sample fit, so it need not be the
        # ensemble median, but it must lie inside a generous envelope.
        assert np.all(point >= lo - 0.05)
        assert np.all(point <= hi + 0.05)

    def test_interval_width_shrinks_with_training_size(self):
        spec = get_spec("overconfident")
        small = sample_forecast(spec, 200, seed=56600)
        large = sample_forecast(spec, 6400, seed=56601)
        grid = np.linspace(0.1, 0.9, 17)
        _, lo_s, hi_s = PlattScaling(n_bootstrap=60, seed=2).fit(
            small.forecasts, small.outcomes
        ).predict_with_interval(grid)
        _, lo_l, hi_l = PlattScaling(n_bootstrap=60, seed=2).fit(
            large.forecasts, large.outcomes
        ).predict_with_interval(grid)
        assert float(np.mean(hi_l - lo_l)) < float(np.mean(hi_s - lo_s))

    def test_isotonic_intervals_are_wider_than_platt_on_small_samples(self):
        s = sample_forecast(get_spec("overconfident"), 300, seed=56602)
        grid = np.linspace(0.1, 0.9, 17)
        _, lo_p, hi_p = PlattScaling(n_bootstrap=60, seed=3).fit(
            s.forecasts, s.outcomes
        ).predict_with_interval(grid)
        _, lo_i, hi_i = IsotonicCalibration(n_bootstrap=60, seed=3).fit(
            s.forecasts, s.outcomes
        ).predict_with_interval(grid)
        assert float(np.mean(hi_i - lo_i)) > float(np.mean(hi_p - lo_p))

    def test_ensemble_size_is_reported(self, overconfident_small):
        s = overconfident_small
        model = IsotonicCalibration(n_bootstrap=17, seed=4).fit(s.forecasts, s.outcomes)
        assert model.n_ensemble == 17

    def test_raw_has_no_ensemble_even_when_asked(self, overconfident_small):
        s = overconfident_small
        model = RawForecast(n_bootstrap=0).fit(s.forecasts, s.outcomes)
        assert model.n_ensemble == 0


class TestAudit:
    def test_raw_row_is_first_and_has_zero_delta(self, overconfident_large):
        s = overconfident_large
        audit = recalibration_audit(s.forecasts, s.outcomes, n_bootstrap=50, seed=1)
        assert audit.results[0].method == "raw"
        assert audit.results[0].delta_brier == 0.0
        assert audit.results[0].verdict == "baseline"

    def test_split_sizes_add_to_the_sample(self, overconfident_large):
        s = overconfident_large
        audit = recalibration_audit(
            s.forecasts, s.outcomes, test_fraction=0.3, n_bootstrap=20, seed=2
        )
        assert audit.n_train + audit.n_test == s.n_samples
        assert audit.n_test == pytest.approx(0.3 * s.n_samples, abs=1)

    def test_delta_is_the_difference_of_the_reported_briers(self, overconfident_large):
        s = overconfident_large
        audit = recalibration_audit(s.forecasts, s.outcomes, n_bootstrap=20, seed=3)
        raw = audit.by_method("raw")
        for r in audit.results:
            assert r.delta_brier == pytest.approx(r.brier - raw.brier, abs=1e-15)

    def test_recalibration_helps_a_badly_miscalibrated_forecast_with_plenty_of_data(self):
        s = sample_forecast(get_spec("biased_high"), 20000, seed=56700)
        audit = recalibration_audit(s.forecasts, s.outcomes, n_bootstrap=200, seed=4)
        assert audit.by_method("platt").verdict == "improved"
        assert audit.by_method("platt").helped

    def test_recalibration_reduces_ece_when_it_helps(self):
        s = sample_forecast(get_spec("biased_high"), 20000, seed=56701)
        audit = recalibration_audit(s.forecasts, s.outcomes, n_bootstrap=50, seed=5)
        assert audit.by_method("platt").ece < audit.by_method("raw").ece

    def test_verdict_vocabulary(self, overconfident_large):
        s = overconfident_large
        audit = recalibration_audit(s.forecasts, s.outcomes, n_bootstrap=50, seed=6)
        allowed = {"baseline", "improved", "worse", "indistinguishable"}
        assert {r.verdict for r in audit.results} <= allowed

    def test_table_lists_every_method(self, overconfident_large):
        s = overconfident_large
        audit = recalibration_audit(s.forecasts, s.outcomes, n_bootstrap=20, seed=7)
        text = audit.table()
        for name in ("raw", "platt", "isotonic"):
            assert name in text

    def test_audit_is_deterministic_in_the_seed(self, overconfident_large):
        s = overconfident_large
        a = recalibration_audit(s.forecasts, s.outcomes, n_bootstrap=30, seed=8)
        b = recalibration_audit(s.forecasts, s.outcomes, n_bootstrap=30, seed=8)
        assert a.by_method("platt").delta_brier == b.by_method("platt").delta_brier
        assert a.by_method("platt").delta_brier_ci == b.by_method("platt").delta_brier_ci

    def test_skill_against_raw_has_the_opposite_sign_to_delta(self, overconfident_large):
        s = overconfident_large
        audit = recalibration_audit(s.forecasts, s.outcomes, n_bootstrap=30, seed=9)
        r = audit.by_method("platt")
        assert (r.brier_skill_vs_raw > 0) == (r.delta_brier < 0)

    def test_subset_of_methods_is_honoured(self):
        s = sample_forecast(get_spec("overconfident"), 2000, seed=56702)
        audit = recalibration_audit(
            s.forecasts, s.outcomes, methods=("raw", "platt"), n_bootstrap=10, seed=10
        )
        assert [r.method for r in audit.results] == ["raw", "platt"]


class TestHonestNegatives:
    """The cases the specification requires to be published, not hidden."""

    def test_recalibration_is_worse_on_a_calibrated_forecast_with_little_data(self):
        # Nothing to fix, so both learned maps only add estimation variance.
        s = sample_forecast(get_spec("calibrated"), 200, seed=56)
        audit = recalibration_audit(s.forecasts, s.outcomes, n_bootstrap=400, seed=56)
        platt = audit.by_method("platt")
        isotonic = audit.by_method("isotonic")
        assert platt.delta_brier > 0.0
        assert isotonic.delta_brier > 0.0
        # Neither may ever be reported as an improvement on a forecast that is
        # already calibrated. At this seed the paired interval still straddles
        # zero, so the honest verdict is "indistinguishable", not "worse".
        assert platt.verdict in ("worse", "indistinguishable")
        assert isotonic.verdict in ("worse", "indistinguishable")

    def test_harm_on_a_calibrated_forecast_is_significant_at_some_seeds(self):
        # Rather than pick the seed that gives the verdict, count them: over
        # 20 seeds at n = 200, how often is the harm large enough to clear a
        # 90 % paired bootstrap interval. The count is the finding.
        spec = get_spec("calibrated")
        worse = 0
        improved = 0
        for k in range(20):
            s = sample_forecast(spec, 200, seed=56800 + k)
            audit = recalibration_audit(s.forecasts, s.outcomes, n_bootstrap=200, seed=k)
            verdicts = {audit.by_method(m).verdict for m in ("platt", "isotonic")}
            worse += "worse" in verdicts
            improved += "improved" in verdicts
        assert worse >= 2, f"expected significant harm at some seeds, got {worse}/20"
        assert improved <= worse, (
            f"recalibration significantly improved a calibrated forecast at "
            f"{improved}/20 seeds against {worse}/20 harmed"
        )

    def test_isotonic_harms_more_often_than_platt_on_small_samples(self):
        sweep = sample_size_sweep(
            get_spec("overconfident"),
            n_samples_grid=[100],
            n_replicates=40,
            seed=56,
        )
        platt = next(r for r in sweep.rows if r.method == "platt")
        isotonic = next(r for r in sweep.rows if r.method == "isotonic")
        assert isotonic.harm_rate > platt.harm_rate
        assert isotonic.harm_rate > 0.5

    def test_the_harm_rate_falls_as_data_grows(self):
        sweep = sample_size_sweep(
            get_spec("overconfident"),
            n_samples_grid=[100, 2000],
            n_replicates=30,
            seed=56,
        )
        small = next(r for r in sweep.rows if r.method == "platt" and r.n_samples == 100)
        large = next(r for r in sweep.rows if r.method == "platt" and r.n_samples == 2000)
        assert large.harm_rate < small.harm_rate

    def test_isotonic_log_score_can_be_worse_than_raw_even_when_brier_improves(self):
        # The two scores disagree because isotonic emits exact 0 and 1.
        s = sample_forecast(get_spec("overconfident"), 400, seed=56703)
        audit = recalibration_audit(s.forecasts, s.outcomes, n_bootstrap=50, seed=11)
        raw = audit.by_method("raw")
        iso = audit.by_method("isotonic")
        assert iso.log_score > raw.log_score


class TestSampleSizeSweep:
    def test_sweep_covers_every_cell(self):
        sweep = sample_size_sweep(
            get_spec("overconfident"),
            n_samples_grid=[100, 400],
            n_replicates=8,
            seed=1,
        )
        assert len(sweep.rows) == 6

    def test_raw_row_has_zero_delta_everywhere(self):
        sweep = sample_size_sweep(
            get_spec("overconfident"), n_samples_grid=[200], n_replicates=5, seed=2
        )
        raw = next(r for r in sweep.rows if r.method == "raw")
        assert raw.mean_delta_brier == 0.0
        assert raw.harm_rate == 0.0

    def test_harm_and_improved_rates_do_not_exceed_one(self):
        sweep = sample_size_sweep(
            get_spec("overconfident"), n_samples_grid=[200], n_replicates=5, seed=3
        )
        for r in sweep.rows:
            assert 0.0 <= r.harm_rate <= 1.0
            assert 0.0 <= r.improved_rate <= 1.0
            assert r.harm_rate + r.improved_rate <= 1.0 + 1e-15

    def test_table_renders(self):
        sweep = sample_size_sweep(
            get_spec("overconfident"), n_samples_grid=[200], n_replicates=4, seed=4
        )
        assert len(sweep.table().splitlines()) == len(sweep.rows) + 2

    def test_crossover_returns_none_when_a_method_never_wins(self):
        sweep = sample_size_sweep(
            get_spec("calibrated"), n_samples_grid=[100, 200], n_replicates=10, seed=5
        )
        assert sweep.crossover("isotonic") is None

    def test_crossover_is_a_swept_sample_size_when_a_method_does_win(self):
        sweep = sample_size_sweep(
            get_spec("biased_high"),
            n_samples_grid=[200, 2000, 8000],
            n_replicates=10,
            seed=6,
        )
        value = sweep.crossover("platt")
        assert value in (200, 2000, 8000)


class TestFittedMapsBeatRawInSample:
    def test_in_sample_brier_never_rises_for_isotonic(self, overconfident_large):
        # Isotonic minimises in-sample squared error subject to monotonicity,
        # so it cannot be worse in sample. Out of sample is another matter.
        s = overconfident_large
        model = IsotonicCalibration().fit(s.forecasts, s.outcomes)
        assert brier_score(model.predict(s.forecasts), s.outcomes) <= brier_score(
            s.forecasts, s.outcomes
        )
