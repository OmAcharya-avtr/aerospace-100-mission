"""Tests for the channel predictors, their confidence output and its gate."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from acmpilot.channel import snr_db_path
from acmpilot.predictor import (
    QUANTILES,
    ChannelPredictor,
    GaussMarkovPredictor,
    PredictivePolicy,
    QuantilePredictor,
    analytic_calibration_report,
    calibration_report,
    make_lag_features,
)
from acmpilot.simulate import run_policy


class TestMakeLagFeatures:
    def test_known_answer_shapes_and_values(self):
        # x = [0, 1, 2, ..., 9], delay 2, n_lags 3.
        # First valid target slot is 2 + 3 - 1 = 4.
        # Row for slot 4: lags are x[2], x[1], x[0] = 2, 1, 0 and the two first
        # differences are 2-1 = 1 and 1-0 = 1. Target is x[4] = 4.
        x = np.arange(10.0)
        features, target, slots = make_lag_features(x, delay_slots=2, n_lags=3)
        assert features.shape == (6, 5)
        np.testing.assert_array_equal(slots, np.arange(4, 10))
        np.testing.assert_allclose(features[0], np.array([2.0, 1.0, 0.0, 1.0, 1.0]))
        assert target[0] == pytest.approx(4.0)

    def test_single_lag_has_no_difference_columns(self):
        features, _, _ = make_lag_features(np.arange(10.0), delay_slots=0, n_lags=1)
        assert features.shape == (10, 1)

    def test_zero_delay_uses_the_current_sample(self):
        features, target, slots = make_lag_features(
            np.arange(10.0), delay_slots=0, n_lags=1
        )
        np.testing.assert_allclose(features[:, 0], target)
        np.testing.assert_array_equal(slots, np.arange(10))

    def test_feature_width_formula(self):
        for n_lags in (1, 2, 4, 8):
            features, _, _ = make_lag_features(
                np.arange(50.0), delay_slots=1, n_lags=n_lags
            )
            assert features.shape[1] == 2 * n_lags - 1

    def test_rejects_negative_delay(self):
        with pytest.raises(ValueError, match="delay_slots"):
            make_lag_features(np.arange(20.0), delay_slots=-1)

    def test_rejects_zero_lags(self):
        with pytest.raises(ValueError, match="n_lags"):
            make_lag_features(np.arange(20.0), delay_slots=0, n_lags=0)

    def test_rejects_series_too_short(self):
        with pytest.raises(ValueError, match="need >"):
            make_lag_features(np.arange(5.0), delay_slots=10, n_lags=4)

    @given(
        delay=st.integers(min_value=0, max_value=10),
        n_lags=st.integers(min_value=1, max_value=6),
    )
    @settings(max_examples=30, deadline=None)
    def test_property_row_count(self, delay, n_lags):
        x = np.arange(80.0)
        features, target, slots = make_lag_features(
            x, delay_slots=delay, n_lags=n_lags
        )
        expected = 80 - (delay + n_lags - 1)
        assert features.shape[0] == expected == target.size == slots.size


class TestGaussMarkovPredictor:
    def test_fit_estimates_three_parameters(self, config):
        train = snr_db_path(config, 20_000, 1)
        pred = GaussMarkovPredictor(delay_slots=10).fit(train)
        assert pred.mean_db == pytest.approx(float(train.mean()), rel=1e-12)
        assert pred.std_db == pytest.approx(float(train.std(ddof=1)), rel=1e-12)
        assert 0.0 < pred.rho < 1.0
        assert pred.rho == pytest.approx(config.rho, abs=0.02)

    def test_not_learned(self, fitted):
        assert fitted[0].learned is False
        assert "not learned" in fitted[0].name

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="fit must be called"):
            GaussMarkovPredictor(delay_slots=5).predict(np.zeros((2, 3)))

    def test_rejects_tiny_training_set(self):
        with pytest.raises(ValueError, match="16 training samples"):
            GaussMarkovPredictor(delay_slots=5).fit(np.zeros(8))

    def test_known_answer_zero_delay_is_the_report_itself(self, config):
        # At d = 0 equation (1) gives rho**0 = 1, so the prediction is exactly
        # the last report and the conditional variance is zero.
        train = snr_db_path(config, 5000, 2)
        pred = GaussMarkovPredictor(delay_slots=0).fit(train)
        features = np.array([[12.0, 11.0, 1.0], [5.0, 4.0, 1.0]])
        centre, spread = pred.predict(features)
        np.testing.assert_allclose(centre, np.array([12.0, 5.0]), rtol=1e-12)
        np.testing.assert_allclose(spread, np.zeros(2), atol=1e-12)

    def test_spread_grows_with_horizon(self, config):
        train = snr_db_path(config, 20_000, 3)
        features = np.array([[12.0, 11.0, 1.0]])
        spreads = [
            float(GaussMarkovPredictor(delay_slots=d).fit(train).predict(features)[1][0])
            for d in (1, 5, 20, 100)
        ]
        assert spreads == sorted(spreads)

    def test_spread_is_constant_across_rows(self, fitted):
        _, spread = fitted[0].predict(np.array([[12.0, 11.0, 10.0, 1.0, 1.0, 2.0, 1.0]][0]
                                              ).reshape(1, -1))
        assert spread.size == 1

    def test_prediction_shrinks_towards_the_mean(self, config):
        train = snr_db_path(config, 20_000, 4)
        pred = GaussMarkovPredictor(delay_slots=30).fit(train)
        high = float(pred.predict(np.array([[40.0, 0.0, 0.0]]))[0][0])
        assert pred.mean_db < high < 40.0

    def test_satisfies_the_protocol(self, fitted):
        assert isinstance(fitted[0], ChannelPredictor)


class TestQuantilePredictor:
    def test_quantile_levels(self):
        assert QUANTILES == (0.1, 0.5, 0.9)

    def test_learned_flag_and_name(self, fitted):
        assert fitted[1].learned is True
        assert fitted[1].name == "learned quantile GBR"

    def test_three_models_fitted(self, fitted):
        assert set(fitted[1].models) == set(QUANTILES)

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="fit must be called"):
            QuantilePredictor(delay_slots=5).predict(np.zeros((2, 3)))

    def test_crossing_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="fit must be called"):
            QuantilePredictor(delay_slots=5).crossing_fraction(np.zeros((2, 3)))

    def test_save_before_fit_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="nothing to save"):
            QuantilePredictor(delay_slots=5).save(tmp_path / "m.joblib")

    def test_rejects_one_dimensional_features(self):
        with pytest.raises(ValueError, match="2-D"):
            QuantilePredictor(delay_slots=1).fit(np.zeros(10), np.zeros(10))

    def test_rejects_row_mismatch(self):
        with pytest.raises(ValueError, match="feature rows"):
            QuantilePredictor(delay_slots=1).fit(np.zeros((10, 3)), np.zeros(9))

    def test_quantiles_ordered_after_repair(self, fitted, config):
        _, learned, delay, n_lags = fitted
        test = snr_db_path(config, 3000, 999)
        features, _, _ = make_lag_features(test, delay_slots=delay, n_lags=n_lags)
        quants = learned.predict_quantiles(features)
        assert np.all(quants[0.1] <= quants[0.5] + 1e-12)
        assert np.all(quants[0.5] <= quants[0.9] + 1e-12)

    def test_spread_is_non_negative(self, fitted, config):
        _, learned, delay, n_lags = fitted
        test = snr_db_path(config, 3000, 998)
        features, _, _ = make_lag_features(test, delay_slots=delay, n_lags=n_lags)
        _, spread = learned.predict(features)
        assert np.all(spread >= 0.0)

    def test_spread_varies_across_rows(self, fitted, config):
        """The learned predictor is heteroscedastic; the analytic one is not.

        This is the only structural thing the learned model can add over
        equations (1)-(2), so it is checked explicitly.
        """
        analytic, learned, delay, n_lags = fitted
        test = snr_db_path(config, 5000, 997)
        features, _, _ = make_lag_features(test, delay_slots=delay, n_lags=n_lags)
        _, learned_spread = learned.predict(features)
        _, analytic_spread = analytic.predict(features)
        assert float(learned_spread.std()) > 0.0
        assert float(analytic_spread.std()) == pytest.approx(0.0, abs=1e-12)

    def test_crossing_fraction_in_unit_interval(self, fitted, config):
        _, learned, delay, n_lags = fitted
        test = snr_db_path(config, 2000, 996)
        features, _, _ = make_lag_features(test, delay_slots=delay, n_lags=n_lags)
        assert 0.0 <= learned.crossing_fraction(features) <= 1.0

    def test_round_trip_joblib(self, fitted, config, tmp_path):
        _, learned, delay, n_lags = fitted
        path = tmp_path / "predictor.joblib"
        learned.save(path)
        restored = QuantilePredictor.load(path)
        test = snr_db_path(config, 1500, 995)
        features, _, _ = make_lag_features(test, delay_slots=delay, n_lags=n_lags)
        np.testing.assert_allclose(
            restored.predict(features)[0], learned.predict(features)[0]
        )
        assert restored.delay_slots == learned.delay_slots

    def test_deterministic_given_the_same_data_and_seed(self, config):
        train = snr_db_path(config, 3000, 5)
        features, target, _ = make_lag_features(train, delay_slots=5, n_lags=3)
        a = QuantilePredictor(delay_slots=5, n_estimators=10).fit(features, target)
        b = QuantilePredictor(delay_slots=5, n_estimators=10).fit(features, target)
        np.testing.assert_allclose(a.predict(features)[0], b.predict(features)[0])

    def test_satisfies_the_protocol(self, fitted):
        assert isinstance(fitted[1], ChannelPredictor)


class TestCalibrationReport:
    def test_fields_present(self, fitted, config):
        _, learned, delay, n_lags = fitted
        test = snr_db_path(config, 4000, 994)
        features, target, _ = make_lag_features(test, delay_slots=delay, n_lags=n_lags)
        report = calibration_report(learned, features, target)
        for key in (
            "coverage_80", "below_q10", "above_q90", "mae_q50", "rmse_q50",
            "mean_spread_db", "crossing_fraction", "n_rows", "pinball_q10",
            "pinball_q50", "pinball_q90",
        ):
            assert key in report

    def test_coverage_near_nominal(self, fitted, config):
        _, learned, delay, n_lags = fitted
        test = snr_db_path(config, 8000, 993)
        features, target, _ = make_lag_features(test, delay_slots=delay, n_lags=n_lags)
        report = calibration_report(learned, features, target)
        assert 0.70 <= report["coverage_80"] <= 0.90

    def test_tail_fractions_sum_with_coverage(self, fitted, config):
        _, learned, delay, n_lags = fitted
        test = snr_db_path(config, 4000, 992)
        features, target, _ = make_lag_features(test, delay_slots=delay, n_lags=n_lags)
        report = calibration_report(learned, features, target)
        total = report["coverage_80"] + report["below_q10"] + report["above_q90"]
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_rmse_at_least_mae(self, fitted, config):
        _, learned, delay, n_lags = fitted
        test = snr_db_path(config, 4000, 991)
        features, target, _ = make_lag_features(test, delay_slots=delay, n_lags=n_lags)
        report = calibration_report(learned, features, target)
        assert report["rmse_q50"] >= report["mae_q50"]

    def test_analytic_report_fields_match(self, fitted, config):
        analytic, learned, delay, n_lags = fitted
        test = snr_db_path(config, 4000, 990)
        features, target, _ = make_lag_features(test, delay_slots=delay, n_lags=n_lags)
        learned_report = calibration_report(learned, features, target)
        analytic_report = analytic_calibration_report(analytic, features, target)
        shared = set(analytic_report) - {"pinball_q10", "pinball_q50", "pinball_q90"}
        assert shared <= set(learned_report)
        assert analytic_report["n_rows"] == learned_report["n_rows"]

    def test_analytic_coverage_near_nominal(self, fitted, config):
        analytic, _, delay, n_lags = fitted
        test = snr_db_path(config, 8000, 989)
        features, target, _ = make_lag_features(test, delay_slots=delay, n_lags=n_lags)
        report = analytic_calibration_report(analytic, features, target)
        assert 0.70 <= report["coverage_80"] <= 0.90

    def test_error_grows_with_horizon(self, config):
        maes = []
        for delay in (1, 10, 50):
            train = snr_db_path(config, 8000, 7)
            predictor = GaussMarkovPredictor(delay_slots=delay).fit(train)
            test = snr_db_path(config, 8000, 8)
            features, target, _ = make_lag_features(test, delay_slots=delay, n_lags=2)
            maes.append(
                analytic_calibration_report(predictor, features, target)["mae_q50"]
            )
        assert maes == sorted(maes)


class TestPredictivePolicy:
    def test_is_causal_and_named(self, fitted):
        analytic, learned, delay, n_lags = fitted
        policy = PredictivePolicy(learned, delay_slots=delay, n_lags=n_lags)
        assert policy.causal is True
        assert "learned quantile GBR" in policy.name
        assert "gate k=0.5" in policy.name

    def test_custom_label(self, fitted):
        policy = PredictivePolicy(
            fitted[1], delay_slots=fitted[2], n_lags=fitted[3], label="custom"
        )
        assert policy.name == "custom"

    def test_rejects_negative_gate(self, fitted):
        with pytest.raises(ValueError, match="gate_k"):
            PredictivePolicy(fitted[1], delay_slots=5, gate_k=-1.0)

    def test_selection_shape_and_range(self, fitted, table, config):
        analytic, learned, delay, n_lags = fitted
        test = snr_db_path(config, 3000, 988)
        policy = PredictivePolicy(learned, delay_slots=delay, n_lags=n_lags)
        chosen = policy.select(table, test, test)
        assert chosen.shape == test.shape
        assert chosen.min() >= 0
        assert chosen.max() <= table.n_modes - 1

    def test_does_not_read_the_true_snr(self, fitted, table, config):
        analytic, learned, delay, n_lags = fitted
        observed = snr_db_path(config, 2000, 987)
        policy = PredictivePolicy(learned, delay_slots=delay, n_lags=n_lags)
        a = policy.select(table, observed, np.zeros_like(observed))
        b = policy.select(table, observed, np.full_like(observed, 99.0))
        np.testing.assert_array_equal(a, b)

    def test_larger_gate_never_selects_higher(self, fitted, table, config):
        analytic, learned, delay, n_lags = fitted
        observed = snr_db_path(config, 3000, 986)
        low = PredictivePolicy(
            learned, delay_slots=delay, n_lags=n_lags, gate_k=0.0
        ).select(table, observed, observed)
        high = PredictivePolicy(
            learned, delay_slots=delay, n_lags=n_lags, gate_k=1.5
        ).select(table, observed, observed)
        assert np.all(high <= low)

    def test_gate_reduces_outage(self, fitted, table, config):
        analytic, learned, delay, n_lags = fitted
        test = snr_db_path(config, 8000, 985)
        outages = []
        for gate in (0.0, 0.5, 1.0):
            policy = PredictivePolicy(
                learned, delay_slots=delay, n_lags=n_lags, gate_k=gate
            )
            outages.append(
                run_policy(
                    table, test, policy, delay_slots=delay
                ).accounting.outage_fraction
            )
        assert outages[0] > outages[1] > outages[2]

    def test_beats_the_floor_policy(self, fitted, table, config):
        analytic, learned, delay, n_lags = fitted
        test = snr_db_path(config, 8000, 984)
        policy = PredictivePolicy(learned, delay_slots=delay, n_lags=n_lags)
        res = run_policy(table, test, policy, delay_slots=delay)
        assert res.accounting.goodput_bit_per_symbol > float(
            table.spectral_efficiencies[0]
        )

    def test_both_predictors_usable_as_policies(self, fitted, table, config):
        analytic, learned, delay, n_lags = fitted
        test = snr_db_path(config, 5000, 983)
        for predictor in (analytic, learned):
            policy = PredictivePolicy(predictor, delay_slots=delay, n_lags=n_lags)
            res = run_policy(table, test, policy, delay_slots=delay)
            assert res.accounting.goodput_bit_per_symbol > 0.0
            assert res.causal is True
