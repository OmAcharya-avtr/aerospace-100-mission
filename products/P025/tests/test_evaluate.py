"""Benchmark harness: protocol wiring, calibration and validation.

These tests run a deliberately small campaign -- eight scenarios, a short
window, a five-tree forest -- because their job is to check that the harness
wires the pieces together correctly and honours its own protocol, not to
measure performance.  The performance measurement is the validation campaign.
"""

from __future__ import annotations

import numpy as np
import pytest

from fdiscope.classifier import FaultClassifier
from fdiscope.evaluate import (
    FAR_START,
    BenchmarkConfig,
    build_cusum_bank,
    build_default_bank,
    calibrate_all_thresholds,
    calibrate_threshold,
    class_labels,
    default_scenario_sets,
    default_signature_specs,
    design_thresholds,
    evaluate_detection,
    evaluate_isolation,
    harvest_training_rows,
    healthy_calibration_runs,
    method_names,
    run_scenarios,
    sequential_alarms,
    sequential_scores,
    window_features_batch,
    window_scores,
)
from fdiscope.faults import FAULT_CLASSES, FaultType
from fdiscope.features import N_FEATURES, window_features
from fdiscope.scenarios import sample_scenarios

SMALL = BenchmarkConfig(det_window=10, iso_window=30, alpha=1e-3)


@pytest.fixture(scope="module")
def small_bank():
    return build_default_bank(SMALL, n_onsets=2)


@pytest.fixture(scope="module")
def small_campaign(small_bank):
    scenarios = sample_scenarios(16, 70000, n_steps=1400)
    runs = run_scenarios(scenarios, SMALL)
    x, y = harvest_training_rows(scenarios, runs, SMALL)
    clf = FaultClassifier(n_estimators=8, max_depth=5, random_state=0).fit(x, y)
    return scenarios, runs, clf, x, y


class TestConfig:
    @pytest.mark.parametrize("kwargs", [{"det_window": 2}, {"iso_window": 1}, {"alpha": 0.0}])
    def test_rejects_bad_parameters(self, kwargs):
        with pytest.raises(ValueError):
            BenchmarkConfig(**kwargs)

    def test_method_names_order(self):
        assert method_names() == ["chi2_short", "chi2_long", "cusum", "glr", "learned"]
        assert method_names(False) == ["chi2_short", "chi2_long", "cusum", "glr"]

    def test_class_labels_match_the_taxonomy(self):
        assert class_labels() == tuple(f.value for f in FAULT_CLASSES)

    def test_design_thresholds_cover_only_the_closed_form_methods(self):
        th = design_thresholds(SMALL)
        assert set(th) == {"chi2_short", "chi2_long", "cusum"}
        assert all(v > 0 for v in th.values())


class TestSignatureSpecs:
    def test_one_spec_per_faulted_class(self):
        specs = default_signature_specs()
        assert set(specs) == set(FAULT_CLASSES) - {FaultType.NONE}
        for fault, spec in specs.items():
            assert spec.kind is fault

    def test_bank_covers_every_faulted_class(self, small_bank):
        assert set(small_bank.faults) == set(FAULT_CLASSES) - {FaultType.NONE}
        assert small_bank.window == SMALL.iso_window


class TestCusumBankConstruction:
    def test_four_signed_channel_directions(self):
        bank = build_cusum_bank(1.0, 5.0)
        assert bank.names == ("ch0_pos", "ch0_neg", "ch1_pos", "ch1_neg")
        directions = np.stack([d.direction for d in bank.detectors.values()])
        assert np.allclose(directions, [[1, 0], [-1, 0], [0, 1], [0, -1]])


class TestTrainingRows:
    def test_shapes_and_label_range(self, small_campaign):
        _, _, _, x, y = small_campaign
        assert x.shape[1] == N_FEATURES
        assert x.shape[0] == y.size
        assert y.min() >= 0 and y.max() < len(FAULT_CLASSES)

    def test_every_scenario_contributes_null_rows(self, small_campaign):
        scenarios, runs, _, _, y = small_campaign
        # Two pre-onset windows per scenario are labelled NONE, plus the four
        # windows of each genuinely healthy scenario.
        n_healthy = sum(1 for s in scenarios if s.label is FaultType.NONE)
        expected_none = 2 * len(scenarios) + 4 * n_healthy
        assert int(np.count_nonzero(y == 0)) == expected_none

    def test_rejects_impossible_offsets(self, small_campaign):
        scenarios, runs, _, _, _ = small_campaign
        with pytest.raises(ValueError, match="no training rows"):
            harvest_training_rows(
                scenarios, runs, SMALL, fault_offsets=(100000,), null_offsets=()
            )


class TestSequentialScores:
    def test_every_method_is_present_and_causal(self, small_campaign, small_bank):
        scenarios, runs, clf, _, _ = small_campaign
        scores = sequential_scores(runs[1], SMALL, small_bank, clf, (300, 800))
        assert set(scores) == set(method_names())
        for name, (score, idx) in scores.items():
            assert score.shape == idx.shape
            assert idx.min() >= 300 and idx.max() < 800, name

    def test_windowed_methods_start_late_enough(self, small_campaign, small_bank):
        scenarios, runs, clf, _, _ = small_campaign
        scores = sequential_scores(runs[1], SMALL, small_bank, clf, (0, 400))
        assert scores["chi2_short"][1][0] == SMALL.det_window - 1
        assert scores["chi2_long"][1][0] == SMALL.iso_window - 1
        assert scores["glr"][1][0] == SMALL.iso_window - 1
        assert scores["learned"][1][0] == SMALL.iso_window - 1
        assert scores["cusum"][1][0] == 0

    def test_alarms_are_thresholded_scores(self, small_campaign, small_bank):
        scenarios, runs, clf, _, _ = small_campaign
        thresholds = dict.fromkeys(method_names(), 1.0)
        scores = sequential_scores(runs[2], SMALL, small_bank, clf, (300, 700))
        alarms = sequential_alarms(runs[2], SMALL, small_bank, clf, thresholds, (300, 700))
        for name in method_names():
            assert np.array_equal(alarms[name][0], scores[name][0] > 1.0)

    def test_classifier_may_be_omitted(self, small_campaign, small_bank):
        scenarios, runs, _, _, _ = small_campaign
        scores = sequential_scores(runs[0], SMALL, small_bank, None, (300, 600))
        assert "learned" not in scores


class TestCalibration:
    def test_calibrate_threshold_known_answer(self):
        # scores 0..99, target 0.10 -> the 0.90 quantile with method "higher"
        # is 90, nudged up by one ULP so "score > threshold" leaves exactly the
        # nine values 91..99 above it.
        scores = np.arange(100.0)
        h = calibrate_threshold(scores, 0.10)
        assert 90.0 < h < 90.0000001
        assert int(np.count_nonzero(scores > h)) == 9

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.5])
    def test_calibrate_threshold_rejects_bad_target(self, bad):
        with pytest.raises(ValueError, match="target_far"):
            calibrate_threshold(np.arange(10.0), bad)

    def test_calibrate_threshold_rejects_empty(self):
        with pytest.raises(ValueError, match="at least one"):
            calibrate_threshold(np.zeros(0), 0.1)

    def test_all_thresholds_are_produced(self, small_campaign, small_bank):
        _, _, clf, _, _ = small_campaign
        _, calib = healthy_calibration_runs(6, 71000, SMALL)
        thresholds = calibrate_all_thresholds(calib, SMALL, small_bank, clf, 0.30)
        assert set(thresholds) == set(method_names())
        assert all(np.isfinite(v) for v in thresholds.values())

    def test_calibration_hits_its_target_on_its_own_data(self, small_campaign, small_bank):
        _, _, clf, _, _ = small_campaign
        _, calib = healthy_calibration_runs(10, 72000, SMALL)
        thresholds = calibrate_all_thresholds(calib, SMALL, small_bank, clf, 0.30)
        for name in method_names():
            fired = 0
            for run in calib:
                score, _ = sequential_scores(
                    run, SMALL, small_bank, clf, (FAR_START, run.residual.shape[0])
                )[name]
                fired += int(np.any(score > thresholds[name]))
            assert fired <= 3, name

    def test_rejects_an_empty_calibration_set(self, small_bank):
        with pytest.raises(ValueError, match="at least one fault-free run"):
            calibrate_all_thresholds([], SMALL, small_bank, None, 0.1)

    @pytest.mark.parametrize("bad", [0.0, 1.0])
    def test_rejects_a_bad_target(self, small_campaign, small_bank, bad):
        _, _, clf, _, _ = small_campaign
        _, calib = healthy_calibration_runs(2, 73000, SMALL)
        with pytest.raises(ValueError, match="target_run_far"):
            calibrate_all_thresholds(calib, SMALL, small_bank, clf, bad)

    def test_healthy_calibration_runs_are_all_fault_free(self):
        scenarios, runs = healthy_calibration_runs(5, 74000, SMALL)
        assert all(s.label is FaultType.NONE for s in scenarios)
        assert all(r.onset_step == -1 for r in runs)


class TestEvaluateDetection:
    def test_bookkeeping_is_self_consistent(self, small_campaign, small_bank):
        scenarios, runs, clf, _, _ = small_campaign
        thresholds = dict.fromkeys(method_names(), 25.0)
        results = evaluate_detection(
            scenarios, runs, SMALL, small_bank, clf, thresholds, delay_horizon=400
        )
        n_faulted = sum(1 for s in scenarios if s.label is not FaultType.NONE)
        n_healthy = len(scenarios) - n_faulted
        for name, m in results.items():
            assert m.n_faulted == n_faulted
            assert m.delays.size + m.censored == n_faulted, name
            assert m.per_run_delay.size == n_faulted
            assert m.far_runs[1] == n_healthy
            assert 0.0 <= m.far_per_sample <= 1.0

    def test_delays_are_non_negative_and_within_the_horizon(self, small_campaign, small_bank):
        scenarios, runs, clf, _, _ = small_campaign
        thresholds = dict.fromkeys(method_names(), 25.0)
        results = evaluate_detection(
            scenarios, runs, SMALL, small_bank, clf, thresholds, delay_horizon=400
        )
        for m in results.values():
            assert np.all(m.delays >= 0.0)
            assert np.all(m.delays < 400.0)

    def test_delays_for_selects_by_class(self, small_campaign, small_bank):
        scenarios, runs, clf, _, _ = small_campaign
        thresholds = dict.fromkeys(method_names(), 25.0)
        results = evaluate_detection(scenarios, runs, SMALL, small_bank, clf, thresholds)
        m = results["chi2_short"]
        total = sum(m.delays_for(f).size for f in FAULT_CLASSES if f is not FaultType.NONE)
        assert total == m.n_faulted

    def test_a_huge_threshold_censors_everything(self, small_campaign, small_bank):
        scenarios, runs, clf, _, _ = small_campaign
        thresholds = dict.fromkeys(method_names(), 1e12)
        results = evaluate_detection(scenarios, runs, SMALL, small_bank, clf, thresholds)
        for m in results.values():
            assert m.censored == m.n_faulted
            assert m.detection_rate == 0.0
            assert m.far_samples[0] == 0


class TestWindowScoresAndIsolation:
    def test_window_scores_shapes(self, small_campaign, small_bank):
        scenarios, runs, clf, _, _ = small_campaign
        pos, neg = window_scores(scenarios, runs, SMALL, small_bank, clf)
        assert set(pos) == set(neg) == {"chi2_long", "cusum", "glr", "learned"}
        sizes = {k: v.size for k, v in pos.items()}
        assert len(set(sizes.values())) == 1
        assert all(v.size > 0 for v in neg.values())

    def test_window_feature_batch_matches_the_reference(self):
        rng = np.random.default_rng(0)
        windows = rng.standard_normal((7, 25, 2))
        batch = window_features_batch(windows)
        for i in range(7):
            assert np.allclose(batch[i], window_features(windows[i]), atol=1e-12)

    def test_isolation_outputs_align(self, small_campaign, small_bank):
        scenarios, runs, clf, _, _ = small_campaign
        out = evaluate_isolation(scenarios, runs, SMALL, small_bank, clf)
        assert set(out) == {"glr", "learned"}
        assert np.array_equal(out["glr"].truth, out["learned"].truth)
        for outcome in out.values():
            assert outcome.truth.size == len(scenarios)
            assert outcome.predicted.min() >= 0
            assert outcome.predicted.max() < len(FAULT_CLASSES)

    def test_isolation_offset_is_honoured(self, small_campaign, small_bank):
        scenarios, runs, clf, _, _ = small_campaign
        aligned = evaluate_isolation(scenarios, runs, SMALL, small_bank, clf)
        shifted = evaluate_isolation(scenarios, runs, SMALL, small_bank, clf, offset=-200)
        assert not np.array_equal(aligned["glr"].predicted, shifted["glr"].predicted)

    def test_isolation_without_a_classifier(self, small_campaign, small_bank):
        scenarios, runs, _, _, _ = small_campaign
        out = evaluate_isolation(scenarios, runs, SMALL, small_bank, None)
        assert set(out) == {"glr"}


class TestScenarioSets:
    def test_default_sets_are_disjoint_and_balanced(self):
        train, test = default_scenario_sets(16, 16)
        assert {s.seed for s in train}.isdisjoint({s.seed for s in test})
        assert len({s.label for s in train}) == len(FAULT_CLASSES)
