"""End-to-end integration: inject, generate residuals, detect, isolate.

One test walks the whole pipeline exactly as a user would; the rest check the
properties that only appear when the stages are composed.
"""

from __future__ import annotations

import numpy as np

from fdiscope.analytic import (
    cusum_threshold_for_arl0,
    normalised_bias_signature,
)
from fdiscope.classifier import FaultClassifier
from fdiscope.detectors import ChiSquaredDetector, CusumDetector, detection_delay
from fdiscope.evaluate import (
    BenchmarkConfig,
    build_default_bank,
    calibrate_all_thresholds,
    evaluate_detection,
    evaluate_isolation,
    harvest_training_rows,
    healthy_calibration_runs,
    method_names,
    run_scenarios,
)
from fdiscope.faults import FaultSpec, FaultType, class_index
from fdiscope.features import feature_matrix
from fdiscope.isolation import isolate_window
from fdiscope.metrics import confusion_report
from fdiscope.plant import PlantConfig, loop_matrices
from fdiscope.residuals import nis_consistency
from fdiscope.scenarios import sample_scenarios
from fdiscope.simulate import LoopConfig, build_filter, simulate_loop

CFG = BenchmarkConfig(det_window=10, iso_window=40, alpha=1e-3)


class TestEndToEndSingleFault:
    def test_inject_generate_detect_isolate(self):
        # 1. Design the loop and the detectors from the model alone.
        plant = PlantConfig()
        kf = build_filter(loop_matrices(plant))
        sigma_rate = float(np.sqrt(plant.gyro_var_rad2_s2))
        bias = 4.0 * sigma_rate
        direction, mu = normalised_bias_signature(kf, [0.0, bias])
        h_cusum = cusum_threshold_for_arl0(2000.0, mu)
        chi = ChiSquaredDetector(window=25, dim=2, alpha=1e-3)
        cusum = CusumDetector(direction=direction, mu=mu, threshold=h_cusum)

        # 2. Inject the fault into the closed loop.
        onset = 600
        spec = FaultSpec(FaultType.SENSOR_BIAS, onset, bias, 1)
        run = simulate_loop(LoopConfig(n_steps=1600, seed=99), spec)

        # 3. The residual must be consistent before the fault and not after.
        pre = nis_consistency(run.residual[300:onset])
        assert pre.consistent
        assert np.mean(run.nis[onset + 200 :]) > 3.0 * np.mean(run.nis[300:onset])

        # 4. Detect, with no alarms before onset and a bounded delay.
        chi_out = chi.run(run.residual)
        cusum_out = cusum.run(run.residual)
        assert not chi_out.alarm[300:onset].any()
        d_chi = detection_delay(chi_out.alarm, onset)
        d_cusum = detection_delay(cusum_out.alarm, onset)
        assert np.isfinite(d_chi) and d_chi < 100
        assert np.isfinite(d_cusum) and d_cusum < 100

        # 5. Isolate with the classical GLR bank on the post-onset window.
        bank = build_default_bank(CFG, n_onsets=2)
        result = isolate_window(run.residual[onset : onset + CFG.iso_window], bank, alpha=1e-3)
        assert result.fault is not FaultType.NONE
        assert result.statistic > result.threshold
        assert 0.0 < result.confidence <= 1.0

        # 6. And with the learned classifier, trained on disjoint seeds.
        scenarios = sample_scenarios(24, 80000, n_steps=1400)
        runs = run_scenarios(scenarios, CFG)
        x, y = harvest_training_rows(scenarios, runs, CFG)
        clf = FaultClassifier(n_estimators=30, max_depth=8, random_state=0).fit(x, y)
        feats, _ = feature_matrix(
            run.residual[onset : onset + CFG.iso_window], CFG.iso_window, 1
        )
        pred = clf.predict_with_confidence(feats)
        assert pred.classes[0] is not FaultType.NONE
        assert pred.detection_score[0] > 0.5

    def test_a_healthy_run_produces_no_isolation(self):
        bank = build_default_bank(CFG, n_onsets=2)
        run = simulate_loop(LoopConfig(n_steps=1200, seed=101))
        declared = 0
        for start in range(300, 1100, 40):
            result = isolate_window(run.residual[start : start + CFG.iso_window], bank, 1e-4)
            declared += int(result.fault is not FaultType.NONE)
        assert declared <= 1, f"{declared} spurious isolations on a fault-free run"


class TestEndToEndCampaign:
    def test_small_campaign_runs_and_reports(self):
        train = sample_scenarios(24, 81000, n_steps=1400)
        test = sample_scenarios(24, 85000, n_steps=1400)
        bank = build_default_bank(CFG, n_onsets=2)
        train_runs = run_scenarios(train, CFG)
        test_runs = run_scenarios(test, CFG)
        x, y = harvest_training_rows(train, train_runs, CFG)
        clf = FaultClassifier(n_estimators=30, max_depth=8, random_state=0).fit(x, y)
        _, calib = healthy_calibration_runs(12, 87000, CFG)
        thresholds = calibrate_all_thresholds(calib, CFG, bank, clf, 0.20)

        results = evaluate_detection(
            test, test_runs, CFG, bank, clf, thresholds, delay_horizon=400
        )
        assert set(results) == set(method_names())
        for name, m in results.items():
            assert m.detection_rate > 0.5, f"{name} detected only {m.detection_rate:.2f}"

        isolation = evaluate_isolation(test, test_runs, CFG, bank, clf)
        labels = tuple(f.value for f in FaultType)
        for name, outcome in isolation.items():
            report = confusion_report(outcome.truth, outcome.predicted, labels)
            assert report.matrix.sum() == len(test)
            assert report.accuracy > 1.0 / len(labels), f"{name} at chance"

    def test_healthy_scenarios_are_never_labelled_faulty_by_construction(self):
        scenarios = sample_scenarios(24, 82000, n_steps=1400)
        healthy = [s for s in scenarios if s.label is FaultType.NONE]
        assert healthy
        for s in healthy:
            assert s.fault.kind is FaultType.NONE
            assert class_index(s.label) == 0
