"""Integration and regression tests for the matched-false-alarm-rate harness.

The sizes here are deliberately small -- these tests prove the harness wires up
correctly and that the matching holds to within its own Monte-Carlo error.  The
numbers that go into the README come from
``validation/validate_matched_far.py``, which runs the same code at the full
size.
"""

from __future__ import annotations

import numpy as np
import pytest

from telemetryool.arl import cusum_window_false_alarm, design_cusum_h
from telemetryool.detectors import (
    CusumDetector,
    EwmaDetector,
    LimitDetector,
    NoveltyDetector,
)
from telemetryool.harness import ScenarioSpec, run_comparison
from telemetryool.novelty import HotellingT2Q
from telemetryool.synthetic import Anomaly, NominalModel, equicorrelation


def _small_suite():
    return [
        LimitDetector(persistence=2),
        EwmaDetector(lam=0.2, persistence=2),
        CusumDetector(k=0.5, persistence=2),
        NoveltyDetector(
            HotellingT2Q(n_components=2), name="t2q", persistence=2, smooth_lam=0.2
        ),
    ]


def _scenarios():
    return [
        ScenarioSpec("step1.5", Anomaly("step", 40, 1.5), 600),
        ScenarioSpec("decorrelate", Anomaly("decorrelate", 40, channels=[0, 1]), 600),
    ]


@pytest.fixture(scope="module")
def result():
    model = NominalModel(4, correlation=equicorrelation(4, 0.6))
    return run_comparison(
        _small_suite(),
        model,
        _scenarios(),
        target_alpha_w=0.05,
        window_length=100,
        n_train_windows=400,
        n_cal_windows=4000,
        n_measure_windows=4000,
        seed=777,
    )


def test_every_method_is_calibrated_to_the_same_in_sample_rate(result) -> None:
    """Calibration is an exact empirical quantile, so every method must land on
    ceil(0.05 * 4000) = 200 alarming calibration windows."""
    for method in result.methods:
        assert method.calibration.n_alarming == 200
        assert method.calibration.achieved_alpha_w == pytest.approx(0.05, abs=1e-12)


def test_measured_false_alarm_rates_agree_within_four_combined_sigma(result) -> None:
    """The combined standard error at n_cal = n_meas = 4000 is
    sqrt(0.05 * 0.95 * 2 / 4000) = 0.004873, so a 4-sigma band is
    [0.0305, 0.0695].  Any method outside that at this size is mis-calibrated."""
    se = result.combined_standard_error()
    assert se == pytest.approx(np.sqrt(0.05 * 0.95 * 2 / 4000), rel=1e-9)
    for method in result.methods:
        z = (method.false_alarm.rate - 0.05) / se
        assert abs(z) < 4.0, f"{method.name}: measured {method.false_alarm.rate:.5f}, z={z:.2f}"


def test_confusion_matrices_are_complete_and_consistent(result) -> None:
    for method in result.methods:
        for name, scenario_result in method.per_scenario.items():
            cm = scenario_result.confusion
            assert cm.n_positive == 600, name
            assert cm.n_negative == 4000, name
            assert cm.total == 4600
            assert cm.false_positive == method.false_alarm.successes
            assert cm.tpr == pytest.approx(
                scenario_result.delay.detection_probability
                + scenario_result.delay.n_early / 600,
                abs=1e-12,
            )


def test_cusum_is_fastest_on_a_univariate_step(result) -> None:
    """At a matched false-alarm rate, CUSUM on the affected channel should beat a
    plain limit check on a 1.5-sigma sustained step.  This is the textbook
    result (Page 1954) and a failure here means the harness is wrong, not that
    the textbook is."""
    by_name = {m.name: m.per_scenario["step1.5"] for m in result.methods}
    cusum = by_name["cusum(k=0.5)"].delay
    ool = by_name["ool(p=2)"].delay
    assert cusum.detection_probability > ool.detection_probability
    assert cusum.mean_delay < ool.mean_delay


def test_only_the_multivariate_method_sees_a_decorrelation(result) -> None:
    """The 'decorrelate' anomaly leaves every marginal distribution unchanged, so
    a univariate monitor is blind to it by construction."""
    by_name = {m.name: m.per_scenario["decorrelate"] for m in result.methods}
    assert by_name["t2q"].delay.detection_probability > 0.5
    for univariate in ("ool(p=2)", "ewma(lam=0.2)", "cusum(k=0.5)"):
        assert by_name[univariate].delay.detection_probability < 0.2


def test_roc_auc_ordering_matches_the_delay_ordering_on_the_step(result) -> None:
    by_name = {m.name: m.per_scenario["step1.5"] for m in result.methods}
    assert by_name["cusum(k=0.5)"].roc.auc > by_name["ool(p=2)"].roc.auc


def test_text_rendering_contains_the_required_elements(result) -> None:
    text = result.to_text()
    assert "Window false-alarm probability" in text
    assert "Confusion matrix" in text
    assert "predicted alarm" in text
    for method in result.methods:
        assert method.name in text
    assert "step1.5" in text and "decorrelate" in text


def test_design_values_are_printed_beside_measurements_when_supplied() -> None:
    model = NominalModel(1)
    design = design_cusum_h(0.05, 0.5, 100)
    analytic = cusum_window_false_alarm(0.5, design.threshold, 100)
    det = CusumDetector(k=0.5)
    res = run_comparison(
        [det],
        model,
        [ScenarioSpec("step2", Anomaly("step", 50, 2.0), 300)],
        target_alpha_w=0.05,
        window_length=100,
        n_train_windows=100,
        n_cal_windows=2000,
        n_measure_windows=2000,
        seed=55,
        design_alpha={det.name: analytic},
    )
    assert res.methods[0].design_alpha_w == pytest.approx(analytic)
    assert "alpha_W(design)" in res.far_table()


def test_fastest_at_matched_far_respects_the_detection_floor(result) -> None:
    assert result.fastest_at_matched_far("step1.5") in {
        m.name for m in result.methods
    }
    # A scenario nobody can see returns the empty string rather than the
    # fastest-but-useless method.
    by_name = {m.name: m.per_scenario["decorrelate"] for m in result.methods}
    assert all(
        by_name[n].delay.detection_probability < 0.5
        for n in ("ool(p=2)", "ewma(lam=0.2)", "cusum(k=0.5)")
    )
    assert result.fastest_at_matched_far("decorrelate") == "t2q"


def test_reproducibility_for_a_fixed_seed() -> None:
    model = NominalModel(2)
    kwargs = dict(
        target_alpha_w=0.05,
        window_length=50,
        n_train_windows=100,
        n_cal_windows=1000,
        n_measure_windows=1000,
        seed=321,
    )
    scen = [ScenarioSpec("step2", Anomaly("step", 20, 2.0), 200)]
    a = run_comparison([CusumDetector(k=0.5)], model, scen, **kwargs)
    b = run_comparison([CusumDetector(k=0.5)], model, scen, **kwargs)
    assert a.methods[0].calibration.threshold == b.methods[0].calibration.threshold
    assert a.methods[0].false_alarm.successes == b.methods[0].false_alarm.successes


def test_harness_validation() -> None:
    model = NominalModel(2)
    scen = [ScenarioSpec("s", Anomaly("step", 5, 1.0), 10)]
    with pytest.raises(ValueError, match="at least one detector"):
        run_comparison([], model, scen)
    with pytest.raises(ValueError, match="at least one scenario"):
        run_comparison([CusumDetector()], model, [])
    with pytest.raises(ValueError, match="names must be unique"):
        run_comparison([CusumDetector(), CusumDetector()], model, scen)
    with pytest.raises(ValueError, match="scenario names must be unique"):
        run_comparison([CusumDetector()], model, scen + scen)
    with pytest.raises(ValueError, match=r"target_alpha_w must lie in \(0, 1\)"):
        run_comparison([CusumDetector()], model, scen, target_alpha_w=0.0)
    with pytest.raises(ValueError, match="n_windows must be >= 1"):
        ScenarioSpec("x", Anomaly("step", 1, 1.0), 0)
