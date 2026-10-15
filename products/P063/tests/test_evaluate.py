"""Integration and regression tests for the benchmarks.

``test_detector_comparison_regression`` and
``test_policy_comparison_regression`` pin the measured numbers of the shipped
declaration at a fixed seed. They are the benchmark/regression tests the medium
class requires. Every pinned number is compared with a tolerance, never for
bit-exact equality.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from safemodetree import (
    NOMINAL_LABEL,
    compare_detectors,
    compare_policies,
    conservatism_sweep,
    false_alarm_rate,
    identification_accuracy,
    make_observations,
    noise_sweep,
    split_observations,
)


@pytest.fixture(scope="module")
def comparison(fmea):
    return compare_detectors(fmea, n_per_fault=400, noise_scale=2.0, seed=63001)


def test_split_partitions_without_overlap(fmea):
    observations = make_observations(fmea, n_per_fault=50, seed=5)
    split = split_observations(observations, seed=6)
    assert sum(split.sizes()) == observations.n_samples
    assert all(size > 0 for size in split.sizes())


def test_false_alarm_rate_and_identification_accuracy_hand_cases():
    predicted = np.array(["nominal", "f1", "f2", "nominal"])
    truth = np.array(["nominal", "nominal", "f2", "f1"])
    # One of two nominal rows was called a fault -> 0.5.
    assert false_alarm_rate(predicted, truth) == pytest.approx(0.5, abs=1e-12)
    # Two faulted rows, one identified correctly -> 0.5.
    assert identification_accuracy(predicted, truth) == pytest.approx(0.5, abs=1e-12)


def test_rates_are_nan_without_the_relevant_class():
    only_faults = np.array(["f1"])
    assert math.isnan(false_alarm_rate(only_faults, only_faults))
    only_nominal = np.array([NOMINAL_LABEL])
    assert math.isnan(identification_accuracy(only_nominal, only_nominal))


def test_detector_comparison_regression(comparison, fmea):
    # Measured on 2026-10-15 with numpy 2.5.3, scikit-learn 1.9.1, Python
    # 3.13.16, seed 63001, noise_scale 2.0, 400 samples per fault. Tolerances
    # are absolute and are wide enough for the floating-point reassociation a
    # different BLAS can introduce, and far too tight to survive a change of
    # method. Re-run validation/validate_calibration.py to refresh them.
    assert comparison["sizes"] == (3200, 1600, 1600)
    assert float(comparison["target_far"]) == pytest.approx(0.13736, abs=5e-4)
    assert float(comparison["ceiling"]) == pytest.approx(0.87406, abs=5e-3)
    baseline = comparison["baseline"]
    learned = comparison["learned"]
    assert baseline.false_alarm_rate == pytest.approx(0.12147, abs=5e-4)
    assert baseline.identification_accuracy == pytest.approx(0.73503, abs=5e-4)
    assert baseline.calibration.ece == pytest.approx(0.02783, abs=5e-4)
    assert learned.false_alarm_rate == pytest.approx(0.07117, abs=5e-3)
    assert learned.identification_accuracy == pytest.approx(0.73503, abs=5e-3)
    assert learned.calibration.ece == pytest.approx(0.02448, abs=5e-3)
    assert learned.brier == pytest.approx(0.16237, abs=5e-3)
    assert baseline.brier is None


def test_the_learned_model_does_not_exceed_the_declared_ceiling(comparison):
    ceiling = float(comparison["ceiling"])
    for key in ("baseline", "learned", "learned_argmax"):
        assert comparison[key].overall_accuracy <= ceiling + 1e-9


def test_matching_brings_the_false_alarm_rates_closer_than_the_argmax(comparison):
    baseline = comparison["baseline"]
    matched = comparison["learned"]
    argmax = comparison["learned_argmax"]
    assert abs(matched.false_alarm_rate - baseline.false_alarm_rate) < abs(
        argmax.false_alarm_rate - baseline.false_alarm_rate
    )


def test_detector_score_row_is_fixed_width(comparison):
    assert len(comparison["baseline"].row()) == len(comparison["learned"].row())


def test_policy_comparison_regression(fmea, comparison):
    # Measured on 2026-10-15 under the versions recorded in
    # validation/validate_environment_output.txt. mode fine_pointing,
    # confidence threshold 0.70, learned detector at the matched operating
    # point.
    scores = compare_policies(fmea, comparison, mode_id="fine_pointing",
                              confidence_threshold=0.7, detector="learned")
    assert scores["oracle"].mean_cost_min == pytest.approx(23.4535, abs=0.05)
    assert scores["always_specific"].mean_cost_min == pytest.approx(35.4866, abs=0.3)
    assert scores["confidence_routed"].mean_cost_min == pytest.approx(44.6322, abs=0.5)
    assert scores["always_conservative"].mean_cost_min == pytest.approx(53.0, abs=1e-9)
    assert float(scores["cost_of_conservatism_vs_oracle"]) == pytest.approx(21.18, abs=0.5)
    assert float(scores["cost_of_conservatism_vs_specific"]) == pytest.approx(9.15, abs=0.5)
    assert scores["always_specific"].n_misrouted == pytest.approx(206, abs=15)
    assert scores["confidence_routed"].n_misrouted <= 20
    assert scores["always_conservative"].n_misrouted == 0


def test_always_conservative_costs_exactly_the_declared_route(fmea, comparison):
    # The conservative route out of fine_pointing is coarse_downmode (8.0 min)
    # then safe_mode_entry (45.0 min). Every commanded event therefore costs
    # 53.0 min, so the mean is 53.0 to within floating-point rounding.
    scores = compare_policies(fmea, comparison, mode_id="fine_pointing", detector="learned")
    assert scores["always_conservative"].mean_cost_min == pytest.approx(53.0, abs=1e-9)
    assert scores["always_conservative"].conservative_fraction == pytest.approx(
        1.0, abs=1e-12
    )


def test_conservatism_eliminates_misroutes_at_a_cost(fmea, comparison):
    scores = compare_policies(fmea, comparison, mode_id="fine_pointing", detector="learned")
    specific = scores["always_specific"]
    routed = scores["confidence_routed"]
    assert routed.n_misrouted < specific.n_misrouted
    assert int(scores["misroutes_avoided"]) > 100
    assert routed.mean_cost_min > specific.mean_cost_min


def test_conservatism_sweep_is_monotone_in_the_conservative_fraction(fmea, comparison):
    rows = conservatism_sweep(fmea, comparison, mode_id="fine_pointing")
    fractions = [row.conservative_fraction for row in rows]
    assert fractions == sorted(fractions)
    assert rows[0].conservative_fraction == pytest.approx(0.0, abs=1e-12)
    assert rows[-1].conservative_fraction == pytest.approx(1.0, abs=1e-12)
    misroutes = [row.n_misrouted for row in rows]
    assert misroutes[-1] == 0
    assert misroutes[0] >= misroutes[-1]


def test_policy_comparison_with_the_baseline_detector_runs(fmea, comparison):
    scores = compare_policies(fmea, comparison, mode_id="coarse_pointing", detector="baseline")
    assert scores["detector"] == "baseline"
    assert scores["mode"] == "coarse_pointing"
    assert math.isfinite(scores["confidence_routed"].mean_cost_min)


def test_compare_policies_rejects_an_unknown_detector(fmea, comparison):
    with pytest.raises(ValueError, match="must be 'learned' or 'baseline'"):
        compare_policies(fmea, comparison, detector="crystal_ball")


def test_compare_policies_rejects_an_unknown_mode(fmea, comparison):
    with pytest.raises(KeyError):
        compare_policies(fmea, comparison, mode_id="nowhere")


def test_policy_score_row_is_fixed_width(fmea, comparison):
    scores = compare_policies(fmea, comparison)
    assert len(scores["oracle"].row()) == len(scores["confidence_routed"].row())


def test_noise_sweep_returns_one_result_per_scale(fmea):
    results = noise_sweep(fmea, noise_scales=(1.0, 2.0), n_per_fault=80, seed=99)
    assert len(results) == 2
    assert [float(r["noise_scale"]) for r in results] == [1.0, 2.0]


def test_false_alarm_rate_rises_with_noise(fmea):
    results = noise_sweep(fmea, noise_scales=(1.0, 2.0, 3.0), n_per_fault=200, seed=77)
    rates = [r["baseline"].false_alarm_rate for r in results]
    assert rates[0] < rates[1] < rates[2]
