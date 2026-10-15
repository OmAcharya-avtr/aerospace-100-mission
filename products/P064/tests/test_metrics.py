"""Known-answer and edge-case tests for the APFD and safety-loss metrics."""

from __future__ import annotations

import math

import pytest

from regressselect.metrics import (
    apfd,
    apfd_single_fault,
    detection_curve,
    first_detection_positions,
    safety_loss,
    selection_cost,
)


def test_apfd_hand_calculated_five_tests_two_faults():
    # Rothermel, Untch, Chu and Harrold, IEEE TSE 27(10), 2001:
    #   APFD = 1 - (TF_1 + ... + TF_m) / (n*m) + 1/(2n)
    # n = 5, m = 2, first detections at positions 1 and 3:
    #   1 - (1 + 3)/(5*2) + 1/10 = 1 - 0.4 + 0.1 = 0.7
    order = ["t1", "t2", "t3", "t4", "t5"]
    detecting = {"f1": ["t1"], "f2": ["t3"]}
    assert apfd(order, detecting) == pytest.approx(0.7, abs=1e-12)


def test_apfd_best_case_all_faults_in_first_test():
    # n = 4, m = 3, every TF_i = 1: 1 - 3/(4*3) + 1/8 = 1 - 0.25 + 0.125 = 0.875
    order = ["a", "b", "c", "d"]
    detecting = {f"f{i}": ["a"] for i in range(3)}
    assert apfd(order, detecting) == pytest.approx(0.875, abs=1e-12)


def test_apfd_worst_case_all_faults_in_last_test():
    # n = 4, m = 3, every TF_i = 4: 1 - 12/12 + 1/8 = 0.125
    order = ["a", "b", "c", "d"]
    detecting = {f"f{i}": ["d"] for i in range(3)}
    assert apfd(order, detecting) == pytest.approx(0.125, abs=1e-12)


def test_apfd_single_test_single_fault():
    # n = 1, m = 1, TF = 1: 1 - 1 + 0.5 = 0.5
    assert apfd(["only"], {"f": ["only"]}) == pytest.approx(0.5, abs=1e-12)


def test_apfd_single_fault_helper_matches_the_general_form():
    order = [f"t{i}" for i in range(10)]
    assert apfd_single_fault(order, ["t4"]) == pytest.approx(
        apfd(order, {"fault": ["t4"]}), abs=1e-12
    )


def test_apfd_rejects_an_undetected_fault():
    with pytest.raises(ValueError, match="detected by no test"):
        apfd(["a", "b"], {"f1": ["a"], "f2": ["zzz"]})


def test_apfd_rejects_empty_inputs():
    with pytest.raises(ValueError, match="empty test order"):
        apfd([], {"f": ["a"]})
    with pytest.raises(ValueError, match="empty fault set"):
        apfd(["a"], {})


def test_apfd_rejects_duplicate_test_ids():
    with pytest.raises(ValueError, match="duplicate"):
        apfd(["a", "a"], {"f": ["a"]})


def test_apfd_lies_strictly_inside_the_unit_interval():
    order = [f"t{i}" for i in range(6)]
    for position in range(6):
        value = apfd_single_fault(order, [f"t{position}"])
        assert 0.0 < value < 1.0


def test_first_detection_positions_omits_undetected_faults():
    got = first_detection_positions(["a", "b"], {"f1": ["b", "a"], "f2": ["c"]})
    assert got == {"f1": 1}


def test_safety_loss_known_answers():
    assert safety_loss(["a", "b"], ["a", "b"]) == pytest.approx(0.0)
    assert safety_loss(["a"], ["a", "b"]) == pytest.approx(0.5)
    assert safety_loss([], ["a", "b", "c", "d"]) == pytest.approx(1.0)


def test_safety_loss_undefined_without_a_failure():
    with pytest.raises(ValueError, match="no test actually failed"):
        safety_loss(["a"], [])


def test_selection_cost_known_answer_and_clipping():
    assert selection_cost(["a", "b"], ["a", "b", "c", "d"]) == pytest.approx(0.5)
    # a selected id outside the universe does not inflate the cost above 1
    assert selection_cost(["a", "zz"], ["a", "b"]) == pytest.approx(0.5)


def test_selection_cost_rejects_an_empty_universe():
    with pytest.raises(ValueError, match="empty suite"):
        selection_cost(["a"], [])


def test_detection_curve_is_monotone_non_decreasing_and_ends_at_one():
    order = ["a", "b", "c", "d"]
    detecting = {"f1": ["c"], "f2": ["a"], "f3": ["d"]}
    curve = detection_curve(order, detecting)
    assert curve[0] == (0, 0.0)
    assert curve[-1][0] == len(order)
    assert curve[-1][1] == pytest.approx(1.0)
    values = [v for _, v in curve]
    assert all(b >= a - 1e-12 for a, b in zip(values, values[1:], strict=False))


def test_detection_curve_never_reaches_one_when_a_fault_is_undetectable():
    curve = detection_curve(["a", "b"], {"f1": ["a"], "f2": ["nope"]})
    assert curve[-1][1] == pytest.approx(0.5)
    assert not math.isclose(curve[-1][1], 1.0)
