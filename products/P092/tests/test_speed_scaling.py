"""The classical speed-scaling predictor. Fitted to nothing, so fully testable."""

from __future__ import annotations

import numpy as np
import pytest

from deadlinerisk.speed_scaling import (
    BOARD_CORES,
    available_fraction,
    predicted_miss_probability,
    predicted_response,
)


def test_known_answer_available_fraction() -> None:
    # HAND CALCULATION with m = 6 cores:
    #   n = 0 -> min(1, 6/1) = 1
    #   n = 5 -> min(1, 6/6) = 1
    #   n = 6 -> min(1, 6/7) = 0.8571428571428571
    #   n = 11 -> min(1, 6/12) = 0.5
    assert available_fraction(6, 0) == pytest.approx(1.0, rel=1e-12, abs=0.0)
    assert available_fraction(6, 5) == pytest.approx(1.0, rel=1e-12, abs=0.0)
    assert available_fraction(6, 6) == pytest.approx(6.0 / 7.0, rel=1e-12, abs=0.0)
    assert available_fraction(6, 11) == pytest.approx(0.5, rel=1e-12, abs=0.0)


def test_single_core_halves_with_one_co_resident_worker() -> None:
    assert available_fraction(1, 1) == pytest.approx(0.5, rel=1e-12, abs=0.0)


def test_fraction_is_non_increasing_in_the_worker_count() -> None:
    values = [available_fraction(BOARD_CORES, n) for n in range(0, 24)]
    assert all(later <= earlier for earlier, later in zip(values[:-1], values[1:], strict=True))


@pytest.mark.parametrize(("cores", "workers"), [(0, 1), (-1, 0), (6, -1)])
def test_fraction_input_validation(cores: int, workers: int) -> None:
    with pytest.raises(ValueError):
        available_fraction(cores, workers)


def test_known_answer_predicted_response() -> None:
    # HAND CALCULATION. idle exec 4 ms, m = 6, n = 11 -> fraction 0.5, so the
    # predicted response is 8 ms.
    predicted = predicted_response(np.asarray([4e-3]), 6, 11)
    assert predicted[0] == pytest.approx(8e-3, rel=1e-12, abs=0.0)


def test_predicted_response_is_the_identity_below_saturation() -> None:
    # Below saturation the available fraction is 1 and the prediction is the
    # idle time itself. Compared with a tolerance on both sides rather than for
    # bit-exact equality: dividing by exactly 1.0 happens to be exact in IEEE
    # 754, but an assertion that depends on that is an assertion about the
    # arithmetic rather than about this function.
    idle = np.asarray([3e-3, 4e-3, 5e-3])
    assert np.allclose(predicted_response(idle, 6, 3), idle, rtol=1e-15, atol=0.0)


def test_known_answer_predicted_miss_probability() -> None:
    # HAND CALCULATION. idle exec = [4, 4, 4, 9] ms, D = 10 ms, m = 6, n = 11
    # so fraction 0.5 and scaled = [8, 8, 8, 18] ms. One of four exceeds 10 ms,
    # so the predicted probability is 0.25.
    idle = np.asarray([4e-3, 4e-3, 4e-3, 9e-3])
    assert predicted_miss_probability(idle, 10e-3, 6, 11) == pytest.approx(
        0.25, rel=1e-12, abs=0.0
    )


def test_predicted_probability_is_non_decreasing_in_contention(campaign) -> None:
    idle = campaign.jobs["mix_c0"].exec_s
    probabilities = [
        predicted_miss_probability(idle, campaign.deadline_s, BOARD_CORES, n)
        for n in range(0, 20)
    ]
    assert all(
        later >= earlier - 1e-15
        for earlier, later in zip(probabilities[:-1], probabilities[1:], strict=True)
    )


@pytest.mark.parametrize(
    ("idle", "deadline"),
    [
        (np.zeros(0), 0.01),
        (np.asarray([-1e-3]), 0.01),
        (np.asarray([1e-3]), 0.0),
        (np.asarray([1e-3]), float("inf")),
    ],
)
def test_miss_probability_input_validation(idle: np.ndarray, deadline: float) -> None:
    with pytest.raises(ValueError):
        predicted_miss_probability(idle, deadline, 6, 2)
