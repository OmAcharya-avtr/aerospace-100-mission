"""Known-answer and validation tests for the four detector statistics."""

from __future__ import annotations

import numpy as np
import pytest

from twininvalidate import (
    BASELINES,
    DETECTORS,
    DetectorSpec,
    cusum_statistic,
    ewma_statistic,
    first_alarm,
    glr_statistic,
    statistic_path,
    variance_cusum_statistic,
)


def test_baselines_are_the_three_declared_ones():
    assert BASELINES == ("cusum", "ewma", "glr")
    assert set(DETECTORS) == {"cusum", "ewma", "glr", "varcusum"}


def test_cusum_known_answer_constant_input():
    # z = 1, 1, 1 with reference k = 0.25.
    # S+_0 = max(0, 0 + 1 - 0.25) = 0.75
    # S+_1 = max(0, 0.75 + 1 - 0.25) = 1.50
    # S+_2 = max(0, 1.50 + 1 - 0.25) = 2.25
    # S-_k stays 0 because -z - k = -1.25 < 0 at every step.
    out = cusum_statistic(np.array([[1.0, 1.0, 1.0]]), reference=0.25)
    assert out[0].tolist() == pytest.approx([0.75, 1.5, 2.25])


def test_cusum_known_answer_lower_arm():
    # z = -1, -1 with k = 0.25 drives the lower arm:
    # S-_0 = max(0, 0 + 1 - 0.25) = 0.75, S-_1 = 1.50; the statistic is the
    # max of the two arms, so 0.75 then 1.50.
    out = cusum_statistic(np.array([[-1.0, -1.0]]), reference=0.25)
    assert out[0].tolist() == pytest.approx([0.75, 1.5])


def test_cusum_resets_at_zero():
    # z = 2, -5, 0 with k = 0.5: S+ = 1.5, then max(0, 1.5 - 5.5) = 0,
    # then max(0, 0 - 0.5) = 0. The lower arm: max(0, -2 - 0.5) = 0, then
    # max(0, 0 + 5 - 0.5) = 4.5, then max(0, 4.5 - 0.5) = 4.0.
    out = cusum_statistic(np.array([[2.0, -5.0, 0.0]]), reference=0.5)
    assert out[0].tolist() == pytest.approx([1.5, 4.5, 4.0])


def test_ewma_known_answer_lambda_one_is_absolute_value():
    # lam = 1: A_k = z_k and sigma = sqrt(1/1) = 1, so g_k = |z_k| exactly.
    out = ewma_statistic(np.array([[2.0, -1.0, 0.5]]), lam=1.0)
    assert out[0].tolist() == pytest.approx([2.0, 1.0, 0.5])


def test_ewma_known_answer_first_sample():
    # lam = 0.1, z_0 = 1: A_0 = 0.1, sigma = sqrt(0.1/1.9) = 0.2294157339...
    # g_0 = 0.1 / 0.2294157339 = 0.4358898944
    out = ewma_statistic(np.array([[1.0]]), lam=0.1)
    assert out[0, 0] == pytest.approx(0.4358898943540674, rel=1e-12)


def test_ewma_known_answer_second_sample():
    # lam = 0.5, z = 2, 0: A_0 = 1.0, A_1 = 0.5.
    # sigma = sqrt(0.5/1.5) = 0.5773502692; g = 1.732050808, 0.866025404
    out = ewma_statistic(np.array([[2.0, 0.0]]), lam=0.5)
    assert out[0].tolist() == pytest.approx([1.7320508075688772, 0.8660254037844386], rel=1e-12)


def test_glr_known_answer_window_one():
    # window = 1: g_k = z_k^2 / 2. z = 2 gives 4/2 = 2.
    out = glr_statistic(np.array([[2.0, 0.0, -3.0]]), window=1)
    assert out[0].tolist() == pytest.approx([2.0, 0.0, 4.5])


def test_glr_known_answer_window_three():
    # z = 2, 0, 0 with window 3.
    # k=0: max over n=1 of 2^2/2 = 2
    # k=1: max(0^2/2, (2+0)^2/4) = max(0, 1) = 1
    # k=2: max(0^2/2, (0+0)^2/4, (2+0+0)^2/6) = 4/6 = 0.6666...
    out = glr_statistic(np.array([[2.0, 0.0, 0.0]]), window=3)
    assert out[0].tolist() == pytest.approx([2.0, 1.0, 2.0 / 3.0])


def test_glr_known_answer_sustained_shift():
    # z = 1, 1, 1, 1 with window 4. At k=3 the best window is the whole four:
    # S_4 = 4, g = 16/8 = 2. Shorter windows give 1/2, 4/4 = 1, 9/6 = 1.5.
    out = glr_statistic(np.array([[1.0, 1.0, 1.0, 1.0]]), window=4)
    assert out[0, 3] == pytest.approx(2.0)


def test_variance_cusum_known_answer():
    # c = 0.5, z = 2, 0, 2:
    # V_0 = max(0, 0 + 4 - 1 - 0.5) = 2.5
    # V_1 = max(0, 2.5 + 0 - 1 - 0.5) = 1.0
    # V_2 = max(0, 1.0 + 4 - 1 - 0.5) = 3.5
    out = variance_cusum_statistic(np.array([[2.0, 0.0, 2.0]]), reference=0.5)
    assert out[0].tolist() == pytest.approx([2.5, 1.0, 3.5])


def test_all_statistics_are_non_negative(in_control):
    for name in DETECTORS:
        assert statistic_path(in_control[:4, :200], name).min() >= 0.0


def test_glr_is_non_decreasing_in_window(in_control):
    z = in_control[:4, :300]
    short = glr_statistic(z, window=5)
    long = glr_statistic(z, window=50)
    assert np.all(long >= short - 1e-12)


def test_glr_window_is_clipped_to_the_stream_length():
    z = np.ones((1, 5))
    assert np.allclose(glr_statistic(z, window=500), glr_statistic(z, window=5))


def test_first_alarm_finds_the_first_crossing():
    stat = np.array([[0.0, 1.0, 5.0, 9.0], [0.0, 0.0, 0.0, 0.0]])
    assert first_alarm(stat, 2.0).tolist() == [2, -1]


def test_first_alarm_is_strict():
    # The comparison is statistic > threshold, so equality is not an alarm.
    stat = np.array([[1.0, 2.0]])
    assert first_alarm(stat, 2.0).tolist() == [-1]
    assert first_alarm(stat, 1.9999).tolist() == [1]


def test_first_alarm_is_monotone_in_threshold(in_control):
    stat = cusum_statistic(in_control[:20, :500])
    low = first_alarm(stat, 2.0)
    high = first_alarm(stat, 5.0)
    both = (low >= 0) & (high >= 0)
    assert np.all(low[both] <= high[both])


def test_one_dimensional_input_is_accepted():
    out = cusum_statistic(np.array([1.0, 1.0]))
    assert out.shape == (1, 2)


@pytest.mark.parametrize(
    "func, kwargs, message",
    [
        (cusum_statistic, {"reference": 0.0}, "reference value must be positive"),
        (cusum_statistic, {"reference": -1.0}, "reference value must be positive"),
        (ewma_statistic, {"lam": 0.0}, r"lambda must lie in \(0, 1\]"),
        (ewma_statistic, {"lam": 1.5}, r"lambda must lie in \(0, 1\]"),
        (glr_statistic, {"window": 0}, "window must be at least 1"),
        (variance_cusum_statistic, {"reference": 0.0}, "reference must be positive"),
    ],
)
def test_detectors_validate_their_design_constants(func, kwargs, message):
    with pytest.raises(ValueError, match=message):
        func(np.ones((1, 4)), **kwargs)


def test_detectors_reject_non_finite_residuals():
    with pytest.raises(ValueError, match="must be finite"):
        cusum_statistic(np.array([[1.0, np.nan]]))
    with pytest.raises(ValueError, match="must be finite"):
        glr_statistic(np.array([[1.0, np.inf]]))


def test_detectors_reject_empty_and_high_dimensional_input():
    with pytest.raises(ValueError, match="at least one sample"):
        cusum_statistic(np.zeros((2, 0)))
    with pytest.raises(ValueError, match="must be 1-D or 2-D"):
        cusum_statistic(np.zeros((2, 2, 2)))


def test_statistic_path_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="unknown detector"):
        statistic_path(np.ones((1, 4)), "kalman")


def test_detector_spec_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="unknown detector"):
        DetectorSpec("kalman")


@pytest.mark.parametrize(
    "spec, fragment",
    [
        (DetectorSpec("cusum"), "CUSUM (k=0.25)"),
        (DetectorSpec("ewma"), "EWMA (lambda=0.1)"),
        (DetectorSpec("glr"), "GLR (window=100)"),
        (DetectorSpec("varcusum"), "variance-CUSUM oracle"),
    ],
)
def test_detector_spec_labels(spec, fragment):
    assert fragment in spec.label()


def test_detector_spec_statistic_dispatches(in_control):
    z = in_control[:3, :100]
    assert np.allclose(DetectorSpec("cusum").statistic(z), cusum_statistic(z, 0.25))
    assert np.allclose(DetectorSpec("ewma").statistic(z), ewma_statistic(z, 0.10))
    assert np.allclose(DetectorSpec("glr").statistic(z), glr_statistic(z, 100))
