"""Detection model: Poisson tail, beam shape, scintillation average, false alarms."""

from __future__ import annotations

import math

import numpy as np
import pytest

from acqsearch import PoissonDwellDetector, poisson_upper_tail


def test_poisson_upper_tail_threshold_one_is_one_minus_exp():
    # P(N >= 1) = 1 - e^-m. At m = 1: 1 - 0.36787944117144233 = 0.6321205588285577
    assert poisson_upper_tail(1, 1.0) == pytest.approx(0.6321205588285577, abs=1e-15)


def test_poisson_upper_tail_threshold_two_hand_calculation():
    # P(N >= 2 | m = 1) = 1 - e^-1 (1 + 1) = 1 - 2/e = 0.26424111765711533
    assert poisson_upper_tail(2, 1.0) == pytest.approx(1.0 - 2.0 / math.e, abs=1e-15)


def test_poisson_upper_tail_threshold_three_hand_calculation():
    # P(N >= 3 | m = 2) = 1 - e^-2 (1 + 2 + 2) = 1 - 5 e^-2 = 0.3233235838169366
    assert poisson_upper_tail(3, 2.0) == pytest.approx(1.0 - 5.0 * math.exp(-2.0), abs=1e-15)


def test_poisson_upper_tail_threshold_zero_is_certain():
    assert poisson_upper_tail(0, 0.0) == 1.0
    assert poisson_upper_tail(0, 17.0) == 1.0


def test_poisson_upper_tail_at_zero_mean_is_zero_for_positive_threshold():
    assert poisson_upper_tail(1, 0.0) == pytest.approx(0.0, abs=1e-15)
    assert poisson_upper_tail(9, 0.0) == pytest.approx(0.0, abs=1e-15)


def test_poisson_upper_tail_is_monotone_in_the_mean():
    means = np.linspace(0.0, 30.0, 200)
    values = np.asarray(poisson_upper_tail(6, means))
    assert np.all(np.diff(values) >= -1e-15)


def test_poisson_upper_tail_is_monotone_in_the_threshold():
    values = [poisson_upper_tail(k, 5.0) for k in range(1, 20)]
    assert all(b <= a + 1e-15 for a, b in zip(values, values[1:], strict=False))


def test_poisson_upper_tail_vectorises():
    out = poisson_upper_tail(2, np.array([0.0, 1.0, 2.0]))
    assert isinstance(out, np.ndarray)
    assert out == pytest.approx(
        np.array([0.0, 1.0 - 2.0 / math.e, 1.0 - 3.0 * math.exp(-2.0)]), abs=1e-15
    )


@pytest.mark.parametrize("bad", [-1, -5])
def test_poisson_upper_tail_rejects_negative_threshold(bad):
    with pytest.raises(ValueError, match="threshold"):
        poisson_upper_tail(bad, 1.0)


def test_poisson_upper_tail_rejects_negative_mean():
    with pytest.raises(ValueError, match="mean_counts"):
        poisson_upper_tail(2, -1.0)


def test_false_alarm_probability_matches_the_poisson_tail(detector):
    tau = 1.4e-3
    expected = poisson_upper_tail(detector.threshold_counts, detector.background_rate_hz * tau)
    assert detector.false_alarm_probability(tau) == pytest.approx(expected, abs=1e-15)


def test_false_alarm_probability_is_zero_for_a_zero_dwell(detector):
    assert detector.false_alarm_probability(0.0) == pytest.approx(0.0, abs=1e-15)


def test_detection_probability_at_large_offset_falls_to_the_false_alarm_floor(detector):
    tau = 1.4e-3
    far = detector.detection_probability(tau, 1.0e-2, 5.0e-5)
    assert far == pytest.approx(detector.false_alarm_probability(tau), abs=1e-12)


def test_detection_probability_decreases_with_offset(detector):
    offsets = np.linspace(0.0, 2.0e-4, 50)
    values = np.asarray(detector.detection_probability(1.4e-3, offsets, 5.0e-5))
    assert np.all(np.diff(values) <= 1e-12)


def test_detection_probability_increases_with_dwell_on_axis(detector):
    values = [detector.detection_probability(t, 0.0, 5.0e-5) for t in np.linspace(1e-4, 5e-3, 40)]
    assert all(b >= a - 1e-12 for a, b in zip(values, values[1:], strict=False))


def test_beam_shape_factor_is_the_gaussian_profile():
    # With no scintillation and threshold 1, P_d = 1 - exp(-(lambda_s(rho) + lambda_b) tau).
    d = PoissonDwellDetector(1.0e4, 0.0, 1, 0.0)
    w, tau, rho = 5.0e-5, 1.0e-3, 2.5e-5
    expected = 1.0 - math.exp(-1.0e4 * math.exp(-2.0 * (rho / w) ** 2) * tau)
    assert d.detection_probability(tau, rho, w) == pytest.approx(expected, rel=1e-12)


def test_irradiance_nodes_have_unit_weight_and_unit_mean(detector):
    values, weights = detector.irradiance_nodes()
    assert weights.sum() == pytest.approx(1.0, abs=1e-12)
    assert float((weights * values).sum()) == pytest.approx(1.0, rel=1e-9)


def test_zero_rytov_variance_uses_a_single_node():
    values, weights = PoissonDwellDetector(1.0e4, 1.0e3, 4, 0.0).irradiance_nodes()
    assert values.shape == (1,)
    assert weights == pytest.approx(np.ones(1))


def test_scintillation_changes_the_detection_probability(detector):
    without = PoissonDwellDetector(
        detector.signal_rate_hz, detector.background_rate_hz, detector.threshold_counts, 0.0
    )
    a = detector.detection_probability(1.4e-3, 3.0e-5, 5.0e-5)
    b = without.detection_probability(1.4e-3, 3.0e-5, 5.0e-5)
    assert abs(a - b) > 1e-4


def test_detection_probability_table_matches_direct_evaluation(detector):
    offsets, values = detector.detection_probability_table(1.4e-3, 5.0e-5, 2.5e-4, 1025)
    probe = np.linspace(0.0, 2.5e-4, 97)
    interpolated = np.interp(probe, offsets, values)
    direct = np.asarray(detector.detection_probability(1.4e-3, probe, 5.0e-5))
    # Measured worst case 1.65e-5 absolute, on the steep shoulder of the beam.
    # The bound asserted here is three times that, not the measured value, so a
    # small change in the quadrature does not fail the suite spuriously.
    assert np.max(np.abs(interpolated - direct)) < 5e-5


def test_detection_probability_vectorises_over_an_array(detector):
    out = detector.detection_probability(1.4e-3, np.array([0.0, 1.0e-4]), 5.0e-5)
    assert isinstance(out, np.ndarray)
    assert out.shape == (2,)
    assert out[0] > out[1]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"signal_rate_hz": -1.0},
        {"background_rate_hz": -1.0},
        {"threshold_counts": 0},
        {"threshold_counts": 2.5},
        {"rytov_variance": -0.1},
        {"quadrature_nodes": 1},
    ],
)
def test_detector_rejects_bad_parameters(kwargs):
    base = {
        "signal_rate_hz": 1.0e4,
        "background_rate_hz": 1.0e3,
        "threshold_counts": 5,
        "rytov_variance": 0.1,
    }
    base.update(kwargs)
    with pytest.raises(ValueError):
        PoissonDwellDetector(**base)


def test_detection_probability_rejects_bad_arguments(detector):
    with pytest.raises(ValueError, match="dwell_s"):
        detector.detection_probability(-1.0, 0.0, 5.0e-5)
    with pytest.raises(ValueError, match="beam_half_angle_rad"):
        detector.detection_probability(1.0e-3, 0.0, 0.0)
    with pytest.raises(ValueError, match="offset_rad"):
        detector.detection_probability(1.0e-3, -1.0e-6, 5.0e-5)


def test_false_alarm_probability_rejects_negative_dwell(detector):
    with pytest.raises(ValueError, match="dwell_s"):
        detector.false_alarm_probability(-1.0)


def test_detection_probability_table_rejects_too_few_points(detector):
    with pytest.raises(ValueError, match="n_points"):
        detector.detection_probability_table(1.0e-3, 5.0e-5, 1.0e-4, 1)
