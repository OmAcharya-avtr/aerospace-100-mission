"""Energy-convention tests, hand-checked."""

from __future__ import annotations

import numpy as np
import pytest

from softdecode.detection import DetectionModel, db_to_ratio, ratio_to_db


def test_db_round_trip():
    assert float(db_to_ratio(10.0)) == pytest.approx(10.0)
    assert float(db_to_ratio(0.0)) == pytest.approx(1.0)
    assert float(ratio_to_db(100.0)) == pytest.approx(20.0)


def test_ratio_to_db_rejects_non_positive():
    with pytest.raises(ValueError, match="strictly positive"):
        ratio_to_db(0.0)


def test_ook_amplitude_hand_calculation(detection):
    # Eb/N0 = 10 dB -> gamma = 10; a = 2 * 1 * sqrt(1 * 10) = 6.32455532...
    assert detection.ook_amplitude(10.0) == pytest.approx(2.0 * np.sqrt(10.0), rel=1e-12)
    # At rate 1/2 the channel-bit energy halves: a = 2 sqrt(5) = 4.47213595
    assert detection.ook_amplitude(10.0, 0.5) == pytest.approx(2.0 * np.sqrt(5.0), rel=1e-12)


def test_ppm_amplitude_hand_calculation(detection):
    # M = 4, Eb/N0 = 10 dB: a = sqrt(2 * 2 * 1 * 10) = sqrt(40) = 6.32455532
    assert detection.ppm_amplitude(10.0, 4) == pytest.approx(np.sqrt(40.0), rel=1e-12)
    # M = 2 reduces to the OOK pulse-energy case a = sqrt(2 * 1 * 10) = sqrt(20)
    assert detection.ppm_amplitude(10.0, 2) == pytest.approx(np.sqrt(20.0), rel=1e-12)


def test_bpsk_amplitude_hand_calculation(detection):
    # Eb/N0 = 10 dB: a = sqrt(2 * 10) = 4.472135955, so a/sigma = sqrt(2 gamma)
    assert detection.bpsk_amplitude(10.0) == pytest.approx(np.sqrt(20.0), rel=1e-12)


def test_ook_ebn0_round_trip(detection):
    for ebn0 in (-3.0, 0.0, 7.5, 20.0):
        a = detection.ook_amplitude(ebn0, 0.5)
        assert detection.ook_ebn0_db(a, 0.5) == pytest.approx(ebn0, abs=1e-10)


def test_n0_convention(detection):
    assert detection.n0 == pytest.approx(2.0 * detection.noise_variance)


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan])
def test_sigma_validation(bad):
    with pytest.raises(ValueError, match="sigma"):
        DetectionModel(bad)


@pytest.mark.parametrize("bad_rate", [0.0, -0.5, 1.5, np.nan])
def test_rate_validation(detection, bad_rate):
    with pytest.raises(ValueError, match="rate"):
        detection.ook_amplitude(10.0, bad_rate)


@pytest.mark.parametrize("bad_order", [3, 0, 1, 6])
def test_ppm_order_validation(detection, bad_order):
    with pytest.raises(ValueError, match="power of two"):
        detection.ppm_amplitude(10.0, bad_order)


def test_ook_ebn0_rejects_non_positive_amplitude(detection):
    with pytest.raises(ValueError, match="amplitude"):
        detection.ook_ebn0_db(0.0)
