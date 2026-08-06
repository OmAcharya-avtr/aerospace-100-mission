"""Synthetic telemetry generator: determinism, spectra, fault injection."""

import numpy as np
import pytest

from jitterscope import band_rms, generate_telemetry, psd


def test_seeded_determinism():
    """Identical args + seed reproduce the record bit-for-bit."""
    _, x1, m1 = generate_telemetry(10, 500, seed=11)
    _, x2, m2 = generate_telemetry(10, 500, seed=11)
    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(m1, m2)
    _, x3, _ = generate_telemetry(10, 500, seed=12)
    assert not np.array_equal(x1, x3)


def test_wheel_tone_present_at_expected_frequency():
    _, x, _ = generate_telemetry(30, 1000, seed=0, wheel_hz=45.0, tone_rms=1e-6)
    f, pxx = psd(x, 1000.0, nperseg=4096)
    # local peak within one bin of 45 Hz
    mask = (f > 40) & (f < 50)
    assert abs(f[mask][np.argmax(pxx[mask])] - 45.0) <= f[1] - f[0]
    # tone band RMS close to requested tone RMS (colored noise floor adds a little)
    rms_tone = band_rms((f, pxx), [(43.0, 47.0)])[0]
    assert rms_tone == pytest.approx(1e-6, rel=0.15)


def test_new_tone_fault_active_after_t_start():
    t, x, mask = generate_telemetry(
        20, 1000, seed=1,
        faults=[{"kind": "new_tone", "t_start": 10.0, "freq_hz": 137.0, "rms": 2e-6}],
    )
    assert not mask[t < 10.0].any()
    assert mask[t >= 10.0].all()
    f1, p1 = psd(x[t < 10.0], 1000.0, nperseg=2048)
    f2, p2 = psd(x[t >= 10.0], 1000.0, nperseg=2048)
    r1 = band_rms((f1, p1), [(132.0, 142.0)])[0]
    r2 = band_rms((f2, p2), [(132.0, 142.0)])[0]
    assert r2 > 5 * r1


def test_band_shift_fault_raises_band_energy():
    t, x, mask = generate_telemetry(
        20, 1000, seed=2,
        faults=[{"kind": "band_shift", "t_start": 10.0, "f_lo": 200.0,
                 "f_hi": 300.0, "factor": 6.0}],
    )
    f1, p1 = psd(x[t < 10.0], 1000.0)
    f2, p2 = psd(x[t >= 10.0], 1000.0)
    r1 = band_rms((f1, p1), [(210.0, 290.0)])[0]
    r2 = band_rms((f2, p2), [(210.0, 290.0)])[0]
    assert r2 > 1.5 * r1
    assert mask[t >= 10.0].all()


def test_transient_fault_marks_bursts_only():
    t, x, mask = generate_telemetry(
        30, 1000, seed=3,
        faults=[{"kind": "transient", "t_start": 5.0, "rate_hz": 0.5,
                 "amp": 5e-6, "decay_s": 0.05, "ring_hz": 120.0}],
    )
    assert mask.any()
    assert not mask[t < 5.0].any()
    # bursts are intermittent: fault duty cycle well below 100 %
    assert mask[t >= 5.0].mean() < 0.5


class TestInputValidation:
    def test_bad_duration_raises(self):
        with pytest.raises(ValueError, match="duration_s"):
            generate_telemetry(-1, 1000)

    def test_bad_fault_kind_raises(self):
        with pytest.raises(ValueError, match="fault kind"):
            generate_telemetry(5, 500, faults=[{"kind": "bogus"}])

    def test_fault_tone_above_nyquist_raises(self):
        with pytest.raises(ValueError, match="freq_hz"):
            generate_telemetry(
                5, 500, faults=[{"kind": "new_tone", "t_start": 1.0, "freq_hz": 400.0}]
            )

    def test_fault_t_start_out_of_range_raises(self):
        with pytest.raises(ValueError, match="t_start"):
            generate_telemetry(
                5, 500, faults=[{"kind": "new_tone", "t_start": 9.0, "freq_hz": 100.0}]
            )

    def test_wheel_above_nyquist_raises(self):
        with pytest.raises(ValueError, match="wheel_hz"):
            generate_telemetry(5, 100, wheel_hz=80.0)
