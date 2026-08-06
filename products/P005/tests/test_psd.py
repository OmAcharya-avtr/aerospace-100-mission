"""PSD known-answer, Parseval-consistency, and input-validation tests."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from jitterscope import band_rms, cumulative_rms, psd

FS = 1000.0


def test_sinusoid_peak_frequency_and_power():
    """Known answer: A*sin(2*pi*f0*t) has total power A^2/2 at f0.

    Hand calculation: A = 2e-6 rad, so power = (2e-6)^2 / 2 = 2e-12 rad^2
    and RMS = A/sqrt(2) = 1.41421e-6 rad.
    """
    a, f0 = 2e-6, 80.0
    t = np.arange(0, 20, 1 / FS)
    x = a * np.sin(2 * np.pi * f0 * t)
    f, pxx = psd(x, FS)
    # Peak must land within one frequency-resolution bin of f0.
    df = f[1] - f[0]
    assert abs(f[np.argmax(pxx)] - f0) <= df
    total_power = np.trapezoid(pxx, f)
    assert total_power == pytest.approx(a**2 / 2, rel=0.01)


def test_sinusoid_band_rms_known_answer():
    """Band RMS around the tone recovers A/sqrt(2); far bands are ~0."""
    a, f0 = 2e-6, 80.0
    t = np.arange(0, 20, 1 / FS)
    x = a * np.sin(2 * np.pi * f0 * t)
    rms = band_rms(psd(x, FS), [(70.0, 90.0), (200.0, 400.0)])
    assert rms[0] == pytest.approx(a / np.sqrt(2), rel=0.01)
    assert rms[1] < 0.01 * rms[0]


def test_white_noise_flat_psd_level():
    """White noise sigma^2 spreads evenly: one-sided PSD = sigma^2/(fs/2).

    Hand calculation: sigma = 1e-6, fs = 1000 -> PSD = 2e-15 u^2/Hz.
    Median-across-bins comparison at 5 % tolerance (Welch estimator
    variance with ~390 averaged segments).
    """
    rng = np.random.default_rng(123)
    sigma = 1e-6
    x = rng.normal(0, sigma, 200_000)
    f, pxx = psd(x, FS)
    expected = sigma**2 / (FS / 2)
    assert np.median(pxx[1:-1]) == pytest.approx(expected, rel=0.05)


def test_cumulative_rms_matches_signal_std():
    rng = np.random.default_rng(5)
    x = rng.normal(0, 3e-6, 100_000)
    f, sig_c = cumulative_rms(psd(x, FS))
    assert sig_c[-1] == pytest.approx(np.std(x), rel=0.03)
    assert np.all(np.diff(sig_c) >= 0)  # monotone by construction


@settings(max_examples=25, deadline=None)
@given(
    seed=st.integers(0, 10_000),
    sigma=st.floats(1e-8, 1e-3),
)
def test_parseval_consistency_property(seed: int, sigma: float):
    """Property: integral of the PSD approximates the signal variance.

    Statistical tolerance 10 % — Welch with hann/50 % overlap on 32768
    samples averages ~63 segments, giving a few-percent standard error
    on the integrated power (Bendat & Piersol 2010, ch. 8).
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, sigma, 32_768)
    f, pxx = psd(x, FS)
    assert np.trapezoid(pxx, f) == pytest.approx(np.var(x), rel=0.10)


class TestInputValidation:
    def test_nan_raises_with_policy_message(self):
        """NaN policy: raise ValueError, never impute (documented)."""
        x = np.ones(100)
        x[3] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            psd(x, FS)

    def test_inf_raises(self):
        x = np.ones(100)
        x[-1] = np.inf
        with pytest.raises(ValueError, match="non-finite"):
            psd(x, FS)

    def test_negative_fs_raises(self):
        with pytest.raises(ValueError, match="fs"):
            psd(np.ones(100), -5.0)

    def test_2d_input_raises(self):
        with pytest.raises(ValueError, match="1-D"):
            psd(np.ones((10, 10)), FS)

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="at least 8"):
            psd(np.ones(4), FS)

    def test_bad_band_order_raises(self):
        f = np.linspace(0, 500, 100)
        p = np.ones(100)
        with pytest.raises(ValueError, match="f_lo < f_hi"):
            band_rms((f, p), [(100.0, 50.0)])

    def test_negative_psd_raises(self):
        f = np.linspace(0, 500, 100)
        p = -np.ones(100)
        with pytest.raises(ValueError, match="non-negative"):
            band_rms((f, p), [(0.0, 100.0)])

    def test_band_outside_grid_gives_zero(self):
        f = np.linspace(0, 500, 100)
        p = np.ones(100)
        assert band_rms((f, p), [(600.0, 700.0)])[0] == 0.0
