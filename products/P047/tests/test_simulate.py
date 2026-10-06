"""Open-loop detector statistics and the closed-loop Monte Carlo."""

from __future__ import annotations

import math

import numpy as np
import pytest

from slotsync.loop import (
    LoopDesign,
    jitter_variance_closed_form,
    jitter_variance_coloured,
)
from slotsync.pulses import half_sine, nyquist_raised_cosine, raised_cosine_time, triangular
from slotsync.scurve import scurve
from slotsync.simulate import (
    measure_ted_autocovariance,
    measure_ted_statistics,
    pulse_table,
    run_timing_loop,
)
from slotsync.ted import TedConfig

NYQUIST = nyquist_raised_cosine(0.5, 4.0)
ZETA = 1.0 / math.sqrt(2.0)
FINE = np.linspace(-0.02, 0.02, 41)


@pytest.mark.parametrize(
    "pulse",
    [triangular(1.0), raised_cosine_time(1.0), half_sine(1.0), NYQUIST],
    ids=lambda p: p.name,
)
def test_pulse_table_interpolation_error_is_bounded(pulse) -> None:
    """The one approximation the sequential loop makes, measured not assumed."""
    lookup = pulse_table(pulse, 16385)
    times = np.linspace(-pulse.half_support, pulse.half_support, 20001)
    exact = pulse.amplitude(times)
    approximate = np.array([lookup(float(t)) for t in times])
    assert np.max(np.abs(exact - approximate)) < 2.0e-6, pulse.name


def test_pulse_table_validation() -> None:
    with pytest.raises(ValueError, match="odd and at least 257"):
        pulse_table(triangular(1.0), 256)


def test_open_loop_mean_reproduces_the_exact_s_curve() -> None:
    """The noisy measurement pipeline and the exact enumeration must agree."""
    config = TedConfig("gardner", "antipodal")
    curve = scurve(config, NYQUIST, np.array([-0.05, 0.0, 0.05]), max_exact_symbols=16)
    for index, offset in enumerate((-0.05, 0.0, 0.05)):
        stats = measure_ted_statistics(
            config, NYQUIST, sample_snr_db=40.0, offset=offset, samples=200000, seed=5
        )
        assert stats.mean == pytest.approx(float(curve.values[index]), abs=5e-4)


def test_open_loop_variance_splits_into_self_noise_and_channel_noise() -> None:
    config = TedConfig("early-late", "antipodal", 0.25, "dd")
    stats = measure_ted_statistics(config, NYQUIST, sample_snr_db=20.0, samples=200000)
    assert stats.self_noise_variance > 0.0
    assert stats.channel_noise_variance > 0.0
    assert stats.variance == pytest.approx(
        stats.self_noise_variance + stats.channel_noise_variance, rel=1e-12
    )


def test_mueller_muller_on_a_nyquist_pulse_has_no_self_noise() -> None:
    stats = measure_ted_statistics(
        TedConfig("mueller-muller", "antipodal"), NYQUIST, sample_snr_db=20.0, samples=100000
    )
    assert stats.self_noise_variance < 1e-9


def test_channel_noise_variance_scales_with_the_noise_power() -> None:
    config = TedConfig("mueller-muller", "antipodal")
    low = measure_ted_statistics(config, NYQUIST, sample_snr_db=20.0, samples=200000)
    high = measure_ted_statistics(config, NYQUIST, sample_snr_db=14.0, samples=200000)
    # 6 dB less SNR is four times the noise power.
    assert high.variance / low.variance == pytest.approx(4.0, rel=0.1)


def test_autocovariance_at_lag_zero_equals_the_variance() -> None:
    config = TedConfig("gardner", "antipodal")
    stats = measure_ted_statistics(config, NYQUIST, sample_snr_db=20.0, samples=100000, seed=9)
    autocovariance = measure_ted_autocovariance(
        config, NYQUIST, sample_snr_db=20.0, max_lag=4, samples=100000, seed=9
    )
    assert autocovariance[0] == pytest.approx(stats.variance, rel=1e-9)


def test_autocovariance_is_essentially_white_for_mueller_muller() -> None:
    autocovariance = measure_ted_autocovariance(
        TedConfig("mueller-muller", "antipodal"),
        NYQUIST,
        sample_snr_db=20.0,
        max_lag=8,
        samples=200000,
    )
    assert np.max(np.abs(autocovariance[1:] / autocovariance[0])) < 0.02


def test_autocovariance_is_strongly_coloured_for_the_early_late_gate() -> None:
    autocovariance = measure_ted_autocovariance(
        TedConfig("early-late", "antipodal", 0.25, "dd"),
        NYQUIST,
        sample_snr_db=20.0,
        max_lag=8,
        samples=200000,
    )
    assert autocovariance[1] / autocovariance[0] < -0.3


def test_closed_loop_acquires_a_static_offset() -> None:
    config = TedConfig("mueller-muller", "antipodal")
    gain = scurve(config, NYQUIST, FINE, max_exact_symbols=16).gain_central_difference
    design = LoopDesign.from_bandwidth(0.01, ZETA, gain)
    run = run_timing_loop(
        config,
        NYQUIST,
        design,
        n_symbols=20000,
        sample_snr_db=30.0,
        true_offset=0.2,
        initial_error=-0.2,
    )
    assert not run.diverged
    assert run.slip_count == 0
    assert abs(run.mean_error) < 0.01
    assert run.jitter_rms < 0.01


def test_closed_loop_jitter_matches_the_coloured_prediction_for_mueller_muller() -> None:
    """The clean case: no self-noise, so all three routes agree."""
    config = TedConfig("mueller-muller", "antipodal")
    gain = scurve(config, NYQUIST, FINE, max_exact_symbols=16).gain_central_difference
    autocovariance = measure_ted_autocovariance(
        config, NYQUIST, sample_snr_db=18.0, max_lag=16, samples=200000
    )
    design = LoopDesign.from_bandwidth(0.004, ZETA, gain)
    run = run_timing_loop(config, NYQUIST, design, n_symbols=120000, sample_snr_db=18.0)
    predicted = jitter_variance_coloured(design, autocovariance)
    white = jitter_variance_closed_form(0.004, gain, autocovariance[0])
    assert run.jitter_variance == pytest.approx(predicted, rel=0.15)
    assert run.jitter_variance == pytest.approx(white, rel=0.2)


def test_white_noise_prediction_badly_overestimates_the_early_late_gate() -> None:
    """A documented divergence: self-noise is not white and the loop filters it better."""
    config = TedConfig("early-late", "antipodal", 0.25, "dd")
    gain = scurve(config, NYQUIST, FINE, max_exact_symbols=16).gain_central_difference
    autocovariance = measure_ted_autocovariance(
        config, NYQUIST, sample_snr_db=20.0, max_lag=16, samples=200000
    )
    design = LoopDesign.from_bandwidth(0.005, ZETA, gain)
    run = run_timing_loop(config, NYQUIST, design, n_symbols=120000, sample_snr_db=20.0)
    white = jitter_variance_closed_form(0.005, gain, autocovariance[0])
    coloured = jitter_variance_coloured(design, autocovariance)
    assert white / coloured > 5.0
    assert run.jitter_variance == pytest.approx(coloured, rel=0.25)


def test_self_noise_produces_a_static_lock_offset_proportional_to_bandwidth() -> None:
    """Gardner locks away from the true centre; Mueller-Mueller does not."""
    offsets = {}
    for name, config in (
        ("gardner", TedConfig("gardner", "antipodal")),
        ("mueller-muller", TedConfig("mueller-muller", "antipodal")),
    ):
        gain = scurve(config, NYQUIST, FINE, max_exact_symbols=16).gain_central_difference
        values = []
        for bandwidth in (0.005, 0.02):
            design = LoopDesign.from_bandwidth(bandwidth, ZETA, gain)
            run = run_timing_loop(
                config, NYQUIST, design, n_symbols=40000, sample_snr_db=50.0, seed=11
            )
            values.append(run.mean_error)
        offsets[name] = values
    assert abs(offsets["mueller-muller"][0]) < 1e-4
    assert abs(offsets["mueller-muller"][1]) < 1e-4
    assert offsets["gardner"][0] > 1e-3
    # Four times the bandwidth, roughly four times the offset.
    assert offsets["gardner"][1] / offsets["gardner"][0] == pytest.approx(4.0, rel=0.3)


def test_loop_slips_at_low_signal_to_noise_ratio() -> None:
    config = TedConfig("mueller-muller", "antipodal")
    gain = scurve(config, NYQUIST, FINE, max_exact_symbols=16).gain_central_difference
    design = LoopDesign.from_bandwidth(0.05, ZETA, gain)
    quiet = run_timing_loop(config, NYQUIST, design, n_symbols=20000, sample_snr_db=25.0)
    noisy = run_timing_loop(config, NYQUIST, design, n_symbols=20000, sample_snr_db=-2.0)
    assert quiet.slip_count == 0
    assert noisy.slip_count > 0


def test_standard_errors_are_reported_and_positive() -> None:
    config = TedConfig("gardner", "antipodal")
    gain = scurve(config, NYQUIST, FINE, max_exact_symbols=16).gain_central_difference
    design = LoopDesign.from_bandwidth(0.01, ZETA, gain)
    run = run_timing_loop(config, NYQUIST, design, n_symbols=60000, sample_snr_db=20.0)
    assert run.jitter_variance_standard_error > 0.0
    assert run.mean_error_standard_error > 0.0
    assert 0.0 < run.jitter_variance_relative_error < 0.25
    assert set(run.summary()) >= {"jitter_variance", "slips", "K_d"}


def test_use_true_decisions_differs_from_sliced_decisions_at_low_snr() -> None:
    config = TedConfig("mueller-muller", "antipodal")
    sliced = measure_ted_statistics(config, NYQUIST, sample_snr_db=0.0, samples=100000)
    ideal = measure_ted_statistics(
        config, NYQUIST, sample_snr_db=0.0, samples=100000, use_true_decisions=True
    )
    assert sliced.variance != pytest.approx(ideal.variance, rel=1e-6)


def test_simulation_validation() -> None:
    config = TedConfig("gardner", "antipodal")
    design = LoopDesign.from_bandwidth(0.01, ZETA, 1.5)
    with pytest.raises(ValueError, match="at least 2000"):
        run_timing_loop(config, NYQUIST, design, n_symbols=100)
    with pytest.raises(ValueError, match=r"discard must lie"):
        run_timing_loop(config, NYQUIST, design, n_symbols=5000, discard=5000)
    with pytest.raises(ValueError, match="at least 1000"):
        measure_ted_statistics(config, NYQUIST, sample_snr_db=10.0, samples=10)
    with pytest.raises(ValueError, match="non-negative"):
        measure_ted_autocovariance(config, NYQUIST, sample_snr_db=10.0, max_lag=-1)


def test_divergence_is_flagged_rather_than_producing_a_meaningless_variance() -> None:
    """A loop started a long way out with a tiny bandwidth never finds lock."""
    config = TedConfig("mueller-muller", "antipodal")
    design = LoopDesign.from_bandwidth(0.0005, ZETA, 1.57)
    run = run_timing_loop(
        config,
        NYQUIST,
        design,
        n_symbols=20000,
        sample_snr_db=20.0,
        initial_error=0.0,
        true_offset=0.0,
        divergence_limit=0.0001,
    )
    assert run.diverged
