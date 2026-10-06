"""The README's worked example, run so that the output in the README is real.

Run from this directory with ``PYTHONPATH=../src``.  Runtime about 8 s.
"""

from __future__ import annotations

import math

import numpy as np

from slotsync import (
    LoopDesign,
    PpmConfig,
    TedConfig,
    cycle_slip_rate_rice,
    half_sine,
    jitter_variance_closed_form,
    jitter_variance_coloured,
    loop_snr_db,
    measure_ted_autocovariance,
    measure_ted_statistics,
    nyquist_raised_cosine,
    ppm_slot_scurve,
    run_ppm_slot_loop,
    run_timing_loop,
    scurve,
)

pulse = nyquist_raised_cosine(rolloff=0.5, truncate_symbols=4.0)
config = TedConfig(detector="gardner", alphabet="antipodal")

# 1. The S-curve is computed exactly, and K_d is its slope at the origin.
curve = scurve(config, pulse, np.linspace(-0.02, 0.02, 41), max_exact_symbols=16)
gain = curve.gain_central_difference
print(f"K_d = {gain:.6f} per symbol, from {curve.pattern_count} enumerated data patterns")
full = scurve(config, pulse, max_exact_symbols=16)
print(
    f"  linear to {full.linear_halfwidth:.3f} symbol, flattens at {full.peak_offset:.3f}, "
    f"reverses at {full.reversal_offset:.3f}"
)

# 2. The loop is designed from that gain, never from a guess.
design = LoopDesign.from_bandwidth(
    noise_bandwidth=0.005, damping=1 / math.sqrt(2), detector_gain=gain
)
print(f"  B_n = {design.noise_bandwidth}, zeta = {design.damping:.4f} -> "
      f"k1 = {design.k_proportional:.6f}, k2 = {design.k_integral:.3e}")

# 3. The detector's own output noise is measured, not assumed.
stats = measure_ted_statistics(config, pulse, sample_snr_db=20.0, samples=400000)
autocovariance = measure_ted_autocovariance(
    config, pulse, sample_snr_db=20.0, max_lag=24, samples=400000
)
print(
    f"  sigma_n^2 = {stats.variance:.6f} "
    f"({100 * stats.self_noise_variance / stats.variance:.0f} % self-noise), "
    f"R[1]/R[0] = {autocovariance[1] / autocovariance[0]:+.4f}"
)

# 4. Two predictions, then the loop itself.
white = jitter_variance_closed_form(0.005, gain, stats.variance)
coloured = jitter_variance_coloured(design, autocovariance)
run = run_timing_loop(config, pulse, design, n_symbols=200000, sample_snr_db=20.0)
print(f"  jitter variance: white {white:.4e}  coloured {coloured:.4e}  "
      f"measured {run.jitter_variance:.4e} +- {run.jitter_variance_standard_error:.1e}")
print(f"  measured / coloured = {run.jitter_variance / coloured:.4f}, "
      f"measured / white = {run.jitter_variance / white:.4f}")
print(f"  rms jitter {run.jitter_rms:.6f} symbol, static lock offset "
      f"{run.mean_error:+.6f}, slips {run.slip_count}")
print(f"  loop SNR {loop_snr_db(run.jitter_variance, full.reversal_offset):.2f} dB, "
      f"Rice slip estimate {cycle_slip_rate_rice(design, autocovariance[0]):.3e} per symbol")

# 5. The same chain for a 4-PPM slot clock, in slot periods.
slot = half_sine(1.0)
ppm = PpmConfig(order=4, delta=0.25)
slot_curve = ppm_slot_scurve(ppm, slot, offsets=np.linspace(-0.001, 0.001, 21), fit_halfwidth=0.001)
slot_design = LoopDesign.from_bandwidth(0.005, 1 / math.sqrt(2), slot_curve.gain_central_difference)
slot_run = run_ppm_slot_loop(ppm, slot, slot_design, n_symbols=40000, sample_snr_db=20.0)
print(f"4-PPM slot clock: K_d = {slot_curve.gain_central_difference:.6f} per slot "
      f"(2 pi = {2 * math.pi:.6f})")
print(f"  rms jitter {slot_run.jitter_rms:.6f} slot, slot error rate "
      f"{slot_run.slot_error_rate:.5f}, loop slips {slot_run.slip_count}")
