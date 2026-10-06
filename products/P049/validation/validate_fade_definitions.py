"""Validation: the fade definitions, their hand-checked answers, and what they cost.

The quantities this product publishes are not uniquely named. This script is
the evidence that (a) each definition is implemented as written, checked against
answers computable by hand from a seven-sample series, and (b) the choice
between them is worth several tens of percent on a real record, which is why
every number this package prints carries its definitions.

Checks
------
1. Hand-calculated known answers on two literal series, including the censored
   case, with the arithmetic shown.
2. The Rice partition identity: with no censoring and every below-threshold
   sample inside a complete run, the measured mean fade duration must be
   exactly ``N / (N - 1)`` times ``outage_fraction / level_crossing_rate``.
   This is an algebraic identity, so it is checked to machine precision rather
   than to a tolerance.
3. The cost of each definitional choice on the specified two-million-sample
   series: level-crossing rate, mean fade duration, outage fraction and
   availability under seven variants.
4. Threshold sweeps of outage probability and availability, with the analytic
   curve beside them.
5. Quantisation: a record logged to 8 bits is where the strict/non-strict
   choice stops being academic.

Runtime: about 10 s on one core.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import numpy as np

from linkoutage.channel import (
    analytic_level_crossing_rate,
    analytic_mean_fade_duration,
    analytic_outage_fraction,
    lognormal_amplitude_series,
)
from linkoutage.fade import FadeDefinitions, fade_statistics

FS = 1.0e6
TAU = 2.0e-4
SI = 0.6
N = 2_000_000

print("=" * 78)
print("validate_fade_definitions.py")
print("=" * 78)

print()
print("[1] Hand-calculated known answers")
print("  (a) a = [1.0, 0.5, 0.5, 1.0, 1.0, 0.4, 1.0], T = 0.6, fs = 1.0 Hz")
print("      mask = [F, T, T, F, F, T, F]")
print("      runs = [1,3) length 2; [5,6) length 1, neither touching an end")
print("      down-crossings at 1 and 5 -> 2;  up-crossings at 3 and 6 -> 2")
print("      mean fade duration = (2 + 1) / 2 samples / 1 Hz = 1.5 s")
print("      outage fraction = 3 / 7;  LCR = 2 / ((7-1)/1) = 1/3 Hz")
a1 = np.array([1.0, 0.5, 0.5, 1.0, 1.0, 0.4, 1.0])
s1 = fade_statistics(a1, 0.6, 1.0)
expect1 = {
    "n_down_crossings": 2,
    "n_up_crossings": 2,
    "n_complete_fades": 2,
    "n_single_sample_fades": 1,
    "mean_fade_duration_s": 1.5,
    "level_crossing_rate_hz": 2.0 / 6.0,
    "outage_fraction": 3.0 / 7.0,
    "availability": 4.0 / 7.0,
}
ok1 = True
for key, want in expect1.items():
    got = getattr(s1, key)
    good = abs(got - want) < 1e-12
    ok1 &= good
    flag = "ok" if good else "FAIL"
    print(f"      {key:<26s} computed {got!r:<24s} expected {want!r:<22s} {flag}")
print(f"      PASS = {ok1}")

print()
print("  (b) a = [0.4, 0.4, 1.0, 1.0, 0.3], T = 0.6, fs = 2.0 Hz  (both ends censored)")
print("      mask = [T, T, F, F, T]")
print("      run [0,2) is left-censored, run [4,5) is right-censored")
print("      index 0 has no predecessor, so there is no down-crossing there -> 1")
print("      no complete fade survives the censoring rule -> mean is nan")
print("      outage fraction = 3 / 5;  LCR = 1 / ((5-1)/2) = 0.5 Hz")
a2 = np.array([0.4, 0.4, 1.0, 1.0, 0.3])
s2 = fade_statistics(a2, 0.6, 2.0)
checks2 = [
    ("n_down_crossings", s2.n_down_crossings, 1),
    ("n_up_crossings", s2.n_up_crossings, 1),
    ("n_runs", s2.n_runs, 2),
    ("n_left_censored", s2.n_left_censored, 1),
    ("n_right_censored", s2.n_right_censored, 1),
    ("n_complete_fades", s2.n_complete_fades, 0),
    ("level_crossing_rate_hz", s2.level_crossing_rate_hz, 0.5),
    ("outage_fraction", s2.outage_fraction, 0.6),
]
ok2 = all(abs(got - want) < 1e-12 for _, got, want in checks2)
for name, got, want in checks2:
    print(f"      {name:<26s} computed {got!r:<24s} expected {want!r:<22s}")
print(f"      mean_fade_duration_s is nan: {np.isnan(s2.mean_fade_duration_s)}")
print(f"      PASS = {ok2 and np.isnan(s2.mean_fade_duration_s)}")

print()
print("  (c) interpolated crossing instants, a = [1.0, 0.4, 1.0], T = 0.6, fs = 1 Hz")
print("      down fraction = (1.0 - 0.6) / (1.0 - 0.4) = 2/3 -> t_down = 2/3 s")
print("      up   fraction = (0.6 - 0.4) / (1.0 - 0.4) = 1/3 -> t_up   = 4/3 s")
print("      duration = 4/3 - 2/3 = 2/3 s, versus 1 s by sample count")
a3 = np.array([1.0, 0.4, 1.0])
s3 = fade_statistics(
    a3, 0.6, 1.0, definitions=FadeDefinitions(duration_convention="interpolated")
)
print(f"      computed {s3.mean_fade_duration_s!r}, expected {2.0 / 3.0!r}")
print(f"      PASS = {abs(s3.mean_fade_duration_s - 2.0 / 3.0) < 1e-12}")

print()
print("[2] Rice partition identity, checked to machine precision")
print("  With nothing censored, below-threshold time is partitioned exactly into")
print("  the complete fades, so mean_fade_duration * LCR * (N-1)/fs = in-fade")
print("  samples / fs, hence mean / (outage/LCR) = N / (N-1) exactly.")
header = f"  {'series':<34s} {'N':>8s} {'residual':>24s} {'1/(N-1)':>24s} {'match':>7s}"
print(header)
print("  " + "-" * (len(header) - 2))
ok_rice = True
for label, arr, fs in [
    ("hand series (a)", a1, 1.0),
    ("500-sample channel realisation", None, FS),
    ("20000-sample channel realisation", None, FS),
]:
    if arr is None:
        n = 500 if "500" in label else 20_000
        realisation = lognormal_amplitude_series(
            n, fs_hz=fs, tau_s=TAU, si=SI, seed=1234
        )
        arr = realisation.amplitude
        # Trim so that neither end is inside a fade.
        mask = arr < 0.6
        first_good = int(np.argmax(~mask))
        last_good = int(len(arr) - 1 - np.argmax(~mask[::-1]))
        arr = arr[first_good : last_good + 1]
    s = fade_statistics(arr, 0.6, fs)
    if s.n_complete_fades == 0:
        print(f"  {label:<34s} {s.n_samples:>8d} {'no complete fade':>24s}")
        continue
    want = 1.0 / (s.n_samples - 1)
    good = abs(s.rice_relative_residual - want) < 1e-12
    ok_rice &= good
    print(
        f"  {label:<34s} {s.n_samples:>8d} {s.rice_relative_residual!r:>24s} "
        f"{want!r:>24s} {'ok' if good else 'FAIL':>7s}"
    )
print(f"  PASS = {ok_rice}")

print()
print("[3] What each definitional choice costs on the specified series")
series = lognormal_amplitude_series(N, fs_hz=FS, tau_s=TAU, si=SI, seed=41)
variants = {
    "published": FadeDefinitions(),
    "interval_count durations": FadeDefinitions(duration_convention="interval_count"),
    "interpolated instants": FadeDefinitions(duration_convention="interpolated"),
    "censored folded in as complete": FadeDefinitions(censoring="include_as_complete"),
    "singles are not fades": FadeDefinitions(count_single_sample_fades=False),
    "non-strict a <= T": FadeDefinitions(strict_below=False),
    "record duration N/fs": FadeDefinitions(record_duration_convention="samples"),
}
header = (
    f"  {'variant':<32s} {'LCR [Hz]':>14s} {'dLCR':>8s} {'MFD [s]':>15s} {'dMFD':>8s} "
    f"{'outage':>10s} {'avail':>10s}"
)
print(header)
print("  " + "-" * (len(header) - 2))
base = None
for label, defs in variants.items():
    s = fade_statistics(series.amplitude, 0.6, FS, definitions=defs)
    if base is None:
        base = (s.level_crossing_rate_hz, s.mean_fade_duration_s)
    print(
        f"  {label:<32s} {s.level_crossing_rate_hz:>14.4f} "
        f"{s.level_crossing_rate_hz / base[0] - 1.0:>+8.2%} "
        f"{s.mean_fade_duration_s:>15.8e} "
        f"{s.mean_fade_duration_s / base[1] - 1.0:>+8.2%} "
        f"{s.outage_fraction:>10.6f} {s.availability:>10.6f}"
    )
print("  The single-sample rule is the dominant choice: -32 % on the rate and")
print("  +43 % on the mean fade duration. Two tools that disagree on that rule")
print("  cannot agree on either statistic, whatever else they get right.")

print()
print("[4] Threshold sweep: outage probability and availability")
header = (
    f"  {'T':>6s} {'outage measured':>16s} {'outage analytic':>16s} {'availability':>13s} "
    f"{'LCR meas [Hz]':>14s} {'LCR anal [Hz]':>14s} {'MFD meas [s]':>14s} "
    f"{'MFD anal [s]':>14s}"
)
print(header)
print("  " + "-" * (len(header) - 2))
for thr in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
    s = fade_statistics(series.amplitude, thr, FS)
    print(
        f"  {thr:>6.2f} {s.outage_fraction:>16.7f} "
        f"{analytic_outage_fraction(thr, SI):>16.7f} {s.availability:>13.7f} "
        f"{s.level_crossing_rate_hz:>14.3f} "
        f"{analytic_level_crossing_rate(thr, si=SI, tau_s=TAU, fs_hz=FS):>14.3f} "
        f"{s.mean_fade_duration_s:>14.6e} "
        f"{analytic_mean_fade_duration(thr, si=SI, tau_s=TAU, fs_hz=FS):>14.6e}"
    )

print()
print("[5] Quantisation: when the strict/non-strict choice stops being academic")
print("  A receiver logs amplitude through an ADC. Quantising the same record to")
print("  8 bits over [0, 2] puts many samples exactly on a code boundary, and if")
print("  the threshold lands on a code the two rules differ by thousands of")
print("  samples rather than by none.")
header = (
    f"  {'bits':>5s} {'threshold':>10s} {'strict LCR':>13s} {'non-strict LCR':>15s} "
    f"{'rel diff':>9s} {'strict outage':>14s} {'non-strict outage':>18s}"
)
print(header)
print("  " + "-" * (len(header) - 2))
for bits in (8, 10, 12):
    step = 2.0 / (2**bits - 1)
    quantised = np.round(series.amplitude / step) * step
    thr_on_code = round(0.6 / step) * step
    s_strict = fade_statistics(quantised, thr_on_code, FS)
    s_loose = fade_statistics(
        quantised, thr_on_code, FS, definitions=FadeDefinitions(strict_below=False)
    )
    rel = s_loose.level_crossing_rate_hz / s_strict.level_crossing_rate_hz - 1.0
    print(
        f"  {bits:>5d} {thr_on_code:>10.6f} {s_strict.level_crossing_rate_hz:>13.3f} "
        f"{s_loose.level_crossing_rate_hz:>15.3f} {rel:>+9.2%} "
        f"{s_strict.outage_fraction:>14.7f} {s_loose.outage_fraction:>18.7f}"
    )
print("  On float data the two rules coincide; on logged data they do not, and")
print("  the difference is not small.")
print()
print("done")
