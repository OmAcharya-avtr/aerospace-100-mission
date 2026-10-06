"""Cross-check X1: level-crossing rate and mean fade duration on a pinned series.

Batch 05 specifies this check as a comparison between two independent
implementations of the same two sample statistics on *the same seeded sample
path*. This script constructs that path from the specification, confirms it
against the one input value the specification supplies, and then reports the
two statistics under definitions that are printed in full beside them.

The series, exactly as specified
--------------------------------
* lognormal irradiance, temporally correlated by a Gauss-Markov (AR(1)) filter
  in log-amplitude,
* driving noise ``numpy.random.default_rng(41).standard_normal(2000000)``:
  seed 41, drawn in ONE call of length 2 000 000,
* ``fs = 1.0e6`` Hz, ``tau = 2.0e-4`` s, hence
  ``rho = exp(-1/(tau*fs)) = exp(-0.005)``, correlation length 200 samples,
* stationary AR(1): ``x[0] = z[0]``, ``x[n] = rho x[n-1] + sqrt(1-rho**2) z[n]``,
  so ``x`` has unit variance at every index,
* scintillation index ``SI = 0.6``, hence ``sigma_lnI**2 = ln(1 + SI)``,
* unit mean irradiance, so ``ln I = -sigma_lnI**2/2 + sigma_lnI x`` and
  ``I = exp(ln I)``,
* amplitude ``a = sqrt(I)``,
* threshold: amplitude 0.6.

What is being compared, and what is not
---------------------------------------
The compared quantities are the **level-crossing rate in Hz** and the **mean
fade duration in seconds**, both as *sample statistics of this series*. The
analytic discrete-time crossing rate of the model is printed at the end as
context and is explicitly **not** the comparand. No wall-clock measurement
appears anywhere in this product.

Runtime: about 4 s on one core.
"""

from __future__ import annotations

import math

import _bootstrap  # noqa: F401
import numpy as np

from linkoutage.channel import (
    analytic_level_crossing_rate,
    analytic_mean_fade_duration,
    analytic_outage_fraction,
    lognormal_amplitude_series,
    rho_from_tau,
    sigma_ln_i_from_si,
)
from linkoutage.fade import FadeDefinitions, fade_runs, fade_statistics

N_SAMPLES = 2_000_000
SEED = 41
FS_HZ = 1.0e6
TAU_S = 2.0e-4
SI = 0.6
THRESHOLD = 0.6
REFERENCE_MEAN_IRRADIANCE = 0.998907972681559

print("=" * 78)
print("validate_cross_check_x1.py -- cross-check X1 (P049 side)")
print("=" * 78)
print()
print("SPECIFIED CONFIGURATION")
print("  model                  : lognormal irradiance, AR(1) Gauss-Markov log-amplitude")
print(f"  n_samples              : {N_SAMPLES}")
print(f"  driving noise          : numpy.random.default_rng({SEED})"
      f".standard_normal({N_SAMPLES})  [ONE call]")
print(f"  fs                     : {FS_HZ!r} Hz")
print(f"  tau                    : {TAU_S!r} s")
print(f"  rho = exp(-1/(tau*fs)) : {rho_from_tau(TAU_S, FS_HZ)!r}")
print(f"  correlation length     : {TAU_S * FS_HZ!r} samples")
print(f"  scintillation index SI : {SI!r}")
print(f"  sigma_lnI**2 = ln(1+SI): {sigma_ln_i_from_si(SI) ** 2!r}")
print(f"  sigma_lnI              : {sigma_ln_i_from_si(SI)!r}")
print("  mean irradiance E[I]   : 1.0 by construction")
print("  amplitude a            : sqrt(I)")
print(f"  amplitude threshold T  : {THRESHOLD!r}")

series = lognormal_amplitude_series(
    N_SAMPLES, fs_hz=FS_HZ, tau_s=TAU_S, si=SI, seed=SEED
)

print()
print("INPUT CHECK -- the series itself, not the compared statistics")
mean_i = float(series.irradiance.mean())
print(f"  computed mean irradiance : {mean_i!r}")
print(f"  specified reference      : {REFERENCE_MEAN_IRRADIANCE!r}")
print(f"  absolute difference      : {abs(mean_i - REFERENCE_MEAN_IRRADIANCE)!r}")
print(f"  relative difference      : "
      f"{abs(mean_i - REFERENCE_MEAN_IRRADIANCE) / REFERENCE_MEAN_IRRADIANCE!r}")
reproduced = abs(mean_i - REFERENCE_MEAN_IRRADIANCE) < 1e-12
print(f"  SERIES REPRODUCED        : {reproduced}")
if not reproduced:
    raise SystemExit(
        "the constructed series does not match the specified mean irradiance; "
        "the statistics below would not be comparable and are not reported"
    )
_c = series.gaussian - series.gaussian.mean()
_lag1 = float(np.dot(_c[:-1], _c[1:]) / np.dot(_c, _c))
print(f"  lag-one correlation of x : {_lag1!r}")

stats = fade_statistics(series.amplitude, THRESHOLD, FS_HZ)

print()
print("DEFINITIONS IN FORCE (the published conventions)")
print(stats.definitions.describe())

print()
print("COMPARED QUANTITIES -- sample statistics of this series")
print(f"  sample count N                 : {stats.n_samples}")
print(f"  record duration (N-1)/fs       : {stats.record_duration_s!r} s")
print(f"  LEVEL-CROSSING RATE            : {stats.level_crossing_rate_hz!r} Hz")
print(f"  MEAN FADE DURATION             : {stats.mean_fade_duration_s!r} s")

print()
print("SUPPORTING COUNTS")
print(f"  down-crossings                 : {stats.n_down_crossings}")
print(f"  up-crossings                   : {stats.n_up_crossings}")
print(f"  fade runs (all)                : {stats.n_runs}")
print(f"  complete fades (in the mean)    : {stats.n_complete_fades}")
print(f"  censored runs (excluded)       : {stats.n_left_censored + stats.n_right_censored}")
print(f"    of which left-censored       : {stats.n_left_censored}")
print(f"    of which right-censored      : {stats.n_right_censored}")
print(f"  single-sample fades (counted)  : {stats.n_single_sample_fades}")
print(f"  in-fade samples                : {stats.in_fade_samples}")
print(f"  OUTAGE FRACTION                : {stats.outage_fraction!r}")
print(f"  availability                   : {stats.availability!r}")
print(f"  median complete fade duration  : {stats.median_fade_duration_s!r} s")
print(f"  longest complete fade          : {stats.max_complete_fade_duration_s!r} s")

print()
print("INTERNAL CONSISTENCY (Rice partition of below-threshold time)")
print(f"  outage_fraction / LCR          : {stats.rice_consistency_mean_fade_duration_s!r} s")
print(f"  relative residual vs the mean  : {stats.rice_relative_residual!r}")
print("  the residual is the combined effect of the one censored run and the")
print("  (N-1)/fs versus N/fs choice of record duration, both stated above")

print()
print("SAMPLING ERROR OF THE COMPARED QUANTITIES")
n_cross = stats.n_down_crossings
se_rate_rel = 1.0 / math.sqrt(n_cross)
runs = fade_runs(series.amplitude, THRESHOLD, FS_HZ)
durations = runs.complete_durations_s()
se_mfd = float(durations.std(ddof=1) / math.sqrt(durations.size))
print(f"  crossings counted              : {n_cross}")
print(f"  approx relative se of the rate : {se_rate_rel!r}  (Poisson, 1/sqrt(n))")
print("  this understates the true error: crossings of a correlated process are")
print("  not independent, so the effective count is below the raw count")
print(f"  standard error of the mean fade duration : {se_mfd!r} s")
print(f"  relative                                 : {se_mfd / stats.mean_fade_duration_s!r}")

print()
print("DEFINITIONAL SENSITIVITY -- for tracing a disagreement to its cause")
variants = {
    "published (sample_count, exclude, singles counted)": FadeDefinitions(),
    "interval_count durations": FadeDefinitions(duration_convention="interval_count"),
    "interpolated crossing instants": FadeDefinitions(duration_convention="interpolated"),
    "censored runs folded in as complete": FadeDefinitions(censoring="include_as_complete"),
    "single-sample excursions not fades": FadeDefinitions(count_single_sample_fades=False),
    "non-strict a <= T": FadeDefinitions(strict_below=False),
    "record duration N/fs": FadeDefinitions(record_duration_convention="samples"),
}
base = None
header = f"  {'variant':<52s} {'LCR [Hz]':>16s} {'MFD [s]':>16s} {'dLCR':>9s} {'dMFD':>9s}"
print(header)
print("  " + "-" * (len(header) - 2))
for label, defs in variants.items():
    s = fade_statistics(series.amplitude, THRESHOLD, FS_HZ, definitions=defs)
    if base is None:
        base = (s.level_crossing_rate_hz, s.mean_fade_duration_s)
        d_rate = 0.0
        d_mfd = 0.0
    else:
        d_rate = s.level_crossing_rate_hz / base[0] - 1.0
        d_mfd = s.mean_fade_duration_s / base[1] - 1.0
    print(
        f"  {label:<52s} {s.level_crossing_rate_hz:>16.6f} "
        f"{s.mean_fade_duration_s:>16.9e} {d_rate:>+9.2%} {d_mfd:>+9.2%}"
    )
print("  A disagreement with the sibling product of the size of one of these rows")
print("  is a definitional difference, not an implementation error.")

print()
print("CONTEXT ONLY -- NOT the compared quantities")
a_rate = analytic_level_crossing_rate(THRESHOLD, si=SI, tau_s=TAU_S, fs_hz=FS_HZ)
a_mfd = analytic_mean_fade_duration(THRESHOLD, si=SI, tau_s=TAU_S, fs_hz=FS_HZ)
a_frac = analytic_outage_fraction(THRESHOLD, SI)
print(f"  analytic discrete-time LCR of the model  : {a_rate!r} Hz")
print(f"  analytic mean fade duration of the model : {a_mfd!r} s")
print(f"  analytic outage fraction Phi(u)          : {a_frac!r}")
print(
    "  sample LCR / analytic LCR - 1            : "
    f"{stats.level_crossing_rate_hz / a_rate - 1.0!r}"
)
print(f"  sample MFD / analytic MFD - 1            : {stats.mean_fade_duration_s / a_mfd - 1.0!r}")
print("  These are model-versus-realisation gaps of one finite sample path, not")
print("  the cross-check. The cross-check compares the two SAMPLE statistics")
print("  above against another implementation's sample statistics on the same")
print("  series.")
print()
print("done")
