"""Validation: the AR(1) lognormal series is the series it claims to be.

Checks
------
1. ``ar1_unit_variance`` reproduces the direct recursion
   ``x[0] = z[0]``, ``x[n] = rho x[n-1] + sqrt(1-rho**2) z[n]`` to within
   floating-point noise over two million samples. The implementation uses
   ``scipy.signal.lfilter`` plus a closed-form correction for the initial
   condition, so this is the check that the correction is right.
2. The filtered series has unit variance and lag-one autocorrelation ``rho``,
   both to within their sampling errors.
3. The lognormal mapping gives the stated mean irradiance.
4. The amplitude threshold maps exactly onto a level of the Gaussian: the mask
   ``a < T`` and the mask ``x < u`` must agree on every single sample, because
   the map from ``x`` to ``a`` is strictly increasing. Any disagreement would
   mean the analytic comparands are computed for a different threshold from the
   measured statistics.

Runtime: about 3 s on one core.
"""

from __future__ import annotations

import math

import _bootstrap  # noqa: F401
import numpy as np
from scipy.signal import lfilter

from linkoutage.channel import (
    amplitude_threshold_to_gaussian_level,
    ar1_unit_variance,
    lognormal_amplitude_series,
    rho_from_tau,
)
from linkoutage.fade import below_threshold

N = 2_000_000
FS = 1.0e6
TAU = 2.0e-4
SI = 0.6
SEED = 41
THRESHOLD = 0.6

print("=" * 78)
print("validate_series_construction.py")
print("=" * 78)

rho = rho_from_tau(TAU, FS)
print(f"rho = exp(-1/(tau*fs)) = {rho!r}")
print(f"correlation length     = {TAU * FS!r} samples")

z = np.random.default_rng(SEED).standard_normal(N)
x = ar1_unit_variance(z, rho)

print()
print("[1] AR(1) recursion against the direct loop")
prefix = 50_000
direct = np.empty(prefix)
direct[0] = z[0]
s = math.sqrt(1.0 - rho * rho)
for i in range(1, prefix):
    direct[i] = rho * direct[i - 1] + s * z[i]
err_prefix = float(np.max(np.abs(direct - x[:prefix])))
print(f"  max |lfilter+correction - direct loop| over first {prefix} samples: {err_prefix!r}")

# A second, independent route over the full length: lfilter with no correction,
# then the analytic discrepancy term subtracted the other way round.
plain = lfilter([s], [1.0, -rho], z)
discrepancy = z[0] * (1.0 - s) * np.exp(np.arange(N, dtype=np.float64) * math.log(rho))
err_full = float(np.max(np.abs((plain + discrepancy) - x)))
print(f"  max |independent reconstruction - x| over all {N} samples:        {err_full!r}")
print(f"  x[0] - z[0] = {float(x[0] - z[0])!r} (must be 0: stationary initial condition)")
print(f"  PASS       = {err_prefix < 1e-12 and err_full == 0.0}")

print()
print("[2] Second-order statistics of x")
var = float(x.var())
centred = x - x.mean()
lag1 = float(np.dot(centred[:-1], centred[1:]) / np.dot(centred, centred))
# Effective independent sample count for a process with correlation length L is
# about N / (2 L); the standard error of the variance scales accordingly.
n_eff = N / (2.0 * TAU * FS)
se_var = math.sqrt(2.0 / n_eff)
print(f"  sample variance      = {var!r}   (target 1, se ~ {se_var!r})")
print(f"  lag-one correlation  = {lag1!r}   (target {rho!r})")
print(f"  PASS                 = {abs(var - 1.0) < 4 * se_var and abs(lag1 - rho) < 1e-3}")

print()
print("[3] Lognormal mapping")
series = lognormal_amplitude_series(N, fs_hz=FS, tau_s=TAU, si=SI, seed=SEED)
mean_i = float(series.irradiance.mean())
var_log_i = float(np.log(series.irradiance).var())
print(f"  mean irradiance          = {mean_i!r}  (target 1.0)")
print(f"  var(ln I)                = {var_log_i!r}  (target {math.log1p(SI)!r})")
_amp_err = float(np.max(np.abs(series.amplitude**2 - series.irradiance)))
print(f"  max |a**2 - I|           = {_amp_err!r}")
scaled = lognormal_amplitude_series(
    200_000, fs_hz=FS, tau_s=TAU, si=SI, seed=7, mean_irradiance=4.0
)
print(f"  mean irradiance at E[I]=4 = {float(scaled.irradiance.mean())!r}")
_ok3 = abs(mean_i - 1.0) < 0.01 and abs(var_log_i - math.log1p(SI)) < 0.01
print(f"  PASS                     = {_ok3}")

print()
print("[4] Threshold maps exactly onto a Gaussian level")
u = amplitude_threshold_to_gaussian_level(THRESHOLD, SI)
by_amplitude = below_threshold(series.amplitude, THRESHOLD)
by_level = series.gaussian < u
mismatch = int(np.count_nonzero(by_amplitude != by_level))
print(f"  u = {u!r}")
print(f"  samples where (a < T) != (x < u): {mismatch}  (must be 0)")
print(f"  PASS = {mismatch == 0}")
print()
print("done")
