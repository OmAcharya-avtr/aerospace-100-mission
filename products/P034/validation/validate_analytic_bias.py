"""Validation 3: an injected constant sensor bias against its analytic effect.

Claim under test (Level 2 requirement): an injected fault with a known analytic
effect produces that effect.  The fault is a constant additive sensor bias; the
effect is the shift it produces in the Kalman filter's innovation sequence.

Derivation
----------
Plant and filter, both exactly as implemented in ``faultinject.target``:

    x_{k+1} = A x_k + B u_k + w_k            (plant)
    z_k     = H x_k + v_k + b                (measurement, with injected bias b)
    xhat^-_k = A xhat_{k-1} + B u_{k-1}      (prediction)
    e_k      = z_k - H xhat^-_k              (innovation)
    xhat_k   = xhat^-_k + K e_k              (update, K frozen at steady state)

Write eps_k = x_k - xhat^-_k for the prediction error.  Substituting the update
into the prediction gives

    eps_{k+1} = A (I - K H) eps_k - A K v_k - A K b + w_k.

Now compare the biased run with the unbiased run of the SAME seed, so that w_k
and v_k are identical in both and cancel.  The control input also cancels,
because within a run the same u_k drives the plant and the predictor -- so the
result below holds for ANY control law, including the saturating one used here.
Writing M = A (I - K H) and Delta for (biased minus unbiased):

    Delta eps_{k+1} = M Delta eps_k - A K b,      Delta eps_0 = 0
    Delta e_k       = H Delta eps_k + b

which solves in closed form to

    Delta e_k = b - H (I - M)^{-1} (I - M^k) A K b                        (1)

with limit, as k -> infinity and provided M is stable,

    Delta e_inf = G b,     G = I - H (I - M)^{-1} A K.                    (2)

The mean shift over a window of N steps is likewise exact:

    mean_k Delta e_k = G b + (1/N) H (I-M)^{-1} (I-M)^{-1} (I - M^N) A K b. (3)

Consequences this script checks, including one that is a negative result:

  3a. Delta e_0 = b exactly: the first innovation after the bias appears is
      shifted by the full bias.
  3b. Every step of the measured Delta e matches Eq. (1).
  3c. The mean shift over the run matches Eq. (3).
  3d. For a bias on the POSITION channel, Eq. (2) gives G b = 0: the filter
      absorbs a constant position bias completely, and an innovation-mean
      monitor becomes blind to it. For a bias on the VELOCITY channel the
      steady-state position-innovation shift is NOT zero. Both are checked.
  3e. Linearity: Delta e / b is independent of b.
  3f. Seed independence: Delta e agrees across noise seeds to floating-point
      rounding, as the derivation requires. Bit identity across seeds is NOT
      expected, because the two runs being differenced round differently.

Run from products/P034/:  python validation/validate_analytic_bias.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from faultinject.faults import Injection  # noqa: E402
from faultinject.harness import nominal_trace, run_case  # noqa: E402
from faultinject.target import kalman_gain, plant_matrices  # noqa: E402
from faultinject.taxonomy import FaultKind  # noqa: E402

N_STEPS = 150
TOL_STEP = 1e-12
TOL_MEAN = 1e-12
TOL_SS = 1e-13
TOL_LINEAR = 1e-12
TOL_SEED = 1e-12

failures: list[str] = []
I2 = np.eye(2)
A, _B = plant_matrices()
K = kalman_gain()
H = np.eye(2)
M = A @ (I2 - K @ H)
G = I2 - H @ np.linalg.inv(I2 - M) @ A @ K


def analytic_steps(b: np.ndarray, n: int) -> np.ndarray:
    """Eq. (1) evaluated for k = 0..n-1. Shape (n, 2)."""
    out = np.zeros((n, 2))
    inv = np.linalg.inv(I2 - M)
    mk = I2.copy()
    for k in range(n):
        out[k] = b - H @ inv @ (I2 - mk) @ A @ K @ b
        mk = mk @ M
    return out


def analytic_mean(b: np.ndarray, n: int) -> np.ndarray:
    """Eq. (3)."""
    inv = np.linalg.inv(I2 - M)
    s = inv @ (I2 - np.linalg.matrix_power(M, n))
    return G @ b + (1.0 / n) * (H @ inv @ s @ A @ K @ b)


def measured_delta(channel: str, offset: float, seed: int, n: int = N_STEPS) -> np.ndarray:
    """Measured biased-minus-nominal innovation sequence. Shape (n, 2)."""
    inj = Injection.create(FaultKind.SENSOR_BIAS, channel, {"offset": offset}, 0, n)
    faulted = run_case([inj], seed, n)
    nominal = nominal_trace(seed, n)
    if len(faulted.innovations) != n or len(nominal.innovations) != n:
        raise AssertionError("expected an innovation at every step with no dropouts")
    out = np.zeros((n, 2))
    for i, ((ka, fp, fv), (kb, np_, nv)) in enumerate(
        zip(faulted.innovations, nominal.innovations, strict=True)
    ):
        if ka != i or kb != i:
            raise AssertionError(f"innovation step mismatch at {i}: {ka} vs {kb}")
        out[i] = (fp - np_, fv - nv)
    return out


print("=" * 78)
print("VALIDATION 3 -- constant sensor bias vs the analytic innovation shift")
print("=" * 78)
print()
print("Steady-state Kalman gain K (frozen, from the Riccati fixed point):")
print(f"  [[{K[0, 0]:.15e}, {K[0, 1]:.15e}],")
print(f"   [{K[1, 0]:.15e}, {K[1, 1]:.15e}]]")
print()
print("Error dynamics M = A (I - K H); eigenvalues (must be inside the unit circle):")
eig = np.linalg.eigvals(M)
print(f"  {eig[0].real:.15f}, {eig[1].real:.15f}")
print()
print("Steady-state shift matrix G = I - H (I - M)^-1 A K:")
print(f"  [[{G[0, 0]: .15e}, {G[0, 1]: .15e}],")
print(f"   [{G[1, 0]: .15e}, {G[1, 1]: .15e}]]")
print()

# ------------------------------------------------------------------ 3a / 3b / 3c
print("3a-3c. Measured vs analytic, bias on each channel, b = 1.0, seed 11")
print()
hdr = (
    f"{'channel':>8} {'Delta e_0 err':>16} {'max step err':>16} "
    f"{'mean meas':>14} {'mean exact':>14} {'mean err':>12}"
)
print(hdr)
print("-" * len(hdr))
for channel, bvec in (("pos", np.array([1.0, 0.0])), ("vel", np.array([0.0, 1.0]))):
    meas = measured_delta(channel, 1.0, 11)
    exact = analytic_steps(bvec, N_STEPS)
    err0 = float(np.max(np.abs(meas[0] - bvec)))
    err_step = float(np.max(np.abs(meas - exact)))
    mean_meas = meas.mean(axis=0)
    mean_exact = analytic_mean(bvec, N_STEPS)
    err_mean = float(np.max(np.abs(mean_meas - mean_exact)))
    print(
        f"{channel:>8} {err0:16.3e} {err_step:16.3e} "
        f"{mean_meas[0]:14.9f} {mean_exact[0]:14.9f} {err_mean:12.3e}"
    )
    if err0 > TOL_STEP:
        failures.append(f"3a {channel}: Delta e_0 != b, error {err0:.3e} > {TOL_STEP:.0e}")
    if err_step > TOL_STEP:
        failures.append(f"3b {channel}: step error {err_step:.3e} > {TOL_STEP:.0e}")
    if err_mean > TOL_MEAN:
        failures.append(f"3c {channel}: mean error {err_mean:.3e} > {TOL_MEAN:.0e}")
print()
print("  'mean meas' and 'mean exact' are the POSITION component of the mean shift")
print(f"  over all {N_STEPS} steps; tolerance on every comparison {TOL_STEP:.0e}.")
print()

# ------------------------------------------------------------------ 3d
print("3d. Steady-state limit, Eq. (2). This is where a naive expectation fails.")
print()
for channel, bvec in (("pos", np.array([1.0, 0.0])), ("vel", np.array([0.0, 1.0]))):
    ss = G @ bvec
    meas = measured_delta(channel, 1.0, 11)
    tail = meas[-1]
    print(f"  bias on {channel}: analytic Delta e_inf = "
          f"[{ss[0]: .12e}, {ss[1]: .12e}]")
    print(f"              measured Delta e at k=149 = [{tail[0]: .12e}, {tail[1]: .12e}]")
    if channel == "pos":
        if abs(ss[0]) > TOL_SS:
            failures.append(f"3d: |G b|_pos = {abs(ss[0]):.3e} should be 0 for a pos bias")
        else:
            print("              PASS  a constant POSITION bias is fully absorbed: the")
            print("                    innovation shift decays to exactly zero, so an")
            print("                    innovation-mean monitor eventually sees nothing.")
    else:
        if abs(ss[0]) < 1.0:
            failures.append(f"3d: |G b|_pos = {abs(ss[0]):.3e} unexpectedly small for a vel bias")
        else:
            print("              PASS  a constant VELOCITY bias leaves a persistent")
            print(f"                    position-innovation shift of {ss[0]:.9f} per unit bias.")
    print()

# ------------------------------------------------------------------ 3e
print("3e. Linearity of the shift in the bias magnitude (Delta e / b must be constant)")
print()
ref = measured_delta("pos", 1.0, 11)
print(f"{'b':>8} {'max |Delta e / b - Delta e_1|':>32}")
print("-" * 42)
worst_lin = 0.0
for b in (0.1, 0.25, 1.0, 4.0, 10.0):
    meas = measured_delta("pos", b, 11)
    err = float(np.max(np.abs(meas / b - ref)))
    worst_lin = max(worst_lin, err)
    print(f"{b:8.2f} {err:32.3e}")
if worst_lin > TOL_LINEAR:
    failures.append(f"3e: linearity error {worst_lin:.3e} > {TOL_LINEAR:.0e}")
print(f"\n  worst linearity deviation {worst_lin:.3e}, tolerance {TOL_LINEAR:.0e}")
print()

# ------------------------------------------------------------------ 3f
print("3f. Seed independence: the derivation says Delta e does not depend on the")
print("    noise realisation. Bit identity is NOT expected here and is not claimed:")
print("    the biased and unbiased runs evaluate different intermediate values, so")
print("    their difference carries different rounding. Bit-identical replay of the")
print("    SAME case is a separate claim, checked in validation 1.")
print()
base = measured_delta("pos", 1.0, 11)
print(f"{'seed':>8} {'max |Delta e(seed) - Delta e(11)|':>36}")
print("-" * 46)
worst_seed = 0.0
for seed in (11, 12, 101, 2026, 999983):
    m = measured_delta("pos", 1.0, seed)
    err = float(np.max(np.abs(m - base)))
    worst_seed = max(worst_seed, err)
    print(f"{seed:>8} {err:36.3e}")
if worst_seed > TOL_SEED:
    failures.append(f"3f: seed dependence {worst_seed:.3e} > {TOL_SEED:.0e}")
print(f"\n  worst seed-to-seed deviation {worst_seed:.3e}, tolerance {TOL_SEED:.0e}")
print()

print("=" * 78)
if failures:
    print(f"RESULT: FAILED ({len(failures)} check(s))")
    for f in failures:
        print(f"  FAILED  {f}")
    sys.exit(1)
print("RESULT: PASS -- all checks in validation 3 passed")
print("=" * 78)
