"""Validation 2 — the RTS smoother beats the forward filter on the same data.

Run from the product root::

    PYTHONPATH=src python validation/smoother_rms.py

Scenario: 1-D constant-velocity target, continuous white-noise
acceleration (Bar-Shalom, Rong Li & Kirubarajan 2001, Ch. 6), position-only
measurements. Truth, process noise and measurement noise are generated from
a seeded numpy Generator, so the numbers below are reproducible exactly.

Checks
------
1. On one fixed seed, RMS position and velocity error of the smoothed
   estimate is lower than that of the filtered estimate.
2. Over a Monte Carlo ensemble, the fraction of seeds where the smoother
   wins is reported (theory says smoothing reduces the *expected* error,
   not necessarily every realisation).
3. The theoretical guarantee P_{k|T} <= P^+_k is checked directly by
   verifying that P^+_k - P_{k|T} is positive semi-definite at every step.
4. Filter consistency is reported via the mean NIS against its
   chi-squared expectation (m degrees of freedom).
"""

from __future__ import annotations

import numpy as np

from estimkit import KalmanFilter, constant_velocity_cwna, min_eigenvalue, rts_smooth

DT = 1.0
Q_PSD = 0.01  # m^2/s^3
R_VAR = 4.0  # m^2  (sigma = 2 m)
STEPS = 300
SEED = 2026
MC_SEEDS = 300


def simulate(rng: np.random.Generator, steps: int = STEPS) -> tuple[np.ndarray, np.ndarray]:
    """Return (truth [steps, 2], measurements [steps, 1])."""
    f, q = constant_velocity_cwna(DT, Q_PSD)
    chol = np.linalg.cholesky(q)
    truth = np.empty((steps, 2))
    x = np.array([0.0, 10.0])
    for k in range(steps):
        x = f @ x + chol @ rng.standard_normal(2)
        truth[k] = x
    z = truth[:, 0:1] + np.sqrt(R_VAR) * rng.standard_normal((steps, 1))
    return truth, z


def run_once(seed: int) -> tuple[float, float, float, float, float, float]:
    rng = np.random.default_rng(seed)
    truth, z = simulate(rng)
    f, q = constant_velocity_cwna(DT, Q_PSD)
    h = np.array([[1.0, 0.0]])
    r = np.array([[R_VAR]])
    kf = KalmanFilter(f, h, q, r)
    res = kf.filter(np.array([0.0, 0.0]), np.diag([100.0, 100.0]), z)
    sm = rts_smooth(res)

    def rms(est: np.ndarray, idx: int) -> float:
        return float(np.sqrt(np.mean((est[:, idx] - truth[:, idx]) ** 2)))

    worst_pd = min(min_eigenvalue(res.p_post[k] - sm.p[k]) for k in range(len(res)))
    return (
        rms(res.x_post, 0),
        rms(sm.x, 0),
        rms(res.x_post, 1),
        rms(sm.x, 1),
        float(np.mean(res.nis)),
        worst_pd,
    )


def main() -> int:
    print("=" * 72)
    print("RTS smoother vs forward Kalman filter - constant-velocity tracking")
    print("=" * 72)
    print(f"dt = {DT} s, acceleration PSD = {Q_PSD} m^2/s^3, "
          f"measurement sigma = {np.sqrt(R_VAR)} m, {STEPS} steps")
    print(f"x0 = [0, 0], P0 = diag(100, 100), seed = {SEED}")

    rp_f, rp_s, rv_f, rv_s, mean_nis, worst_pd = run_once(SEED)
    print()
    print("Single seed (seed = %d):" % SEED)
    print(f"  RMS position  filter   : {rp_f:.6f} m")
    print(f"  RMS position  smoother : {rp_s:.6f} m"
          f"   ({100.0 * (1 - rp_s / rp_f):.2f} % reduction)")
    print(f"  RMS velocity  filter   : {rv_f:.6f} m/s")
    print(f"  RMS velocity  smoother : {rv_s:.6f} m/s"
          f"   ({100.0 * (1 - rv_s / rv_f):.2f} % reduction)")
    pos_ok = rp_s < rp_f
    vel_ok = rv_s < rv_f
    print(f"  verdict position       : {'PASS' if pos_ok else 'FAIL'}")
    print(f"  verdict velocity       : {'PASS' if vel_ok else 'FAIL'}")

    print()
    print("Covariance ordering P^+_k - P_{k|T} >= 0 at every step:")
    print(f"  worst minimum eigenvalue : {worst_pd:.3e}  "
          f"({'PASS' if worst_pd >= -1e-12 else 'FAIL'}, tol -1e-12)")
    pd_ok = worst_pd >= -1e-12

    print()
    print(f"Filter consistency: mean NIS = {mean_nis:.4f} for m = 1 dof "
          "(expectation 1.0)")

    print()
    print(f"Monte Carlo over {MC_SEEDS} seeds (0 ... {MC_SEEDS - 1}):")
    wins_p = 0
    wins_v = 0
    fp = np.empty(MC_SEEDS)
    sp = np.empty(MC_SEEDS)
    fv = np.empty(MC_SEEDS)
    sv = np.empty(MC_SEEDS)
    for i in range(MC_SEEDS):
        a, b, c, d, _, _ = run_once(i)
        fp[i], sp[i], fv[i], sv[i] = a, b, c, d
        wins_p += int(b < a)
        wins_v += int(d < c)
    print(f"  mean RMS position  filter / smoother : {fp.mean():.6f} / {sp.mean():.6f} m")
    print(f"  mean RMS velocity  filter / smoother : {fv.mean():.6f} / {sv.mean():.6f} m/s")
    print(f"  smoother wins on position : {wins_p}/{MC_SEEDS}")
    print(f"  smoother wins on velocity : {wins_v}/{MC_SEEDS}")
    mc_ok = sp.mean() < fp.mean() and sv.mean() < fv.mean()
    print(f"  verdict (ensemble means)  : {'PASS' if mc_ok else 'FAIL'}")

    ok = pos_ok and vel_ok and pd_ok and mc_ok
    print()
    print("=" * 72)
    print(f"OVERALL: {'PASS' if ok else 'FAIL'}")
    print("=" * 72)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
