"""Validation 1 — the filter reproduces the analytic steady-state Riccati solution.

Run from the product root::

    PYTHONPATH=src python3 validation/v1_riccati_steady_state.py

CASE A (hand-solved).  Scalar random walk ``x_k = x_{k-1} + w``, ``z = x + v``,
``Var(w) = q``, ``Var(v) = r``.  At the fixed point

    update : p⁺ = (1 − K) p,  K = p/(p + r)      predict: p = p⁺ + q

so ``p = p r/(p + r) + q`` → ``p² − q p − q r = 0`` →
``p = ½(q + sqrt(q² + 4 q r))``, ``K = p/(p + r)``, ``p⁺ = p − q``.
With ``q = r = 1`` this is the golden ratio φ = 1.618033988749895.

CASE B (published closed form).  Two-state constant-velocity model with
discrete white-noise acceleration.  Kalata's tracking index gives the exact
steady-state α-β gains:

    Λ = σ_a T²/σ_v,  ρ = (4 + Λ − sqrt(8Λ + Λ²))/4,
    α = 1 − ρ²,      β = 2(2 − α) − 4 sqrt(1 − α),   K_∞ = [α, β/T]ᵀ

Kalata, P. R. (1984), "The Tracking Index: A Generalized Parameter for α-β
and α-β-γ Target Trackers", *IEEE Transactions on Aerospace and Electronic
Systems* AES-20(2), 174-182.

CASE C (independent solver).  The same DARE solved by
``scipy.linalg.solve_discrete_are`` on the filtering dual ``(Fᵀ, Hᵀ, Q, R)``.
SciPy is used only here, never inside the library.

CASE D (dynamic convergence).  A running :class:`navbench.KalmanFilter` is
stepped 600 times on a CWNA model and its covariance/gain are compared with
the steady-state solution, confirming that the recursion the filter actually
executes converges to the algebra above.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from navbench import (  # noqa: E402
    KalmanFilter,
    constant_velocity_cwna,
    constant_velocity_dwna,
    random_walk,
    steady_state_riccati,
)

FMT = "{:>32s} : {}"


def _line(label: str, value: object) -> None:
    print(FMT.format(label, value))


def case_a() -> bool:
    print("=" * 78)
    print("CASE A - scalar random walk, hand-solved algebraic Riccati equation")
    print("=" * 78)
    ok = True
    for q, r in ((1.0, 1.0), (0.25, 4.0), (2.0, 0.5), (1e-3, 1e3)):
        f, h, qm, rm = random_walk(q, r)
        p_prior, p_post, gain, iters = steady_state_riccati(f, h, qm, rm, tol=1e-15)
        p_hand = 0.5 * (q + np.sqrt(q * q + 4.0 * q * r))
        k_hand = p_hand / (p_hand + r)
        p_post_hand = p_hand - q
        d_p = abs(float(p_prior[0, 0]) - p_hand)
        d_k = abs(float(gain[0, 0]) - k_hand)
        d_pp = abs(float(p_post[0, 0]) - p_post_hand)
        rel = max(d_p / p_hand, d_k / k_hand, d_pp / abs(p_post_hand))
        passed = rel < 1e-12
        ok &= passed
        print(f"\n  q = {q}, r = {r}   (converged in {iters} iterations)")
        _line("hand  P^-_inf", f"{p_hand:.15e}")
        _line("filter P^-_inf", f"{float(p_prior[0, 0]):.15e}   |diff| = {d_p:.3e}")
        _line("hand  K_inf", f"{k_hand:.15e}")
        _line("filter K_inf", f"{float(gain[0, 0]):.15e}   |diff| = {d_k:.3e}")
        _line("hand  P^+_inf", f"{p_post_hand:.15e}")
        _line("filter P^+_inf", f"{float(p_post[0, 0]):.15e}   |diff| = {d_pp:.3e}")
        _line("worst relative deviation", f"{rel:.3e}")
        _line("verdict (rel tol 1e-12)", "PASS" if passed else "FAIL")
    return ok


def case_b_c() -> bool:
    print()
    print("=" * 78)
    print("CASE B/C - 2-state constant velocity (DWNA): Kalata alpha-beta and SciPy DARE")
    print("=" * 78)
    ok = True
    for dt, sigma_a, sigma_v in ((1.0, 0.1, 2.0), (0.5, 1.0, 0.5), (2.0, 0.02, 10.0)):
        f, q = constant_velocity_dwna(dt, sigma_a)
        h = np.array([[1.0, 0.0]])
        r = np.array([[sigma_v**2]])
        p_prior, _p_post, gain, iters = steady_state_riccati(f, h, q, r, tol=1e-15)
        lam = sigma_a * dt**2 / sigma_v
        rho = (4.0 + lam - np.sqrt(8.0 * lam + lam * lam)) / 4.0
        alpha = 1.0 - rho**2
        beta = 2.0 * (2.0 - alpha) - 4.0 * np.sqrt(1.0 - alpha)
        k_hand = np.array([alpha, beta / dt])
        d_k = float(np.max(np.abs(gain.ravel() - k_hand)))

        from scipy.linalg import solve_discrete_are

        x_dare = solve_discrete_are(f.T, h.T, q, r)
        d_dare = float(np.max(np.abs(p_prior - x_dare)))
        passed = d_k < 1e-12 and d_dare < 1e-10
        ok &= passed
        print(
            f"\n  dt = {dt} s, sigma_a = {sigma_a} m/s^2, sigma_v = {sigma_v} m"
            f"   (converged in {iters} iterations)"
        )
        _line("tracking index Lambda", f"{lam:.15f}")
        _line("Kalata K_inf = [a, b/T]", f"[{k_hand[0]:.15f}, {k_hand[1]:.15f}]")
        _line("filter K_inf", f"[{gain[0, 0]:.15f}, {gain[1, 0]:.15f}]")
        _line("max |diff| gain", f"{d_k:.3e}")
        _line("max |P^- - scipy DARE|", f"{d_dare:.3e}")
        _line("verdict (tol 1e-12 / 1e-10)", "PASS" if passed else "FAIL")
    return ok


def case_d() -> bool:
    print()
    print("=" * 78)
    print("CASE D - a RUNNING KalmanFilter converges to the steady-state solution")
    print("=" * 78)
    ok = True
    for dt, q_psd, sigma_z in ((1.0, 0.05, 3.0), (0.1, 2.0, 0.5)):
        f, q = constant_velocity_cwna(dt, q_psd)
        h = np.array([[1.0, 0.0]])
        r = np.array([[sigma_z**2]])
        p_inf, p_post_inf, k_inf, _ = steady_state_riccati(f, h, q, r, tol=1e-15)
        kf = KalmanFilter(f, h, q, r, np.zeros(2), np.diag([1e6, 1e4]))
        rng = np.random.default_rng(4242)
        last = None
        for _ in range(600):
            kf.predict()
            last = kf.update(np.array([rng.standard_normal() * sigma_z]))
        d_p = float(np.max(np.abs(kf.p - p_post_inf)))
        d_k = float(np.max(np.abs(np.asarray(last["gain"]) - k_inf)))  # type: ignore[index]
        rel_p = d_p / float(np.max(np.abs(p_post_inf)))
        rel_k = d_k / float(np.max(np.abs(k_inf)))
        passed = max(rel_p, rel_k) < 1e-12
        ok &= passed
        print(f"\n  dt = {dt} s, q_psd = {q_psd} m^2/s^3, sigma_z = {sigma_z} m, 600 steps")
        _line("steady-state P^+_inf", np.array2string(p_post_inf, precision=12))
        _line("filter P^+ after 600 steps", np.array2string(kf.p, precision=12))
        _line("max |diff| (relative)", f"{d_p:.3e} ({rel_p:.3e})")
        _line("steady-state K_inf", np.array2string(k_inf.ravel(), precision=12))
        _line("filter K after 600 steps", np.array2string(np.asarray(last["gain"]).ravel(),  # type: ignore[index]
                                                          precision=12))
        _line("max |diff| gain (relative)", f"{d_k:.3e} ({rel_k:.3e})")
        _line("verdict (rel tol 1e-12)", "PASS" if passed else "FAIL")
    return ok


def main() -> int:
    np.set_printoptions(precision=15, suppress=False)
    results = [case_a(), case_b_c(), case_d()]
    print()
    print("=" * 78)
    print(f"OVERALL v1: {'PASS' if all(results) else 'FAIL'}")
    print("=" * 78)
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
