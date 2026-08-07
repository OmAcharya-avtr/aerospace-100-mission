"""Validation 1 — steady-state Riccati solutions vs hand and published algebra.

Run from the product root::

    PYTHONPATH=src python validation/riccati_steady_state.py

Case A (hand-solved). Scalar random walk x_k = x_{k-1} + w, z = x + v,
Var(w) = q, Var(v) = r. The steady-state predicted variance solves
p = p r/(p + r) + q, i.e. p^2 - q p - q r = 0, so
p = (q + sqrt(q^2 + 4 q r))/2. With q = r = 1 this is the golden ratio.
Full arithmetic in validation/VALIDATION.md.

Case B (published closed form). 2-state constant-velocity model with
discrete white-noise acceleration. Kalata's tracking index gives the exact
steady-state alpha-beta gains:
  Lambda = sigma_a T^2 / sigma_v
  rho    = (4 + Lambda - sqrt(8 Lambda + Lambda^2)) / 4
  alpha  = 1 - rho^2
  beta   = 2(2 - alpha) - 4 sqrt(1 - alpha)
  K_inf  = [alpha, beta/T]^T
Reference: Kalata, P. R., "The Tracking Index: A Generalized Parameter for
alpha-beta and alpha-beta-gamma Target Trackers", IEEE Transactions on
Aerospace and Electronic Systems, Vol. AES-20, No. 2, 1984.

Case C (independent solver cross-check). The same 2-state DARE solved by
scipy.linalg.solve_discrete_are on the dual pair (F^T, H^T, Q, R). SciPy is
used here only, not by the library.
"""

from __future__ import annotations

import numpy as np

from estimkit import constant_velocity_dwna, random_walk, steady_state

FMT = "{:>28s} : {}"


def _line(label: str, value: object) -> None:
    print(FMT.format(label, value))


def case_a() -> bool:
    print("=" * 72)
    print("CASE A - scalar random walk, hand-solved algebraic Riccati equation")
    print("=" * 72)
    ok = True
    for q, r in ((1.0, 1.0), (0.25, 4.0), (2.0, 0.5)):
        f, h, qm, rm = random_walk(q, r)
        p_prior, p_post, gain, iters = steady_state(f, h, qm, rm, tol=1e-15)

        p_hand = 0.5 * (q + np.sqrt(q * q + 4.0 * q * r))
        k_hand = p_hand / (p_hand + r)
        p_post_hand = p_hand - q  # from P^- = P^+ + q

        d_p = abs(float(p_prior[0, 0]) - p_hand)
        d_k = abs(float(gain[0, 0]) - k_hand)
        d_pp = abs(float(p_post[0, 0]) - p_post_hand)
        passed = max(d_p, d_k, d_pp) < 1e-12
        ok &= passed
        print(f"\n  q = {q}, r = {r}   (converged in {iters} iterations)")
        _line("hand  P^-_inf", f"{p_hand:.15f}")
        _line("filter P^-_inf", f"{float(p_prior[0, 0]):.15f}   |diff| = {d_p:.3e}")
        _line("hand  K_inf", f"{k_hand:.15f}")
        _line("filter K_inf", f"{float(gain[0, 0]):.15f}   |diff| = {d_k:.3e}")
        _line("hand  P^+_inf", f"{p_post_hand:.15f}")
        _line("filter P^+_inf", f"{float(p_post[0, 0]):.15f}   |diff| = {d_pp:.3e}")
        _line("verdict (tol 1e-12)", "PASS" if passed else "FAIL")
    return ok


def case_b_c() -> bool:
    print()
    print("=" * 72)
    print("CASE B/C - 2-state constant velocity (DWNA): Kalata alpha-beta and DARE")
    print("=" * 72)
    ok = True
    for dt, sigma_a, sigma_v in ((1.0, 0.1, 2.0), (0.5, 1.0, 0.5), (2.0, 0.02, 10.0)):
        f, q = constant_velocity_dwna(dt, sigma_a)
        h = np.array([[1.0, 0.0]])
        r = np.array([[sigma_v**2]])
        p_prior, _p_post, gain, iters = steady_state(f, h, q, r, tol=1e-15)

        lam = sigma_a * dt**2 / sigma_v
        rho = (4.0 + lam - np.sqrt(8.0 * lam + lam * lam)) / 4.0
        alpha = 1.0 - rho**2
        beta = 2.0 * (2.0 - alpha) - 4.0 * np.sqrt(1.0 - alpha)
        k_hand = np.array([alpha, beta / dt])
        d_k = float(np.max(np.abs(gain.ravel() - k_hand)))

        try:
            from scipy.linalg import solve_discrete_are

            x_dare = solve_discrete_are(f.T, h.T, q, r)
            d_dare = float(np.max(np.abs(p_prior - x_dare)))
            dare_txt = f"{d_dare:.3e}"
            dare_ok = d_dare < 1e-10
        except ImportError:  # pragma: no cover - scipy is present in this env
            dare_txt = "scipy unavailable - skipped"
            dare_ok = True

        passed = d_k < 1e-12 and dare_ok
        ok &= passed
        print(f"\n  dt = {dt} s, sigma_a = {sigma_a} m/s^2, sigma_v = {sigma_v} m"
              f"   (converged in {iters} iterations)")
        _line("tracking index Lambda", f"{lam:.15f}")
        _line("Kalata K_inf = [a, b/T]", f"[{k_hand[0]:.15f}, {k_hand[1]:.15f}]")
        _line("filter K_inf", f"[{gain[0, 0]:.15f}, {gain[1, 0]:.15f}]")
        _line("max |diff| gain", f"{d_k:.3e}")
        _line("max |P^- - scipy DARE|", dare_txt)
        _line("verdict (tol 1e-12 / 1e-10)", "PASS" if passed else "FAIL")
    return ok


def main() -> int:
    np.set_printoptions(precision=15)
    ok_a = case_a()
    ok_bc = case_b_c()
    print()
    print("=" * 72)
    print(f"OVERALL: {'PASS' if (ok_a and ok_bc) else 'FAIL'}")
    print("=" * 72)
    return 0 if (ok_a and ok_bc) else 1


if __name__ == "__main__":
    raise SystemExit(main())
