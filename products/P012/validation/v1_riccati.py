"""V1 — the linear filter reproduces the analytic steady-state Riccati solution.

Run from ``products/P012/``::

    PYTHONPATH=src python validation/v1_riccati.py

What is checked
---------------
For a linear time-invariant, observable, controllable-from-noise system the
Kalman filter covariance converges to the unique stabilising solution of the
discrete algebraic Riccati equation, independently of the initial covariance.
The DARE solution is computed by ``scipy.linalg.solve_discrete_are`` (an
independent implementation based on a generalised Schur decomposition), so this
is a genuine cross-check of the recursion against an analytic reference and not
a self-consistency test.

Reference: Bar-Shalom, Rong Li & Kirubarajan (2001), *Estimation with
Applications to Tracking and Navigation*, Wiley, §5.2.6; Anderson & Moore
(1979), *Optimal Filtering*, Prentice-Hall, Ch. 4.

Additional checks
-----------------
* Convergence from three wildly different initial covariances to the same
  fixed point (uniqueness of the stabilising solution).
* Residual of the Riccati equation itself, evaluated at the returned solution.
* The scalar case, where the fixed point has a closed form that can be checked
  by hand.
"""

from __future__ import annotations

import numpy as np

from navbench.kf import KalmanFilter, steady_state_riccati
from navbench.models import ConstantVelocity

TOL_ABS = 1e-8


def _rule(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def scalar_closed_form() -> bool:
    r"""Scalar case: hand-checkable fixed point.

    For ``F = f``, ``H = 1`` the posterior is ``p⁺ = p⁻R/(p⁻+R)`` and the prior
    fixed point satisfies ``p = f² p R/(p + R) + q``. Multiplying by ``(p+R)``:

        p² + pR = f² p R + q p + q R
        p² + p(R − f²R − q) − qR = 0

    With ``f = 0.9``, ``q = 0.5``, ``R = 2.0`` the coefficient is
    ``2 − 0.81·2 − 0.5 = −0.12``, so

        p² − 0.12 p − 1.0 = 0
        p = (0.12 + sqrt(0.0144 + 4))/2 = 1.061798...

    and the posterior is ``pR/(p+R)``.
    """
    f, q, r = 0.9, 0.5, 2.0
    coeff = r - f * f * r - q
    p_hand = (-coeff + np.sqrt(coeff ** 2 + 4.0 * q * r)) / 2.0
    post_hand = p_hand * r / (p_hand + r)
    p_prior, p_post, k = steady_state_riccati([[f]], [[q]], [[1.0]], [[r]])
    err = abs(float(p_prior[0, 0]) - p_hand)
    print(f"hand-derived positive root p        = {p_hand:.15f}")
    print(f"steady_state_riccati prior variance = {float(p_prior[0, 0]):.15f}")
    print(f"|difference|                        = {err:.3e}   (tolerance {TOL_ABS:.0e})")
    print(f"hand-derived posterior variance     = {post_hand:.15f}")
    print(f"steady_state_riccati posterior      = {float(p_post[0, 0]):.15f}")
    err_post = abs(float(p_post[0, 0]) - post_hand)
    print(f"|difference|                        = {err_post:.3e}   (tolerance {TOL_ABS:.0e})")
    print(f"steady-state gain K                 = {float(k[0, 0]):.15f}")
    ok = err < TOL_ABS and err_post < TOL_ABS
    print(f"RESULT: {'PASS' if ok else 'FAILED'}")
    return ok


def cv_convergence() -> bool:
    """2-D constant-velocity model: filter recursion vs the DARE solution."""
    model = ConstantVelocity(dt=1.0, q_psd=0.1, sigma_pos=5.0, dim=2)
    f, q, h, r = model.f(), model.q(1.0), model.h(), model.r()
    p_prior, p_post, k_ss = steady_state_riccati(f, q, h, r)

    residual = f @ (
        p_prior - p_prior @ h.T @ np.linalg.solve(h @ p_prior @ h.T + r, h @ p_prior)
    ) @ f.T + q - p_prior
    res_norm = float(np.max(np.abs(residual)))
    print("analytic steady-state PRIOR covariance P^-:")
    print(np.array2string(p_prior, precision=9, suppress_small=False))
    print("\nanalytic steady-state POSTERIOR covariance P^+:")
    print(np.array2string(p_post, precision=9))
    print("\nanalytic steady-state gain K:")
    print(np.array2string(k_ss, precision=9))
    print(f"\nmax |Riccati residual| at the returned solution = {res_norm:.3e}")

    ok = res_norm < 1e-9
    print("\nConvergence of the recursion from three different initial covariances:")
    print(f"{'P0':>28s} {'steps':>7s} {'max|P_f - P_analytic|':>24s} {'max|K_f - K_ss|':>18s}")
    for label, p0 in (
        ("1e4 · I", 1.0e4 * np.eye(4)),
        ("1e-2 · I", 1.0e-2 * np.eye(4)),
        ("diag(1e6, 1e-3, 1e6, 1e-3)", np.diag([1e6, 1e-3, 1e6, 1e-3])),
    ):
        kf = KalmanFilter(f=f, q=q, h=h, r=r, x=np.zeros(4), p=p0)
        n_steps = 400
        for _ in range(n_steps):
            kf.predict()
            info = kf.update(np.zeros(2))
        dp = float(np.max(np.abs(kf.p - p_post)))
        dk = float(np.max(np.abs(info.gain - k_ss)))
        print(f"{label:>28s} {n_steps:7d} {dp:24.6e} {dk:18.6e}")
        ok = ok and dp < 1e-9 and dk < 1e-9
    print(f"\nRESULT: {'PASS' if ok else 'FAILED'}  (tolerance 1e-9 on both P and K)")
    return ok


def main() -> int:
    """Run V1 and return a process exit code."""
    np.set_printoptions(linewidth=110)
    _rule("V1a — scalar steady state against a hand-derived closed form")
    ok_a = scalar_closed_form()
    _rule("V1b — 2-D constant-velocity model: recursion vs scipy.solve_discrete_are")
    ok_b = cv_convergence()
    _rule("V1 SUMMARY")
    print(f"V1a scalar closed form : {'PASS' if ok_a else 'FAILED'}")
    print(f"V1b CV DARE agreement  : {'PASS' if ok_b else 'FAILED'}")
    return 0 if (ok_a and ok_b) else 1


if __name__ == "__main__":
    raise SystemExit(main())
