"""Level-1 validation: RK4 attitude propagation vs closed-form constant-ω solution.

For constant body angular velocity the kinematic equation q̇ = ½ q ⊗ [0, ω]
has the exact solution q(t) = q0 ⊗ exp_q(ω t) (Markley & Crassidis 2014,
Eq. 3.25). This script reports the maximum attitude angle error of the RK4
integrator against that analytic solution over a 60 s propagation for several
step sizes, plus the observed convergence order, and the norm drift with
renormalization disabled.

Run from products/P007/:  python validation/check_rk4_vs_analytic.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quatkit import (  # noqa: E402
    angle_between,
    closed_form_constant_omega,
    propagate,
)

OMEGA = np.array([0.10, 0.20, -0.15])  # rad/s, constant, |ω| ≈ 0.269 rad/s
T_END = 60.0  # s
Q0 = np.array([1.0, 0.0, 0.0, 0.0])


def max_err(dt: float) -> float:
    times = np.arange(0.0, T_END + 1e-9, dt)
    qs = propagate(Q0, lambda t: OMEGA, times)
    q_exact = closed_form_constant_omega(Q0, OMEGA, times)
    return float(np.max(angle_between(qs, q_exact)))


def main() -> int:
    print("RK4 vs analytic closed form, constant ω = [0.10, 0.20, -0.15] rad/s")
    print(f"|ω| = {np.linalg.norm(OMEGA):.4f} rad/s, t ∈ [0, {T_END:.0f}] s, q0 = identity")
    print("=" * 72)
    dts = [0.4, 0.2, 0.1, 0.05]
    errs = [max_err(dt) for dt in dts]
    for dt, err in zip(dts, errs):
        print(f"  dt = {dt:5.2f} s : max attitude angle error = {err:.3e} rad")
    orders = [np.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
    print(f"  observed convergence order (log2 error ratios): "
          f"{', '.join(f'{o:.2f}' for o in orders)} (theory: 4 for global RK4 error)")

    # Norm drift without renormalization (documents why we renormalize).
    times = np.arange(0.0, T_END + 1e-9, 0.05)
    qs_raw = propagate(Q0, lambda t: OMEGA, times, renormalize=False)
    drift = float(np.max(np.abs(np.linalg.norm(qs_raw, axis=1) - 1.0)))
    qs_norm = propagate(Q0, lambda t: OMEGA, times, renormalize=True)
    drift_n = float(np.max(np.abs(np.linalg.norm(qs_norm, axis=1) - 1.0)))
    print(f"  norm drift over 60 s, dt=0.05 s : {drift:.3e} (renormalize=False)")
    print(f"                                    {drift_n:.3e} (renormalize=True)")

    ok = errs[-1] < 1e-9 and drift_n < 1e-14 and all(o > 3.5 for o in orders)
    print()
    print(f"Criteria: err(dt=0.05) < 1e-9 rad, order > 3.5, renormalized drift < 1e-14 "
          f"-> {'ALL PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
