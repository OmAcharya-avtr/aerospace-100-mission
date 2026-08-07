"""Validation 3 — the UKF reduces to the linear KF on a linear-Gaussian system.

Run from the product root::

    PYTHONPATH=src python validation/ukf_vs_kf_linear.py

Claim under test. For an affine transition and affine measurement the
scaled unscented transform reproduces the exact mean and covariance (see
the estimkit.ukf module docstring), so the UKF recursion must coincide with
the linear Kalman filter up to floating-point round-off, for *any*
admissible (alpha, beta, kappa).

Method. One seeded measurement sequence from a 2-state constant-velocity
model is run through KalmanFilter and UnscentedKalmanFilter over a grid of
sigma-point parameters. The worst deviation over all 200 time steps is
reported for the filtered mean, filtered covariance, Kalman gain and
innovation covariance, and for the RTS-smoothed mean and covariance (which
additionally exercises the effective-transition construction used for the
unscented RTS smoother).

Tolerance and the round-off model (stated, not hidden). The scaled
transform places the sigma points at alpha*sqrt(n+kappa) standard
deviations from the mean and then divides the resulting spread by
2*alpha^2*(n+kappa). Cancellation error entering at the level
eps*|x| in the sigma points is therefore amplified by roughly
1/alpha^2 in the reconstructed moments. The expected *relative* agreement
is thus of order eps/alpha^2, which for alpha = 1e-3 is 2.2e-10 -- i.e.
small alpha genuinely costs significant digits. The pass criterion is
applied to the *relative* deviation (each quantity normalised by its own
peak magnitude over the run) with a tolerance of 1e-9; absolute deviations
and the predicted eps/alpha^2 bound are printed alongside.
"""

from __future__ import annotations

import numpy as np

from estimkit import (
    KalmanFilter,
    UnscentedKalmanFilter,
    constant_velocity_cwna,
    rts_smooth,
)

DT = 1.0
Q_PSD = 0.05
R_VAR = 9.0
STEPS = 200
SEED = 7
REL_TOL = 1e-9
EPS = float(np.finfo(float).eps)

PARAM_GRID = [
    (1.0, 2.0, 0.0),
    (1.0, 0.0, 1.0),
    (0.5, 2.0, 1.0),
    (1e-1, 2.0, 0.0),
    (1e-2, 2.0, 0.0),
    (1e-3, 2.0, 0.0),
]


def make_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Seeded truth and measurements for the linear-Gaussian test case."""
    rng = np.random.default_rng(SEED)
    f, q = constant_velocity_cwna(DT, Q_PSD)
    h = np.array([[1.0, 0.0]])
    r = np.array([[R_VAR]])
    chol = np.linalg.cholesky(q)
    x = np.array([0.0, 20.0])
    truth = np.empty((STEPS, 2))
    for k in range(STEPS):
        x = f @ x + chol @ rng.standard_normal(2)
        truth[k] = x
    z = truth[:, 0:1] + np.sqrt(R_VAR) * rng.standard_normal((STEPS, 1))
    return f, h, q, r, z


def main() -> int:
    f, h, q, r, z = make_data()
    x0 = np.array([0.0, 0.0])
    p0 = np.diag([100.0, 100.0])

    kf = KalmanFilter(f, h, q, r)
    ref = kf.filter(x0, p0, z)
    ref_sm = rts_smooth(ref)

    scales = {
        "x": float(np.max(np.abs(ref.x_post))),
        "P": float(np.max(np.abs(ref.p_post))),
        "K": float(np.max(np.abs(ref.gain))),
        "S": float(np.max(np.abs(ref.innovation_cov))),
        "xs": float(np.max(np.abs(ref_sm.x))),
        "Ps": float(np.max(np.abs(ref_sm.p))),
    }

    print("=" * 92)
    print("UKF -> KF reduction on a linear-Gaussian system")
    print("=" * 92)
    print(f"2-state constant velocity, dt = {DT} s, acceleration PSD = {Q_PSD} m^2/s^3,")
    print(f"measurement sigma = {np.sqrt(R_VAR)} m, {STEPS} steps, seed = {SEED}.")
    print(f"KF state at final step: {ref.x_post[-1]}")
    print("Normalisation scales (peak |value| over the run): "
          + ", ".join(f"{k} = {v:.4g}" for k, v in scales.items()))
    print(f"Relative tolerance: {REL_TOL:g};  machine eps = {EPS:.3e}")
    print()
    header = (
        f"{'alpha':>8} {'beta':>5} {'kappa':>6} | {'abs dx':>10} {'abs dP':>10} "
        f"{'abs dK':>10} {'abs dS':>10} {'abs dx_s':>10} {'abs dP_s':>10} | "
        f"{'worst rel':>10} {'eps/a^2':>10} | verdict"
    )
    print(header)
    print("-" * len(header))

    overall = True
    worst_rel_all = 0.0
    for alpha, beta, kappa in PARAM_GRID:
        ukf = UnscentedKalmanFilter(
            f=lambda x, _f=f: _f @ x,
            h=lambda x, _h=h: _h @ x,
            process_noise=q,
            measurement_noise=r,
            alpha=alpha,
            beta=beta,
            kappa=kappa,
        )
        res = ukf.filter(x0, p0, z)
        sm = rts_smooth(res)

        absdev = {
            "x": float(np.max(np.abs(res.x_post - ref.x_post))),
            "P": float(np.max(np.abs(res.p_post - ref.p_post))),
            "K": float(np.max(np.abs(res.gain - ref.gain))),
            "S": float(np.max(np.abs(res.innovation_cov - ref.innovation_cov))),
            "xs": float(np.max(np.abs(sm.x - ref_sm.x))),
            "Ps": float(np.max(np.abs(sm.p - ref_sm.p))),
        }
        worst_rel = max(absdev[k] / scales[k] for k in absdev)
        worst_rel_all = max(worst_rel_all, worst_rel)
        predicted = EPS / alpha**2
        ok = worst_rel < REL_TOL
        overall &= ok
        print(
            f"{alpha:8.3g} {beta:5.2g} {kappa:6.2g} | {absdev['x']:10.3e} "
            f"{absdev['P']:10.3e} {absdev['K']:10.3e} {absdev['S']:10.3e} "
            f"{absdev['xs']:10.3e} {absdev['Ps']:10.3e} | {worst_rel:10.3e} "
            f"{predicted:10.3e} | {'PASS' if ok else 'FAIL'}"
        )

    print()
    print(f"Worst relative deviation over the whole grid: {worst_rel_all:.3e} "
          f"(tolerance {REL_TOL:g})")
    print("Observation: the relative deviation tracks the predicted eps/alpha^2 "
          "round-off amplification of the scaled transform; the algebra is exact, "
          "the arithmetic is not.")
    print()
    print("=" * 92)
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
    print("=" * 92)
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
