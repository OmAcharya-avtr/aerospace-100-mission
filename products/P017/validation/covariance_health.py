"""Validation 4 — Joseph form preserves symmetry and positive definiteness.

Run from the product root::

    PYTHONPATH=src python validation/covariance_health.py

Check 1 (long run). A 4-state constant-velocity filter (position and
velocity in two axes) is run for 200 000 update steps with seeded
measurements. After every step the covariance is checked for exact
symmetry (max |P - P^T|) and for positive definiteness (minimum
eigenvalue). Both extremes over the whole run are reported.

Check 2 (ill-conditioned, reduced precision). The classic stress case for
the covariance update: two nearly parallel measurement rows,
H = [[1, 1], [1, 1 + delta]] with delta = 1e-3, and very small measurement
noise, so the information carried by the second row is of relative size
delta^2 = 1e-6. In float32 (eps = 1.19e-7) that information sits at the
edge of representable precision. The Joseph form
P = (I - KH)P(I - KH)^T + K R K^T is compared against the short form
P = (I - KH)P for the same gain, in float32, over repeated updates.

Check 3 (sub-optimal gain, exact arithmetic). The Joseph form is valid for
*any* gain; the short form is valid only for the optimal Kalman gain. An
over-relaxed gain K = 1.5 K_opt -- the sort of thing produced by gain
scheduling, fixed-gain (alpha-beta) implementations, coefficient
quantisation, or a deliberately detuned filter -- is applied to both forms
in double precision.

Together these are the numerical argument for the Joseph form made
concrete. See Bierman, *Factorization Methods for Discrete Sequential
Estimation*, Academic Press 1977, for why even the Joseph form eventually
loses and a factorised (UD / square-root) filter is required.

All results are reported as measured, whichever way they come out.
"""

from __future__ import annotations

import numpy as np

from estimkit import KalmanFilter, covariance_health

STEPS = 200_000
SEED = 31337


def check_long_run() -> tuple[bool, dict[str, float]]:
    print("=" * 78)
    print("CHECK 1 - symmetry and positive definiteness over a long Joseph-form run")
    print("=" * 78)
    dt = 0.1
    # 4-state [x, vx, y, vy], continuous white-noise acceleration in each axis.
    f = np.array(
        [
            [1.0, dt, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, dt],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    q_psd = 0.02
    block = q_psd * np.array([[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]])
    q = np.zeros((4, 4))
    q[0:2, 0:2] = block
    q[2:4, 2:4] = block
    h = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    r = np.diag([4.0, 4.0])

    kf = KalmanFilter(f, h, q, r)
    rng = np.random.default_rng(SEED)
    x = np.array([0.0, 5.0, 0.0, -3.0])
    p = np.diag([1e4, 1e2, 1e4, 1e2])
    chol_q = np.linalg.cholesky(q)
    truth = np.array([0.0, 5.0, 0.0, -3.0])
    chol_r = np.linalg.cholesky(r)

    worst_asym = 0.0
    worst_min_eig = np.inf
    worst_cond = 0.0
    final_trace = 0.0
    for _ in range(STEPS):
        truth = f @ truth + chol_q @ rng.standard_normal(4)
        z = h @ truth + chol_r @ rng.standard_normal(2)
        x, p = kf.predict(x, p)
        res = kf.update(x, p, z)
        x, p = res.x, res.p
        health = covariance_health(p)
        worst_asym = max(worst_asym, health["asymmetry"])
        worst_min_eig = min(worst_min_eig, health["min_eig"])
        worst_cond = max(worst_cond, health["condition"])
        final_trace = health["trace"]

    print(f"steps                        : {STEPS}")
    print(f"max |P - P^T| over the run   : {worst_asym:.3e}  (exact symmetry => 0)")
    print(f"min eigenvalue over the run  : {worst_min_eig:.6e}  (must stay > 0)")
    print(f"max condition number         : {worst_cond:.4f}")
    print(f"final trace(P)               : {final_trace:.6f}")
    ok = worst_asym == 0.0 and worst_min_eig > 0.0
    print(f"verdict                      : {'PASS' if ok else 'FAIL'}")
    return ok, {
        "asymmetry": worst_asym,
        "min_eig": worst_min_eig,
        "condition": worst_cond,
        "trace": final_trace,
    }


DELTA = 1e-3
R_F32 = 1e-8
INFLATION = 1e-5
N_F32_UPDATES = 500


def _f32_updates(joseph: bool, n_updates: int = N_F32_UPDATES) -> tuple[float, float]:
    """Repeat an ill-conditioned update in float32; return (min eig, max asymmetry)."""
    dtype = np.float32
    h = np.array([[1.0, 1.0], [1.0, 1.0 + DELTA]], dtype=dtype)
    r = (np.eye(2) * R_F32).astype(dtype)
    p = np.eye(2, dtype=dtype)
    eye = np.eye(2, dtype=dtype)
    infl = (np.eye(2) * INFLATION).astype(dtype)
    min_eig = np.inf
    max_asym = 0.0
    for _ in range(n_updates):
        s = h @ p @ h.T + r
        k = (p @ h.T) @ np.linalg.inv(s)
        if joseph:
            a = eye - k @ h
            p = (a @ p @ a.T + k @ r @ k.T).astype(dtype)
        else:
            p = ((eye - k @ h) @ p).astype(dtype)
        eig = np.linalg.eigvalsh(p.astype(np.float64))
        min_eig = min(min_eig, float(eig[0]))
        max_asym = max(max_asym, float(np.max(np.abs(p - p.T))))
        # Re-inflate so the covariance does not simply run to zero, mimicking
        # a process-noise injection between updates.
        p = (p + infl).astype(dtype)
    return min_eig, max_asym


def check_ill_conditioned() -> tuple[bool, dict[str, float]]:
    print()
    print("=" * 78)
    print("CHECK 2 - Joseph vs short form, ill-conditioned H, float32 arithmetic")
    print("=" * 78)
    j_eig, j_asym = _f32_updates(joseph=True)
    s_eig, s_asym = _f32_updates(joseph=False)
    print(f"H = [[1, 1], [1, 1+{DELTA:g}]], R = {R_F32:g} I, float32 (eps = 1.19e-07),")
    print(f"{N_F32_UPDATES} updates with {INFLATION:g} I re-inflation between updates.")
    print()
    print(f"{'form':>14} | {'min eigenvalue':>16} | {'max |P - P^T|':>14}")
    print("-" * 52)
    print(f"{'Joseph':>14} | {j_eig:16.6e} | {j_asym:14.3e}")
    print(f"{'short (I-KH)P':>14} | {s_eig:16.6e} | {s_asym:14.3e}")
    print()
    joseph_pd = j_eig > 0.0
    print(f"Joseph form stayed positive definite : {joseph_pd}")
    print(f"Short form stayed positive definite  : {s_eig > 0.0}")
    print(f"verdict (Joseph stays PD)            : {'PASS' if joseph_pd else 'FAIL'}")
    return joseph_pd, {
        "joseph_min_eig": j_eig,
        "joseph_asym": j_asym,
        "short_min_eig": s_eig,
        "short_asym": s_asym,
    }


def check_suboptimal_gain() -> tuple[bool, dict[str, float]]:
    print()
    print("=" * 78)
    print("CHECK 3 - sub-optimal gain K = 1.5 K_opt, double precision")
    print("=" * 78)
    from estimkit import joseph_update, simple_update

    p = np.array([[1.0, 0.2], [0.2, 0.5]])
    h = np.array([[1.0, 0.0]])
    r = np.array([[0.01]])
    s = h @ p @ h.T + r
    k_opt = p @ h.T @ np.linalg.inv(s)
    factor = 1.5
    k = factor * k_opt

    p_j = joseph_update(p, k, h, r)
    p_s = simple_update(p, k, h)
    eig_j = float(np.linalg.eigvalsh(0.5 * (p_j + p_j.T))[0])
    eig_s = float(np.linalg.eigvalsh(0.5 * (p_s + p_s.T))[0])

    print(f"P = {p.tolist()}, H = {h.tolist()}, R = {r.tolist()}")
    print(f"optimal gain K_opt = {k_opt.ravel()}, applied gain = {k.ravel()} "
          f"({factor} x optimal)")
    print()
    print(f"{'form':>14} | {'min eigenvalue':>16} | {'max |P - P^T|':>14}")
    print("-" * 52)
    print(f"{'Joseph':>14} | {eig_j:16.6e} | {float(np.max(np.abs(p_j - p_j.T))):14.3e}")
    print(f"{'short (I-KH)P':>14} | {eig_s:16.6e} | {float(np.max(np.abs(p_s - p_s.T))):14.3e}")
    print()
    ok = eig_j > 0.0 and eig_s < 0.0
    print("Joseph stays positive definite for a gain the short form cannot handle:")
    print(f"  Joseph min eig > 0 : {eig_j > 0.0}")
    print(f"  short  min eig < 0 : {eig_s < 0.0}   (K H > I along the measured direction,")
    print("                        so (I - K H) P has a negative eigenvalue)")
    print(f"verdict                              : {'PASS' if ok else 'FAIL'}")
    return ok, {"joseph_min_eig": eig_j, "short_min_eig": eig_s}


def main() -> int:
    ok1, _ = check_long_run()
    ok2, _ = check_ill_conditioned()
    ok3, _ = check_suboptimal_gain()
    ok = ok1 and ok2 and ok3
    print()
    print("=" * 78)
    print(f"OVERALL: {'PASS' if ok else 'FAIL'}")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
