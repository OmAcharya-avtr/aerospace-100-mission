"""V2 — NEES and NIS inside their chi-squared bounds, and provably outside them.

Run from ``products/P012/``::

    PYTHONPATH=src python validation/v2_consistency.py

What is checked
---------------
1. A **correctly specified** filter: NEES and NIS averaged over independent
   Monte Carlo runs must sit inside their chi-squared acceptance regions, and
   the fraction of time steps inside the per-step region must be close to the
   nominal 95 %.
2. **Mis-specified** filters: the true process noise is scaled by a factor
   ``s_true`` while the filter keeps the nominal ``Q``. The filter must leave
   the bounds, in the predicted direction — optimistic (ANEES > 1) when the
   truth is noisier than assumed, conservative (ANEES < 1) when it is quieter.
   The smallest mis-specification that the test detects at the Monte Carlo size
   used is reported, so the *power* of the test is a measured quantity.
3. A mis-specified **measurement** noise, which NIS detects without truth.
4. **Whiteness** of the innovations: for the correct filter the normalised
   autocorrelation must sit inside ``±1.96/√K``; for a filter with far too
   small a ``Q`` the innovations become correlated and leave the band.

Reference: Bar-Shalom, Rong Li & Kirubarajan (2001), *Estimation with
Applications to Tracking and Navigation*, Wiley, §5.4.
"""

from __future__ import annotations

import numpy as np

from navbench.bench import run_linear_mc
from navbench.consistency import innovation_autocorrelation, whiteness_band
from navbench.kf import KalmanFilter
from navbench.models import ConstantVelocity

MODEL = ConstantVelocity(dt=1.0, q_psd=0.1, sigma_pos=5.0, dim=2)
N_RUNS = 200
N_STEPS = 160
BURN_IN = 20
SEED = 4242


def _rule(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def correct_filter() -> bool:
    """Correctly specified filter: expect both tests to pass."""
    res = run_linear_mc(
        MODEL, n_runs=N_RUNS, n_steps=N_STEPS, seed=SEED, burn_in=BURN_IN,
        q_true_scale=1.0, label="correct",
    )
    print(f"Monte Carlo: M = {res.n_runs} runs x K = {res.n_steps} scored steps "
          f"(burn-in {BURN_IN}), state dim n = {MODEL.n}, measurement dim m = {MODEL.m}")
    print(f"wall clock: {res.seconds:.2f} s")
    print("\n" + res.nees_report.summary())
    print(res.nis_report.summary())
    print(f"\nposition RMSE = {res.rmse_position:.4f} m")
    ok = res.nees_report.passed and res.nis_report.passed
    print(f"RESULT: {'PASS' if ok else 'FAILED'}")
    return ok


def misspecified_sweep() -> tuple[bool, list[tuple[float, float, bool]]]:
    """Sweep the truth/filter process-noise mismatch and record when the test breaks."""
    rows: list[tuple[float, float, bool]] = []
    print(f"{'s_true':>9s} {'ANEES':>8s} {'NEES 95% CI':>24s} {'in?':>5s} "
          f"{'ANIS':>8s} {'NIS 95% CI':>24s} {'in?':>5s} {'RMSE [m]':>10s}")
    for s_true in (0.05, 0.2, 0.5, 0.8, 1.0, 1.25, 2.0, 5.0, 20.0, 100.0):
        res = run_linear_mc(
            MODEL, n_runs=N_RUNS, n_steps=N_STEPS, seed=SEED, burn_in=BURN_IN,
            q_true_scale=s_true, label=f"s={s_true:g}",
        )
        nr, ir = res.nees_report, res.nis_report
        rows.append((s_true, res.anees, nr.passed))
        print(
            f"{s_true:9.3g} {res.anees:8.4f} "
            f"[{nr.mean_ci[0]:9.4f},{nr.mean_ci[1]:9.4f}] {'yes' if nr.passed else 'NO':>5s} "
            f"{res.anis:8.4f} "
            f"[{ir.mean_ci[0]:9.4f},{ir.mean_ci[1]:9.4f}] {'yes' if ir.passed else 'NO':>5s} "
            f"{res.rmse_position:10.4f}"
        )
    detected = [r for r in rows if r[0] != 1.0 and not r[2]]
    correct_ok = all(r[2] for r in rows if r[0] == 1.0)
    # Direction check: optimistic above 1, conservative below 1.
    direction_ok = all(
        (r[1] > 1.0) if r[0] > 1.0 else (r[1] < 1.0) for r in rows if r[0] != 1.0
    )
    nearest = min((abs(np.log10(r[0])), r[0]) for r in detected)[1] if detected else float("nan")
    print(f"\ncorrectly specified case inside bounds : {'yes' if correct_ok else 'NO'}")
    print(f"mis-specified cases rejected            : {len(detected)} of "
          f"{len([r for r in rows if r[0] != 1.0])}")
    print(f"ANEES moves in the predicted direction  : {'yes' if direction_ok else 'NO'}")
    print(f"smallest |log10 s| rejected at M={N_RUNS}   : s_true = {nearest:g}")
    ok = correct_ok and direction_ok and len(detected) >= 7
    print(f"RESULT: {'PASS' if ok else 'FAILED'}")
    return ok, rows


def misspecified_measurement_noise() -> bool:
    """Filter told the wrong R: NIS must detect it without any truth data."""
    print(f"{'R_filter/R_true':>16s} {'ANIS':>9s} {'NIS 95% CI':>24s} {'detected?':>10s}")
    ok = True
    for ratio in (0.25, 1.0, 4.0):
        model_filter = ConstantVelocity(
            dt=MODEL.dt, q_psd=MODEL.q_psd,
            sigma_pos=MODEL.sigma_pos * np.sqrt(ratio), dim=MODEL.dim,
        )
        nis = np.zeros((N_RUNS, N_STEPS - BURN_IN))
        for i in range(N_RUNS):
            rng = np.random.default_rng(SEED + 7919 * i)
            x0 = np.array([0.0, 10.0, 0.0, -5.0])
            _, zs = MODEL.simulate(x0, N_STEPS, rng, q_true_scale=1.0)
            p0 = np.diag([100.0, 25.0] * 2)
            kf = KalmanFilter(
                f=MODEL.f(), q=MODEL.q(1.0), h=MODEL.h(), r=model_filter.r(),
                x=x0 + np.linalg.cholesky(p0) @ rng.standard_normal(4), p=p0.copy(),
            )
            for k in range(N_STEPS):
                kf.predict()
                info = kf.update(zs[k])
                if k >= BURN_IN:
                    nis[i, k - BURN_IN] = info.nis
        from navbench.consistency import assess

        rep = assess(nis, dof=MODEL.m, label=f"NIS[R x {ratio:g}]")
        detected = not rep.passed
        print(f"{ratio:16.3g} {rep.normalised_mean:9.4f} "
              f"[{rep.mean_ci[0]:9.4f},{rep.mean_ci[1]:9.4f}] "
              f"{('DETECTED' if detected else 'inside'):>10s}")
        ok = ok and (detected if ratio != 1.0 else not detected)
    print(f"RESULT: {'PASS' if ok else 'FAILED'}")
    return ok


def whiteness() -> bool:
    """Innovation whiteness for a correct filter and for a badly under-tuned Q."""
    band = None
    ok = True
    print(f"{'case':>26s} {'max|rho(1..10)|':>17s} {'band ±':>10s} {'verdict':>10s}")
    for label, q_scale_filter, s_true in (
        ("correct Q", 1.0, 1.0),
        ("Q 100x too small", 1.0, 100.0),
    ):
        rng = np.random.default_rng(SEED + 31)
        x0 = np.array([0.0, 10.0, 0.0, -5.0])
        n_long = 4000
        _, zs = MODEL.simulate(x0, n_long, rng, q_true_scale=s_true)
        kf = KalmanFilter(
            f=MODEL.f(), q=MODEL.q(q_scale_filter), h=MODEL.h(), r=MODEL.r(),
            x=x0.copy(), p=np.diag([100.0, 25.0] * 2),
        )
        nus = []
        for k in range(n_long):
            kf.predict()
            info = kf.update(zs[k])
            if k >= 200:
                nus.append(info.innovation)
        nus_a = np.array(nus)
        rho = innovation_autocorrelation(nus_a, max_lag=10)
        band = whiteness_band(nus_a.shape[0])
        worst = float(np.max(np.abs(rho)))
        white = worst <= band
        expect_white = s_true == 1.0
        print(f"{label:>26s} {worst:17.5f} {band:10.5f} "
              f"{('white' if white else 'CORRELATED'):>10s}")
        print(f"{'':>26s} rho = {np.array2string(rho, precision=4)}")
        ok = ok and (white == expect_white)
    print(f"RESULT: {'PASS' if ok else 'FAILED'}")
    return ok


def main() -> int:
    """Run V2 and return a process exit code."""
    np.set_printoptions(linewidth=140)
    _rule("V2a — correctly specified filter: NEES and NIS inside the chi-squared bounds")
    ok_a = correct_filter()
    _rule("V2b — process-noise mis-specification sweep: leaving the bounds")
    ok_b, _ = misspecified_sweep()
    _rule("V2c — measurement-noise mis-specification detected by NIS alone")
    ok_c = misspecified_measurement_noise()
    _rule("V2d — innovation whiteness")
    ok_d = whiteness()
    _rule("V2 SUMMARY")
    for name, ok in (
        ("V2a correct filter inside bounds", ok_a),
        ("V2b mis-specified filter leaves bounds", ok_b),
        ("V2c NIS detects wrong R", ok_c),
        ("V2d whiteness", ok_d),
    ):
        print(f"{name:<44s}: {'PASS' if ok else 'FAILED'}")
    return 0 if all((ok_a, ok_b, ok_c, ok_d)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
