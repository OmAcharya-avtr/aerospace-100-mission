"""Validation 6 — learned adaptive Q vs fixed hand-tuned Q vs classical Mehra IAE.

Run from the product root::

    PYTHONPATH=src python3 validation/v6_adaptive_q_benchmark.py

THE TEST.  A 1-D CWNA constant-velocity truth is generated with an
acceleration PSD ``q̃_true = q̃_nom · 10^u``, ``u ~ U(−1.5, 1.5)`` — i.e. the
真 process noise is anywhere from 1/32 to 32 times the value the filter was
hand-tuned for.  Three filters see exactly the same measurements:

  * **fixed**   — ``Q = Q_nom``.  This is the *strong* baseline: ``q̃_nom`` is
    the geometric centre of the test distribution, so it is the best possible
    single fixed choice, not a strawman.
  * **mehra**   — classical innovation-based adaptive estimation
    (Mehra 1970/1972; Mohamed & Schwarz 1999 Eq. (12)), ``Q̂ = K Ĉ Kᵀ``,
    projected onto the scalar knob ``λ = tr Q̂ / tr Q_nom``.
  * **learned** — bootstrap ensemble of gradient-boosted trees on innovation
    statistics, with an ensemble-spread confidence output.

All three tune the same single scalar, use the same window (40) and the same
re-estimation cadence (20 steps), and are strictly causal.

TRAIN/TEST SPLIT.  Training uses seeds ``S + i`` for ``i ∈ [0, N_TRAIN)``;
the held-out runs use seeds ``S + 100000 + i``.  The two sets are disjoint by
construction, and no held-out run contributes to fitting.

SCORING.  Position RMSE (the thing users look at) **and** the chi-squared
consistency statistics (the thing that actually matters downstream), plus the
accuracy with which each adaptive method recovers ``u`` itself.

WHOEVER WINS, WINS.  The result printed below is the measured result.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from navbench import (  # noqa: E402
    LearnedAdaptiveQ,
    chi2_bounds,
    constant_velocity_cwna,
    generate_adaptive_dataset,
    nees,
    nis,
    run_adaptive_kf,
    simulate_linear_system,
)

SEED = 20260812
N_TRAIN_RUNS = 150
N_TEST_RUNS = 60
N_STEPS = 400
DT = 1.0
Q_NOM = 0.05  # m^2/s^3
SIGMA_Z = 3.0  # m
BURN_IN = 60
WINDOW = 40
CADENCE = 20


def main() -> int:
    t_start = time.perf_counter()
    print("=" * 78)
    print("v6 - adaptive process-noise tuning: learned vs fixed vs classical")
    print("=" * 78)
    print(
        f"  q_nom = {Q_NOM} m^2/s^3, sigma_z = {SIGMA_Z} m, dt = {DT} s, "
        f"{N_STEPS} steps/run, burn-in {BURN_IN}"
    )
    print(f"  true log10 scale u ~ U(-1.5, 1.5)  =>  q_true in [{Q_NOM / 10**1.5:.5f}, "
          f"{Q_NOM * 10**1.5:.5f}] m^2/s^3")
    print(f"  window {WINDOW} steps, re-estimation every {CADENCE} steps")

    t0 = time.perf_counter()
    x_train, y_train, run_idx = generate_adaptive_dataset(
        n_runs=N_TRAIN_RUNS, n_steps=N_STEPS, dt=DT, q_nominal_psd=Q_NOM,
        sigma_z=SIGMA_Z, window=WINDOW, stride=CADENCE, seed=SEED,
    )
    t_data = time.perf_counter() - t0
    print(f"\n  dataset: {x_train.shape[0]} windows from {N_TRAIN_RUNS} runs "
          f"({np.unique(run_idx).size} distinct), {x_train.shape[1]} features, "
          f"generated in {t_data:.2f} s")

    t0 = time.perf_counter()
    model = LearnedAdaptiveQ(n_members=5, random_state=SEED).fit(x_train, y_train)
    t_fit = time.perf_counter() - t0
    print(f"  model: 5-member bootstrap GradientBoostingRegressor ensemble, "
          f"fitted in {t_fit:.2f} s")

    f, q_nom = constant_velocity_cwna(DT, Q_NOM)
    h = np.array([[1.0, 0.0]])
    r = np.array([[SIGMA_Z**2]])
    p0 = np.diag([100.0, 10.0])

    tuners = ("fixed", "mehra", "learned")
    metrics: dict[str, dict[str, list[float]]] = {
        t: {"rmse_pos": [], "rmse_vel": [], "nees": [], "nis": [], "log_err": [],
            "conf": [], "scale_final": []}
        for t in tuners
    }
    u_values: list[float] = []
    conf_pairs: list[tuple[float, float]] = []

    t0 = time.perf_counter()
    for i in range(N_TEST_RUNS):
        rng = np.random.default_rng(SEED + 100000 + i)
        u = float(rng.uniform(-1.5, 1.5))
        u_values.append(u)
        _, q_true = constant_velocity_cwna(DT, Q_NOM * 10.0**u)
        truth, meas = simulate_linear_system(
            f, h, q_true, r, np.array([0.0, 1.0]), N_STEPS, rng
        )
        for tuner in tuners:
            res = run_adaptive_kf(
                f=f, h=h, q_nominal=q_nom, r=r, x0=np.zeros(2), p0=p0,
                measurements=meas, tuner=tuner,
                model=model if tuner == "learned" else None,
                window=WINDOW, update_every=CADENCE,
            )
            err = truth[BURN_IN:] - res.states[BURN_IN:]
            m = metrics[tuner]
            m["rmse_pos"].append(float(np.sqrt(np.mean(err[:, 0] ** 2))))
            m["rmse_vel"].append(float(np.sqrt(np.mean(err[:, 1] ** 2))))
            m["nees"].append(float(np.mean(nees(err, res.covariances[BURN_IN:]))))
            nv = nis(res.innovations[BURN_IN:], res.innovation_covs[BURN_IN:])
            m["nis"].append(float(np.nanmean(nv)))
            lam = float(res.scales[-1])
            m["scale_final"].append(lam)
            m["log_err"].append(abs(np.log10(lam) - u))
            c = float(np.nanmean(res.confidences)) if tuner == "learned" else float("nan")
            m["conf"].append(c)
            if tuner == "learned":
                conf_pairs.append((c, abs(np.log10(lam) - u)))
    t_test = time.perf_counter() - t0
    print(f"  {N_TEST_RUNS} held-out runs x 3 tuners evaluated in {t_test:.2f} s")

    lo_e, hi_e = chi2_bounds(2, N_TEST_RUNS)
    lo_i, hi_i = chi2_bounds(1, N_TEST_RUNS)
    print()
    print("-" * 78)
    print("RESULTS on the held-out runs (mean over runs)")
    print("-" * 78)
    hdr = (f"  {'tuner':<10}{'pos RMSE [m]':>14}{'vel RMSE [m/s]':>16}"
           f"{'ANEES (2)':>12}{'ANIS (1)':>11}{'|log10 lam - u|':>17}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    summary = {}
    for t in tuners:
        m = metrics[t]
        summary[t] = {
            "rmse_pos": float(np.mean(m["rmse_pos"])),
            "rmse_vel": float(np.mean(m["rmse_vel"])),
            "nees": float(np.mean(m["nees"])),
            "nis": float(np.mean(m["nis"])),
            "log_err": float(np.mean(m["log_err"])),
        }
        s = summary[t]
        le = "n/a (fixed)" if t == "fixed" else f"{s['log_err']:.4f}"
        print(f"  {t:<10}{s['rmse_pos']:>14.5f}{s['rmse_vel']:>16.6f}"
              f"{s['nees']:>12.4f}{s['nis']:>11.4f}{le:>17}")
    print()
    print(f"  ANEES 95 % acceptance band over {N_TEST_RUNS} runs (dof 2): "
          f"[{lo_e:.4f}, {hi_e:.4f}]")
    print(f"  ANIS  95 % acceptance band over {N_TEST_RUNS} runs (dof 1): "
          f"[{lo_i:.4f}, {hi_i:.4f}]")
    for t in tuners:
        s = summary[t]
        v_e = "inside" if lo_e <= s["nees"] <= hi_e else (
            "ABOVE (optimistic)" if s["nees"] > hi_e else "BELOW (pessimistic)")
        v_i = "inside" if lo_i <= s["nis"] <= hi_i else (
            "ABOVE (optimistic)" if s["nis"] > hi_i else "BELOW (pessimistic)")
        print(f"    {t:<10} NEES {v_e:<20} NIS {v_i}")

    # Paired comparison: how many runs does each method win?
    print()
    print("-" * 78)
    print("PAIRED comparison, run by run (position RMSE)")
    print("-" * 78)
    for a, b in (("learned", "fixed"), ("learned", "mehra"), ("mehra", "fixed")):
        wa = np.array(metrics[a]["rmse_pos"])
        wb = np.array(metrics[b]["rmse_pos"])
        wins = int(np.sum(wa < wb))
        diff = wa - wb
        se = float(np.std(diff, ddof=1) / np.sqrt(diff.size))
        print(f"  {a} beats {b} on {wins}/{N_TEST_RUNS} runs; "
              f"mean RMSE difference {float(np.mean(diff)):+.5f} m "
              f"+/- {1.96 * se:.5f} (95 % CI on the paired mean)")

    # Stratify by the size of the mis-specification.
    print()
    print("-" * 78)
    print("STRATIFIED by |u| (size of the Q mis-specification)")
    print("-" * 78)
    u_arr = np.array(u_values)
    for lo_u, hi_u, label in ((0.0, 0.5, "|u| <= 0.5  (within ~3x)"),
                              (0.5, 1.0, "0.5 < |u| <= 1.0 (3x-10x)"),
                              (1.0, 1.6, "|u| > 1.0   (>10x)")):
        sel = (np.abs(u_arr) > lo_u) & (np.abs(u_arr) <= hi_u)
        if not np.any(sel):
            continue
        print(f"  {label}  (n = {int(np.sum(sel))})")
        for t in tuners:
            rp = np.array(metrics[t]["rmse_pos"])[sel]
            nn = np.array(metrics[t]["nees"])[sel]
            print(f"    {t:<10} pos RMSE {float(np.mean(rp)):8.5f} m, "
                  f"mean NEES {float(np.mean(nn)):7.4f}")

    # Is each adaptive method actually ADAPTING, or just saturating?
    print()
    print("-" * 78)
    print("IS EACH METHOD ACTUALLY ADAPTING? (lambda in force at the last step)")
    print("-" * 78)
    u_arr_all = np.array(u_values)
    for t in ("mehra", "learned"):
        lam = np.array(metrics[t]["scale_final"])
        at_hi = float(np.mean(lam >= 63.99))
        at_lo = float(np.mean(lam <= (1.0 / 64.0) * 1.0001))
        corr = float(np.corrcoef(np.log10(lam), u_arr_all)[0, 1])
        print(f"  {t}:")
        print(f"    lambda  min {float(lam.min()):.4f}  median {float(np.median(lam)):.4f}  "
              f"max {float(lam.max()):.4f}")
        print(f"    fraction pinned at the UPPER clip (64)   : {at_hi:.4f}")
        print(f"    fraction pinned at the LOWER clip (1/64) : {at_lo:.4f}")
        print(f"    correlation( log10 lambda , true u )     : {corr:+.4f}")
    print()
    print("  READ THIS BEFORE BELIEVING THE CONSISTENCY COLUMN. A method pinned at a")
    print("  clip is not adapting: it is applying a constant inflation. Its NEES then")
    print("  reflects that constant inflation, not any inference about the true Q.")

    # Confidence output behaviour.
    print()
    print("-" * 78)
    print("CONFIDENCE OUTPUT of the learned tuner")
    print("-" * 78)
    conf = np.array([c for c, _ in conf_pairs])
    lerr = np.array([e for _, e in conf_pairs])
    corr = float(np.corrcoef(conf, lerr)[0, 1])
    hi_conf = lerr[conf >= np.median(conf)]
    lo_conf = lerr[conf < np.median(conf)]
    print(f"  mean confidence exp(-std) over held-out runs : {float(np.mean(conf)):.4f}")
    print(f"  range                                        : "
          f"[{float(np.min(conf)):.4f}, {float(np.max(conf)):.4f}]")
    print(f"  correlation(confidence, |log10 lambda - u|)  : {corr:+.4f}")
    print(f"  mean |log10 lam - u| in the high-confidence half : {float(np.mean(hi_conf)):.4f}")
    print(f"  mean |log10 lam - u| in the low-confidence half  : {float(np.mean(lo_conf)):.4f}")
    print("  A useful confidence output should be NEGATIVELY correlated with the error")
    print("  (more confident -> smaller error). The measured sign and magnitude are")
    print("  reported above and repeated verbatim in MODEL_CARD.md.")

    # Verdict.
    print()
    print("=" * 78)
    print("VERDICT (stated as measured, not as hoped)")
    print("=" * 78)
    best_rmse = min(tuners, key=lambda t: summary[t]["rmse_pos"])
    nees_dist = {t: abs(summary[t]["nees"] - 2.0) for t in tuners}
    best_nees = min(tuners, key=lambda t: nees_dist[t])
    print(f"  lowest mean position RMSE : {best_rmse}  "
          f"({summary[best_rmse]['rmse_pos']:.5f} m)")
    print(f"  NEES closest to dof = 2   : {best_nees}  "
          f"({summary[best_nees]['nees']:.4f})")
    if "learned" in (best_rmse, best_nees):
        print("  The learned tuner wins at least one of the two criteria.")
    else:
        print("  THE LEARNED TUNER WINS NEITHER CRITERION. The classical/fixed baseline")
        print("  is better on this benchmark and that is the reported result.")
    if best_rmse != best_nees:
        print("  NOTE: the RMSE winner and the consistency winner are NOT the same method.")
        print("  Reporting only RMSE would have hidden that.")
    lam_m = np.array(metrics["mehra"]["scale_final"])
    if float(np.mean(lam_m >= 63.99)) > 0.9:
        print()
        print("  IMPORTANT CAVEAT ON THE CONSISTENCY WINNER: the Mehra scheme is pinned at")
        print("  its upper clip on essentially every held-out run, and its correlation with")
        print("  the true scale is zero. It is therefore NOT adapting - it is applying a")
        print("  constant 64x inflation of Q. That makes the filter pessimistic (ANEES below")
        print("  the lower bound), which happens to land nearer dof = 2 than the optimistic")
        print("  alternatives. Calling it 'the consistency winner' without this sentence")
        print("  would be misleading, so the sentence is printed here and repeated in")
        print("  MODEL_CARD.md and README.md.")
    print(f"\n  total wall time: {time.perf_counter() - t_start:.1f} s")
    # This script reports; it does not gate. It exits 0 whichever method wins,
    # because an honest negative result is a valid outcome, not a test failure.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
