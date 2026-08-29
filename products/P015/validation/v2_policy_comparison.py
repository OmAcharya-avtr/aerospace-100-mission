"""V2 -- policy comparison on Scenario B with paired confidence intervals.

Run from ``products/P015/``::

    PYTHONPATH=src python validation/v2_policy_comparison.py

Protocol
--------
1. The learned outage predictor is fitted on TRAIN_SEEDS only.
2. Every free parameter of every policy -- the fixed threshold, the two
   hysteresis thresholds and the learned policy's probability threshold -- is
   selected on TUNE_SEEDS, which are disjoint from both TRAIN_SEEDS and
   TEST_SEEDS. The baselines get exactly the same tuning budget as the learned
   policy; nothing is selected on the test seeds.
3. All policies are then scored once on TEST_SEEDS. Because every policy sees
   the same traces, differences are analysed as **paired** per-trial
   differences, which is far more powerful than comparing marginal intervals.
4. A difference whose 95 % paired interval contains zero is reported as
   INDISTINGUISHABLE, not as a win for either policy.

The channel model is simulated, not measured (see ``DATASET_CARD.md``).
"""

from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, "src")

from linkswitch import (  # noqa: E402
    TEST_SEEDS,
    TRAIN_SEEDS,
    TUNE_SEEDS,
    AlwaysOpticalPolicy,
    AlwaysRfPolicy,
    ClairvoyantPolicy,
    FixedThresholdPolicy,
    HysteresisPolicy,
    OutagePredictor,
    evaluate_selection,
    make_features,
    make_labels,
    mean_ci,
    optimal_fixed_threshold_db,
    paired_diff_ci,
    scenario_b_operational,
    simulate_trace,
)

CONF = 0.95
MAX_TRAIN_ROWS = 25_000
HORIZON = 4
N_MEMBERS = 5

SC = scenario_b_operational()


def score(sel: np.ndarray, trace, guard: int | None = None):
    """Score a selection on one trace with the scenario's rates."""
    return evaluate_selection(
        sel,
        trace.optical_up,
        trace.rf_up,
        rate_optical_bps=SC.rate_optical_bps,
        rate_rf_bps=SC.rate_rf_bps,
        dt_s=SC.dt_s,
        switch_penalty_steps=SC.switch_penalty_steps if guard is None else guard,
    )


def mean_tput(sels, traces, guard: int | None = None) -> float:
    """Mean throughput [Mb/s] over a set of traces."""
    return float(np.mean([score(s, t, guard).throughput_bps for s, t in zip(sels, traces)]) / 1e6)


def banner(text: str) -> None:
    """Print a section banner."""
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


def main() -> int:
    """Run the V2 comparison; return 0 (this script reports, it does not assert)."""
    t0 = time.time()
    banner("Scenario B -- operational hybrid link (simulated, not measured)")
    o, r = SC.optical, SC.rf
    print(f"  optical  mean margin              : {o.mean_margin_db:.2f} dB")
    print(f"           sigma_I^2                : {o.sigma_i2:.3f}")
    print(f"           correlation time         : {o.correlation_time_s * 1e3:.3f} ms")
    print(f"           telemetry noise          : {o.telemetry_noise_db:.2f} dB rms")
    print(f"           obscuration prob / median: {o.obscuration_prob:.3f} / "
          f"{o.obscuration_median_db:.1f} dB, tau = {o.obscuration_time_s:.1f} s")
    print(f"  RF       clear-sky margin         : {r.clear_sky_margin_db:.2f} dB")
    print(f"           k, alpha (PLACEHOLDERS)  : {r.k_itu:.3f}, {r.alpha_itu:.3f}")
    print(f"           rain prob / median rate  : {r.rain_prob:.3f} / "
          f"{r.median_rain_rate_mm_per_h:.1f} mm/h, tau = {r.rain_correlation_time_s:.0f} s")
    print(f"  link     R_o / R_r                : {SC.rate_optical_bps / 1e9:.2f} Gb/s / "
          f"{SC.rate_rf_bps / 1e6:.0f} Mb/s   (q = {SC.rate_ratio:.3f})")
    print(f"           dt / trace length        : {SC.dt_s * 1e3:.1f} ms / "
          f"{SC.n_steps} samples = {SC.n_steps * SC.dt_s:.1f} s")
    print(f"           handover guard           : {SC.switch_penalty_steps} sample "
          f"= {SC.switch_penalty_steps * SC.dt_s * 1e3:.1f} ms")
    print(f"  seeds    train / tune / test      : {len(TRAIN_SEEDS)} / {len(TUNE_SEEDS)} / "
          f"{len(TEST_SEEDS)}")

    tune_traces = [simulate_trace(SC, s) for s in TUNE_SEEDS]
    test_traces = [simulate_trace(SC, s) for s in TEST_SEEDS]
    print(f"\n  measured channel statistics over the {len(TEST_SEEDS)} test traces:")
    print(f"    optical available fraction : "
          f"{np.mean([t.optical_up.mean() for t in test_traces]):.4f}")
    print(f"    RF available fraction      : "
          f"{np.mean([t.rf_up.mean() for t in test_traces]):.4f}")
    print(f"    raining fraction           : "
          f"{np.mean([(t.rain_rate_mm_per_h > 0).mean() for t in test_traces]):.4f}")

    # ------------------------------------------------------------ train
    banner("Learned predictor: training (TRAIN_SEEDS only)")
    t_fit = time.time()
    pred = OutagePredictor(
        horizon=HORIZON, n_members=N_MEMBERS, max_iter=60, max_leaf_nodes=15,
        max_bins=64, random_state=0,
    )
    x_tr, y_tr = pred.dataset([simulate_trace(SC, s) for s in TRAIN_SEEDS])
    pred.fit(x_tr, y_tr, max_rows=MAX_TRAIN_ROWS)
    fit_s = time.time() - t_fit
    print(f"  horizon H                : {HORIZON} samples = {HORIZON * SC.dt_s * 1e3:.0f} ms")
    print(f"  ensemble members         : {N_MEMBERS} (bootstrap resamples, HistGradientBoosting)")
    print(f"  rows available / used    : {x_tr.shape[0]:,} / {pred.n_train_rows_:,}")
    print(f"  positive-label rate      : {pred.train_positive_rate_:.4f}")
    print(f"  fit wall time            : {fit_s:.1f} s")

    t_p = time.time()
    p_tune = [pred.predict_outage(make_features(t))[0] for t in tune_traces]
    p_test_pairs = [pred.predict_outage(make_features(t)) for t in test_traces]
    p_test = [a for a, _ in p_test_pairs]
    s_test = [b for _, b in p_test_pairs]
    print(f"  inference wall time      : {time.time() - t_p:.1f} s for "
          f"{(len(tune_traces) + len(test_traces)) * SC.n_steps:,} rows")

    # ------------------------------------------------------------ tune
    banner("Parameter selection on TUNE_SEEDS (never on TEST_SEEDS)")
    grid_t = np.arange(-6.0, 3.01, 0.25)
    tune_fixed = {
        float(t): mean_tput([FixedThresholdPolicy(t).select(tr) for tr in tune_traces],
                            tune_traces)
        for t in grid_t
    }
    t_fixed = max(tune_fixed, key=tune_fixed.get)
    print(f"  fixed threshold grid     : {grid_t[0]:.2f} .. {grid_t[-1]:.2f} dB step 0.25 "
          f"({len(grid_t)} values)")
    print(f"  best fixed threshold     : {t_fixed:+.2f} dB  "
          f"({tune_fixed[t_fixed]:.2f} Mb/s on tune seeds)")

    best_h, best_hv = None, -np.inf
    n_h = 0
    for lo in np.arange(-6.0, 2.01, 0.5):
        for up in np.arange(lo, 3.01, 0.5):
            n_h += 1
            v = mean_tput([HysteresisPolicy(lo, up).select(tr) for tr in tune_traces],
                          tune_traces)
            if v > best_hv:
                best_h, best_hv = (float(lo), float(up)), v
    print(f"  hysteresis grid          : {n_h} (lower, upper) pairs, step 0.5 dB")
    print(f"  best hysteresis          : lower {best_h[0]:+.2f} dB, upper {best_h[1]:+.2f} dB  "
          f"({best_hv:.2f} Mb/s on tune seeds)")

    p_grid = np.round(np.arange(0.10, 0.996, 0.02), 3)
    tune_p = {}
    for p in p_grid:
        sels = []
        for pm in p_tune:
            sel = pm <= p
            sel[0] = True
            sels.append(sel)
        tune_p[float(p)] = mean_tput(sels, tune_traces)
    p_tuned = max(tune_p, key=tune_p.get)
    p_derived = 1.0 - SC.rate_ratio
    print(f"  learned p* grid          : 0.10 .. 0.99 step 0.02 ({len(p_grid)} values)")
    print(f"  cost-derived p* = 1 - q  : {p_derived:.3f}  "
          f"({tune_p[min(tune_p, key=lambda k: abs(k - p_derived))]:.2f} Mb/s on tune seeds)")
    print(f"  best learned p*          : {p_tuned:.3f}  ({tune_p[p_tuned]:.2f} Mb/s on tune seeds)")

    t_analytic = optimal_fixed_threshold_db(
        SC.optical.margin_mean_db, SC.optical.margin_std_db, SC.rho,
        SC.rate_optical_bps, SC.rate_rf_bps,
    )
    print(f"\n  analytic T* from the scintillation-only closed form: {t_analytic:+.4f} dB")
    print("  (Scenario B is not stationary Gaussian, so this is a reference point,")
    print("   not the optimum -- see V1 and the VALIDATION.md discussion.)")

    # ------------------------------------------------------------ test
    banner(f"Held-out evaluation on {len(TEST_SEEDS)} TEST_SEEDS")
    selections: dict[str, list[np.ndarray]] = {
        "always_optical": [AlwaysOpticalPolicy().select(t) for t in test_traces],
        "always_rf": [AlwaysRfPolicy().select(t) for t in test_traces],
        "fixed_analytic": [FixedThresholdPolicy(t_analytic).select(t) for t in test_traces],
        "fixed_tuned": [FixedThresholdPolicy(t_fixed).select(t) for t in test_traces],
        "hysteresis_tuned": [
            HysteresisPolicy(best_h[0], best_h[1]).select(t) for t in test_traces
        ],
        "clairvoyant": [ClairvoyantPolicy().select(t) for t in test_traces],
    }
    for label, p in (("learned_derived", p_derived), ("learned_tuned", p_tuned)):
        sels = []
        for pm in p_test:
            sel = pm <= p
            sel[0] = True
            sels.append(sel)
        selections[label] = sels

    order = [
        "always_optical", "always_rf", "fixed_analytic", "fixed_tuned",
        "hysteresis_tuned", "learned_derived", "learned_tuned", "clairvoyant",
    ]
    metrics = {k: [score(s, t) for s, t in zip(selections[k], test_traces)] for k in order}
    print(f"\n  {'policy':<18} {'throughput [Mb/s], 95% CI':<30} {'outage frac':<22} "
          f"{'switches/s':<20}")
    for k in order:
        tp = mean_ci([m.throughput_bps / 1e6 for m in metrics[k]], CONF)
        of = mean_ci([m.outage_fraction for m in metrics[k]], CONF)
        sw = mean_ci([m.switches_per_s for m in metrics[k]], CONF)
        print(f"  {k:<18} {tp.mean:8.2f} [{tp.low:7.2f},{tp.high:7.2f}]  "
              f"{of.mean:7.4f} [{of.low:6.4f},{of.high:6.4f}]  "
              f"{sw.mean:7.2f} [{sw.low:6.2f},{sw.high:6.2f}]")
    print("\n  ('clairvoyant' is non-causal: it reads the true optical state at the step it")
    print("   is deciding. It is an upper reference, not a deployable policy, and with a")
    print("   non-zero handover guard it is not even an upper bound.)")

    banner(
        f"Paired per-trial differences in throughput (95 pct Student-t, "
        f"n = {len(TEST_SEEDS)})"
    )
    print(f"  {'comparison':<40} {'mean diff [Mb/s]':>17} {'95% CI':>24}   verdict")
    pairs = [
        ("fixed_tuned", "always_optical"),
        ("hysteresis_tuned", "always_optical"),
        ("learned_tuned", "always_optical"),
        ("hysteresis_tuned", "fixed_tuned"),
        ("learned_derived", "fixed_tuned"),
        ("learned_tuned", "fixed_tuned"),
        ("learned_tuned", "hysteresis_tuned"),
        ("learned_tuned", "learned_derived"),
        ("fixed_tuned", "fixed_analytic"),
    ]
    for a, b in pairs:
        d = paired_diff_ci(
            [m.throughput_bps / 1e6 for m in metrics[a]],
            [m.throughput_bps / 1e6 for m in metrics[b]],
            CONF,
        )
        verdict = ("A > B" if d.mean > 0 else "B > A") if d.excludes_zero() else "INDISTINGUISHABLE"
        print(f"  {a + ' - ' + b:<40} {d.mean:17.3f} [{d.low:10.3f},{d.high:10.3f}]   {verdict}")

    banner("Predictor quality on the test traces")
    y_true = np.concatenate([make_labels(t, HORIZON) for t in test_traces])
    p_all = np.concatenate(p_test)
    s_all = np.concatenate(s_test)
    brier = float(np.mean((p_all - y_true.astype(float)) ** 2))
    base_rate = float(y_true.mean())
    brier_base = float(np.mean((base_rate - y_true.astype(float)) ** 2))
    print(f"  rows                            : {y_true.size:,}")
    print(f"  positive rate (outage within H) : {base_rate:.4f}")
    print(f"  Brier score (model)             : {brier:.6f}")
    print(f"  Brier score (constant base rate): {brier_base:.6f}")
    print(f"  Brier skill score               : {1.0 - brier / brier_base:.4f}")
    print(f"  mean ensemble std (confidence)  : {float(s_all.mean()):.6f}")
    print(f"  mean ensemble std | p in (.1,.9): "
          f"{float(s_all[(p_all > 0.1) & (p_all < 0.9)].mean()):.6f}")
    print("\n  reliability (calibration) table:")
    print(f"  {'bin':<14} {'n':>10} {'mean p':>10} {'observed':>10} {'gap':>9}")
    edges = np.array([0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0])
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p_all >= lo) & (p_all < hi) if hi < 1.0 else (p_all >= lo) & (p_all <= hi)
        if not m.any():
            continue
        mp = float(p_all[m].mean())
        obs = float(y_true[m].mean())
        ece += m.mean() * abs(mp - obs)
        print(f"  [{lo:.2f},{hi:.2f})   {int(m.sum()):10,} {mp:10.4f} {obs:10.4f} "
              f"{mp - obs:+9.4f}")
    print(f"\n  expected calibration error (ECE): {ece:.4f}")

    banner("Sensitivity to the handover guard (same test traces, retuned nothing)")
    print("  Throughput [Mb/s] as the guard grows. No parameter is re-tuned, so this")
    print("  shows how brittle each policy's tuned setting is to the guard assumption.")
    print(f"\n  {'guard [samples]':<16} {'guard [ms]':<12}", end="")
    show = ["always_optical", "fixed_tuned", "hysteresis_tuned", "learned_tuned"]
    for k in show:
        print(f"{k:>19}", end="")
    print()
    for g in (0, 1, 2, 5, 10):
        print(f"  {g:<16} {g * SC.dt_s * 1e3:<12.1f}", end="")
        for k in show:
            print(f"{mean_tput(selections[k], test_traces, guard=g):19.2f}", end="")
        print()

    print(f"\n  total wall time: {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
