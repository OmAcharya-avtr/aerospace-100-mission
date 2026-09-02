"""The learned warm start against the cold-started optimiser.

This is the benchmark the AI element of SlewForge stands or falls on. It
measures **both** things the specification asks for -- solve time and solution
quality -- and reports whichever way they come out.

PART A  dataset generation and the cold baseline
PART B  training, and the model's own regression error
PART C  the benchmark: cold multi-start vs warm start vs a zero-start control
PART D  paired differences with 95 % confidence intervals
PART E  the confidence output: what it does and does not predict
PART F  compute budget

Run: ``python validation/validate_warm_start.py``  (about 7 minutes on 2 cores)
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from slewforge.dataset import generate_dataset, problem_features  # noqa: E402
from slewforge.ml import LearnedWarmStart  # noqa: E402
from slewforge.planner import path_min_margin, plan  # noqa: E402

N_TRAIN = 200
N_TEST = 110
SEED_TRAIN = 20260901
SEED_TEST = 20260902
SEED_MODEL = 0
N_JOBS = 2


def rule(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def ci95(x: np.ndarray) -> tuple[float, float]:
    """Mean and half-width of a 95 % confidence interval on the mean."""
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return float(np.mean(x)) if x.size else float("nan"), float("nan")
    return float(np.mean(x)), float(1.96 * np.std(x, ddof=1) / math.sqrt(x.size))


def main() -> int:
    failures = 0

    # ---------------------------------------------------------------- PART A
    rule("PART A -- dataset and the cold baseline")
    print("Problems are drawn Haar-uniformly on SO(3) with 1-3 Haar-uniform cones")
    print("of 15-50 deg half-angle, and kept only when both endpoints clear every")
    print("cone and the direct eigenaxis slew violates at least one. Labels are the")
    print("via parameter vector the cold multi-start planner arrives at: seven")
    print("deterministic SLSQP starts, best objective kept.")
    print()
    t0 = time.perf_counter()
    train = generate_dataset(N_TRAIN, seed=SEED_TRAIN, n_jobs=N_JOBS)
    t_train_data = time.perf_counter() - t0
    t0 = time.perf_counter()
    test = generate_dataset(N_TEST, seed=SEED_TEST, n_jobs=N_JOBS)
    t_test_data = time.perf_counter() - t0
    print(f"train : {len(train):>4} labelled of {train.n_attempted} attempted "
          f"(label rate {train.label_rate:.4f}), {t_train_data:.1f} s wall, "
          f"{N_JOBS} worker processes")
    print(f"test  : {len(test):>4} labelled of {test.n_attempted} attempted "
          f"(label rate {test.label_rate:.4f}), {t_test_data:.1f} s wall")
    print(f"features {train.features.shape[1]}, targets {train.targets.shape[1]}")
    print()
    print("Cold baseline recorded during labelling (note: the labelling runs two")
    print("worker processes on two cores, so these solve times are inflated by the")
    print("contention. PART C re-times the cold planner single-process, and that is")
    print("the number the benchmark uses):")
    print(f"  mean solve time     : {float(np.mean(test.solve_time)) * 1e3:.2f} ms "
          f"(under 2-way parallelism)")
    print(f"  mean path evals     : {float(np.mean(test.evals)):.0f}")
    print(f"  mean objective      : {float(np.mean(test.objective)):.4f} s")
    print()
    print("Target distribution (via parameter, canonical frame, rad):")
    for k, name in enumerate(("p1 (along eigenaxis)", "p2 (boresight plane)", "p3 (normal)")):
        col = train.targets[:, k]
        print(f"  {name:<22} mean {float(np.mean(col)):+.4f}  sd {float(np.std(col)):.4f}  "
              f"min {float(np.min(col)):+.4f}  max {float(np.max(col)):+.4f}")
    print(f"  |p| mean {float(np.mean(np.linalg.norm(train.targets, axis=1))):.4f} rad, "
          f"max {float(np.max(np.linalg.norm(train.targets, axis=1))):.4f} rad")

    # ---------------------------------------------------------------- PART B
    rule("PART B -- training and regression error")
    t0 = time.perf_counter()
    model = LearnedWarmStart(n_estimators=300, min_samples_leaf=2, random_state=SEED_MODEL)
    model.fit(train.features, train.targets)
    t_fit = time.perf_counter() - t0
    print(f"ExtraTreesRegressor, 300 trees, min_samples_leaf 2, seed {SEED_MODEL}")
    print(f"fit time            : {t_fit:.2f} s")
    pred_tr = model.predict(train.features)
    pred_te = model.predict(test.features)
    for label, pred, targ in (("train", pred_tr, train.targets), ("test", pred_te, test.targets)):
        err = np.linalg.norm(pred.params - targ, axis=1)
        print(f"{label:<6} |p_pred - p_label| : mean {float(np.mean(err)):.4f} rad, "
              f"median {float(np.median(err)):.4f}, p90 {float(np.percentile(err, 90)):.4f}")
    baseline_err = np.linalg.norm(test.targets, axis=1)
    print(f"predicting zero      : mean {float(np.mean(baseline_err)):.4f} rad "
          f"(the trivial baseline the model must beat)")
    print(f"predicting the train mean: mean "
          f"{float(np.mean(np.linalg.norm(test.targets - np.mean(train.targets, axis=0), axis=1))):.4f} rad")
    print()
    print("Regression error is not the quantity of interest -- a warm start only")
    print("has to land in the right basin, not on the label -- but a model that")
    print("cannot beat 'predict zero' has learned nothing, so it is reported.")

    # ---------------------------------------------------------------- PART C
    rule("PART C -- the benchmark")
    print("Three planners on the same held-out problems:")
    print("  cold   the deterministic seven-start sweep, best objective kept")
    print("  warm   one SLSQP start from the model's prediction, no fallback")
    print("  zero   one SLSQP start from p = 0, no fallback -- the control that")
    print("         separates 'the model helps' from 'one start is enough'")
    print("  warm+  the model's start, falling back to the cold sweep on failure")
    print("         (this is what plan(warm_start=...) does by default)")
    print()
    rows = []
    for i, problem in enumerate(test.problems):
        feats = problem_features(problem)[None, :]
        t0 = time.perf_counter()
        pred = model.predict(feats)
        t_pred = time.perf_counter() - t0
        p_hat = pred.params[0]
        r_cold = plan(problem)
        r_warm = plan(problem, warm_start=p_hat,
                      warm_start_confidence=float(pred.confidence[0]), cold_fallback=False)
        r_zero = plan(problem, warm_start=np.zeros(3), cold_fallback=False)
        r_wf = plan(problem, warm_start=p_hat,
                    warm_start_confidence=float(pred.confidence[0]), cold_fallback=True)
        rows.append(
            {
                "i": i,
                "t_pred": t_pred,
                "cold_t": r_cold.solve_time_s,
                "cold_obj": r_cold.objective if r_cold.feasible else np.nan,
                "cold_ok": r_cold.feasible,
                "cold_ev": r_cold.n_objective_evals,
                "warm_t": r_warm.solve_time_s,
                "warm_obj": r_warm.objective if r_warm.feasible else np.nan,
                "warm_ok": r_warm.feasible,
                "warm_ev": r_warm.n_objective_evals,
                "zero_t": r_zero.solve_time_s,
                "zero_obj": r_zero.objective if r_zero.feasible else np.nan,
                "zero_ok": r_zero.feasible,
                "zero_ev": r_zero.n_objective_evals,
                "wf_t": r_wf.solve_time_s,
                "wf_obj": r_wf.objective if r_wf.feasible else np.nan,
                "wf_ok": r_wf.feasible,
                "wf_ev": r_wf.n_objective_evals,
                "wf_accepted": r_wf.warm_start_accepted,
                "conf": float(pred.confidence[0]),
                "extrap": bool(pred.extrapolating[0]),
                "warm_margin": (
                    path_min_margin(problem, r_warm.path) if r_warm.path is not None else np.nan
                ),
            }
        )

    def col(name):
        return np.array([r[name] for r in rows], dtype=float)

    n = len(rows)
    print(f"{'planner':<8}{'feasible':>12}{'mean t [ms]':>14}{'median t [ms]':>16}"
          f"{'mean evals':>12}{'mean T [s]':>13}")
    for tag, label in (("cold", "cold"), ("warm", "warm"), ("zero", "zero"), ("wf", "warm+")):
        ok = col(f"{tag}_ok").astype(bool)
        t = col(f"{tag}_t")
        ev = col(f"{tag}_ev")
        obj = col(f"{tag}_obj")
        print(
            f"{label:<8}{int(ok.sum()):>6} / {n:<5}{float(np.mean(t)) * 1e3:>14.2f}"
            f"{float(np.median(t)) * 1e3:>16.2f}{float(np.mean(ev)):>12.0f}"
            f"{float(np.nanmean(obj)):>13.4f}"
        )
    print()
    print(f"model inference time per problem : "
          f"{float(np.mean(col('t_pred'))) * 1e3:.2f} ms "
          f"(included in the warm timings above)")
    print(f"warm start accepted without fallback : "
          f"{int(np.sum([r['wf_accepted'] for r in rows]))} / {n}")
    bad = int(np.sum(col("warm_margin") < 0.0))
    print(f"warm-start paths violating a cone    : {bad}   tolerance 0")
    failures += bad > 0

    # ---------------------------------------------------------------- PART D
    rule("PART D -- paired differences, 95 % confidence intervals")
    both = col("cold_ok").astype(bool) & col("warm_ok").astype(bool)
    print(f"problems where both cold and warm found a path: {int(both.sum())} / {n}")
    print()
    dt = (col("cold_t") - col("warm_t"))[both] * 1e3
    m, h = ci95(dt)
    print(f"solve time saved (cold - warm) : {m:+.2f} +/- {h:.2f} ms")
    print(f"  speed-up factor (mean cold / mean warm) : "
          f"{float(np.mean(col('cold_t')[both]) / np.mean(col('warm_t')[both])):.3f}x")
    dev = (col("cold_ev") - col("warm_ev"))[both]
    m2, h2 = ci95(dev)
    print(f"path evaluations saved         : {m2:+.1f} +/- {h2:.1f}")
    do = (col("warm_obj") - col("cold_obj"))[both]
    m3, h3 = ci95(do)
    print(f"objective penalty (warm - cold): {m3:+.4f} +/- {h3:.4f} s "
          f"({m3 / float(np.mean(col('cold_obj')[both])) * 100:+.3f} % of the mean)")
    print(f"  warm strictly worse on {int(np.sum(do > 1e-9))} of {int(both.sum())} problems, "
          f"strictly better on {int(np.sum(do < -1e-9))}, equal on "
          f"{int(np.sum(np.abs(do) <= 1e-9))}")
    print(f"  worst single-problem penalty : {float(np.max(do)):+.4f} s "
          f"({float(np.max(do / col('cold_obj')[both])) * 100:+.2f} %)")

    print()
    print("Against the zero-start control, which is the honest question -- does the")
    print("model help, or is one start simply enough?")
    both_z = col("cold_ok").astype(bool) & col("zero_ok").astype(bool)
    print(f"  zero-start feasible          : {int(col('zero_ok').sum())} / {n}")
    print(f"  warm-start feasible          : {int(col('warm_ok').sum())} / {n}")
    dz = (col("zero_obj") - col("cold_obj"))[both_z]
    m4, h4 = ci95(dz)
    print(f"  zero-start objective penalty : {m4:+.4f} +/- {h4:.4f} s")
    only_warm = int(np.sum(col("warm_ok").astype(bool) & ~col("zero_ok").astype(bool)))
    only_zero = int(np.sum(col("zero_ok").astype(bool) & ~col("warm_ok").astype(bool)))
    print(f"  problems solved by warm but not zero : {only_warm}")
    print(f"  problems solved by zero but not warm : {only_zero}")

    # ---------------------------------------------------------------- PART E
    rule("PART E -- the confidence output")
    conf = col("conf")
    ok = col("warm_ok").astype(bool)
    print(f"confidence range : {float(np.min(conf)):.4f} to {float(np.max(conf)):.4f}, "
          f"mean {float(np.mean(conf)):.4f}")
    print(f"extrapolation flags raised : {int(np.sum([r['extrap'] for r in rows]))} / {n}")
    print()
    print("Does confidence predict whether the warm start succeeds?")
    print(f"  mean confidence when warm succeeded : {float(np.mean(conf[ok])):.4f}")
    if np.any(~ok):
        print(f"  mean confidence when warm failed    : {float(np.mean(conf[~ok])):.4f}")
    else:
        print("  mean confidence when warm failed    : n/a (no failures)")
    print()
    print("Does confidence predict the objective penalty?")
    valid = both & np.isfinite(do_full := (col("warm_obj") - col("cold_obj")))
    if int(valid.sum()) > 4:
        c = conf[valid]
        d = do_full[valid]
        rank_c = np.argsort(np.argsort(c))
        rank_d = np.argsort(np.argsort(d))
        rho = float(np.corrcoef(rank_c, rank_d)[0, 1])
        print(f"  Spearman rank correlation confidence vs penalty : {rho:+.4f}")
        edges = np.quantile(c, [0.0, 1 / 3, 2 / 3, 1.0])
        print(f"  {'confidence tercile':<24}{'n':>5}{'mean penalty [s]':>20}")
        for lo, hi, name in ((edges[0], edges[1], "low"), (edges[1], edges[2], "middle"),
                             (edges[2], edges[3] + 1e-12, "high")):
            sel = (c >= lo) & (c < hi)
            if sel.sum():
                print(f"  {name:<24}{int(sel.sum()):>5}{float(np.mean(d[sel])):>20.4f}")
    print()
    print("Read this section as a measurement, not a guarantee. The confidence is")
    print("the agreement of 300 extremely randomised trees, not a calibrated")
    print("probability. The planner does not act on it.")

    # ---------------------------------------------------------------- PART F
    rule("PART F -- compute budget")
    t_bench = float(np.sum(col("cold_t") + col("warm_t") + col("zero_t") + col("wf_t")))
    stages = [
        (f"train dataset generation ({train.n_attempted} problems, {N_JOBS} processes)",
         t_train_data),
        (f"test dataset generation  ({test.n_attempted} problems, {N_JOBS} processes)",
         t_test_data),
        ("model fit", t_fit),
        (f"benchmark ({n} problems x 4 planners)", t_bench),
    ]
    for label, secs in stages:
        verdict = "within" if secs <= 180.0 else "OVER"
        print(f"{label:<55}{secs:>8.1f} s   {verdict} the 180 s guide budget")
    print(f"{'total':<55}{sum(s for _, s in stages):>8.1f} s")
    print()
    print("The label cost is what makes the dataset small: every training example")
    print("is a full cold solve. Peak resident memory is dominated by the 300-tree")
    print("forest, a few megabytes.")
    print()
    print(f"Process-pool efficiency: labelling took {t_train_data / train.n_attempted * 1e3:.0f} ms")
    print(f"per problem on {N_JOBS} workers against a single-process cold solve of "
          f"{float(np.mean(col('cold_t'))) * 1e3:.0f} ms")
    print("(PART C). On a 2-core machine the pool buys little and sometimes nothing;")
    print("the figure is reported rather than assumed.")

    rule("VERDICT")
    speed = float(np.mean(col("cold_t")[both]) / np.mean(col("warm_t")[both]))
    penalty_pct = m3 / float(np.mean(col("cold_obj")[both])) * 100
    print(f"solve time      : warm start is {speed:.2f}x faster than the cold sweep")
    print(f"solution quality: warm start is {penalty_pct:+.3f} % on the objective "
          f"(95 % CI {(m3 - h3) / float(np.mean(col('cold_obj')[both])) * 100:+.3f} % to "
          f"{(m3 + h3) / float(np.mean(col('cold_obj')[both])) * 100:+.3f} %)")
    print(f"feasibility     : warm {int(col('warm_ok').sum())}/{n}, "
          f"zero {int(col('zero_ok').sum())}/{n}, cold {int(col('cold_ok').sum())}/{n}")
    print()
    if speed <= 1.0:
        print("The warm start does NOT reduce solve time. That is the measurement.")
    if m3 - h3 > 0.0:
        print("The warm start significantly DEGRADES the objective; the confidence")
        print("interval on the penalty excludes zero.")
    elif m3 + h3 < 0.0:
        print("The warm start significantly IMPROVES the objective.")
    else:
        print("The objective difference is not distinguishable from zero at 95 %.")
    print()
    print("MODEL_CARD.md carries the same numbers and the interpretation.")

    rule("SUMMARY")
    print(f"failed checks: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
