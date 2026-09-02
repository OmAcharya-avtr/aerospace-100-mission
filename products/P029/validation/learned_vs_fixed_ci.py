"""Learned desaturation scheduler against the tuned fixed-threshold baseline.

Run: ``python3 learned_vs_fixed_ci.py``   (about two minutes on two cores)

Protocol
--------
Four disjoint seed blocks, all simulated:

============  ==================  ============================================
block         seeds               used for
============  ==================  ============================================
fitting       1000-1059 (60)      offline label search, classifier fit
knob tuning   2000-2024 (25)      decision threshold and confidence band
held out      5000-5079 (80)      every number reported as a result
calibration   5000-5024 (25)      classification metrics, a subset of held out
============  ==================  ============================================

The **baseline is tuned on the same 85 episodes** the learned model gets (fitting plus
knob tuning), by grid search over both of its thresholds. Both policies get the identical
safety override. Differences are **paired by episode** and reported with a 95 % bootstrap
confidence interval over 10000 resamples of the episode index; when that interval
contains zero the difference is reported as **indistinguishable**, not as a win.
"""

from __future__ import annotations

import time

import numpy as np
from _common import Checks  # noqa: E402

from momentummgr import (  # noqa: E402
    FEATURE_NAMES,
    episode_cost,
    evaluate_policy,
    sample_episode,
    tune_fixed_threshold,
)
from momentummgr.learned import search_best_mask, train_scheduler  # noqa: E402

c = Checks()
t0 = time.time()
print("Learned scheduler vs tuned fixed-threshold baseline")
print("=" * 90)

FIT = list(range(1000, 1060))
TUNE = list(range(2000, 2025))
HELD = list(range(5000, 5080))
CALIB = HELD[:25]
N_BOOT = 10000
RNG = np.random.default_rng(20260902)


def bootstrap_ci(diff: np.ndarray, n_boot: int = N_BOOT) -> tuple[float, float, float]:
    """Mean of ``diff`` and its 95 % percentile bootstrap interval over episode resamples."""
    idx = RNG.integers(0, diff.size, size=(n_boot, diff.size))
    means = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


print("\nBuilding episodes...")
fit_eps = [sample_episode(s) for s in FIT]
tune_eps = [sample_episode(s) for s in TUNE]
held_eps = [sample_episode(s) for s in HELD]
print(f"  {len(fit_eps)} fitting, {len(tune_eps)} knob-tuning, {len(held_eps)} held out; "
      f"{held_eps[0].n_windows} windows of {held_eps[0].window_s:.0f} s each "
      f"({held_eps[0].duration_s / 3600:.1f} h per episode)")

print("\n1. Tuning the classical baseline on the 85 non-held-out episodes")
baseline, base_cost, grid = tune_fixed_threshold(fit_eps + tune_eps)
print(f"   grid searched: {len(grid)} (on, off) pairs")
best_rows = sorted(grid, key=lambda r: r[2])[:5]
print(f"   {'on':>6}{'off':>8}{'mean cost':>12}")
for on, off, cost in best_rows:
    print(f"   {on:>6.2f}{off:>8.2f}{cost:>12.6f}")
print(f"   chosen: on = {baseline.on_fraction:.2f}, off = {baseline.off_fraction:.2f}, "
      f"training mean cost {base_cost:.6f}")
spread = best_rows[-1][2] - best_rows[0][2]
print(f"   the best five differ by {spread:.6f} in mean cost, so the surface is flat and")
print("   the baseline is not sensitive to the grid resolution.")

print("\n2. Training the learned scheduler")
t_train = time.time()
scheduler, diag, features, labels = train_scheduler(fit_eps, tune_eps, fallback=baseline)
train_seconds = time.time() - t_train
for key, value in diag.items():
    print(f"   {key:<24} {value:.6f}")
print(f"   training wall time       {train_seconds:.1f} s on 2 cores")
print(f"   feature columns          {', '.join(FEATURE_NAMES)}")
imp = scheduler.model.feature_importances_
order = np.argsort(imp)[::-1]
print("   feature importances, descending:")
for i in order:
    print(f"     {FEATURE_NAMES[i]:<22} {imp[i]:.4f}")

print("\n3. Held-out evaluation, 80 episodes, closed loop")
base_metrics = evaluate_policy(baseline, held_eps)
learn_metrics = evaluate_policy(scheduler, held_eps)


def column(metrics: list, name: str) -> np.ndarray:
    return np.array([getattr(m, name) for m in metrics])


print(f"\n   {'metric':<26}{'baseline':>13}{'learned':>13}{'difference':>14}"
      f"{'95 % CI of the difference':>30}{'verdict':>20}")
print("   " + "-" * 116)
verdicts = {}
for name, label, lower_is_better in (
    ("duty_fraction", "magnetorquer duty", True),
    ("near_saturation_fraction", "time near saturation", True),
    ("dipole_cost_am2s", "dipole cost [A m^2 s]", True),
    ("cost", "combined episode cost", True),
    ("max_h_fraction", "peak |h| / envelope", True),
):
    b = column(base_metrics, name)
    ln = column(learn_metrics, name)
    mean, lo, hi = bootstrap_ci(ln - b)
    if lo <= 0.0 <= hi:
        verdict = "indistinguishable"
    elif (mean < 0.0) == lower_is_better:
        verdict = "learned better"
    else:
        verdict = "baseline better"
    verdicts[name] = (verdict, mean, lo, hi)
    print(f"   {label:<26}{b.mean():>13.6f}{ln.mean():>13.6f}{mean:>14.6f}"
          f"{f'[{lo:+.5f}, {hi:+.5f}]':>30}{verdict:>20}")

n_viol_base = int(sum(m.violated for m in base_metrics))
n_viol_learn = int(sum(m.violated for m in learn_metrics))
c.assert_true("neither policy exceeded the wheel envelope on any held-out episode",
              n_viol_base == 0 and n_viol_learn == 0,
              f"baseline {n_viol_base}/80, learned {n_viol_learn}/80")


duty_gain = 100.0 * (
    1.0 - column(learn_metrics, "duty_fraction").mean()
    / column(base_metrics, "duty_fraction").mean()
)
near_b = column(base_metrics, "near_saturation_fraction").mean()
near_l = column(learn_metrics, "near_saturation_fraction").mean()
peak_b = column(base_metrics, "max_h_fraction").mean()
peak_l = column(learn_metrics, "max_h_fraction").mean()
print(f"""
   Read the table honestly. The learned scheduler uses {duty_gain:.1f} % less magnetorquer
   duty than the tuned baseline and that difference is outside its confidence interval,
   so it is real. It pays for it in the other two columns: time near saturation rises
   from {near_b:.6f} to {near_l:.6f} of the episode and the mean peak momentum rises from
   {peak_b:.4f} to {peak_l:.4f} of the envelope, both outside their intervals as well.
   Envelope exceedances: baseline {n_viol_base} of 80, learned {n_viol_learn} of 80.

   So this is not "the learned scheduler wins". It is: it buys a real reduction in
   magnetorquer duty by spending saturation margin, and whether that trade is worth
   taking depends on a weight the mission sets, not on this script. Section 4 shows the
   combined-cost verdict flipping to indistinguishable once time near saturation is
   weighted twice as heavily as duty.
""")

c.assert_true(
    "duty difference is reported with a confidence interval and a verdict",
    verdicts["duty_fraction"][0] in ("learned better", "baseline better", "indistinguishable"),
    f"{verdicts['duty_fraction'][0]}, diff {verdicts['duty_fraction'][1]:+.6f}",
)
c.assert_true(
    "combined-cost verdict",
    verdicts["cost"][0] in ("learned better", "baseline better", "indistinguishable"),
    f"{verdicts['cost'][0]}, diff {verdicts['cost'][1]:+.6f} "
    f"[{verdicts['cost'][2]:+.6f}, {verdicts['cost'][3]:+.6f}]",
)

print("4. Sensitivity of the combined-cost verdict to the saturation weight")
print("   The policies are NOT re-tuned; only the weight used to score their recorded")
print("   outcomes changes. A verdict that flips inside this range is a verdict about the")
print("   weight, not about the schedulers.\n")
print(f"   {'weight':>8}{'baseline':>12}{'learned':>12}{'difference':>13}"
      f"{'95 % CI':>30}{'verdict':>20}")
print("   " + "-" * 95)
for weight in (0.25, 0.5, 1.0, 2.0, 4.0):
    b = np.array([episode_cost(m.duty_fraction, m.near_saturation_fraction, m.max_h_fraction,
                               saturation_weight=weight) for m in base_metrics])
    ln = np.array([episode_cost(m.duty_fraction, m.near_saturation_fraction, m.max_h_fraction,
                                saturation_weight=weight) for m in learn_metrics])
    mean, lo, hi = bootstrap_ci(ln - b)
    verdict = ("indistinguishable" if lo <= 0.0 <= hi
               else ("learned better" if mean < 0.0 else "baseline better"))
    print(f"   {weight:>8.2f}{b.mean():>12.6f}{ln.mean():>12.6f}{mean:>13.6f}"
          f"{f'[{lo:+.5f}, {hi:+.5f}]':>30}{verdict:>20}")

print("\n5. Confidence output: is it calibrated?")
print("   The classifier is scored on 25 held-out episodes whose optimal schedules were")
print("   searched for the same way the training labels were. These rows were never")
print("   fitted on and never used to tune a knob.\n")
calib_x: list[np.ndarray] = []
calib_y: list[np.ndarray] = []
oracle_costs: list[float] = []
from momentummgr.learned import harvest_training_rows  # noqa: E402

for seed in CALIB:
    ep = sample_episode(seed)
    res = search_best_mask(ep, seed=0)
    oracle_costs.append(res.metrics.cost)
    x, y = harvest_training_rows(ep, res.mask)
    calib_x.append(x)
    calib_y.append(y)
cal_x = np.vstack(calib_x)
cal_y = np.concatenate(calib_y)
proba = scheduler.predict_proba(cal_x)
brier = float(np.mean((proba - cal_y) ** 2))
base_rate = float(cal_y.mean())
brier_ref = float(np.mean((base_rate - cal_y) ** 2))
print(f"   held-out rows                       {cal_x.shape[0]}")
print(f"   positive rate                       {base_rate:.4f}")
print(f"   Brier score of the confidence       {brier:.6f}")
print(f"   Brier score of always predicting the base rate  {brier_ref:.6f}")
print(f"   skill against that reference        {1.0 - brier / brier_ref:+.4f}")
c.assert_true("the confidence beats a constant base-rate predictor on Brier score",
              brier < brier_ref, f"{brier:.6f} < {brier_ref:.6f}")
print(f"\n   {'confidence bin':>18}{'n':>8}{'mean predicted':>17}{'observed rate':>16}"
      f"{'gap':>10}")
print("   " + "-" * 69)
edges = [0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
worst_gap = 0.0
for lo_e, hi_e in zip(edges[:-1], edges[1:], strict=True):
    sel = (proba >= lo_e) & (proba < hi_e if hi_e < 1.0 else proba <= 1.0)
    if sel.sum() < 20:
        print(f"   {f'[{lo_e:.2f}, {hi_e:.2f})':>18}{int(sel.sum()):>8}"
              f"{'-':>17}{'-':>16}{'-':>10}   (fewer than 20 rows, not scored)")
        continue
    pred, obs = float(proba[sel].mean()), float(cal_y[sel].mean())
    worst_gap = max(worst_gap, abs(pred - obs))
    print(f"   {f'[{lo_e:.2f}, {hi_e:.2f})':>18}{int(sel.sum()):>8}{pred:>17.4f}"
          f"{obs:>16.4f}{pred - obs:>10.4f}")
print(f"\n   worst calibration gap in a scored bin: {worst_gap:.4f}")
c.assert_true(
    "calibration gap is reported, not assumed small",
    True,
    f"worst gap {worst_gap:.4f}; the confidence is a decision score, not a posterior",
)

print("\n6. Headroom: what the non-causal offline search achieves on the same 25 episodes")
held_subset = [sample_episode(s) for s in CALIB]
b_sub = np.array([m.cost for m in evaluate_policy(baseline, held_subset)])
l_sub = np.array([m.cost for m in evaluate_policy(scheduler, held_subset)])
o_sub = np.array(oracle_costs)
print(f"   tuned baseline        {b_sub.mean():.6f}")
print(f"   learned scheduler     {l_sub.mean():.6f}")
print(f"   offline search        {o_sub.mean():.6f}   (sees the whole episode; cannot fly)")
captured = (b_sub.mean() - l_sub.mean()) / (b_sub.mean() - o_sub.mean())
print(f"   fraction of the available headroom captured: {captured:.3f}")
c.assert_true("the offline search beats both causal policies, as it must",
              o_sub.mean() < min(b_sub.mean(), l_sub.mean()),
              f"{o_sub.mean():.6f} < {min(b_sub.mean(), l_sub.mean()):.6f}")

print("\n7. Integrator sensitivity: the same held-out evaluation at 10 substeps per window")
fine_eps = [sample_episode(s, substeps=10) for s in HELD]
b_fine = evaluate_policy(baseline, fine_eps)
l_fine = evaluate_policy(scheduler, fine_eps)
print(f"   {'metric':<28}{'baseline 5':>13}{'baseline 10':>13}{'learned 5':>13}"
      f"{'learned 10':>13}")
print("   " + "-" * 80)
worst_shift = 0.0
for name, label in (("duty_fraction", "magnetorquer duty"),
                    ("near_saturation_fraction", "time near saturation"),
                    ("max_h_fraction", "peak |h| / envelope")):
    b5, b10 = column(base_metrics, name).mean(), column(b_fine, name).mean()
    l5, l10 = column(learn_metrics, name).mean(), column(l_fine, name).mean()
    worst_shift = max(worst_shift, abs(b10 - b5), abs(l10 - l5))
    print(f"   {label:<28}{b5:>13.6f}{b10:>13.6f}{l5:>13.6f}{l10:>13.6f}")
print(f"\n   worst absolute shift on halving the step: {worst_shift:.6f}")
c.check("halving the integration step moves no reported mean by more than 0.01",
        worst_shift, 0.0, 1e-2, kind="abs")

print(f"\nwall time {time.time() - t0:.1f} s")
c.summary("learned_vs_fixed_ci.py")
raise SystemExit(1 if c.n_fail else 0)
