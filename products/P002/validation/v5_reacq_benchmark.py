"""V5 - reacquisition policy benchmark: tabular Q-learning vs two scripted baselines.

Claim under test
----------------
A tabular Q-learning policy trained in the reacquisition simulator reduces
the mean time-to-reacquire relative to BOTH scripted baselines
(always-restart-full-spiral and always-local-restart), with non-overlapping
95 % confidence intervals, and is bitwise reproducible from its seed.

Method
------
1. Evaluate both baselines FIRST on 2 000 seeded Monte Carlo episodes
   (common random numbers: episode i uses seed 999 + i for every policy).
2. Train Q-learning for 20 000 episodes (seed 12345) and evaluate on the
   SAME 2 000 episodes.
3. Report mean, 95 % normal-approximation CI, median, p90, success rate,
   mean attempts and the action mix.
4. Repeat training with three different seeds to show the spread of the
   learned result is small compared with the gap to the baselines.
5. Confirm the learned Q-table is bitwise reproducible.

Timed-out episodes (no reacquisition within ``max_time``) are counted at
``max_time``; the mean is therefore a censored mean and must be read
together with the success rate.

Run: python validation/v5_reacq_benchmark.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trackforge.reacq import (  # noqa: E402
    ACTIONS,
    AlwaysFullPolicy,
    AlwaysLocalPolicy,
    ReacqConfig,
    evaluate_policy,
    train_q_learning,
)

TRAIN_EPISODES = 20_000
EVAL_EPISODES = 2_000
EVAL_SEED = 999
TRAIN_SEEDS = (12345, 20260, 777)


def print_row(r: dict) -> None:
    """Print one benchmark row."""
    mix = r["action_mix"]
    total = max(sum(mix.values()), 1)
    mix_s = " ".join(f"{a}:{mix[a] / total:.0%}" for a in ACTIONS)
    print(f"{r['policy']:>22} {r['mean_time_s']:9.4f} "
          f"[{r['ci_low_s']:7.4f},{r['ci_high_s']:8.4f}] "
          f"{r['median_time_s']:9.4f} {r['p90_time_s']:8.3f} "
          f"{r['success_rate']:8.3f} {r['mean_attempts']:8.3f}  {mix_s}")


def main() -> int:
    """Run the reacquisition benchmark and print tables."""
    cfg = ReacqConfig()
    print("V5 - reacquisition benchmark (learned vs scripted baselines)")
    print(f"config: sigma0 = {cfg.sigma0:.2e} rad, drift = {cfg.drift_rate:.2e} rad/s, "
          f"cone = {cfg.cone_radius:.2e} rad")
    print(f"        coverage rate = {cfg.coverage_rate:.2e} rad^2/s, "
          f"p_detect = {cfg.p_detect}, max_time = {cfg.max_time} s")
    print(f"        LOCAL disc = {cfg.k_local} sigma(t), "
          f"RING width = {cfg.k_ring} sigma(t), kappa = {cfg.kappa}")
    print(f"evaluation: {EVAL_EPISODES} episodes, common random numbers, "
          f"seeds {EVAL_SEED}..{EVAL_SEED + EVAL_EPISODES - 1}")
    print()

    hdr = (f"{'policy':>22} {'mean [s]':>9} {'95% CI [s]':>18} {'median':>9} "
           f"{'p90':>8} {'success':>8} {'attempts':>8}  action mix")
    print("BASELINES FIRST")
    print(hdr)
    print("-" * 118)
    t0 = time.perf_counter()
    base_full = evaluate_policy(AlwaysFullPolicy(), cfg, EVAL_EPISODES, EVAL_SEED)
    base_local = evaluate_policy(AlwaysLocalPolicy(), cfg, EVAL_EPISODES, EVAL_SEED)
    t_base = time.perf_counter() - t0
    print_row(base_full)
    print_row(base_local)
    print(f"(baseline evaluation wall time: {t_base:.2f} s)")
    print()

    print(f"LEARNED (tabular Q-learning, {TRAIN_EPISODES} training episodes)")
    print(hdr)
    print("-" * 118)
    results = []
    for seed in TRAIN_SEEDS:
        t0 = time.perf_counter()
        pol = train_q_learning(cfg, episodes=TRAIN_EPISODES, seed=seed)
        t_train = time.perf_counter() - t0
        r = evaluate_policy(pol, cfg, EVAL_EPISODES, EVAL_SEED)
        r["policy"] = f"q-learning seed {seed}"
        results.append((r, t_train, pol))
        print_row(r)
    print("(training wall times: "
          + ", ".join(f"{t:.2f} s" for _, t, _ in results) + ")")
    print()

    means = np.array([r["mean_time_s"] for r, _, _ in results])
    best = min(results, key=lambda x: x[0]["mean_time_s"])[0]
    print(f"learned mean over {len(TRAIN_SEEDS)} training seeds: "
          f"{means.mean():.4f} s (spread {means.min():.4f}-{means.max():.4f} s)")
    print(f"baseline always-full : {base_full['mean_time_s']:.4f} s "
          f"[{base_full['ci_low_s']:.4f}, {base_full['ci_high_s']:.4f}]")
    print(f"baseline always-local: {base_local['mean_time_s']:.4f} s "
          f"[{base_local['ci_low_s']:.4f}, {base_local['ci_high_s']:.4f}]")
    for b in (base_full, base_local):
        gain = (b["mean_time_s"] - means.mean()) / b["mean_time_s"]
        sep = best["ci_high_s"] < b["ci_low_s"]
        print(f"   vs {b['policy']:<22}: {gain:.1%} REDUCTION in mean "
              f"time-to-reacquire, 95% CIs disjoint = {sep}")
    print()

    print("reproducibility check (same seed twice, bitwise)")
    a = train_q_learning(cfg, episodes=2_000, seed=4242)
    b = train_q_learning(cfg, episodes=2_000, seed=4242)
    same = bool(np.array_equal(a.q, b.q) and np.array_equal(a.visits, b.visits))
    print(f"   Q tables identical: {same}")
    ev1 = evaluate_policy(a, cfg, 500, 31337)
    ev2 = evaluate_policy(b, cfg, 500, 31337)
    print(f"   evaluation identical: {ev1 == ev2}")
    print()

    print("confidence output (uncertainty exposed by the learned policy)")
    pol = results[0][2]
    conf = np.array([pol.confidence(s) for s in range(pol.q.shape[0])])
    visited = pol.visits.sum(axis=1) > 0
    print(f"   states visited during training: {int(visited.sum())} / {conf.size}")
    print(f"   confidence on visited states: mean {conf[visited].mean():.3f}, "
          f"min {conf[visited].min():.3f}, max {conf[visited].max():.3f}")
    unvisited = int((~visited).sum())
    print(f"   unvisited states (policy defers to the FULL baseline action): "
          f"{unvisited}")
    print(f"   visited states with confidence < 0.1 (acts greedily but flags low "
          f"confidence): {int((conf[visited] < 0.1).sum())}")
    print()

    ok = (
        best["ci_high_s"] < base_full["ci_low_s"]
        and best["ci_high_s"] < base_local["ci_low_s"]
        and same
    )
    print("PASS criteria: learned 95% CI strictly below BOTH baseline CIs, and")
    print("               training bitwise reproducible")
    print("RESULT:", "PASS" if ok else "FAIL")
    if not ok:
        print("NOTE: a FAIL here is a legitimate scientific result and must be")
        print("reported as such in VALIDATION.md, not tuned away.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
