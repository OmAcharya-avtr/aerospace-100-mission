"""Validation 5: the learned null-motion policy against the classical alternatives.

The classical laws were implemented and validated first (validations 1-4); the
model's labels are the output of a short-horizon lookahead oracle built on top
of them.  This script trains the policy once, on fixed seeds, and reports the
result as it came out.

Benchmark design
----------------
Five configurations run over the same held-out seeded manoeuvre suite:

* ``pinv``      -- Moore-Penrose pseudo-inverse, no null motion (reference for
                   how bad an unregularised inverse gets)
* ``sr``        -- singularity-robust inverse, no null motion (the baseline)
* ``gsr``       -- generalised SR inverse with dither, no null motion
* ``sr+grad``   -- SR inverse plus classical manipulability-gradient null motion
* ``sr+learned``-- SR inverse plus the learned null-motion policy

Differences are reported as paired differences against ``sr`` with a
bootstrap 95% confidence interval; a difference whose interval straddles zero
is reported as indistinguishable rather than as a win.

Run: ``python validation/validate_nullmotion_ml.py``  (about 2 minutes)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cmgsteer.arrays import pyramid_array  # noqa: E402
from cmgsteer.dataset import generate_policy_dataset, manoeuvre_suite  # noqa: E402
from cmgsteer.ml import LearnedNullMotion  # noqa: E402
from cmgsteer.nullmotion import GradientNullMotion, NoNullMotion  # noqa: E402
from cmgsteer.simulate import run_steering  # noqa: E402

TRAIN_SEED = 1234
TEST_SEED = 5678
BENCH_SEED = 9012
BOOTSTRAP_SEED = 4242
N_TRAIN = 900
N_TEST = 300
N_BENCH = 16
MAX_GIMBAL_RATE = 2.0
N_BOOTSTRAP = 10000


def bootstrap_ci(values, rng, level=0.95):
    """Percentile bootstrap confidence interval for the mean of ``values``."""
    values = np.asarray(values, dtype=float)
    idx = rng.integers(0, values.size, size=(N_BOOTSTRAP, values.size))
    means = values[idx].mean(axis=1)
    lo = float(np.percentile(means, 100.0 * (1.0 - level) / 2.0))
    hi = float(np.percentile(means, 100.0 * (1.0 + level) / 2.0))
    return lo, hi


def main() -> int:
    print("=" * 78)
    print("CMGSteer validation 5 -- learned null motion vs classical null motion")
    print("=" * 78)
    array = pyramid_array()
    print(f"array: {array.n_cmgs}-CMG pyramid, capacity {array.total_momentum_capacity} N*m*s")
    print(f"seeds: train {TRAIN_SEED}, test {TEST_SEED}, benchmark {BENCH_SEED}")

    print("\n## 1. Dataset generation")
    t0 = time.perf_counter()
    train = generate_policy_dataset(
        array, N_TRAIN, seed=TRAIN_SEED, horizon=25, n_candidates=9, stride=17, n_manoeuvres=20
    )
    t_train_gen = time.perf_counter() - t0
    t0 = time.perf_counter()
    test = generate_policy_dataset(
        array, N_TEST, seed=TEST_SEED, horizon=25, n_candidates=9, stride=17, n_manoeuvres=10
    )
    t_test_gen = time.perf_counter() - t0
    print(f"train {train.n_samples} samples in {t_train_gen:.2f} s")
    print(f"test  {test.n_samples} samples in {t_test_gen:.2f} s")
    print(f"features {train.features.shape[1]}, horizon {train.horizon} steps, "
          f"candidates {train.candidates.size}")

    print("\n### Oracle headroom on the horizon objective [N*m*s]")
    print(f"{'split':>8} {'k = 0 (plain SR)':>18} {'best candidate':>16} "
          f"{'gradient policy':>17} {'oracle gain':>13}")
    for name, data in (("train", train), ("test", test)):
        gain = 1.0 - data.best_scores.mean() / data.zero_scores.mean()
        print(
            f"{name:>8} {data.zero_scores.mean():>18.6e} {data.best_scores.mean():>16.6e} "
            f"{data.gradient_scores.mean():>17.6e} {gain:>12.2%}"
        )
    grad_gain = 1.0 - test.gradient_scores.mean() / test.zero_scores.mean()
    print(f"classical gradient policy against plain SR on the same objective: {grad_gain:+.2%}")

    print("\n### Label distribution")
    for q in (0, 10, 25, 50, 75, 90, 100):
        print(f"  train percentile {q:>3}: {np.percentile(train.coefficients, q):+.4f}")
    print(f"  train mean {train.coefficients.mean():+.4f}, "
          f"std {train.coefficients.std():.4f}")
    at_edge = float(np.mean(np.abs(train.coefficients) > 0.99))
    print(f"  fraction of labels at a grid edge (|k| > 0.99): {at_edge:.4f}")

    print("\n## 2. Training")
    policy = LearnedNullMotion(
        max_null_rate=0.5,
        n_estimators=5,
        hidden_layer_sizes=(64, 32),
        alpha=1e-4,
        max_iter=400,
        random_state=0,
    )
    t0 = time.perf_counter()
    policy.fit(train.features, train.coefficients)
    t_fit = time.perf_counter() - t0
    print(f"5 x MLPRegressor(64, 32), early stopping, fitted in {t_fit:.2f} s")
    print(f"total dataset + training time: {t_train_gen + t_test_gen + t_fit:.2f} s")

    print("\n## 3. Label-level accuracy on the held-out set")
    pred, spread = policy.predict(test.features)
    conf = policy.confidence(spread)
    resid = pred - test.coefficients
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((test.coefficients - test.coefficients.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    print(f"  mean absolute error   : {np.mean(np.abs(resid)):.6f}")
    print(f"  rms error             : {np.sqrt(np.mean(resid**2)):.6f}")
    print(f"  label standard deviation: {test.coefficients.std():.6f}")
    print(f"  R^2                   : {r2:.6f}")
    print(f"  predicted mean {pred.mean():+.4f}, std {pred.std():.4f}")
    print(f"  sign agreement with the oracle label: "
          f"{np.mean(np.sign(pred) == np.sign(test.coefficients)):.4f}")

    print("\n## 4. Oracle gap at the predicted coefficient")
    print("the horizon score at the predicted k, by linear interpolation on the")
    print("candidate grid, against the best and the k = 0 scores")
    scores_at_pred = np.array(
        [np.interp(p, test.candidates, row) for p, row in zip(pred, test.candidate_scores)]
    )
    zero = test.zero_scores
    best = test.best_scores
    print(f"  mean score at predicted k : {scores_at_pred.mean():.6e} N*m*s")
    print(f"  mean score at k = 0       : {zero.mean():.6e} N*m*s")
    print(f"  mean score at the oracle k: {best.mean():.6e} N*m*s")
    captured = (zero.mean() - scores_at_pred.mean()) / (zero.mean() - best.mean())
    print(f"  fraction of the oracle gain captured: {captured:.2%}")
    print(f"  states where the policy is worse than k = 0: "
          f"{np.mean(scores_at_pred > zero):.2%}")

    print("\n### Oracle gap by confidence decile (lowest confidence first)")
    order = np.argsort(conf)
    deciles = np.array_split(order, 10)
    print(f"{'decile':>7} {'mean confidence':>17} {'mean |label error|':>20} "
          f"{'mean normalised gap':>21}")
    gap = (scores_at_pred - best) / np.maximum(zero, 1e-30)
    for i, idx in enumerate(deciles):
        print(
            f"{i + 1:>7} {conf[idx].mean():>17.6f} {np.mean(np.abs(resid[idx])):>20.6f} "
            f"{gap[idx].mean():>21.6f}"
        )
    corr_conf = float(np.corrcoef(conf, np.abs(resid))[0, 1])
    corr_spread = float(np.corrcoef(spread, np.abs(resid))[0, 1])
    print(f"Pearson r(confidence, |label error|) = {corr_conf:+.4f}")
    print(f"Pearson r(ensemble spread, |label error|) = {corr_spread:+.4f}")
    print(f"mean ensemble spread / rms label error = "
          f"{spread.mean() / np.sqrt(np.mean(resid**2)):.4f}")

    print("\n## 5. Closed-loop benchmark on a held-out manoeuvre suite")
    suite = manoeuvre_suite(
        array, N_BENCH, seed=BENCH_SEED, n_segments=3, segment_duration=6.0, dt=0.02
    )
    print(f"{N_BENCH} manoeuvres x {suite.profiles[0].n_steps} steps, dt "
          f"{suite.profiles[0].dt} s, gimbal-rate limit {MAX_GIMBAL_RATE} rad/s")

    configs = [
        ("pinv", "pinv", NoNullMotion()),
        ("sr", "sr", NoNullMotion()),
        ("gsr", "gsr", NoNullMotion()),
        ("sr+grad", "sr", GradientNullMotion(gain=1.0, max_rate=0.5)),
        ("sr+learned", "sr", policy),
    ]
    results: dict[str, dict[str, np.ndarray]] = {}
    timings: dict[str, float] = {}
    for label, method, null_policy in configs:
        path = []
        net = []
        min_m = []
        rms = []
        sat = []
        t0 = time.perf_counter()
        for profile, start in suite:
            history = run_steering(
                array,
                start,
                profile,
                method=method,
                null_policy=null_policy,
                max_gimbal_rate=MAX_GIMBAL_RATE,
            )
            path.append(history.total_momentum_error_path)
            net.append(history.accumulated_momentum_error)
            min_m.append(history.min_measure)
            rms.append(history.rms_torque_error)
            sat.append(history.n_rate_limited)
        timings[label] = time.perf_counter() - t0
        results[label] = {
            "path": np.array(path),
            "net": np.array(net),
            "min_measure": np.array(min_m),
            "rms": np.array(rms),
            "saturated": np.array(sat, dtype=float),
        }

    print(
        f"\n{'configuration':>12} {'path err [N*m*s]':>18} {'net err [N*m*s]':>17} "
        f"{'min m':>10} {'rms tau err':>13} {'sat steps':>11} {'wall [s]':>10}"
    )
    for label, _, _ in configs:
        r = results[label]
        print(
            f"{label:>12} {r['path'].mean():>18.6e} {r['net'].mean():>17.6e} "
            f"{r['min_measure'].mean():>10.6f} {r['rms'].mean():>13.6e} "
            f"{r['saturated'].mean():>11.2f} {timings[label]:>10.2f}"
        )

    print("\n### Paired differences against the plain SR inverse, "
          "bootstrap 95% confidence intervals")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    base = results["sr"]
    print(
        f"{'configuration':>12} {'metric':>12} {'mean difference':>18} "
        f"{'95% CI':>30} {'wins':>7} {'verdict':>18}"
    )
    verdicts = {}
    for label, _, _ in configs:
        if label == "sr":
            continue
        for metric, better_is_lower in (("path", True), ("net", True), ("min_measure", False)):
            diff = results[label][metric] - base[metric]
            lo, hi = bootstrap_ci(diff, rng)
            wins = int(np.count_nonzero(diff < 0 if better_is_lower else diff > 0))
            if lo <= 0.0 <= hi:
                verdict = "indistinguishable"
            elif (hi < 0.0) == better_is_lower:
                verdict = "better than SR"
            else:
                verdict = "worse than SR"
            verdicts[(label, metric)] = verdict
            print(
                f"{label:>12} {metric:>12} {diff.mean():>18.6e} "
                f"{f'[{lo:+.4e}, {hi:+.4e}]':>30} {f'{wins}/{N_BENCH}':>7} {verdict:>18}"
            )

    print("\n## 6. Runtime per steering step")
    steps_total = N_BENCH * suite.profiles[0].n_steps
    print(f"{'configuration':>12} {'us per step':>14} {'vs sr':>10}")
    sr_us = timings["sr"] / steps_total * 1e6
    for label, _, _ in configs:
        us = timings[label] / steps_total * 1e6
        print(f"{label:>12} {us:>14.1f} {us / sr_us:>10.2f}x")

    print("\n## 7. Verdict")
    print(f"oracle headroom on the horizon objective (test split): "
          f"{1.0 - test.best_scores.mean() / test.zero_scores.mean():.2%}")
    print(f"fraction of that headroom the learned policy captures: {captured:.2%}")
    print(f"closed-loop path-error change, sr+learned vs sr: "
          f"{results['sr+learned']['path'].mean() / base['path'].mean() - 1.0:+.2%} "
          f"({verdicts[('sr+learned', 'path')]})")
    print(f"closed-loop path-error change, sr+grad vs sr:    "
          f"{results['sr+grad']['path'].mean() / base['path'].mean() - 1.0:+.2%} "
          f"({verdicts[('sr+grad', 'path')]})")
    print(f"closed-loop net-error change, sr+learned vs sr:  "
          f"{results['sr+learned']['net'].mean() / base['net'].mean() - 1.0:+.2%} "
          f"({verdicts[('sr+learned', 'net')]})")
    print(f"closed-loop net-error change, sr+grad vs sr:     "
          f"{results['sr+grad']['net'].mean() / base['net'].mean() - 1.0:+.2%} "
          f"({verdicts[('sr+grad', 'net')]})")
    print(f"pinv path error vs sr: "
          f"{results['pinv']['path'].mean() / base['path'].mean() - 1.0:+.2%}")
    print("\nThis is the measured outcome, reported as it came out; see MODEL_CARD.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
