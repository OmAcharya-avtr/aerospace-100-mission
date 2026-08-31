"""Validation 5: the learned allocator against the exact QP it imitates.

The classical allocators were built and validated first (validations 1-4);
this script is the measurement of what the learned surrogate buys and what it
costs. Three axes, all on the same held-out data:

* **allocation error** -- the 2-norm of the torque the command actually
  produces minus the torque asked for;
* **constraint satisfaction** -- the fraction of predictions that respect the
  actuator bounds, and the size of the violations when they do not;
* **runtime** -- per-allocation wall clock, single-solve and batched.

plus the confidence output's usefulness as an error predictor.

Everything is trained and evaluated once, on fixed seeds, and reported as it
came out. Run: ``python validation/validate_ml_vs_qp.py``
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alloclab.allocation import qp_allocate  # noqa: E402
from alloclab.dataset import (  # noqa: E402
    generate_dataset,
    reference_thruster_cluster,
    torque_scale,
)
from alloclab.ml import LearnedAllocator  # noqa: E402

TRAIN_SEED = 1234
TEST_SEED = 5678
N_TRAIN = 4000
N_TEST = 2000
MODEL_SEED = 0
N_ESTIMATORS = 5
HIDDEN = (64, 48)
MAX_ITER = 300
BOUND_TOL = 1e-9


def summarize(name, err, viol, tag=""):
    ok = float(np.mean(viol <= BOUND_TOL))
    print(
        f"{name:<26}{np.mean(err):>14.6e}{np.median(err):>14.6e}"
        f"{np.percentile(err, 95):>14.6e}{np.max(err):>14.6e}{ok * 100:>12.2f}{tag:>8}"
    )


def main() -> None:
    eset = reference_thruster_cluster(1.0, 0.5)
    scale = torque_scale(eset)
    print("=" * 96)
    print("VALIDATION 5: learned allocator versus the exact QP")
    print("=" * 96)
    print("effector set        : reference thruster cluster, m=8, u in [0,1] N, arm 0.5 m")
    print(f"AMS boundary radius : {scale:.6f} N*m")
    print(f"train / test        : {N_TRAIN} / {N_TEST} samples, seeds {TRAIN_SEED} / {TEST_SEED}")
    print(f"model               : {N_ESTIMATORS} x MLPRegressor{HIDDEN}, max_iter={MAX_ITER}, "
          f"random_state={MODEL_SEED}")
    print("labels              : qp_allocate(u_pref=0, gamma=1e12), i.e. minimum total thrust")

    t0 = time.perf_counter()
    train = generate_dataset(eset, N_TRAIN, seed=TRAIN_SEED)
    t_train_data = time.perf_counter() - t0
    t0 = time.perf_counter()
    test = generate_dataset(eset, N_TEST, seed=TEST_SEED)
    t_test_data = time.perf_counter() - t0

    t0 = time.perf_counter()
    model = LearnedAllocator(
        eset,
        n_estimators=N_ESTIMATORS,
        hidden_layer_sizes=HIDDEN,
        max_iter=MAX_ITER,
        random_state=MODEL_SEED,
    ).fit(train.torques, train.health, train.commands)
    t_fit = time.perf_counter() - t0

    print("\n--- compute budget (2 CPU cores, no GPU) ---")
    print(f"  training-set generation ({N_TRAIN} QP solves) : {t_train_data:7.2f} s")
    print(f"  test-set generation ({N_TEST} QP solves)      : {t_test_data:7.2f} s")
    print(f"  model fit ({N_ESTIMATORS} MLPs)                        : {t_fit:7.2f} s")
    print(f"  total                                        : "
          f"{t_train_data + t_test_data + t_fit:7.2f} s")
    print(f"  train attainable fraction : {train.attainable_fraction:.4f}")
    print(f"  test  attainable fraction : {test.attainable_fraction:.4f}")

    # ---- predictions -------------------------------------------------
    t0 = time.perf_counter()
    raw = model.predict(test.torques, test.health, clip=False)
    t_ml_batch = time.perf_counter() - t0
    clipped_cmd = eset.clip(raw.commands)

    t0 = time.perf_counter()
    for i in range(200):
        model.predict(test.torques[i : i + 1], test.health[i : i + 1])
    t_ml_single = (time.perf_counter() - t0) / 200

    # Build the degraded effector sets outside the timed loop so the QP timing
    # measures the solve, not the bookkeeping a real flight system would have
    # done once when the failure was declared.
    degraded_sets = [
        eset.with_failures(np.flatnonzero(test.health[i] == 0.0)) for i in range(N_TEST)
    ]
    qp_cmds = np.zeros_like(raw.commands)
    t0 = time.perf_counter()
    for i in range(N_TEST):
        d = degraded_sets[i]
        qp_cmds[i] = qp_allocate(d, test.torques[i], u_pref=d.lower).commands
    t_qp_total = time.perf_counter() - t0
    t_qp_single = t_qp_total / N_TEST

    err_qp = np.linalg.norm(test.torques - qp_cmds @ eset.matrix.T, axis=1)
    err_ml = np.linalg.norm(test.torques - raw.commands @ eset.matrix.T, axis=1)
    err_ml_clip = np.linalg.norm(test.torques - clipped_cmd @ eset.matrix.T, axis=1)
    viol_qp = eset.bound_violation(qp_cmds)
    viol_ml = eset.bound_violation(raw.commands)
    viol_clip = eset.bound_violation(clipped_cmd)

    print("\n" + "=" * 96)
    print("5a: allocation error [N*m] and constraint satisfaction, all 2000 test samples")
    print("=" * 96)
    print(
        f"{'allocator':<26}{'mean':>14}{'median':>14}{'p95':>14}{'max':>14}{'% in bounds':>12}"
    )
    summarize("exact QP", err_qp, viol_qp)
    summarize("learned (raw)", err_ml, viol_ml)
    summarize("learned (clipped)", err_ml_clip, viol_clip)

    print("\n--- restricted to the 'attainable' samples (QP meets the command exactly) ---")
    att = test.attainable
    print(
        f"{'allocator':<26}{'mean':>14}{'median':>14}{'p95':>14}{'max':>14}{'% in bounds':>12}"
    )
    summarize("exact QP", err_qp[att], viol_qp[att])
    summarize("learned (raw)", err_ml[att], viol_ml[att])
    summarize("learned (clipped)", err_ml_clip[att], viol_clip[att])

    print("\n--- bound violations of the raw learned output ---")
    over = viol_ml > BOUND_TOL
    print(f"  samples violating at least one bound : {int(over.sum())} / {N_TEST} "
          f"({100.0 * over.mean():.2f}%)")
    print(f"  max violation                        : {viol_ml.max():.6e} N "
          f"({100.0 * viol_ml.max() / 1.0:.2f}% of max thrust)")
    print(f"  mean violation over violating samples: "
          f"{viol_ml[over].mean() if over.any() else 0.0:.6e} N")
    per_eff = np.maximum(
        np.maximum(eset.lower - raw.commands, raw.commands - eset.upper), 0.0
    )
    print("  per-effector max violation [N]       : "
          f"{np.array2string(per_eff.max(axis=0), precision=4)}")
    print(f"  QP violating samples                 : "
          f"{int((viol_qp > BOUND_TOL).sum())} / {N_TEST}")

    print("\n" + "=" * 96)
    print("5b: runtime")
    print("=" * 96)
    print(f"  exact QP, single solve                    : {t_qp_single * 1e6:9.1f} us")
    print(f"  learned, single solve (batch of 1)        : {t_ml_single * 1e6:9.1f} us")
    print(f"  learned, batched {N_TEST} at once           : "
          f"{t_ml_batch / N_TEST * 1e6:9.1f} us per sample")
    print(f"  speed-up, single-solve                    : "
          f"{t_qp_single / t_ml_single:9.2f} x")
    print(f"  speed-up, batched                         : "
          f"{t_qp_single / (t_ml_batch / N_TEST):9.2f} x")

    print("\n" + "=" * 96)
    print("5c: error by number of failed effectors")
    print("=" * 96)
    n_failed = (test.health == 0.0).sum(axis=1)
    print(f"{'n failed':<10}{'count':>8}{'QP mean err':>16}{'ML mean err':>16}"
          f"{'ML/QP':>10}{'ML % in bounds':>16}")
    for k in sorted(set(n_failed.tolist())):
        sel = n_failed == k
        qp_m = float(np.mean(err_qp[sel]))
        ml_m = float(np.mean(err_ml[sel]))
        ratio = ml_m / qp_m if qp_m > 0 else float("inf")
        ok = 100.0 * float(np.mean(viol_ml[sel] <= BOUND_TOL))
        print(f"{k:<10}{int(sel.sum()):>8}{qp_m:>16.6e}{ml_m:>16.6e}{ratio:>10.1f}{ok:>16.2f}")

    print("\n" + "=" * 96)
    print("5d: is the confidence output useful?")
    print("=" * 96)
    conf = raw.confidence
    spread = raw.std.mean(axis=1)
    r_conf = float(np.corrcoef(conf, err_ml)[0, 1])
    r_spread = float(np.corrcoef(spread, err_ml)[0, 1])
    print(f"  Pearson r(confidence, allocation error)   : {r_conf:+.4f}")
    print(f"  Pearson r(ensemble spread, error)         : {r_spread:+.4f}")
    print(f"  mean ensemble spread / rms command error  : "
          f"{spread.mean() / np.sqrt(np.mean((raw.commands - qp_cmds) ** 2)):.4f}")
    print(f"\n{'confidence decile':<20}{'count':>8}{'mean error':>16}{'% in bounds':>14}")
    order = np.argsort(conf)
    for d in range(10):
        sel = order[d * N_TEST // 10 : (d + 1) * N_TEST // 10]
        print(
            f"{d + 1:<20}{len(sel):>8}{float(np.mean(err_ml[sel])):>16.6e}"
            f"{100.0 * float(np.mean(viol_ml[sel] <= BOUND_TOL)):>14.2f}"
        )

    print("\n" + "=" * 96)
    print("5e: verdict")
    print("=" * 96)
    ratio = float(np.mean(err_ml)) / float(np.mean(err_qp))
    print(f"  mean allocation error, learned / QP       : {ratio:.2f} x")
    print(f"  learned predictions inside actuator bounds: "
          f"{100.0 * float(np.mean(viol_ml <= BOUND_TOL)):.2f} %")
    print(f"  QP predictions inside actuator bounds     : "
          f"{100.0 * float(np.mean(viol_qp <= BOUND_TOL)):.2f} %")
    print(f"  single-solve speed-up                     : {t_qp_single / t_ml_single:.2f} x")
    print("\n  The QP wins on accuracy and on constraint satisfaction; the learned")
    print("  allocator wins on runtime only when its output is batched. The exact")
    print("  numbers above are the claim -- see MODEL_CARD.md sec. 7-9.")


if __name__ == "__main__":
    main()
