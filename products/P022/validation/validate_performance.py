"""Validation 6: performance benchmark.

Wall-clock cost of every operation a user is likely to put in a loop, on the
two-core reference machine.  ``tests/test_performance.py`` asserts loose bounds
on the same quantities so that an algorithmic regression fails the suite; this
script is where the actual numbers live.

Run: ``python validation/validate_performance.py``
"""

from __future__ import annotations

import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cmgsteer.arrays import pyramid_array  # noqa: E402
from cmgsteer.dataset import manoeuvre_suite, rollout_score  # noqa: E402
from cmgsteer.ml import LearnedNullMotion  # noqa: E402
from cmgsteer.nullmotion import (  # noqa: E402
    GradientNullMotion,
    PreferredAngleNullMotion,
    unit_null_vector,
)
from cmgsteer.simulate import rest_to_rest_profile, run_steering  # noqa: E402
from cmgsteer.singularity import (  # noqa: E402
    classify_singularity,
    manipulability_gradient,
    momentum_envelope,
    singularity_measure,
)
from cmgsteer.steering import (  # noqa: E402
    gsr_inverse_steer,
    pseudo_inverse_steer,
    sr_inverse_steer,
)

D = np.array([0.30, -0.50, 0.80, 0.20])
TAU = np.array([0.10, -0.05, 0.20])


def timed(fn, repeats):
    fn()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - start) / repeats * 1e6


def section_1_primitives():
    print("\n## 1. Per-call cost of the primitives (4-CMG pyramid)")
    array = pyramid_array()
    jac = array.jacobian(D)
    rows = [
        ("array.momentum", lambda: array.momentum(D), 5000),
        ("array.jacobian", lambda: array.jacobian(D), 5000),
        ("singularity_measure", lambda: singularity_measure(jac), 5000),
        ("manipulability_gradient", lambda: manipulability_gradient(array, D), 2000),
        ("unit_null_vector", lambda: unit_null_vector(array, D), 2000),
        ("classify_singularity", lambda: classify_singularity(array, D), 2000),
        ("pseudo_inverse_steer", lambda: pseudo_inverse_steer(array, D, TAU), 2000),
        ("sr_inverse_steer", lambda: sr_inverse_steer(array, D, TAU), 2000),
        ("gsr_inverse_steer", lambda: gsr_inverse_steer(array, D, TAU), 2000),
    ]
    print(f"{'operation':>26} {'us per call':>14} {'calls per second':>18}")
    for name, fn, repeats in rows:
        us = timed(fn, repeats)
        print(f"{name:>26} {us:>14.2f} {1e6 / us:>18,.0f}")


def section_2_scaling():
    print("\n## 2. Scaling with the number of CMGs")
    print(f"{'n CMGs':>8} {'jacobian [us]':>15} {'measure [us]':>14} {'sr steer [us]':>15} "
          f"{'gradient [us]':>15}")
    for n in (4, 6, 8, 12, 16):
        array = pyramid_array(n_cmgs=n)
        d = np.linspace(0.1, 1.1, n)
        jac = array.jacobian(d)
        print(
            f"{n:>8} {timed(lambda: array.jacobian(d), 3000):>15.2f} "
            f"{timed(lambda: singularity_measure(jac), 3000):>14.2f} "
            f"{timed(lambda: sr_inverse_steer(array, d, TAU), 1500):>15.2f} "
            f"{timed(lambda: manipulability_gradient(array, d), 1500):>15.2f}"
        )


def section_3_policies():
    print("\n## 3. Per-call cost of the null-motion policies")
    array = pyramid_array()
    data_free = np.zeros(4)
    policies = [
        ("GradientNullMotion", GradientNullMotion(gain=1.0, max_rate=0.5), 1000),
        (
            "PreferredAngleNullMotion",
            PreferredAngleNullMotion(preferred=data_free, gain=1.0),
            1000,
        ),
    ]
    print(f"{'policy':>26} {'us per call':>14}")
    for name, policy, repeats in policies:
        print(f"{name:>26} {timed(lambda: policy.rates(array, D, TAU), repeats):>14.2f}")
    rng = np.random.default_rng(0)
    features = rng.normal(size=(200, 20))
    labels = rng.uniform(-1.0, 1.0, 200)
    learned = LearnedNullMotion(
        n_estimators=5, hidden_layer_sizes=(64, 32), max_iter=60
    ).fit(features, labels)
    print(f"{'LearnedNullMotion':>26} {timed(lambda: learned.rates(array, D, TAU), 200):>14.2f}")
    print("the learned policy is dominated by five scikit-learn predict calls on a")
    print("single sample; it cannot be batched inside a sequential control loop")


def section_4_runs():
    print("\n## 4. Whole-run throughput")
    array = pyramid_array()
    profile = rest_to_rest_profile([0.2, 0.3, 0.93], 1.5, 20.0, 0.02)
    print(f"profile: {profile.n_steps} steps of {profile.dt} s")
    print(f"{'configuration':>26} {'wall [s]':>10} {'us per step':>14} {'steps per second':>18}")
    for label, method, policy in (
        ("pinv", "pinv", None),
        ("sr", "sr", None),
        ("gsr", "gsr", None),
        ("sr + gradient", "sr", GradientNullMotion(gain=1.0, max_rate=0.5)),
    ):
        start = time.perf_counter()
        run_steering(array, np.zeros(4), profile, method=method, null_policy=policy)
        elapsed = time.perf_counter() - start
        print(
            f"{label:>26} {elapsed:>10.3f} {elapsed / profile.n_steps * 1e6:>14.1f} "
            f"{profile.n_steps / elapsed:>18,.0f}"
        )


def section_5_mapping_and_datasets():
    print("\n## 5. Surface mapping and dataset generation")
    array = pyramid_array()
    for n_points in (500, 2000, 8000):
        start = time.perf_counter()
        momenta, _ = momentum_envelope(array, n_points=n_points)
        elapsed = time.perf_counter() - start
        print(
            f"momentum_envelope({n_points:>5}) : {elapsed:>7.3f} s, "
            f"{momenta.shape[0]} points, {elapsed / max(momenta.shape[0], 1) * 1e6:.1f} us/point"
        )
    suite = manoeuvre_suite(array, 1, seed=1, n_segments=1, segment_duration=3.0, dt=0.02)
    profile, start_angles = suite.profiles[0], suite.initial_deltas[0]
    window = profile.torques[:100]
    fast = timed(
        lambda: rollout_score(array, start_angles, window, profile.dt, 0.5, fast=True), 10
    )
    slow = timed(
        lambda: rollout_score(array, start_angles, window, profile.dt, 0.5, fast=False), 10
    )
    print(f"rollout of 100 steps, fused single-SVD path : {fast:>10.1f} us "
          f"({fast / 100:.1f} us/step)")
    print(f"rollout of 100 steps, public path           : {slow:>10.1f} us "
          f"({slow / 100:.1f} us/step)")
    print(f"speedup of the fused path: {slow / fast:.2f}x")


def section_6_memory():
    print("\n## 6. Memory footprint of a recorded run")
    array = pyramid_array()
    for n_steps in (500, 2000, 8000):
        profile = rest_to_rest_profile([0.2, 0.3, 0.93], 1.5, n_steps * 0.02, 0.02)
        history = run_steering(array, np.zeros(4), profile, method="sr")
        total = sum(
            getattr(history, name).nbytes
            for name in (
                "times",
                "deltas",
                "momentum",
                "gimbal_rates",
                "commanded_torque",
                "achieved_torque",
                "torque_error",
                "momentum_error",
                "measure",
                "min_singular_value",
                "lam",
                "rate_limited",
                "null_rates",
            )
        )
        print(
            f"{n_steps:>6} steps: {total / 1024:>9.1f} KiB "
            f"({total / n_steps:.1f} bytes per step)"
        )


def main() -> int:
    print("=" * 78)
    print("CMGSteer validation 6 -- performance benchmark")
    print("=" * 78)
    print(f"{platform.python_version()} on {platform.machine()}, numpy {np.__version__}")
    print("2 CPU cores, no GPU")
    section_1_primitives()
    section_2_scaling()
    section_3_policies()
    section_4_runs()
    section_5_mapping_and_datasets()
    section_6_memory()
    print("\nOVERALL: measurements only, no pass/fail criterion")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
