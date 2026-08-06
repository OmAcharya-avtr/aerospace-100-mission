"""V6 - performance benchmark (simulation throughput and compute budget).

Reports the wall-clock cost of every expensive operation in the package on
the build machine, so the 3-minute mission compute budget can be checked and
so the loose thresholds in tests/test_performance.py can be justified.

Run: python validation/v6_performance.py
"""

from __future__ import annotations

import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import scipy  # noqa: E402

from trackforge.dynamics import JitterPSD, synthesize_jitter  # noqa: E402
from trackforge.reacq import (  # noqa: E402
    AlwaysLocalPolicy,
    ReacqConfig,
    evaluate_policy,
    train_q_learning,
)
from trackforge.scan import (  # noqa: E402
    GaussianUncertainty,
    coverage_fraction,
    spiral_scan,
)
from trackforge.sim import (  # noqa: E402
    DEFAULT_SCENARIO,
    Scenario,
    run_episode,
    run_monte_carlo,
    sim_steps_per_second,
)


def timed(label: str, fn, *args, **kwargs) -> tuple[str, float, object]:
    """Run ``fn`` once and return (label, wall time [s], result)."""
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return label, time.perf_counter() - t0, out


def main() -> int:
    """Run the performance benchmark and print a table."""
    print("V6 - performance benchmark")
    print(f"python {platform.python_version()}, numpy {np.__version__}, "
          f"scipy {scipy.__version__}")
    print(f"platform: {platform.platform()}, machine {platform.machine()}")
    print(f"CPU count (os.cpu_count): {__import__('os').cpu_count()}")
    print()

    print("A) closed-loop tracking throughput (sim.sim_steps_per_second)")
    print(f"   {'sim duration [s]':>18} {'steps':>9} {'wall [s]':>10} "
          f"{'steps/s':>12} {'x realtime':>11}")
    rates = []
    for dur in (0.2, 1.0, 4.0):
        p = sim_steps_per_second(DEFAULT_SCENARIO, duration=dur)
        rates.append(p["steps_per_second"])
        print(f"   {dur:18.1f} {p['steps']:9d} {p['wall_time_s']:10.4f} "
              f"{p['steps_per_second']:12.0f} {p['realtime_factor']:11.2f}")
    print(f"   median throughput: {np.median(rates):.0f} steps/s")
    print()

    print("B) component wall times (single run each)")
    u = GaussianUncertainty(3e-4)
    rows = [
        timed("spiral_scan (9987 pts)", spiral_scan, u, 2e-5),
        timed("coverage_fraction (2e5 samples)", coverage_fraction,
              spiral_scan(u, 2e-5), u, 200_000, np.random.default_rng(0)),
        timed("synthesize_jitter (2^20 samples)", synthesize_jitter,
              JitterPSD(1e-12, 3.0), 2**20, 5000.0, np.random.default_rng(0)),
        timed("run_episode (default scenario)", run_episode, DEFAULT_SCENARIO,
              seed=1, keep_series=False),
        timed("run_monte_carlo (20 episodes)", run_monte_carlo,
              Scenario(track_duration=1.0, spike_time=0.5), n_episodes=20),
        timed("train_q_learning (20 000 ep)", train_q_learning,
              ReacqConfig(), episodes=20_000, seed=12345),
        timed("evaluate_policy (2 000 ep)", evaluate_policy,
              AlwaysLocalPolicy(), ReacqConfig(), 2_000, 999),
    ]
    print(f"   {'operation':>34} {'wall [s]':>10}")
    total = 0.0
    for label, dt, _ in rows:
        total += dt
        print(f"   {label:>34} {dt:10.4f}")
    print(f"   {'TOTAL':>34} {total:10.4f}")
    print()

    print("C) compute-budget check (mission limit: 3 min = 180 s per training")
    print("   or Monte Carlo run)")
    train_t = next(dt for label, dt, _ in rows if label.startswith("train_q"))
    eval_t = next(dt for label, dt, _ in rows if label.startswith("evaluate_policy"))
    print(f"   Q-learning training (20 000 episodes): {train_t:.2f} s "
          f"({train_t / 180:.1%} of budget)")
    print(f"   Monte Carlo evaluation (2 000 episodes): {eval_t:.2f} s "
          f"({eval_t / 180:.1%} of budget)")
    print(f"   full V5 benchmark (3 trainings + 5 evaluations): "
          f"~{3 * train_t + 5 * eval_t:.1f} s")
    print()

    ok = train_t < 180.0 and eval_t < 180.0 and np.median(rates) > 5_000
    print("PASS criteria: training < 180 s, MC evaluation < 180 s, "
          "throughput > 5 000 steps/s")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
