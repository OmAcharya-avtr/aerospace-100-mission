"""Validation V3: performance benchmark (Monte Carlo throughput, runtimes).

Measures, on the machine actually running this script:
  - Monte Carlo sampling throughput (samples/s) at several sample counts
  - end-to-end scenario report runtime
  - surrogate prediction throughput vs equivalent Monte Carlo cost

Run: python validation/v3_performance.py
"""

from __future__ import annotations

import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from beamtwin.budget import LinkParams  # noqa: E402
from beamtwin.channel import ChannelParams, sample_received_power_dbm  # noqa: E402
from beamtwin.scenario import load_scenario, run_twin  # noqa: E402
from beamtwin.surrogate import FadeSurrogate, default_model_path, features_from_params  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPEATS = 3


def _timeit(fn, repeats: int = REPEATS) -> float:
    """Best-of-`repeats` wall time in seconds."""
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> int:
    link = LinkParams(range_m=10_000.0, attenuation_db_per_km=2.5, rx_sensitivity_dbm=-30.0)
    channel = ChannelParams(cn2=5e-16, pointing_jitter_rad=5e-6)

    lines = [
        "V3 — Performance benchmark",
        "=" * 62,
        f"platform: {platform.platform()}",
        f"python:   {platform.python_version()}   numpy: {np.__version__}",
        f"cpu_count: {__import__('os').cpu_count()}",
        f"timing: best of {REPEATS} runs",
        "",
        "Monte Carlo sampling throughput",
        f"{'n_samples':>12}  {'wall_s':>9}  {'samples/s':>14}",
    ]
    throughputs = []
    for n in (10_000, 100_000, 1_000_000):
        dt = _timeit(lambda n=n: sample_received_power_dbm(link, channel, n_samples=n, seed=1))
        tp = n / dt
        throughputs.append(tp)
        lines.append(f"{n:>12,}  {dt:>9.4f}  {tp:>14,.0f}")

    peak = max(throughputs)
    lines += ["", f"peak throughput: {peak:,.0f} samples/s", ""]

    # End-to-end scenario report
    scenario_path = ROOT / "examples" / "link_10km.yaml"
    scenario = load_scenario(scenario_path)
    dt_report = _timeit(lambda: run_twin(scenario))
    lines += [
        "End-to-end scenario report (examples/link_10km.yaml, "
        f"n={scenario.n_samples:,} MC samples)",
        f"  wall time: {dt_report:.4f} s",
        "",
    ]

    # Surrogate vs Monte Carlo cost per query
    model_path = default_model_path()
    if model_path.exists():
        surrogate = FadeSurrogate.load(model_path)
        feats = np.tile(features_from_params(link, channel), (1000, 1))
        dt_sur = _timeit(lambda: surrogate.predict_log10(feats))
        per_query_sur = dt_sur / 1000.0
        dt_mc = _timeit(
            lambda: sample_received_power_dbm(link, channel, n_samples=100_000, seed=1)
        )
        lines += [
            "Surrogate vs Monte Carlo, per fade-probability query",
            f"  surrogate (batch of 1000):     {per_query_sur * 1e6:>10.2f} us/query "
            f"({1.0 / per_query_sur:,.0f} queries/s)",
            f"  Monte Carlo (1e5 samples):     {dt_mc * 1e6:>10.2f} us/query "
            f"({1.0 / dt_mc:,.0f} queries/s)",
            f"  speed-up: {dt_mc / per_query_sur:,.0f}x",
            "",
        ]
    else:
        lines += ["Surrogate model not found; skipping surrogate timing.", ""]

    # Requirement R13: 1e5-sample scenario report must complete in < 5 s.
    bound_ok = dt_report < 5.0
    lines += [
        f"R13 bound: end-to-end report < 5 s  ->  measured {dt_report:.4f} s  "
        f"{'PASS' if bound_ok else 'FAIL'}",
        f"STATUS: {'PASS' if bound_ok else 'FAIL'}",
    ]
    text = "\n".join(lines)
    print(text)
    (Path(__file__).parent / "v3_performance.txt").write_text(text + "\n", encoding="utf-8")
    return 0 if bound_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
