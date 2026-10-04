"""Performance benchmark (Level 3), with its measurement method stated.

Every number below is a WORKSTATION/CONTAINER measurement.  It is not a
measurement on flight or edge hardware and must not be quoted as one.

Measurement method
------------------
* ``time.perf_counter`` wall clock; its measured resolution is printed.
* Each stage is repeated and the minimum, median and maximum are all shown.
  The build container has ONE CPU core and runs several build agents
  concurrently, so the maximum is dominated by contention; the minimum is the
  closest estimate of the work itself.  The mean is deliberately not reported.
* Peak memory is ``tracemalloc``, which counts Python-level allocations only
  and is therefore a lower bound (the SGP4 C++ propagator's own memory is
  outside it).

Scaling
-------
The contact scan is ``O(n_sat^2 n_t)`` for inter-satellite links.  The script
measures it at three constellation sizes and reports the observed exponent
against the expected 2, so the complexity claim in the README is a measurement
rather than an assertion.
"""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime, timedelta

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from constellink.availability import (  # noqa: E402
    LinkAvailabilityModel,
    LogisticBaseline,
    grouped_split,
)
from constellink.benchmark import BenchmarkResult, benchmark, environment  # noqa: E402
from constellink.constellation import walker_delta  # noqa: E402
from constellink.contacts import all_contact_windows  # noqa: E402
from constellink.flow import ilp_max_flow  # noqa: E402
from constellink.graph import ContactGraph, TimeExpandedGraph  # noqa: E402
from constellink.routing import shortest_route  # noqa: E402
from constellink.synthdata import (  # noqa: E402
    DatasetConfig,
    default_stations,
    generate_dataset,
)

EPOCH = datetime(2026, 4, 1, tzinfo=UTC)
HOURS = 3.0
STEP_S = 60.0
REPEATS = 3


def main() -> int:
    env = environment()
    print("Performance benchmark")
    print("=" * 94)
    print(env.format_block())
    print("")
    print(f"configuration: 24/4/1 Walker at 550 km, {HOURS:g} h horizon, "
          f"{STEP_S:g} s scan step, {len(default_stations())} ground stations")
    print(f"repeats per stage: {REPEATS}")
    print("")

    const = walker_delta(24, 4, 1, 53.0, 550.0, EPOCH, max_epoch_age_days=2.0)
    stations = default_stations()
    t1 = EPOCH + timedelta(hours=HOURS)
    results = []

    results.append(benchmark("ephemeris, 24 sat x 181 epochs",
                             lambda: const.ephemeris(EPOCH, t1, STEP_S),
                             repeats=REPEATS))
    eph = const.ephemeris(EPOCH, t1, STEP_S)
    results.append(benchmark(
        "contact scan, 276 ISL + 120 ground pairs",
        lambda: all_contact_windows(eph, const.satellites, stations),
        repeats=REPEATS))
    windows = all_contact_windows(eph, const.satellites, stations)
    cg = ContactGraph(windows=windows, t0=EPOCH, t1=t1)
    results.append(benchmark("time-expanded unroll, 180 slots",
                             lambda: TimeExpandedGraph.from_contact_graph(
                                 cg, STEP_S, default_rate_bps=1e8),
                             repeats=REPEATS))
    teg = TimeExpandedGraph.from_contact_graph(cg, STEP_S, default_rate_bps=1e8)
    src, dst = "W00-00", "AWARUA"
    results.append(benchmark("dijkstra, one source-destination pair",
                             lambda: shortest_route(teg, src, dst),
                             repeats=REPEATS))
    results.append(benchmark("dijkstra, all 27 nodes to AWARUA",
                             lambda: [shortest_route(teg, n, dst)
                                      for n in cg.nodes if n != dst],
                             repeats=REPEATS))
    # One repeat only: the HiGHS solve is the slowest stage by an order of
    # magnitude and three repeats would push this script past the 3 min
    # per-script compute budget stated in the README.
    results.append(benchmark("ILP max flow (HiGHS), 1 pair, 1 repeat",
                             lambda: ilp_max_flow(teg, src, dst,
                                                  flow_unit_bits=6.0e9),
                             repeats=1))

    print(BenchmarkResult.header())
    print("-" * len(BenchmarkResult.header()))
    for r in results:
        print(r.format_row())
    print("")
    print(f"graph size: {len(cg.nodes)} nodes, {len(windows)} contact windows, "
          f"{teg.n_te_nodes} time-expanded nodes, {len(teg.edges)} edges")
    print("")

    print("Contact-scan scaling in satellite count (expected exponent 2)")
    head = (f"  {'n_sat':>7}{'pairs':>8}{'min [ms]':>12}{'median [ms]':>14}"
            f"{'ms per pair':>14}")
    print(head)
    print("  " + "-" * (len(head) - 2))
    sizes, times = [], []
    for n_sat, n_planes in ((8, 2), (16, 4), (24, 4)):
        c = walker_delta(n_sat, n_planes, 1, 53.0, 550.0, EPOCH,
                         max_epoch_age_days=2.0)
        e = c.ephemeris(EPOCH, t1, STEP_S)
        r = benchmark(f"scan n={n_sat}",
                      lambda c=c, e=e: all_contact_windows(e, c.satellites, []),
                      repeats=REPEATS, measure_memory=False)
        pairs = n_sat * (n_sat - 1) // 2
        sizes.append(n_sat)
        times.append(r.min_s)
        print(f"  {n_sat:>7}{pairs:>8}{r.min_s * 1e3:>12.3f}"
              f"{r.median_s * 1e3:>14.3f}{r.min_s * 1e3 / pairs:>14.4f}")
    exponent = float(np.polyfit(np.log(sizes), np.log(times), 1)[0])
    print(f"  fitted exponent d(log t)/d(log n_sat) = {exponent:.3f} "
          f"(expected 2 for the pair loop; the ground-station leg is linear "
          f"and is excluded from this sweep)")
    print("")

    print("Model training and inference")
    cfg = DatasetConfig(horizon_hours=18.0)
    gen = benchmark("dataset generation, 18 h horizon",
                    lambda: generate_dataset(cfg), repeats=1)
    data = generate_dataset(cfg)
    train, test = grouped_split(data, test_fraction=0.3, seed=0)
    fit_logi = benchmark("logistic baseline fit",
                         lambda: LogisticBaseline().fit(data.x[train],
                                                        data.y[train]),
                         repeats=REPEATS)
    fit_model = benchmark("learned model fit (5 members, sigmoid)",
                          lambda: LinkAvailabilityModel(seed=1).fit(
                              data.x[train], data.y[train]),
                          repeats=1)
    model = LinkAvailabilityModel(seed=1).fit(data.x[train], data.y[train])
    infer = benchmark(f"learned model inference, {test.size} rows",
                      lambda: model.predict_with_uncertainty(data.x[test]),
                      repeats=REPEATS)
    print(BenchmarkResult.header())
    print("-" * len(BenchmarkResult.header()))
    for r in (gen, fit_logi, fit_model, infer):
        print(r.format_row())
    print("")
    print(f"dataset: {len(data)} rows, {train.size} train / {test.size} test")
    print("")

    total_s = sum(r.median_s * r.repeats for r in results) + \
        gen.median_s + fit_logi.median_s * REPEATS + fit_model.median_s + \
        infer.median_s * REPEATS
    print("=" * 94)
    print(f"sum of median x repeats over every timed stage: {total_s:.2f} s")
    print("Every stage is a container measurement; see the environment block "
          "above.")
    print("OVERALL: REPORTED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
