"""Validation 2: ILP vs known optima, and the greedy-vs-ILP gap.

Part A -- hand-solved instances.  Four small instances whose optimum is worked
out by hand (the arithmetic is reproduced in the output and in
tests/test_scheduler.py).  The ILP must reproduce the hand optimum exactly.

Part B -- randomized small instances.  Seeded random instances with n <= 12
are solved by exhaustive enumeration of all feasible subsets (the definition
of the optimum) and compared with the ILP; the greedy gap is tabulated.

Part C -- realistic instances.  Passes of the ISS fixture TLE over an
8-station synthetic network, with the setup (slew/acquisition) time swept to
vary the conflict density.  Greedy is compared with the ILP optimum.

Run: python validation/validate_scheduler.py
"""

from __future__ import annotations

import random
import sys
import time
from datetime import datetime, timedelta, timezone
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from passplanner import (  # noqa: E402
    ClimatologyAvailability,
    Station,
    find_passes,
    schedule_greedy,
    schedule_ilp,
)
from passplanner.fixtures import ISS_2008  # noqa: E402
from passplanner.passes import Pass  # noqa: E402
from passplanner.scheduler import passes_conflict  # noqa: E402

T0 = datetime(2026, 3, 1, tzinfo=timezone.utc)
S1 = Station(name="S1", lat_deg=0.0, lon_deg=0.0, data_rate_gbps=1.0)
S2 = Station(name="S2", lat_deg=10.0, lon_deg=20.0, data_rate_gbps=1.0)


def mk(sat, station, a, b):
    return Pass(sat, station, T0 + timedelta(seconds=a), T0 + timedelta(seconds=b),
                T0 + timedelta(seconds=0.5 * (a + b)), 45.0)


def brute_force(passes, setup_time_s=0.0):
    best, best_set = 0.0, ()
    for k in range(len(passes) + 1):
        for combo in combinations(range(len(passes)), k):
            if any(passes_conflict(passes[i], passes[j], setup_time_s)
                   for i, j in combinations(combo, 2)):
                continue
            val = sum(passes[i].duration_s * passes[i].station.data_rate_gbps for i in combo)
            if val > best:
                best, best_set = val, combo
    return best, best_set


HAND_CASES = [
    ("A greedy trap: 1 station, 1 satellite; A=[0,1000] v1000, B=[0,600] v600, "
     "C=[700,1400] v700",
     [mk("SAT-A", S1, 0, 1000), mk("SAT-A", S1, 0, 600), mk("SAT-A", S1, 700, 1400)],
     0.0, 1300.0,
     "feasible sets: {A}=1000, {B}=600, {C}=700, {B,C}=1300 (B ends 600 < 700 = C rise); "
     "A conflicts with both -> optimum {B,C} = 1300"),
    ("B satellite constraint across stations: D=S1[0,500] v500, E=S2[100,900] v800",
     [mk("SAT-B", S1, 0, 500), mk("SAT-B", S2, 100, 900)],
     0.0, 800.0,
     "same satellite, intervals overlap on [100,500] -> only one may be used; "
     "max(500, 800) = 800"),
    ("C mixed constraints: F=SAT-A@S1[0,400] v400, G=SAT-B@S1[200,900] v700, "
     "H=SAT-A@S2[0,300] v300",
     [mk("SAT-A", S1, 0, 400), mk("SAT-B", S1, 200, 900), mk("SAT-A", S2, 0, 300)],
     0.0, 1000.0,
     "conflicts F-G (station S1, overlap 200-400) and F-H (satellite SAT-A, overlap 0-300); "
     "G-H share nothing -> {G,H} = 700 + 300 = 1000 > {F} = 400"),
    ("D setup time 120 s: I=[0,200] v200, J=[260,500] v240 (60 s raw gap)",
     [mk("SAT-A", S1, 0, 200), mk("SAT-A", S1, 260, 500)],
     120.0, 240.0,
     "padding by 120 s makes [0,320] and [260,620] overlap -> only one; max(200,240) = 240"),
]


def part_a(w):
    w("PART A -- hand-solved instances (rate 1 Gbit/s, so value [Gbit] = duration [s])")
    w("")
    ok = True
    for title, inst, setup, hand_opt, reasoning in HAND_CASES:
        ilp = schedule_ilp(inst, setup_time_s=setup)
        greedy = schedule_greedy(inst, setup_time_s=setup)
        bf, _ = brute_force(inst, setup)
        match = abs(ilp.total_value - hand_opt) < 1e-9 and abs(bf - hand_opt) < 1e-9
        ok = ok and match
        gap = 0.0 if hand_opt == 0 else 100.0 * (hand_opt - greedy.total_value) / hand_opt
        w(f"  {title}")
        w(f"    hand solution : {reasoning}")
        w(f"    hand optimum  : {hand_opt:.1f} Gbit")
        w(f"    brute force   : {bf:.1f} Gbit")
        w(f"    ILP           : {ilp.total_value:.1f} Gbit  "
          f"({len(ilp.selected)} pass(es))  -> {'MATCH' if match else 'MISMATCH'}")
        w(f"    greedy        : {greedy.total_value:.1f} Gbit  (gap {gap:.4f} %)")
        w("")
    return ok


def part_b(w):
    w("PART B -- randomized instances vs exhaustive enumeration (seed 20260306)")
    w("")
    rng = random.Random(20260306)
    w(f"  {'#':>3} {'n':>3} {'brute [Gbit]':>13} {'ILP [Gbit]':>12} {'greedy [Gbit]':>14} "
      f"{'gap [%]':>9}")
    ok = True
    gaps = []
    for case in range(20):
        n = rng.randint(5, 12)
        passes = []
        for _ in range(n):
            start = rng.uniform(0.0, 4000.0)
            passes.append(mk(rng.choice(["SAT-A", "SAT-B"]), rng.choice([S1, S2]),
                             start, start + rng.uniform(100.0, 1200.0)))
        bf, _ = brute_force(passes)
        ilp = schedule_ilp(passes)
        greedy = schedule_greedy(passes)
        gap = 0.0 if bf == 0 else 100.0 * (bf - greedy.total_value) / bf
        gap = 0.0 if abs(gap) < 1e-9 else gap
        gaps.append(gap)
        if abs(ilp.total_value - bf) > 1e-6:
            ok = False
        w(f"  {case:>3} {n:>3} {bf:>13.1f} {ilp.total_value:>12.1f} "
          f"{greedy.total_value:>14.1f} {gap:>9.3f}")
    w("")
    w(f"  ILP == exhaustive optimum on all {len(gaps)} instances: {ok}")
    w(f"  greedy gap: mean {sum(gaps) / len(gaps):.3f} %, max {max(gaps):.3f} %, "
      f"instances with gap > 0: {sum(1 for g in gaps if g > 1e-9)}/{len(gaps)}")
    w("")
    return ok


def part_c(w):
    w("PART C -- realistic instances: ISS fixture TLE over an 8-station synthetic network")
    w("  (stations are fictional placeholders; monthly clear-sky priors 0.50-0.64)")
    w("")
    stations = [
        Station(name=f"GS{i}", lat_deg=-60.0 + 15.0 * i, lon_deg=-180.0 + 40.0 * i,
                alt_km=1.0, min_elevation_deg=10.0, data_rate_gbps=10.0,
                monthly_clear_prob=tuple([round(0.50 + 0.02 * i, 2)] * 12))
        for i in range(8)]
    t0 = datetime(2008, 9, 20, 12, 0, tzinfo=timezone.utc)
    passes = []
    for st in stations:
        passes.extend(find_passes(ISS_2008, st, t0, t0 + timedelta(days=1)))
    passes.sort(key=lambda p: p.t_rise)
    availability = ClimatologyAvailability.from_stations(stations)
    w(f"  candidate passes in 24 h: {len(passes)}")
    w("")
    w(f"  {'setup [s]':>10} {'conflicts':>10} {'greedy [Gbit]':>14} {'ILP [Gbit]':>12} "
      f"{'gap [%]':>9} {'ILP t [s]':>10}")
    ok = True
    for setup in (0.0, 300.0, 600.0, 1200.0, 1800.0, 3600.0):
        n_conf = sum(1 for i, j in combinations(range(len(passes)), 2)
                     if passes_conflict(passes[i], passes[j], setup))
        greedy = schedule_greedy(passes, availability, setup_time_s=setup)
        t_a = time.perf_counter()
        ilp = schedule_ilp(passes, availability, setup_time_s=setup)
        dt = time.perf_counter() - t_a
        gap = 0.0 if ilp.total_value == 0 else 100.0 * (
            ilp.total_value - greedy.total_value) / ilp.total_value
        if greedy.total_value > ilp.total_value + 1e-6:
            ok = False
        w(f"  {setup:>10.0f} {n_conf:>10d} {greedy.total_value:>14.1f} "
          f"{ilp.total_value:>12.1f} {gap:>9.3f} {dt:>10.2f}")
    w("")
    return ok


def main() -> int:
    lines = []
    w = lines.append
    w("PassPlanner validation 2 -- scheduler optimality and greedy gap")
    w(f"run: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    w("solver: PuLP + CBC (bundled)")
    w("")
    ok_a = part_a(w)
    ok_b = part_b(w)
    ok_c = part_c(w)
    verdict = "PASS" if (ok_a and ok_b and ok_c) else "FAIL"
    w(f"VERDICT: {verdict}  (A hand optima {ok_a}, B exhaustive {ok_b}, "
      f"C greedy <= ILP {ok_c})")
    text = "\n".join(lines)
    print(text)
    (Path(__file__).parent / "validate_scheduler_output.txt").write_text(text + "\n")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
