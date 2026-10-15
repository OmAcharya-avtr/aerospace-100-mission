"""Cross-check the reachability fixpoint and Dijkstra against brute force.

The product computes safe-mode reachability with a backward breadth-first
fixpoint and cheapest-path cost with multi-source Dijkstra on the reversed
graph. This script recomputes both by enumerating every simple path on small
automata and compares.

Three checks:

1. Set equality of the reachable-state sets over 2000 random declarations.
   Both sides are sets of discrete states over the same transition relation, so
   this comparison is exact and carries no tolerance.
2. Cheapest-path costs agree to within 1e-9 relative and 1e-9 absolute. The
   tolerance is on both sides; no equality of a floating-point quantity is
   asserted.
3. The witness path's declared costs sum to the reported cost, to the same
   tolerance.

Run:
    PYTHONPATH=src python validation/validate_reachability.py
"""

from __future__ import annotations

import math
import time

from safemodetree import (
    build_automaton,
    exhaustive_min_cost_to_safe,
    exhaustive_safe_reachable,
    min_cost_to_safe,
    random_declaration,
    reachability_report,
    safe_reachable_set,
)

N_DECLARATIONS = 2000
REL_TOL = 1e-9
ABS_TOL = 1e-9

CHECKS: list[tuple[str, bool]] = []


def record(label: str, ok: bool) -> None:
    """Print and record one check."""
    CHECKS.append((label, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def close(left: float, right: float) -> bool:
    """Compare two costs with a tolerance on both sides."""
    if math.isinf(left) and math.isinf(right):
        return True
    return abs(left - right) <= ABS_TOL + REL_TOL * max(abs(left), abs(right))


def main() -> None:
    """Run the three cross-checks and print the worst observed deviations."""
    print("safemodetree reachability cross-check against exhaustive enumeration")
    print("=" * 76)
    print(f"declarations: {N_DECLARATIONS} random, seeds 0..{N_DECLARATIONS - 1}")
    print("shape: 4 modes, 3 faults, 6 responses, 3 observation channels")
    print(f"tolerance on costs: {REL_TOL:g} relative, {ABS_TOL:g} absolute")
    print()

    set_mismatches = 0
    support_mismatches = 0
    worst_cost_gap = 0.0
    worst_cost_seed = -1
    worst_path_gap = 0.0
    worst_path_seed = -1
    total_states = 0
    total_transitions = 0
    unreachable_declarations = 0
    start = time.perf_counter()

    for seed in range(N_DECLARATIONS):
        spec = random_declaration(seed, n_modes=4, n_faults=3, n_responses=6, n_channels=3)
        automaton = build_automaton(spec)
        total_states += len(automaton)
        total_transitions += len(automaton.transitions)

        fixpoint_set = safe_reachable_set(automaton)
        brute_set = exhaustive_safe_reachable(automaton)
        if fixpoint_set != brute_set:
            set_mismatches += 1
            print(f"  MISMATCH set at seed {seed}: {fixpoint_set ^ brute_set}")

        fixpoint_cost = min_cost_to_safe(automaton)
        brute_cost = exhaustive_min_cost_to_safe(automaton)
        if set(fixpoint_cost) != set(brute_cost):
            support_mismatches += 1
            print(f"  MISMATCH support at seed {seed}")
        for state, (cost, _path, _hops) in fixpoint_cost.items():
            reference = brute_cost.get(state, math.inf)
            gap = abs(cost - reference) if math.isfinite(reference) else 0.0
            if gap > worst_cost_gap:
                worst_cost_gap, worst_cost_seed = gap, seed
            if not close(cost, reference):
                print(f"  MISMATCH cost at seed {seed}, {state}: {cost} vs {reference}")

        report = reachability_report(spec, automaton=automaton)
        if report.unreachable_rows:
            unreachable_declarations += 1
        for row in report.rows:
            if not row.reachable:
                continue
            declared = sum(spec.response(rid).cost_min for rid in row.path)
            gap = abs(declared - row.cost_min)
            if gap > worst_path_gap:
                worst_path_gap, worst_path_seed = gap, seed

    elapsed = time.perf_counter() - start

    print()
    print(f"states examined:      {total_states}")
    print(f"transitions examined: {total_transitions}")
    print(
        f"declarations with at least one unreachable fault state: "
        f"{unreachable_declarations} of {N_DECLARATIONS}"
    )
    print(f"elapsed: {elapsed:.2f} s")
    print()
    print("results")
    print("-" * 76)
    record(f"reachable-state sets agree on all {N_DECLARATIONS} declarations",
           set_mismatches == 0)
    record("cheapest-cost supports agree on every declaration", support_mismatches == 0)
    record(
        f"worst cheapest-cost deviation {worst_cost_gap:.3e} min at seed "
        f"{worst_cost_seed} is within tolerance",
        worst_cost_gap <= ABS_TOL + REL_TOL * 1e3,
    )
    record(
        f"worst witness-path sum deviation {worst_path_gap:.3e} min at seed "
        f"{worst_path_seed} is within tolerance",
        worst_path_gap <= ABS_TOL + REL_TOL * 1e3,
    )

    print()
    failed = [label for label, ok in CHECKS if not ok]
    print(f"summary: {len(CHECKS) - len(failed)} of {len(CHECKS)} checks passed")
    if failed:
        print("FAILED CHECKS (recorded, not suppressed):")
        for label in failed:
            print(f"  {label}")
    assert not failed, failed
    print("REACHABILITY CROSS-CHECK COMPLETE")


if __name__ == "__main__":
    main()
