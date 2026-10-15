"""The planted unreachable-safe-mode case must be detected, and localised.

Two planted defects in ``specs/planted_unreachable.yaml``:

1. ``sensor_fail`` has exactly one declared response, into ``degraded``, which
   is declared absorbing and not safe. No declared safe mode is therefore
   reachable from ``(ops, sensor_fail)``.
2. ``orphan_safe`` is declared safe and is the target of no response, so no
   fault state can reach it.

A third planted case is built in memory: the shipped reaction-wheel
declaration with ``safe_mode_entry`` removed, which strands every fault state
in ``fine_pointing`` behind ``survival_entry`` at 480.0 min, except for the one
fault whose own response targets ``sun_safe`` directly. That case checks that
the checker reports the change in the witness rather than a false negative, and
it is also where the first draft of this script predicted the wrong answer:
removing ``safe_mode_entry`` does not strand ``sun_safe``, because
``bat_cell_bypass`` still targets it.

Run:
    PYTHONPATH=src python validation/validate_planted_case.py
"""

from __future__ import annotations

import math

from safemodetree import (
    AutomatonState,
    analyse,
    build_automaton,
    exhaustive_safe_reachable,
    load_spec,
    packaged_spec_path,
    reachability_report,
    safe_reachable_set,
)

CHECKS: list[tuple[str, bool]] = []


def record(label: str, ok: bool) -> None:
    """Print and record one check."""
    CHECKS.append((label, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def section_planted_file() -> None:
    """The shipped planted declaration."""
    spec = load_spec(packaged_spec_path("planted_unreachable"))
    report = analyse(spec)
    print("1. the shipped planted declaration")
    print("-" * 76)
    print(report.report())
    print()
    by_state = {row.state: row for row in report.reachability.rows}
    sensor = by_state[AutomatonState("ops", "sensor_fail")]
    actuator = by_state[AutomatonState("ops", "actuator_fail")]
    record("(ops, sensor_fail) is reported as unable to reach a safe mode",
           not sensor.reachable)
    record("its reported cost is infinite", math.isinf(sensor.cost_min))
    record("(ops, actuator_fail) is still reported as reachable", actuator.reachable)
    record(
        f"its cheapest cost is the declared 11.0 min (measured {actuator.cost_min:.6f})",
        abs(actuator.cost_min - 11.0) <= 1e-9,
    )
    record("orphan_safe is reported as an unreachable declared safe mode",
           report.reachability.unreachable_safe_modes == ("orphan_safe",))
    record("exactly one fault state is reported unreachable",
           len(report.reachability.unreachable_rows) == 1)
    automaton = build_automaton(spec)
    record(
        "exhaustive enumeration agrees with the fixpoint on the planted case",
        safe_reachable_set(automaton) == exhaustive_safe_reachable(automaton),
    )
    print()


def section_planted_in_memory() -> None:
    """The shipped declaration with its safe-mode entry removed."""
    spec = load_spec(packaged_spec_path("reaction_wheel_fmea"))
    stripped = spec.__class__(
        name="reaction_wheel_fmea_without_safe_mode_entry",
        modes=spec.modes,
        faults=spec.faults,
        responses=tuple(r for r in spec.responses if r.id != "safe_mode_entry"),
        conditions=spec.conditions,
        observability=spec.observability,
        initial_mode=spec.initial_mode,
    )
    print("2. the shipped declaration with safe_mode_entry removed")
    print("-" * 76)
    intact = reachability_report(spec)
    broken = reachability_report(stripped)
    print(broken.table())
    print()
    record("the intact declaration reaches a safe mode from every fault state",
           intact.all_reachable)
    record("the stripped declaration still reaches one, through survival_entry",
           broken.all_reachable)
    record(
        f"its worst cost rises from {intact.worst_cost_min:.1f} min to "
        f"{broken.worst_cost_min:.1f} min",
        broken.worst_cost_min > intact.worst_cost_min,
    )
    # Removing safe_mode_entry does NOT strand sun_safe: bat_cell_bypass
    # declares to_mode sun_safe at 60.0 min and clears its own fault, so one
    # route into sun_safe survives. Every other fault state falls back to
    # survival_entry at 480.0 min. The checker reports exactly that, and the
    # first draft of this script expected the wrong answer.
    paths = {row.path for row in broken.rows}
    record(
        f"the surviving witness paths are exactly {sorted(paths)}",
        paths == {("survival_entry",), ("bat_cell_bypass",)},
    )
    record(
        "the one route into sun_safe that survives is bat_cell_bypass at 60.0 min",
        any(
            row.path == ("bat_cell_bypass",)
            and row.safe_mode == "sun_safe"
            and abs(row.cost_min - 60.0) <= 1e-9
            for row in broken.rows
        ),
    )
    record(
        "no declared safe mode becomes unreachable, because bat_cell_bypass remains",
        broken.unreachable_safe_modes == (),
    )
    record(
        "14 of 16 fault states fall back to survival_entry at 480.0 min "
        "(bat_cell_short keeps its own route in both operational modes)",
        sum(1 for row in broken.rows if row.path == ("survival_entry",)) == 14,
    )
    print()


def main() -> None:
    """Run both planted cases."""
    print("safemodetree planted-case detection")
    print("=" * 76)
    print()
    section_planted_file()
    section_planted_in_memory()
    failed = [label for label, ok in CHECKS if not ok]
    print(f"summary: {len(CHECKS) - len(failed)} of {len(CHECKS)} checks passed")
    if failed:
        print("FAILED CHECKS (recorded, not suppressed):")
        for label in failed:
            print(f"  {label}")
    assert not failed, failed
    print("PLANTED-CASE DETECTION COMPLETE")


if __name__ == "__main__":
    main()
