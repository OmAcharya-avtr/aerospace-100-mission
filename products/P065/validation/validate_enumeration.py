"""Enumeration completeness cross-checked against exhaustive enumeration.

The grid has ``4n`` cells for ``n`` control actions, so for small ``n`` the
whole power set of filled cells can be enumerated and the checker's answer
compared against a brute-force count that shares no code with it.

Two independent cross-checks:

1. **Exhaustive, n = 1 and n = 2.** Every subset of the ``4n`` cells is filled
   with a UCA and every subset of the remainder with an exclusion, and the
   reported ``(covered, total)`` is compared with ``len(subset_union)``.
2. **Seeded random, n up to 8.** 2000 random fill patterns, with the covered
   count compared against a set union computed directly from the pattern.
"""

from __future__ import annotations

import itertools
import random

from _support import Emitter

from hazardtrace import (
    UCA_CLASSES,
    Analysis,
    Component,
    ComponentKind,
    ControlAction,
    ControlStructure,
    Exclusion,
    Uca,
    build_grid,
    run_checks,
)

GOOD_JUSTIFICATION = "a justification long enough to satisfy the twenty-character policy"
SEED = 20261015


def structure(n_actions: int) -> ControlStructure:
    """A control structure with ``n_actions`` actions on one controller."""
    return ControlStructure(
        components=(
            Component(id="CTRL-1", name="c", kind=ComponentKind.CONTROLLER),
            Component(id="ACT-1", name="a", kind=ComponentKind.ACTUATOR),
            Component(id="PROC-1", name="p", kind=ComponentKind.PROCESS),
        ),
        control_actions=tuple(
            ControlAction(
                id=f"CA-{i}", name=f"action {i}", controller="CTRL-1",
                actuator="ACT-1", process="PROC-1",
            )
            for i in range(n_actions)
        ),
    )


def build(n_actions: int, uca_cells: set[int], exclusion_cells: set[int]) -> Analysis:
    """Build an analysis whose filled cells are the given flat cell indices."""
    def cell(index: int) -> tuple[str, int]:
        return (f"CA-{index // 4}", index % 4)

    return Analysis(
        title="generated",
        structure=structure(n_actions),
        ucas=tuple(
            Uca(
                id=f"U-{index}", control_action=cell(index)[0],
                uca_class=UCA_CLASSES[cell(index)[1]], description="d",
            )
            for index in sorted(uca_cells)
        ),
        exclusions=tuple(
            Exclusion(
                control_action=cell(index)[0], uca_class=UCA_CLASSES[cell(index)[1]],
                justification=GOOD_JUSTIFICATION,
            )
            for index in sorted(exclusion_cells)
        ),
    )


def main() -> int:
    """Run both cross-checks. Returns a process exit code."""
    emit = Emitter(__file__)
    emit("hazardtrace validation: UCA enumeration completeness")
    emit()
    emit("Reference: STPA Handbook (Leveson and Thomas, 2018), step 3 - each")
    emit("control action is considered in four classes, so the grid is 4n cells")
    emit("and completeness is a count against a denominator known in advance.")
    emit()

    # -- 1. exhaustive ----------------------------------------------------
    emit("A. exhaustive cross-check")
    total_cases = 0
    mismatches = 0
    gap_mismatches = 0
    for n_actions in (1, 2):
        cells = list(range(4 * n_actions))
        for uca_size in range(len(cells) + 1):
            for uca_cells in itertools.combinations(cells, uca_size):
                remaining = [c for c in cells if c not in uca_cells]
                for excl_size in range(len(remaining) + 1):
                    for excl_cells in itertools.combinations(remaining, excl_size):
                        analysis = build(n_actions, set(uca_cells), set(excl_cells))
                        grid = build_grid(analysis)
                        expected_total = 4 * n_actions
                        expected_covered = len(set(uca_cells) | set(excl_cells))
                        total_cases += 1
                        if grid.completeness() != (expected_covered, expected_total):
                            mismatches += 1
                        report = run_checks(analysis)
                        ht001 = sum(1 for f in report.findings if f.code == "HT001")
                        if ht001 != expected_total - expected_covered:
                            gap_mismatches += 1
        emit(f"   n={n_actions}: {4 * n_actions} cells, cumulative cases {total_cases}")
    emit(f"   cases enumerated        : {total_cases}")
    emit(f"   completeness mismatches : {mismatches}")
    emit(f"   HT001-count mismatches  : {gap_mismatches}")
    emit.check("exhaustive completeness matches brute force", mismatches == 0)
    emit.check("HT001 count equals uncovered-cell count", gap_mismatches == 0)
    emit()

    # -- 2. seeded random -------------------------------------------------
    emit("B. seeded random cross-check")
    rng = random.Random(SEED)
    trials = 2000
    bad_total = 0
    bad_covered = 0
    bad_partition = 0
    bad_determinism = 0
    for _ in range(trials):
        n_actions = rng.randint(0, 8)
        cells = list(range(4 * n_actions))
        rng.shuffle(cells)
        split = rng.randint(0, len(cells))
        uca_cells = set(cells[:split])
        exclusion_cells = set(rng.sample(cells, rng.randint(0, len(cells))))
        analysis = build(n_actions, uca_cells, exclusion_cells)
        grid = build_grid(analysis)
        if grid.total != 4 * n_actions:
            bad_total += 1
        if grid.covered != len(uca_cells | exclusion_cells):
            bad_covered += 1
        if grid.covered + len(grid.gaps) != grid.total:
            bad_partition += 1
        if run_checks(analysis).findings != run_checks(analysis).findings:
            bad_determinism += 1
    emit(f"   trials                  : {trials}  (seed {SEED})")
    emit("   actions per trial       : 0 to 8")
    emit(f"   wrong cell totals       : {bad_total}")
    emit(f"   wrong covered counts    : {bad_covered}")
    emit(f"   broken partitions       : {bad_partition}")
    emit(f"   non-deterministic runs  : {bad_determinism}")
    emit.check("grid size is always 4n", bad_total == 0)
    emit.check("covered count always equals the filled-cell union", bad_covered == 0)
    emit.check("covered plus gaps always equals the total", bad_partition == 0)
    emit.check("run_checks is deterministic", bad_determinism == 0)
    return emit.finish()


if __name__ == "__main__":
    raise SystemExit(main())
