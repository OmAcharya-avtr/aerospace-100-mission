"""The graph and set identities the checks rest on, measured over random cases.

The quantities here are counts over finite sets, so these identities are exact
statements in integer arithmetic and are checked as equalities. The only
floating-point quantities in the package are the coverage ratios, and the one
identity involving them is checked with a tolerance on both sides.

One claim searched here was FALSIFIED and is reported as such: the total
finding count is not monotone when a cell is filled. The counterexample is
pinned in ``tests/test_regression.py`` and the fact is documented in the
README under Limitations.
"""

from __future__ import annotations

import math
import random

from _support import Emitter

from hazardtrace import (
    UCA_CLASSES,
    Analysis,
    CellState,
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
TRIALS = 1500
TOL = 1e-12


def random_analysis(rng: random.Random) -> Analysis:
    """A random analysis with 0 to 6 control actions and random filled cells."""
    n_actions = rng.randint(0, 6)
    components = (
        Component(id="CTRL-1", name="c", kind=ComponentKind.CONTROLLER),
        Component(id="ACT-1", name="a", kind=ComponentKind.ACTUATOR),
        Component(id="PROC-1", name="p", kind=ComponentKind.PROCESS),
    )
    actions = tuple(
        ControlAction(
            id=f"CA-{i}", name=f"action {i}", controller="CTRL-1", actuator="ACT-1",
            process="PROC-1",
        )
        for i in range(n_actions)
    )
    cells = [(a, c) for a in range(n_actions) for c in range(4)]
    uca_cells = [rng.choice(cells) for _ in range(rng.randint(0, 10))] if cells else []
    excl_cells = (
        rng.sample(cells, rng.randint(0, min(len(cells), 8))) if cells else []
    )
    ucas = tuple(
        Uca(
            id=f"U-{index}", control_action=f"CA-{a}", uca_class=UCA_CLASSES[c],
            description="d",
        )
        for index, (a, c) in enumerate(uca_cells)
    )
    exclusions = tuple(
        Exclusion(
            control_action=f"CA-{a}", uca_class=UCA_CLASSES[c],
            justification=GOOD_JUSTIFICATION,
        )
        for a, c in excl_cells
    )
    return Analysis(
        title="generated",
        structure=ControlStructure(components=components, control_actions=actions),
        ucas=ucas,
        exclusions=exclusions,
    )


def main() -> int:
    """Measure every identity. Returns a process exit code."""
    emit = Emitter(__file__)
    emit("hazardtrace validation: grid and finding identities")
    emit()
    emit(f"   trials                  : {TRIALS}  (seed {SEED})")
    emit("   actions per trial       : 0 to 6, cells filled at random")
    emit()

    rng = random.Random(SEED)
    counters = {
        "grid size is 4n": 0,
        "states partition the grid": 0,
        "covered plus gaps equals total": 0,
        "HT001 count equals the gap count": 0,
        "HT003 count equals the conflict count": 0,
        "every cell appears exactly once": 0,
        "run_checks is deterministic": 0,
        "declaration order does not change the findings": 0,
        "enumeration ratio agrees with the grid to 1e-12": 0,
        "enumeration ratio lies in [0, 1]": 0,
        "excluding a gap cell removes exactly that gap": 0,
        "adding a UCA to a filled cell leaves the gap set alone": 0,
        "adding an orphan component adds exactly one HT016": 0,
    }
    applicable = dict.fromkeys(counters, 0)
    non_monotone = 0
    monotone_applicable = 0

    for _ in range(TRIALS):
        analysis = random_analysis(rng)
        grid = build_grid(analysis)
        report = run_checks(analysis)
        n_actions = len(analysis.structure.control_actions)

        def record(name: str, ok: bool) -> None:
            applicable[name] += 1
            if ok:
                counters[name] += 1

        record("grid size is 4n", grid.total == 4 * n_actions)
        record(
            "states partition the grid",
            sum(grid.count(s) for s in CellState) == grid.total,
        )
        record(
            "covered plus gaps equals total",
            grid.covered + len(grid.gaps) == grid.total,
        )
        record(
            "HT001 count equals the gap count",
            sum(1 for f in report.findings if f.code == "HT001")
            == grid.count(CellState.GAP),
        )
        record(
            "HT003 count equals the conflict count",
            sum(1 for f in report.findings if f.code == "HT003")
            == grid.count(CellState.CONFLICT),
        )
        keys = [(c.control_action, c.uca_class) for c in grid.cells]
        record("every cell appears exactly once", len(keys) == len(set(keys)))
        record(
            "run_checks is deterministic",
            run_checks(analysis).findings == report.findings,
        )

        ucas = list(analysis.ucas)
        exclusions = list(analysis.exclusions)
        rng.shuffle(ucas)
        rng.shuffle(exclusions)
        shuffled = Analysis(
            title=analysis.title, structure=analysis.structure,
            ucas=tuple(ucas), exclusions=tuple(exclusions),
        )
        record(
            "declaration order does not change the findings",
            sorted((f.code, f.subjects) for f in run_checks(shuffled).findings)
            == sorted((f.code, f.subjects) for f in report.findings),
        )

        ratio = report.coverage.enumeration
        if ratio.defined:
            record(
                "enumeration ratio agrees with the grid to 1e-12",
                math.isclose(
                    ratio.fraction, grid.covered / grid.total, rel_tol=TOL, abs_tol=TOL
                ),
            )
            record(
                "enumeration ratio lies in [0, 1]",
                -TOL <= ratio.fraction <= 1.0 + TOL,
            )

        if grid.gaps:
            target = grid.gaps[0]
            extended = Analysis(
                title=analysis.title, structure=analysis.structure, ucas=analysis.ucas,
                exclusions=analysis.exclusions
                + (
                    Exclusion(
                        control_action=target.control_action,
                        uca_class=target.uca_class,
                        justification=GOOD_JUSTIFICATION,
                    ),
                ),
            )
            new_grid = build_grid(extended)
            removed = {(c.control_action, c.uca_class) for c in grid.gaps} - {
                (c.control_action, c.uca_class) for c in new_grid.gaps
            }
            record(
                "excluding a gap cell removes exactly that gap",
                new_grid.total == grid.total
                and new_grid.covered == grid.covered + 1
                and removed == {(target.control_action, target.uca_class)},
            )

        filled = [c for c in grid.cells if c.state is CellState.UCA]
        if filled:
            target = filled[0]
            extended = Analysis(
                title=analysis.title, structure=analysis.structure,
                ucas=analysis.ucas
                + (
                    Uca(
                        id="U-extra", control_action=target.control_action,
                        uca_class=target.uca_class, description="d",
                    ),
                ),
                exclusions=analysis.exclusions,
            )
            record(
                "adding a UCA to a filled cell leaves the gap set alone",
                [(c.control_action, c.uca_class) for c in build_grid(extended).gaps]
                == [(c.control_action, c.uca_class) for c in grid.gaps],
            )

        orphaned = Analysis(
            title=analysis.title,
            structure=ControlStructure(
                components=analysis.structure.components
                + (Component(id="SNS-ORPHAN", name="orphan", kind=ComponentKind.SENSOR),),
                control_actions=analysis.structure.control_actions,
                feedback=analysis.structure.feedback,
            ),
            ucas=analysis.ucas,
            exclusions=analysis.exclusions,
        )
        added = [f for f in run_checks(orphaned).findings if f not in report.findings]
        record(
            "adding an orphan component adds exactly one HT016",
            [(f.code, f.subjects) for f in added] == [("HT016", ("SNS-ORPHAN",))],
        )

        # The FALSIFIED claim: filling a gap cell never increases the finding
        # count. Measured here rather than asserted, because it is false.
        if grid.gaps:
            target = grid.gaps[0]
            filled_analysis = Analysis(
                title=analysis.title, structure=analysis.structure,
                ucas=analysis.ucas
                + (
                    Uca(
                        id="U-filled", control_action=target.control_action,
                        uca_class=target.uca_class, description="d", hazards=(),
                    ),
                ),
                exclusions=analysis.exclusions,
            )
            monotone_applicable += 1
            if len(run_checks(filled_analysis).findings) > len(report.findings):
                non_monotone += 1

    emit(f"   {'identity':<58}{'holds':<10}of applicable")
    for name, passed in counters.items():
        emit(f"   {name:<58}{passed:<10}{applicable[name]}")
    emit()
    for name, passed in counters.items():
        emit.check(name, passed == applicable[name], f"{passed}/{applicable[name]}")

    emit()
    emit("   The claim that was FALSIFIED, reported as a measurement:")
    emit("     filling one enumeration gap with a UCA increased the total finding")
    emit(f"     count in {non_monotone} of {monotone_applicable} applicable trials.")
    emit("     Filling a cell removes one HT001 and can add HT006 and HT010, so the")
    emit("     finding COUNT is not a progress measure. The enumeration ratio is.")
    emit("     Counterexample pinned in tests/test_regression.py as")
    emit("     test_filling_an_enumeration_gap_can_increase_the_finding_count.")
    emit.check(
        "the non-monotonicity is real and is reproduced here",
        non_monotone > 0,
        f"{non_monotone}/{monotone_applicable} trials",
    )
    return emit.finish()


if __name__ == "__main__":
    raise SystemExit(main())
