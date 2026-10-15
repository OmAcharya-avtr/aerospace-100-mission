"""Property tests over the grid and finding algebra.

The domain here is discrete: the quantities are counts over finite sets and the
identities are set identities, so these are exact statements and are asserted
as such. The only floating-point quantities in the package are the coverage
ratios, and those are compared with a tolerance.
"""

from __future__ import annotations

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

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

GOOD_JUSTIFICATION = "a justification long enough to satisfy the policy"

CONTROLLER = Component(id="CTRL-1", name="c", kind=ComponentKind.CONTROLLER)
ACTUATOR = Component(id="ACT-1", name="a", kind=ComponentKind.ACTUATOR)
PROCESS = Component(id="PROC-1", name="p", kind=ComponentKind.PROCESS)


def make_analysis(
    n_actions: int,
    uca_cells: list[tuple[int, int]],
    exclusion_cells: list[tuple[int, int]],
) -> Analysis:
    """Build an analysis with ``n_actions`` actions and the given filled cells.

    ``uca_cells`` and ``exclusion_cells`` are ``(action index, class index)``
    pairs. Duplicate exclusion cells are collapsed, because the parser forbids
    two exclusions of one cell.
    """
    actions = tuple(
        ControlAction(
            id=f"CA-{i}", name=f"action {i}", controller="CTRL-1", actuator="ACT-1",
            process="PROC-1",
        )
        for i in range(n_actions)
    )
    ucas = tuple(
        Uca(
            id=f"U-{index}",
            control_action=f"CA-{a}",
            uca_class=UCA_CLASSES[c],
            description="d",
            hazards=(),
        )
        for index, (a, c) in enumerate(uca_cells)
    )
    seen: set[tuple[int, int]] = set()
    exclusions: list[Exclusion] = []
    for a, c in exclusion_cells:
        if (a, c) in seen:
            continue
        seen.add((a, c))
        exclusions.append(
            Exclusion(
                control_action=f"CA-{a}",
                uca_class=UCA_CLASSES[c],
                justification=GOOD_JUSTIFICATION,
            )
        )
    return Analysis(
        title="generated",
        structure=ControlStructure(
            components=(CONTROLLER, ACTUATOR, PROCESS), control_actions=actions
        ),
        ucas=ucas,
        exclusions=tuple(exclusions),
    )


cell_lists = st.lists(
    st.tuples(st.integers(min_value=0, max_value=5), st.integers(min_value=0, max_value=3)),
    max_size=12,
)


@st.composite
def analyses(draw):
    """Analyses with 0 to 6 control actions and arbitrary filled cells."""
    n_actions = draw(st.integers(min_value=0, max_value=6))
    if n_actions == 0:
        return make_analysis(0, [], [])
    bound = n_actions - 1
    ucas = [
        (a % (bound + 1), c)
        for a, c in draw(
            st.lists(
                st.tuples(
                    st.integers(min_value=0, max_value=bound),
                    st.integers(min_value=0, max_value=3),
                ),
                max_size=12,
            )
        )
    ]
    exclusions = [
        (a % (bound + 1), c)
        for a, c in draw(
            st.lists(
                st.tuples(
                    st.integers(min_value=0, max_value=bound),
                    st.integers(min_value=0, max_value=3),
                ),
                max_size=12,
            )
        )
    ]
    return make_analysis(n_actions, ucas, exclusions)


@given(analyses())
@settings(max_examples=150, deadline=None)
def test_grid_size_is_exactly_four_per_control_action(analysis):
    grid = build_grid(analysis)
    assert grid.total == 4 * len(analysis.structure.control_actions)


@given(analyses())
@settings(max_examples=150, deadline=None)
def test_cell_states_partition_the_grid(analysis):
    grid = build_grid(analysis)
    assert sum(grid.count(state) for state in CellState) == grid.total
    assert grid.covered + len(grid.gaps) == grid.total


@given(analyses())
@settings(max_examples=150, deadline=None)
def test_ht001_count_equals_the_number_of_uncovered_cells(analysis):
    report = run_checks(analysis)
    gaps = sum(1 for f in report.findings if f.code == "HT001")
    assert gaps == report.grid.total - report.grid.covered


@given(analyses())
@settings(max_examples=150, deadline=None)
def test_every_cell_appears_exactly_once(analysis):
    grid = build_grid(analysis)
    keys = [(c.control_action, c.uca_class) for c in grid.cells]
    assert len(keys) == len(set(keys))


@given(analyses())
@settings(max_examples=100, deadline=None)
def test_run_checks_is_deterministic(analysis):
    first = run_checks(analysis)
    second = run_checks(analysis)
    assert first.findings == second.findings
    assert first.coverage.as_items() == second.coverage.as_items()


@given(analyses(), st.randoms(use_true_random=False))
@settings(max_examples=100, deadline=None)
def test_declaration_order_does_not_change_the_finding_multiset(analysis, rng):
    ucas = list(analysis.ucas)
    exclusions = list(analysis.exclusions)
    rng.shuffle(ucas)
    rng.shuffle(exclusions)
    shuffled = Analysis(
        title=analysis.title,
        structure=analysis.structure,
        ucas=tuple(ucas),
        exclusions=tuple(exclusions),
    )
    before = sorted((f.code, f.subjects) for f in run_checks(analysis).findings)
    after = sorted((f.code, f.subjects) for f in run_checks(shuffled).findings)
    assert before == after


@given(analyses())
@settings(max_examples=100, deadline=None)
def test_excluding_a_gap_cell_removes_exactly_that_gap(analysis):
    grid = build_grid(analysis)
    assume(grid.gaps)
    target = grid.gaps[0]
    extended = Analysis(
        title=analysis.title,
        structure=analysis.structure,
        ucas=analysis.ucas,
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
    assert new_grid.total == grid.total
    assert new_grid.covered == grid.covered + 1
    removed = {(c.control_action, c.uca_class) for c in grid.gaps} - {
        (c.control_action, c.uca_class) for c in new_grid.gaps
    }
    assert removed == {(target.control_action, target.uca_class)}


@given(analyses())
@settings(max_examples=100, deadline=None)
def test_adding_a_uca_to_a_covered_cell_leaves_the_gap_set_alone(analysis):
    grid = build_grid(analysis)
    covered = [c for c in grid.cells if c.state is CellState.UCA]
    assume(covered)
    target = covered[0]
    extended = Analysis(
        title=analysis.title,
        structure=analysis.structure,
        ucas=analysis.ucas
        + (
            Uca(
                id="U-extra",
                control_action=target.control_action,
                uca_class=target.uca_class,
                description="d",
            ),
        ),
        exclusions=analysis.exclusions,
    )
    new_grid = build_grid(extended)
    assert [(c.control_action, c.uca_class) for c in new_grid.gaps] == [
        (c.control_action, c.uca_class) for c in grid.gaps
    ]


@given(analyses())
@settings(max_examples=100, deadline=None)
def test_adding_an_orphan_component_adds_exactly_one_ht016(analysis):
    before = run_checks(analysis).findings
    extended = Analysis(
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
    after = run_checks(extended).findings
    added = [f for f in after if f not in before]
    assert [f.code for f in added] == ["HT016"]
    assert added[0].subjects == ("SNS-ORPHAN",)


@given(analyses())
@settings(max_examples=100, deadline=None)
def test_enumeration_ratio_is_consistent_with_the_grid(analysis):
    report = run_checks(analysis)
    ratio = report.coverage.enumeration
    assert (ratio.numerator, ratio.denominator) == report.grid.completeness()
    if ratio.defined:
        assert ratio.fraction == pytest.approx(
            report.grid.covered / report.grid.total, rel=1e-12, abs=1e-12
        )
        assert 0.0 - 1e-12 <= ratio.fraction <= 1.0 + 1e-12
    else:
        assert report.grid.total == 0
