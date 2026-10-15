"""The enumeration grid. Hand-counted answers are in the comments."""

from __future__ import annotations

from hazardtrace import CellState, UcaClass, build_grid, parse_analysis

TWO_ACTIONS = """
meta:
  title: t
control_structure:
  controllers:
    - id: CTRL-1
      name: c
  actuators:
    - id: ACT-1
      name: a
  controlled_processes:
    - id: PROC-1
      name: p
  control_actions:
    - id: CA-1
      name: one
      controller: CTRL-1
      actuator: ACT-1
      process: PROC-1
    - id: CA-2
      name: two
      controller: CTRL-1
      actuator: ACT-1
      process: PROC-1
"""


def _grid(extra: str = ""):
    return build_grid(parse_analysis(TWO_ACTIONS + extra))


def test_grid_size_is_four_times_the_action_count():
    # Hand count: 2 control actions x 4 UCA classes = 8 cells.
    grid = _grid()
    assert grid.total == 8
    assert grid.covered == 0
    assert len(grid.gaps) == 8


def test_grid_is_empty_when_there_are_no_control_actions():
    grid = build_grid(parse_analysis("meta:\n  title: t\n"))
    assert grid.total == 0
    assert grid.covered == 0
    assert grid.completeness() == (0, 0)


def test_grid_order_is_action_major_class_minor():
    grid = _grid()
    assert [(c.control_action, c.uca_class.value) for c in grid.cells[:5]] == [
        ("CA-1", "not_provided"),
        ("CA-1", "provided"),
        ("CA-1", "wrong_timing_or_order"),
        ("CA-1", "stopped_too_soon_or_applied_too_long"),
        ("CA-2", "not_provided"),
    ]


def test_one_uca_covers_one_cell():
    # Hand count: 1 of 8 cells covered, 7 gaps.
    grid = _grid(
        "ucas:\n  - id: U-1\n    control_action: CA-1\n    uca_class: provided\n"
        "    description: d\n"
    )
    assert grid.completeness() == (1, 8)
    assert len(grid.gaps) == 7
    cell = grid.cell("CA-1", UcaClass.PROVIDED)
    assert cell is not None
    assert cell.state is CellState.UCA
    assert cell.uca_ids == ("U-1",)


def test_two_ucas_in_one_cell_still_cover_one_cell():
    # Hand count: 2 UCAs, 1 covered cell, 7 gaps. Coverage counts cells, not UCAs.
    grid = _grid(
        "ucas:\n"
        "  - id: U-1\n    control_action: CA-1\n    uca_class: provided\n    description: d\n"
        "  - id: U-2\n    control_action: CA-1\n    uca_class: provided\n    description: e\n"
    )
    assert grid.completeness() == (1, 8)
    cell = grid.cell("CA-1", UcaClass.PROVIDED)
    assert cell is not None
    assert cell.uca_ids == ("U-1", "U-2")


def test_exclusion_covers_a_cell():
    grid = _grid(
        "exclusions:\n  - control_action: CA-2\n    uca_class: not_provided\n"
        "    justification: a justification long enough to pass the policy\n"
    )
    assert grid.completeness() == (1, 8)
    cell = grid.cell("CA-2", UcaClass.NOT_PROVIDED)
    assert cell is not None
    assert cell.state is CellState.EXCLUDED
    assert cell.justification.startswith("a justification")


def test_uca_and_exclusion_in_one_cell_is_a_conflict():
    grid = _grid(
        "ucas:\n  - id: U-1\n    control_action: CA-1\n    uca_class: provided\n"
        "    description: d\n"
        "exclusions:\n  - control_action: CA-1\n    uca_class: provided\n"
        "    justification: a justification long enough to pass the policy\n"
    )
    cell = grid.cell("CA-1", UcaClass.PROVIDED)
    assert cell is not None
    assert cell.state is CellState.CONFLICT
    # A conflict still counts as covered: somebody did look at the cell.
    assert grid.completeness() == (1, 8)
    assert len(grid.conflicts) == 1


def test_states_partition_the_grid():
    grid = _grid(
        "ucas:\n  - id: U-1\n    control_action: CA-1\n    uca_class: provided\n"
        "    description: d\n"
        "exclusions:\n  - control_action: CA-2\n    uca_class: provided\n"
        "    justification: a justification long enough to pass the policy\n"
    )
    total = sum(grid.count(state) for state in CellState)
    assert total == grid.total
    assert grid.count(CellState.UCA) == 1
    assert grid.count(CellState.EXCLUDED) == 1
    assert grid.count(CellState.GAP) == 6
    assert grid.count(CellState.CONFLICT) == 0


def test_cell_lookup_of_an_unknown_action_returns_none():
    assert _grid().cell("CA-9", UcaClass.PROVIDED) is None


def test_gap_state_is_the_only_uncovered_one():
    assert CellState.GAP.is_covered is False
    for state in (CellState.UCA, CellState.EXCLUDED, CellState.CONFLICT):
        assert state.is_covered is True
