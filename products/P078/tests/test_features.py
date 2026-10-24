"""Features are static. The absence of label leakage is asserted, not asserted about."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from conftest import tiny_case

from cmdguard.features import FEATURE_NAMES, NEIGHBOUR_WINDOW_S, feature_frame_rows, feature_matrix
from cmdguard.model import CommandDictionary, CommandSpec, Verity

_T, _F, _U = Verity.TRUE, Verity.FALSE, Verity.UNKNOWN


def test_feature_names_are_twenty_and_unique() -> None:
    assert len(FEATURE_NAMES) == 20
    assert len(set(FEATURE_NAMES)) == 20


def test_matrix_shape_and_dtype(cases: dict) -> None:
    for case in cases.values():
        matrix = feature_matrix(case)
        assert matrix.shape == (len(case.sequence), 20)
        assert matrix.dtype == np.float64
        assert np.isfinite(matrix).all()


def test_empty_sequence_gives_an_empty_matrix() -> None:
    assert feature_matrix(tiny_case([])).shape == (0, 20)


def test_features_are_hand_checkable() -> None:
    """One command, every field known.

    FIRE: irreversible, two preconditions of which p is unverified in the
    initial state and q is contradicted, draws 50 W against a 100 W ceiling so
    the draw fraction is 0.5, holds 10 s, declares a 7 s gap, is gated to one
    mode, writes one effect which is false, sets no mode, is in no exclusion
    group. It is the only step, so the position fraction is 0, both time
    deltas are 0, it occurs once and has no neighbours.
    """
    commands = [
        CommandSpec(
            identifier="FIRE",
            required_modes=("NOMINAL",),
            preconditions={"p": _T, "q": _T},
            effects={"r": _F},
            resource_draw={"power_w": 50.0},
            hold_s=10.0,
            min_gap_after_s=7.0,
            irreversible=True,
        )
    ]
    case = tiny_case(
        [(0.0, "FIRE")], commands=commands, predicates={"p": _U, "q": _F}, limits={"power_w": 100.0}
    )
    row = dict(zip(FEATURE_NAMES, feature_matrix(case)[0], strict=True))
    assert row["irreversible"] == 1.0
    assert row["n_preconditions"] == 2.0
    assert row["n_unverified_preconditions"] == 1.0
    assert row["n_false_preconditions"] == 1.0
    assert row["power_draw_w"] == pytest.approx(50.0)
    assert row["data_draw_mbps"] == 0.0
    assert row["draw_fraction_of_ceiling"] == pytest.approx(0.5)
    assert row["hold_s"] == pytest.approx(10.0)
    assert row["min_gap_after_s"] == pytest.approx(7.0)
    assert row["n_required_modes"] == 1.0
    assert row["n_effects"] == 1.0
    assert row["n_effects_false"] == 1.0
    assert row["sets_mode"] == 0.0
    assert row["exclusion_group_size"] == 0.0
    assert row["position_fraction"] == 0.0
    assert row["dt_prev_s"] == 0.0
    assert row["dt_next_s"] == 0.0
    assert row["n_occurrences"] == 1.0
    assert row["neighbours_within_60s"] == 0.0


def test_neighbour_window_counts_both_sides() -> None:
    steps = [(0.0, "SET_P"), (30.0, "SET_P"), (200.0, "SET_P")]
    matrix = feature_matrix(tiny_case(steps))
    column = FEATURE_NAMES.index("neighbours_within_60s")
    assert matrix[0, column] == 1.0
    assert matrix[1, column] == 1.0
    assert matrix[2, column] == 0.0
    assert NEIGHBOUR_WINDOW_S == 60.0


def test_position_fraction_spans_zero_to_one() -> None:
    matrix = feature_matrix(tiny_case([(0.0, "SET_P"), (1.0, "SET_P"), (2.0, "SET_P")]))
    column = FEATURE_NAMES.index("position_fraction")
    assert matrix[0, column] == 0.0
    assert matrix[-1, column] == 1.0


def test_undeclared_command_row_is_zero_except_position_and_timing() -> None:
    matrix = feature_matrix(tiny_case([(0.0, "SET_P"), (5.0, "GHOST")]))
    row = dict(zip(FEATURE_NAMES, matrix[1], strict=True))
    assert row["irreversible"] == 0.0
    assert row["n_preconditions"] == 0.0
    assert row["dt_prev_s"] == pytest.approx(5.0)
    assert row["position_fraction"] == 1.0


def test_features_do_not_read_the_forward_walk(cases: dict) -> None:
    """Changing the violations without changing the declarations the features
    read must leave the matrix bit-identical.

    The edit used is the mode-machine transition relation, which the forward
    walk consults for transition legality and which no feature reads. On
    nominal_science_pass, deleting the NOMINAL -> SCIENCE transition turns a
    clean case into one with violations; the feature matrix must not move.
    """
    from cmdguard.model import ModeMachine
    from cmdguard.trace import forward_execute

    case = cases["nominal_science_pass"]
    crippled = replace(
        case,
        machine=ModeMachine(
            modes=case.machine.modes,
            transitions={"SAFE": ("NOMINAL",), "NOMINAL": ("SAFE",), "SCIENCE": ("NOMINAL",)},
        ),
    )
    assert forward_execute(case).n_violations == 0
    assert forward_execute(crippled).n_violations > 0
    assert np.array_equal(feature_matrix(case), feature_matrix(crippled))


def test_feature_matrix_is_deterministic(cases: dict) -> None:
    for case in cases.values():
        assert np.array_equal(feature_matrix(case), feature_matrix(case))


def test_frame_rows_match_the_matrix(cases: dict) -> None:
    case = cases["resource_and_exclusion"]
    rows = feature_frame_rows(case)
    matrix = feature_matrix(case)
    assert len(rows) == matrix.shape[0]
    for index, row in enumerate(rows):
        assert list(row) == list(FEATURE_NAMES)
        assert np.allclose(list(row.values()), matrix[index])


def test_exclusion_group_size_counts_declared_members() -> None:
    commands = [
        CommandSpec(identifier="A", exclusion_group="g"),
        CommandSpec(identifier="B", exclusion_group="g"),
        CommandSpec(identifier="C"),
    ]
    matrix = feature_matrix(tiny_case([(0.0, "A"), (1.0, "C")], commands=commands))
    column = FEATURE_NAMES.index("exclusion_group_size")
    assert matrix[0, column] == 2.0
    assert matrix[1, column] == 0.0


def test_draw_fraction_uses_the_largest_ratio() -> None:
    commands = [
        CommandSpec(identifier="A", resource_draw={"power_w": 10.0, "data_rate_mbps": 30.0})
    ]
    case = tiny_case(
        [(0.0, "A")], commands=commands, limits={"power_w": 100.0, "data_rate_mbps": 40.0}
    )
    column = FEATURE_NAMES.index("draw_fraction_of_ceiling")
    assert feature_matrix(case)[0, column] == pytest.approx(0.75)


def test_dictionary_without_limits_gives_zero_fraction() -> None:
    commands = [CommandSpec(identifier="A", resource_draw={"power_w": 10.0})]
    case = tiny_case([(0.0, "A")], commands=commands, limits={})
    assert isinstance(case.dictionary, CommandDictionary)
    column = FEATURE_NAMES.index("draw_fraction_of_ceiling")
    assert feature_matrix(case)[0, column] == 0.0
