"""The bridge network: one polynomial, four independent routes to it.

What is cited and what is derived is stated in :mod:`reliablock.bridge` and
repeated here because it matters: the closed form
``R = 2p^5 - 5p^4 + 2p^3 + 2p^2`` is **derived** in that module's docstring by
pivotal decomposition on the bridge edge. No page number and no published
figure is quoted for it. These tests are the verification that stands in place
of a citation.
"""

from __future__ import annotations

import pytest

from reliablock.bridge import (
    BRIDGE_CUT_SETS,
    BRIDGE_PATH_SETS,
    bridge_diagram,
    bridge_reliability_equal,
    bridge_reliability_inclusion_exclusion,
    bridge_reliability_pivotal,
)
from reliablock.importance import boolean_system_unreliability
from reliablock.quantify import quantify
from reliablock.structure import enumerate_states, minimal_cut_sets, minimal_path_sets

from conftest import abs_, rel


class TestDerivedClosedForm:
    def test_the_derived_value_at_p_nine(self) -> None:
        # 2(0.59049) - 5(0.6561) + 2(0.729) + 2(0.81)
        #   = 1.18098 - 3.2805 + 1.458 + 1.62 = 0.97848
        assert bridge_reliability_equal(0.9) == rel(0.97848, 1e-14)

    @pytest.mark.parametrize(
        ("p", "expected"),
        [
            (0.0, 0.0),
            (1.0, 1.0),
            (0.5, 0.5),
        ],
    )
    def test_boundary_and_symmetry_points(self, p: float, expected: float) -> None:
        # p = 0.5 gives exactly 0.5: 2/32 - 5/16 + 2/8 + 2/4
        #   = 0.0625 - 0.3125 + 0.25 + 0.5 = 0.5. The bridge is self-dual.
        assert bridge_reliability_equal(p) == abs_(expected, 1e-15)

    def test_self_duality(self) -> None:
        # A self-dual structure satisfies R(p) + R(1 - p) = 1. The bridge is
        # self-dual because its minimal cut sets and minimal path sets have the
        # same shape, which is visible in BRIDGE_CUT_SETS and BRIDGE_PATH_SETS.
        for p in (0.0, 0.1, 0.25, 0.5, 0.73, 0.9, 1.0):
            assert bridge_reliability_equal(p) + bridge_reliability_equal(1.0 - p) == rel(
                1.0, 1e-14
            )

    def test_p_out_of_range_is_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            bridge_reliability_equal(1.5)


class TestFourRoutes:
    @pytest.mark.parametrize("p", [0.0, 0.1, 0.5, 0.9, 0.99, 1.0])
    def test_all_four_routes_agree(self, p: float) -> None:
        derived = bridge_reliability_equal(p)
        pivotal = bridge_reliability_pivotal(p, p, p, p, p)
        inclusion = bridge_reliability_inclusion_exclusion({f"e{i}": p for i in range(1, 6)})
        package = quantify(bridge_diagram(p), 1.0).reliability
        for name, value in (
            ("pivotal", pivotal),
            ("inclusion-exclusion", inclusion),
            ("reliablock.quantify", package),
        ):
            assert value == abs_(derived, 1e-14), f"{name} at p={p}"

    @pytest.mark.parametrize("p", [0.1, 0.5, 0.9, 0.99])
    def test_exhaustive_state_enumeration_agrees(self, p: float) -> None:
        diagram = bridge_diagram(p)
        unreliabilities = {f"e{i}": 1.0 - p for i in range(1, 6)}
        assert 1.0 - boolean_system_unreliability(diagram, unreliabilities) == abs_(
            bridge_reliability_equal(p), 1e-14
        )

    def test_non_identical_edges_agree_between_two_routes(self) -> None:
        values = (0.91, 0.87, 0.95, 0.80, 0.99)
        mapping = {f"e{i + 1}": v for i, v in enumerate(values)}
        pivotal = bridge_reliability_pivotal(*values)
        inclusion = bridge_reliability_inclusion_exclusion(mapping)
        diagram = bridge_diagram(
            edge_unreliabilities={k: 1.0 - v for k, v in mapping.items()}, mission_hours=1.0
        )
        package = quantify(diagram, 1.0).reliability
        assert pivotal == rel(inclusion, 1e-13)
        assert package == rel(pivotal, 1e-13)

    def test_inclusion_exclusion_reports_a_missing_edge(self) -> None:
        with pytest.raises(KeyError, match="missing edge reliabilities"):
            bridge_reliability_inclusion_exclusion({"e1": 0.9})


class TestStructure:
    def test_declared_cut_sets_match_the_enumeration(self) -> None:
        diagram = bridge_diagram(0.9)
        assert set(minimal_cut_sets(diagram)) == set(BRIDGE_CUT_SETS)

    def test_declared_path_sets_match_the_enumeration(self) -> None:
        diagram = bridge_diagram(0.9)
        assert set(minimal_path_sets(diagram)) == set(BRIDGE_PATH_SETS)

    def test_exhaustive_enumeration_finds_the_same_cut_sets(self) -> None:
        diagram = bridge_diagram(0.9)
        result = enumerate_states(diagram)
        assert result["states"] == 32
        assert result["n_minimal"] == 4
        assert set(result["minimal_cut_sets"]) == set(BRIDGE_CUT_SETS)

    def test_sixteen_of_the_thirty_two_states_fail(self) -> None:
        # The bridge is self-dual, so exactly half the states fail at p = 1/2,
        # which is the count of failed states regardless of p.
        assert enumerate_states(bridge_diagram(0.9))["failed_states"] == 16

    def test_the_bridge_is_not_series_parallel(self) -> None:
        # Both routes that a series-parallel decomposition would give are
        # wrong, and they bracket the true answer, which is why the bridge is
        # the right test case.
        p = 0.9
        two_disjoint_paths = 1.0 - (1.0 - p * p) ** 2
        merged = (1.0 - (1.0 - p) ** 2) ** 2
        exact = bridge_reliability_equal(p)
        assert two_disjoint_paths < exact < merged


class TestDeclaredDiagram:
    def test_declaring_both_p_and_unreliabilities_is_refused(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            bridge_diagram(0.9, edge_unreliabilities={"e1": 0.1})

    def test_declaring_neither_is_refused(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            bridge_diagram()

    def test_a_missing_edge_unreliability_is_reported(self) -> None:
        with pytest.raises(KeyError, match="is missing"):
            bridge_diagram(edge_unreliabilities={"e1": 0.1}, mission_hours=1.0)

    def test_the_one_shot_form_is_time_independent(self) -> None:
        diagram = bridge_diagram(0.9)
        for hours in (0.0, 1.0, 1e6):
            assert quantify(diagram, hours).reliability == rel(0.97848, 1e-14)

    def test_the_exponential_form_reaches_the_declared_unreliability(self) -> None:
        diagram = bridge_diagram(
            edge_unreliabilities={f"e{i}": 0.1 for i in range(1, 6)}, mission_hours=250.0
        )
        for block in diagram.blocks():
            assert block.distribution.unreliability(250.0) == rel(0.1, 1e-13)
        assert quantify(diagram, 250.0).reliability == rel(0.97848, 1e-13)

    def test_the_notes_record_the_cut_and_path_sets(self) -> None:
        notes = " ".join(bridge_diagram(0.9).notes)
        assert "e1,e4" in notes.replace(" ", "")
        assert "e1,e2" in notes.replace(" ", "")

    def test_high_reliability_edges_keep_their_unreliability(self) -> None:
        # Q = 2q^2 + O(q^3) for the bridge, recoverable only in Q space.
        q = 1.0e-7
        diagram = bridge_diagram(
            edge_unreliabilities={f"e{i}": q for i in range(1, 6)}, mission_hours=1.0
        )
        result = quantify(diagram, 1.0)
        assert result.unreliability == rel(2.0 * q * q, 1e-5)
        # The reliability is 1 - 2e-14, which float64 can just represent; the
        # point is that Q is read off the Q-space recursion and not recovered
        # from R. At q = 1e-9 the reliability IS exactly 1.0 and Q would be
        # lost entirely; that case is asserted next.
        assert result.reliability == rel(1.0 - 2.0 * q * q, 1e-12)

    def test_very_high_reliability_edges_lose_the_answer_in_r_space(self) -> None:
        q = 1.0e-9
        diagram = bridge_diagram(
            edge_unreliabilities={f"e{i}": q for i in range(1, 6)}, mission_hours=1.0
        )
        result = quantify(diagram, 1.0)
        assert result.unreliability == rel(2.0 * q * q, 1e-5)
        # R is exactly 1.0, so 1 - R would report Q = 0 rather than 2e-18.
        assert result.reliability == 1.0
        assert 1.0 - result.reliability == 0.0
