"""Cut sets and path sets, every one cross-checked against exhaustive enumeration.

The structural recursion in :mod:`reliablock.structure` and the ``2**n`` walk in
:func:`~reliablock.structure.enumerate_states` share no code path: the first
combines child cut sets, the second evaluates the structure function on every
state and keeps the minimal failures. Where they disagree, one is wrong.
"""

from __future__ import annotations

import pytest

from reliablock.cases import CASES
from reliablock.diagram import Diagram, KOfN, Network, Parallel, Series, Standby
from reliablock.structure import (
    MAX_ENUMERATION_EVENTS,
    basic_events,
    enumerate_states,
    minimal_cut_sets,
    minimal_path_sets,
    structure_function,
)

from conftest import exponential_block, fixed_block


def _as_sorted(sets: tuple[frozenset[str], ...]) -> list[list[str]]:
    return sorted(sorted(s) for s in sets)


class TestKnownAnswers:
    def test_series_cut_sets_are_the_singletons(self, series_pair: Diagram) -> None:
        assert _as_sorted(minimal_cut_sets(series_pair)) == [["a"], ["b"]]

    def test_series_path_set_is_the_whole_chain(self, series_pair: Diagram) -> None:
        assert _as_sorted(minimal_path_sets(series_pair)) == [["a", "b"]]

    def test_parallel_cut_set_is_the_whole_group(self, parallel_pair: Diagram) -> None:
        assert _as_sorted(minimal_cut_sets(parallel_pair)) == [["a", "b"]]

    def test_parallel_path_sets_are_the_singletons(self, parallel_pair: Diagram) -> None:
        assert _as_sorted(minimal_path_sets(parallel_pair)) == [["a"], ["b"]]

    def test_two_of_three_cut_sets_are_the_pairs(self, two_of_three: Diagram) -> None:
        assert _as_sorted(minimal_cut_sets(two_of_three)) == [
            ["a", "b"],
            ["a", "c"],
            ["b", "c"],
        ]

    def test_two_of_three_path_sets_are_the_pairs(self, two_of_three: Diagram) -> None:
        assert _as_sorted(minimal_path_sets(two_of_three)) == [
            ["a", "b"],
            ["a", "c"],
            ["b", "c"],
        ]

    def test_bridge_cut_sets_match_the_standard_four(self, bridge_network: Diagram) -> None:
        assert _as_sorted(minimal_cut_sets(bridge_network)) == [
            ["e1", "e2"],
            ["e1", "e3", "e5"],
            ["e2", "e3", "e4"],
            ["e4", "e5"],
        ]

    def test_bridge_path_sets_match_the_standard_four(self, bridge_network: Diagram) -> None:
        assert _as_sorted(minimal_path_sets(bridge_network)) == [
            ["e1", "e3", "e5"],
            ["e1", "e4"],
            ["e2", "e3", "e4"],
            ["e2", "e5"],
        ]

    def test_standby_cut_sets_include_the_switch(self, cold_standby: Diagram) -> None:
        # Failure needs the primary down AND (the switch failed OR the spare
        # down). The switch is a basic event in its own right.
        assert _as_sorted(minimal_cut_sets(cold_standby)) == [
            ["pair::switch", "prime"],
            ["prime", "spare"],
        ]

    def test_standby_switch_is_a_basic_event(self, cold_standby: Diagram) -> None:
        # Root-first walk: the standby node is the root, so its switch is
        # appended before its children are reached.
        assert basic_events(cold_standby) == ("pair::switch", "prime", "spare")

    def test_shared_load_cut_set_is_parallel(self, shared_load: Diagram) -> None:
        assert _as_sorted(minimal_cut_sets(shared_load)) == [["u1", "u2"]]


class TestAgainstExhaustiveEnumeration:
    @pytest.mark.parametrize("name", sorted(CASES))
    def test_built_in_cases_agree(self, name: str) -> None:
        diagram = CASES[name]
        if len(basic_events(diagram)) > MAX_ENUMERATION_EVENTS:
            pytest.skip(
                f"{name} has {len(basic_events(diagram))} basic events, above the "
                f"declared exhaustive-enumeration cap of {MAX_ENUMERATION_EVENTS}"
            )
        structural = set(minimal_cut_sets(diagram))
        exhaustive = set(enumerate_states(diagram)["minimal_cut_sets"])
        assert structural == exhaustive

    @pytest.mark.parametrize(
        "fixture",
        [
            "series_pair",
            "parallel_pair",
            "two_of_three",
            "cold_standby",
            "shared_load",
            "bridge_network",
        ],
    )
    def test_fixtures_agree(self, fixture: str, request: pytest.FixtureRequest) -> None:
        diagram = request.getfixturevalue(fixture)
        structural = set(minimal_cut_sets(diagram))
        exhaustive = set(enumerate_states(diagram)["minimal_cut_sets"])
        assert structural == exhaustive

    def test_nested_structure_agrees(self) -> None:
        diagram = Diagram(
            "nested",
            Series(
                items=(
                    Parallel(
                        items=(
                            Series(
                                items=(
                                    exponential_block("a1", 1e-5),
                                    exponential_block("a2", 1e-5),
                                )
                            ),
                            exponential_block("b", 1e-5),
                        ),
                        label="branch",
                    ),
                    KOfN(
                        k=2,
                        items=tuple(exponential_block(n, 2e-5) for n in ("c", "d", "e")),
                        label="triple",
                    ),
                ),
                label="chain",
            ),
        )
        structural = set(minimal_cut_sets(diagram))
        exhaustive = set(enumerate_states(diagram)["minimal_cut_sets"])
        assert structural == exhaustive

    def test_enumeration_refuses_above_the_cap(self) -> None:
        items = tuple(exponential_block(f"b{i}", 1e-5) for i in range(MAX_ENUMERATION_EVENTS + 1))
        with pytest.raises(ValueError, match="refused"):
            enumerate_states(Diagram("wide", Series(items=items)))


class TestStructureFunctionProperties:
    def test_every_cut_set_fails_the_system(self) -> None:
        for diagram in CASES.values():
            events = set(basic_events(diagram))
            for cut in minimal_cut_sets(diagram):
                assert not structure_function(diagram, events - cut)

    def test_removing_one_element_from_a_cut_set_restores_the_system(self) -> None:
        for diagram in CASES.values():
            events = set(basic_events(diagram))
            for cut in minimal_cut_sets(diagram):
                for element in sorted(cut):
                    restored = (events - cut) | {element}
                    assert structure_function(diagram, restored)

    def test_every_path_set_succeeds(self) -> None:
        for diagram in CASES.values():
            for path in minimal_path_sets(diagram):
                assert structure_function(diagram, path)

    def test_removing_one_element_from_a_path_set_fails_the_system(self) -> None:
        for diagram in CASES.values():
            for path in minimal_path_sets(diagram):
                for element in sorted(path):
                    assert not structure_function(diagram, path - {element})

    def test_all_working_succeeds_and_all_failed_fails(self) -> None:
        for diagram in CASES.values():
            assert structure_function(diagram, basic_events(diagram))
            assert not structure_function(diagram, [])

    def test_cut_sets_are_returned_in_a_reproducible_order(self) -> None:
        for diagram in CASES.values():
            assert minimal_cut_sets(diagram) == minimal_cut_sets(diagram)
            orders = [len(c) for c in minimal_cut_sets(diagram)]
            assert orders == sorted(orders)

    def test_no_cut_set_contains_another(self) -> None:
        for diagram in CASES.values():
            cuts = minimal_cut_sets(diagram)
            for i, first in enumerate(cuts):
                for second in cuts[i + 1 :]:
                    assert not first <= second
                    assert not second <= first


class TestNetworkEnumeration:
    def test_series_network_has_one_path_and_two_cuts(self) -> None:
        diagram = Diagram(
            "chain",
            Network(
                edges=(("s", "m", fixed_block("e1", 0.1)), ("m", "t", fixed_block("e2", 0.1))),
                source="s",
                sink="t",
            ),
        )
        assert _as_sorted(minimal_path_sets(diagram)) == [["e1", "e2"]]
        assert _as_sorted(minimal_cut_sets(diagram)) == [["e1"], ["e2"]]

    def test_parallel_network_has_two_paths_and_one_cut(self) -> None:
        diagram = Diagram(
            "two wires",
            Network(
                edges=(("s", "t", fixed_block("e1", 0.1)), ("s", "t", fixed_block("e2", 0.1))),
                source="s",
                sink="t",
            ),
        )
        assert _as_sorted(minimal_path_sets(diagram)) == [["e1"], ["e2"]]
        assert _as_sorted(minimal_cut_sets(diagram)) == [["e1", "e2"]]

    def test_two_bridges_in_sequence_agrees_with_enumeration(self) -> None:
        layout = (
            ("s", "a", "e1"),
            ("s", "b", "e2"),
            ("a", "b", "e3"),
            ("a", "m", "e4"),
            ("b", "m", "e5"),
            ("m", "t", "e6"),
        )
        diagram = Diagram(
            "bridge then a link",
            Network(
                edges=tuple((x, y, fixed_block(n, 0.1)) for x, y, n in layout),
                source="s",
                sink="t",
            ),
        )
        assert set(minimal_cut_sets(diagram)) == set(
            enumerate_states(diagram)["minimal_cut_sets"]
        )


class TestStandbyChainCutSets:
    def test_two_cold_standbys(self) -> None:
        diagram = Diagram(
            "two spares",
            Standby(
                primary=exponential_block("p", 1e-5),
                standbys=(exponential_block("s1", 1e-5), exponential_block("s2", 1e-5)),
                switch_reliability=0.99,
                label="sb",
            ),
        )
        assert _as_sorted(minimal_cut_sets(diagram)) == [
            ["p", "s1", "s2"],
            ["p", "sb::switch"],
        ]
        assert set(minimal_cut_sets(diagram)) == set(
            enumerate_states(diagram)["minimal_cut_sets"]
        )
