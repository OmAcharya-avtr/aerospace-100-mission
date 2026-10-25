"""Importance measures: hand-calculated answers, identities, and the refusals.

The Birnbaum measure is a derivative, so the reference it is scored against is
a central finite difference of the system unreliability -- the only check that
shares no expression with the implementation.
"""

from __future__ import annotations

import pytest

from reliablock.diagram import Diagram, KOfN, Parallel, Series
from reliablock.importance import (
    boolean_system_unreliability,
    importance_table,
    pivot_unreliability,
    system_unreliability_at,
)

from conftest import abs_, exponential_block, rel

MISSION = 1000.0


def _series_parallel() -> Diagram:
    return Diagram(
        "series parallel",
        Series(
            items=(
                exponential_block("single", 1.0e-4),
                Parallel(
                    items=(exponential_block("a", 3.0e-4), exponential_block("b", 3.0e-4)),
                    label="pair",
                ),
            ),
            label="chain",
        ),
    )


class TestHandCalculated:
    def test_series_block_birnbaum_equals_the_rest_of_the_chain(self) -> None:
        # For a block in series with the rest, Q(q_i = 1) = 1 and
        # Q(q_i = 0) = Q_rest, so I_B = 1 - Q_rest = R_rest.
        diagram = _series_parallel()
        unreliabilities = {
            b.name: b.distribution.unreliability(MISSION) for b in diagram.blocks()
        }
        rows = {r.event: r for r in importance_table(diagram, MISSION)}
        q_pair = unreliabilities["a"] * unreliabilities["b"]
        assert rows["single"].birnbaum == rel(1.0 - q_pair, 1e-12)

    def test_parallel_block_birnbaum_is_hand_calculable(self) -> None:
        # For block a inside the parallel pair, with the single block in
        # series: Q_sys = 1 - R_single (1 - q_a q_b), so
        # dQ/dq_a = R_single q_b.
        diagram = _series_parallel()
        unreliabilities = {
            b.name: b.distribution.unreliability(MISSION) for b in diagram.blocks()
        }
        rows = {r.event: r for r in importance_table(diagram, MISSION)}
        expected = (1.0 - unreliabilities["single"]) * unreliabilities["b"]
        assert rows["a"].birnbaum == rel(expected, 1e-12)

    def test_fussell_vesely_of_a_singleton_cut_set(self) -> None:
        diagram = _series_parallel()
        unreliabilities = {
            b.name: b.distribution.unreliability(MISSION) for b in diagram.blocks()
        }
        q_system = system_unreliability_at(diagram, unreliabilities)
        rows = {r.event: r for r in importance_table(diagram, MISSION)}
        assert rows["single"].fussell_vesely == rel(
            unreliabilities["single"] / q_system, 1e-12
        )

    def test_fussell_vesely_of_a_pair_cut_set(self) -> None:
        diagram = _series_parallel()
        unreliabilities = {
            b.name: b.distribution.unreliability(MISSION) for b in diagram.blocks()
        }
        q_system = system_unreliability_at(diagram, unreliabilities)
        rows = {r.event: r for r in importance_table(diagram, MISSION)}
        expected = unreliabilities["a"] * unreliabilities["b"] / q_system
        assert rows["a"].fussell_vesely == rel(expected, 1e-12)

    def test_identical_blocks_have_identical_importance(self) -> None:
        rows = {r.event: r for r in importance_table(_series_parallel(), MISSION)}
        assert rows["a"].birnbaum == rows["b"].birnbaum
        assert rows["a"].fussell_vesely == rows["b"].fussell_vesely

    def test_the_table_is_sorted_by_fussell_vesely(self) -> None:
        rows = importance_table(_series_parallel(), MISSION)
        values = [r.fussell_vesely for r in rows]
        assert values == sorted(values, reverse=True)

    def test_cut_set_count_is_reported(self) -> None:
        rows = {r.event: r for r in importance_table(_series_parallel(), MISSION)}
        assert rows["single"].cut_sets == 1
        assert rows["a"].cut_sets == 1

    def test_a_voter_dominates_a_two_of_three_group(self) -> None:
        # Measured at 100 h, where every block is highly reliable and the
        # order-one cut set therefore dominates. At 1000 h the same diagram has
        # q_channel = 0.237 and the channel PAIRS dominate the voter instead,
        # which is the correct answer at that point and is why the mission
        # duration is stated here rather than inherited.
        diagram = Diagram(
            "tmr",
            Series(
                items=(
                    KOfN(
                        k=2,
                        items=tuple(exponential_block(f"c{i}", 3.0e-4) for i in range(3)),
                        label="channels",
                    ),
                    exponential_block("voter", 1.0e-4),
                ),
                label="tmr",
            ),
        )
        rows = importance_table(diagram, 100.0)
        # The voter is a cut set of order one, so its Birnbaum is near 1.
        assert rows[0].event == "voter"
        assert rows[0].birnbaum > 0.99


class TestFiniteDifference:
    @pytest.mark.parametrize("step", [1e-5, 1e-6, 1e-7])
    def test_birnbaum_matches_a_central_finite_difference(self, step: float) -> None:
        diagram = _series_parallel()
        base = {b.name: b.distribution.unreliability(MISSION) for b in diagram.blocks()}
        for row in importance_table(diagram, MISSION):
            high = dict(base)
            low = dict(base)
            high[row.event] = min(base[row.event] + step, 1.0)
            low[row.event] = max(base[row.event] - step, 0.0)
            difference = (
                system_unreliability_at(diagram, high) - system_unreliability_at(diagram, low)
            ) / (high[row.event] - low[row.event])
            assert row.birnbaum == rel(difference, 1e-6)

    def test_birnbaum_is_exact_for_a_multilinear_structure(self) -> None:
        # Q_sys is multilinear in the q_i for any coherent structure with
        # independent components, so the forward difference over the whole
        # interval [0, 1] IS the derivative, exactly, not approximately.
        diagram = _series_parallel()
        base = {b.name: b.distribution.unreliability(MISSION) for b in diagram.blocks()}
        for row in importance_table(diagram, MISSION):
            high = pivot_unreliability(
                diagram, MISSION, row.event, 1.0, base_unreliabilities=base
            )
            low = pivot_unreliability(
                diagram, MISSION, row.event, 0.0, base_unreliabilities=base
            )
            assert row.birnbaum == rel(high - low, 1e-14)

    def test_criticality_identity(self) -> None:
        diagram = _series_parallel()
        base = {b.name: b.distribution.unreliability(MISSION) for b in diagram.blocks()}
        q_system = system_unreliability_at(diagram, base)
        for row in importance_table(diagram, MISSION):
            pinned = pivot_unreliability(
                diagram, MISSION, row.event, 0.0, base_unreliabilities=base
            )
            assert row.criticality == rel((q_system - pinned) / q_system, 1e-11)


class TestExhaustiveCrossCheck:
    @pytest.mark.parametrize(
        "diagram_factory",
        [
            _series_parallel,
            lambda: Diagram(
                "k of n",
                KOfN(
                    k=2,
                    items=tuple(exponential_block(f"u{i}", 2.0e-4) for i in range(4)),
                    label="g",
                ),
            ),
        ],
    )
    def test_structural_and_exhaustive_quantification_agree(
        self, diagram_factory  # type: ignore[no-untyped-def]
    ) -> None:
        diagram = diagram_factory()
        base = {b.name: b.distribution.unreliability(MISSION) for b in diagram.blocks()}
        assert system_unreliability_at(diagram, base) == abs_(
            boolean_system_unreliability(diagram, base), 1e-14
        )

    def test_bridge_quantification_agrees_with_enumeration(
        self, bridge_network: Diagram
    ) -> None:
        base = {name: 0.1 for name in bridge_network.block_names()}
        assert system_unreliability_at(bridge_network, base) == abs_(
            boolean_system_unreliability(bridge_network, base), 1e-14
        )


class TestRefusals:
    def test_a_standby_diagram_is_refused(self, cold_standby: Diagram) -> None:
        with pytest.raises(ValueError, match="not defined for this diagram"):
            importance_table(cold_standby, MISSION)

    def test_a_shared_load_diagram_is_refused(self, shared_load: Diagram) -> None:
        with pytest.raises(ValueError, match="dependent quantification"):
            importance_table(shared_load, MISSION)

    def test_the_refusal_names_the_offending_node_kind(self, shared_load: Diagram) -> None:
        with pytest.raises(ValueError, match="shared_load"):
            importance_table(shared_load, MISSION)

    def test_an_unknown_event_is_reported(self) -> None:
        with pytest.raises(KeyError, match="unknown basic event"):
            pivot_unreliability(_series_parallel(), MISSION, "nonesuch", 0.0)

    def test_exhaustive_route_refuses_above_the_cap(self) -> None:
        from reliablock.structure import MAX_ENUMERATION_EVENTS

        items = tuple(
            exponential_block(f"u{i}", 1e-5) for i in range(MAX_ENUMERATION_EVENTS + 1)
        )
        diagram = Diagram("wide", Series(items=items))
        base = {b.name: 0.1 for b in diagram.blocks()}
        with pytest.raises(ValueError, match="refused"):
            boolean_system_unreliability(diagram, base)


class TestCommonCauseImportance:
    def test_a_common_cause_event_dominates_its_group(self, beta_pair: Diagram) -> None:
        # Measured at 100 h, where each member's unreliability is 0.0296 and
        # the shared event at beta = 0.1 is 0.0030 against an independent pair
        # at 7.1e-4. The shared event is four times the pair, which is the
        # point. At 1000 h the members' q is 0.237, the independent pair is
        # 0.056 and it dominates the shared event instead -- correctly, and
        # that is why the duration is stated here.
        from reliablock.commoncause import apply_common_cause

        rows = {r.event: r for r in importance_table(apply_common_cause(beta_pair), 100.0)}
        assert "pair::common_cause" in rows
        assert rows["pair::common_cause"].fussell_vesely > rows["a"].fussell_vesely

    def test_a_common_cause_event_stops_dominating_when_the_members_are_unreliable(
        self, beta_pair: Diagram
    ) -> None:
        # The companion to the test above, so that the regime dependence is
        # recorded as a measurement rather than discovered later.
        from reliablock.commoncause import apply_common_cause

        rows = {r.event: r for r in importance_table(apply_common_cause(beta_pair), 1000.0)}
        assert rows["pair::common_cause"].fussell_vesely < rows["a"].fussell_vesely

    def test_the_common_cause_event_has_birnbaum_near_one(self, beta_pair: Diagram) -> None:
        from reliablock.commoncause import apply_common_cause

        rows = {r.event: r for r in importance_table(apply_common_cause(beta_pair), 100.0)}
        assert rows["pair::common_cause"].birnbaum > 0.999

    def test_str_is_one_readable_line(self, beta_pair: Diagram) -> None:
        row = importance_table(beta_pair, MISSION)[0]
        text = str(row)
        assert row.event in text
        assert "FV=" in text
        assert "\n" not in text
