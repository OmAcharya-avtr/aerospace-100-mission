"""Tests for the declared diagram: what it accepts, and everything it refuses.

The refusals are the product. A declaration that does not say what it means has
to be rejected, because the alternative is a defaulted value and a reliability
figure with no complaint attached.
"""

from __future__ import annotations

import pytest
import yaml

from reliablock.diagram import (
    Block,
    Diagram,
    KOfN,
    Network,
    Parallel,
    Series,
    SharedLoad,
    Standby,
    load_diagram,
    node_from_mapping,
)
from reliablock.distributions import Exponential, Weibull

from conftest import exponential_block, fixed_block


class TestBlockValidation:
    def test_empty_name_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty string"):
            Block(name="  ", distribution=Exponential(1e-5))

    def test_non_distribution_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="FailureDistribution"):
            Block(name="a", distribution=0.5)  # type: ignore[arg-type]


class TestStructureValidation:
    def test_empty_series_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="has no items"):
            Diagram("d", Series(items=()))

    def test_parallel_with_one_item_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="not redundancy"):
            Diagram("d", Parallel(items=(exponential_block("a", 1e-5),)))

    @pytest.mark.parametrize("k", [0, 4, -1])
    def test_k_outside_one_to_n_is_rejected(self, k: int) -> None:
        items = tuple(exponential_block(n, 1e-5) for n in "abc")
        with pytest.raises(ValueError, match="k must lie in 1..n|at least 2"):
            Diagram("d", KOfN(k=k, items=items))

    def test_non_integer_k_is_rejected(self) -> None:
        items = tuple(exponential_block(n, 1e-5) for n in "abc")
        with pytest.raises(ValueError, match="non-integer k"):
            Diagram("d", KOfN(k=2.0, items=items))  # type: ignore[arg-type]

    def test_duplicate_block_names_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate block names"):
            Diagram(
                "d",
                Series(items=(exponential_block("a", 1e-5), exponential_block("a", 2e-5))),
            )

    def test_standby_without_a_primary_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="has no primary"):
            Diagram("d", Standby(standbys=(exponential_block("s", 1e-5),)))

    def test_standby_without_standbys_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="no standby units"):
            Diagram("d", Standby(primary=exponential_block("p", 1e-5)))

    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_standby_switch_reliability_out_of_range(self, bad: float) -> None:
        with pytest.raises(ValueError, match="not a probability"):
            Diagram(
                "d",
                Standby(
                    primary=exponential_block("p", 1e-5),
                    standbys=(exponential_block("s", 1e-5),),
                    switch_reliability=bad,
                ),
            )

    def test_standby_hazard_fraction_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="standby_hazard_fraction"):
            Diagram(
                "d",
                Standby(
                    primary=exponential_block("p", 1e-5),
                    standbys=(exponential_block("s", 1e-5),),
                    switch_reliability=1.0,
                    standby_hazard_fraction=1.5,
                ),
            )

    def test_shared_load_needs_exactly_two_units(self) -> None:
        with pytest.raises(ValueError, match="takes\n?.*exactly 2|exactly\n?.*2"):
            Diagram(
                "d",
                SharedLoad(
                    units=(
                        exponential_block("a", 1e-5),
                        exponential_block("b", 1e-5),
                        exponential_block("c", 1e-5),
                    ),
                    shared_rate_factor=0.5,
                ),
            )

    @pytest.mark.parametrize("bad", [0.0, -0.5, 1.5])
    def test_shared_rate_factor_out_of_range(self, bad: float) -> None:
        with pytest.raises(ValueError, match="shared_rate_factor"):
            Diagram(
                "d",
                SharedLoad(
                    units=(exponential_block("a", 1e-5), exponential_block("b", 1e-5)),
                    shared_rate_factor=bad,
                ),
            )

    def test_shared_load_rejects_a_non_exponential_unit(self) -> None:
        with pytest.raises(ValueError, match="exponential distribution"):
            Diagram(
                "d",
                SharedLoad(
                    units=(
                        exponential_block("a", 1e-5),
                        Block(name="b", distribution=Weibull(shape=2.0, scale_hours=1e4)),
                    ),
                    shared_rate_factor=0.5,
                ),
            )

    def test_a_beta_factor_on_a_k_equals_n_group_is_refused(self) -> None:
        # k = n is a series group, not a redundant one. A beta factor there is
        # meaningless and the arithmetic makes the group look BETTER, measured
        # at a ratio of 0.918939 for k = n = 6 at beta = 0.1 in
        # validation/validate_commoncause_output.txt.
        items = tuple(exponential_block(n, 1e-5) for n in "abc")
        with pytest.raises(ValueError, match="series group and not a redundant one"):
            Diagram("d", KOfN(k=3, items=items, beta_factor=0.1))

    def test_a_beta_factor_below_k_equals_n_is_accepted(self) -> None:
        items = tuple(exponential_block(n, 1e-5) for n in "abc")
        assert Diagram("d", KOfN(k=2, items=items, beta_factor=0.1)).declared_betas

    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_beta_factor_out_of_range(self, bad: float) -> None:
        with pytest.raises(ValueError, match="beta factor is the"):
            Diagram(
                "d",
                Parallel(
                    items=(exponential_block("a", 1e-5), exponential_block("b", 1e-5)),
                    beta_factor=bad,
                ),
            )


class TestNetworkValidation:
    def test_empty_network_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="has no edges"):
            Diagram("d", Network(edges=()))

    def test_self_loop_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="self-loop"):
            Diagram("d", Network(edges=(("s", "s", fixed_block("e", 0.1)),), source="s", sink="t"))

    def test_source_equal_to_sink_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="source == sink"):
            Diagram(
                "d",
                Network(edges=(("s", "a", fixed_block("e", 0.1)),), source="s", sink="s"),
            )

    def test_disconnected_sink_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="can never succeed"):
            Diagram(
                "d",
                Network(edges=(("s", "a", fixed_block("e", 0.1)),), source="s", sink="t"),
            )

    def test_edge_on_no_path_is_rejected(self) -> None:
        # e2 dangles off vertex a and lies on no s-t path, so its reliability
        # could never enter the answer.
        with pytest.raises(ValueError, match="lie on no"):
            Diagram(
                "d",
                Network(
                    edges=(
                        ("s", "t", fixed_block("e1", 0.1)),
                        ("s", "x", fixed_block("e2", 0.1)),
                    ),
                    source="s",
                    sink="t",
                ),
            )


class TestWalkingAndIntrospection:
    def test_block_names_are_in_declaration_order(self, bridge_network: Diagram) -> None:
        assert bridge_network.block_names() == ("e1", "e2", "e3", "e4", "e5")

    def test_is_boolean_is_false_with_a_standby(self, cold_standby: Diagram) -> None:
        assert cold_standby.is_boolean is False

    def test_is_boolean_is_false_with_shared_load(self, shared_load: Diagram) -> None:
        assert shared_load.is_boolean is False

    def test_is_boolean_is_true_for_series_parallel(self, parallel_pair: Diagram) -> None:
        assert parallel_pair.is_boolean is True

    def test_declared_betas_reports_the_label(self, beta_pair: Diagram) -> None:
        assert beta_pair.declared_betas == {"pair": 0.1}

    def test_declared_betas_empty_without_one(self, parallel_pair: Diagram) -> None:
        assert parallel_pair.declared_betas == {}

    def test_nodes_of_kind(self, bridge_network: Diagram) -> None:
        assert len(bridge_network.nodes_of_kind("network")) == 1

    def test_diagram_with_no_blocks_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="has no items|declares no blocks"):
            Diagram("d", Series(items=()))

    def test_empty_diagram_name_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty string"):
            Diagram("", exponential_block("a", 1e-5))


class TestDeclarationLoading:
    def test_round_trip_through_yaml(self) -> None:
        source = Diagram(
            "round trip",
            Series(
                items=(
                    fixed_block("deploy", 1e-3),
                    Parallel(
                        items=(exponential_block("a", 1e-5), exponential_block("b", 1e-5)),
                        beta_factor=0.1,
                        label="pair",
                    ),
                    KOfN(
                        k=2,
                        items=tuple(exponential_block(n, 2e-5) for n in ("x", "y", "z")),
                        label="triple",
                    ),
                    Standby(
                        primary=exponential_block("p", 3e-5),
                        standbys=(exponential_block("s", 3e-5),),
                        switch_reliability=0.99,
                        standby_hazard_fraction=0.2,
                        label="sb",
                    ),
                ),
                label="chain",
            ),
            description="every node kind that round-trips",
        )
        body = yaml.safe_load(source.to_yaml())
        rebuilt = Diagram(
            name=body["name"],
            root=node_from_mapping(body["root"]),
            description=body.get("description", ""),
        )
        assert rebuilt.to_yaml() == source.to_yaml()

    def test_network_round_trips(self, bridge_network: Diagram) -> None:
        body = yaml.safe_load(bridge_network.to_yaml())
        rebuilt = Diagram(name=body["name"], root=node_from_mapping(body["root"]))
        assert rebuilt.to_yaml() == bridge_network.to_yaml()

    def test_unknown_node_key_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="does not accept"):
            node_from_mapping({"kind": "series", "itemz": []})

    def test_unknown_node_kind_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown node kind"):
            node_from_mapping({"kind": "majority_voter", "items": []})

    def test_standby_without_a_declared_switch_reliability_is_rejected(self) -> None:
        # The switching reliability is the term this package exists to surface,
        # so a declaration that omits it is an error and not a default of 1.0.
        with pytest.raises(ValueError, match="implicit"):
            node_from_mapping(
                {
                    "kind": "standby",
                    "primary": {
                        "kind": "block",
                        "name": "p",
                        "distribution": {"kind": "exponential", "rate_per_hour": 1e-5},
                    },
                    "standbys": [],
                }
            )

    def test_items_must_be_a_list(self) -> None:
        with pytest.raises(ValueError, match="must be a list"):
            node_from_mapping({"kind": "series", "items": "not a list"})

    def test_k_of_n_requires_k(self) -> None:
        with pytest.raises(ValueError, match="requires 'k'"):
            node_from_mapping({"kind": "k_of_n", "items": []})

    def test_network_edge_needs_from_to_block(self) -> None:
        with pytest.raises(ValueError, match="network edge"):
            node_from_mapping({"kind": "network", "edges": [{"from": "s", "to": "t"}]})

    def test_load_diagram_round_trip(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        source = Diagram(
            "from file",
            Parallel(
                items=(exponential_block("a", 1e-5), exponential_block("b", 1e-5)),
                beta_factor=0.05,
                label="pair",
            ),
        )
        path = tmp_path / "d.yaml"
        path.write_text(source.to_yaml())
        assert load_diagram(path).to_yaml() == source.to_yaml()

    def test_load_diagram_rejects_an_unknown_top_level_key(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = tmp_path / "d.yaml"
        path.write_text("name: x\nroot: {}\nextra: 1\n")
        with pytest.raises(ValueError, match="does not accept"):
            load_diagram(path)

    def test_load_diagram_rejects_a_scalar_document(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = tmp_path / "d.yaml"
        path.write_text("just a string\n")
        with pytest.raises(ValueError, match="mapping at the top level"):
            load_diagram(path)

    def test_load_diagram_requires_name_and_root(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = tmp_path / "d.yaml"
        path.write_text("name: x\n")
        with pytest.raises(ValueError, match="requires 'name' and 'root'"):
            load_diagram(path)
