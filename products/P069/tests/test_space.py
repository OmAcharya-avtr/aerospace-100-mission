"""Parameter and ScenarioSpace: declaration, encoding, embedding and the edge caveat."""

from __future__ import annotations

import math

import numpy as np
import pytest

from scenariocov import Parameter, ScenarioSpace


class TestParameterDeclaration:
    def test_categorical_levels_are_kept_in_order(self) -> None:
        p = Parameter.categorical("lighting", ["full_sun", "terminator", "eclipse"])
        assert p.levels == ("full_sun", "terminator", "eclipse")
        assert p.n_levels == 3
        assert p.edges == ()

    def test_continuous_labels_are_generated_from_edges(self) -> None:
        # Edges 0, 1, 2, 4 give bins [0,1), [1,2) and the closed top bin [2,4].
        p = Parameter.continuous("speed", [0.0, 1.0, 2.0, 4.0], units="m/s")
        assert p.levels == ("[0,1)", "[1,2)", "[2,4]")
        assert p.lower == pytest.approx(0.0)
        assert p.upper == pytest.approx(4.0)
        assert p.units == "m/s"

    def test_explicit_labels_are_used(self) -> None:
        p = Parameter.continuous("q", [0.0, 1.0, 2.0], labels=["low", "high"])
        assert p.levels == ("low", "high")

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"name": "", "kind": "categorical", "levels": ("a", "b")}, "non-empty"),
            ({"name": "x", "kind": "ordinal", "levels": ("a", "b")}, "categorical"),
            ({"name": "x", "kind": "categorical", "levels": ("a",)}, "at least 2 levels"),
            ({"name": "x", "kind": "categorical", "levels": ("a", "a")}, "unique"),
            (
                {"name": "x", "kind": "continuous", "levels": ("a", "b"), "edges": (0.0, 1.0)},
                "len(edges)",
            ),
            (
                {"name": "x", "kind": "categorical", "levels": ("a", "b"), "edges": (0.0, 1.0)},
                "must not declare edges",
            ),
        ],
    )
    def test_invalid_declarations_raise(self, kwargs: dict, message: str) -> None:
        with pytest.raises(ValueError, match=re_escape(message)):
            Parameter(**kwargs)

    def test_non_increasing_edges_raise(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            Parameter(name="x", kind="continuous", levels=("a", "b"), edges=(0.0, 0.0, 1.0))

    def test_two_edges_are_not_enough_for_a_parameter(self) -> None:
        with pytest.raises(ValueError, match="at least 3 edges"):
            Parameter.continuous("x", [0.0, 1.0])

    def test_lower_and_upper_reject_categorical(self) -> None:
        p = Parameter.categorical("x", ["a", "b"])
        with pytest.raises(TypeError, match="continuous"):
            _ = p.lower
        with pytest.raises(TypeError, match="continuous"):
            _ = p.upper


class TestLevelIndex:
    def test_known_answers_for_the_half_open_convention(self) -> None:
        # Edges 0, 1, 2, 4. Hand-worked: 0.0 -> bin 0 (lower edge included),
        # 0.999 -> 0, 1.0 -> 1 (left-closed), 1.999 -> 1, 2.0 -> 2, 4.0 -> 2
        # because the top bin is closed at the top.
        p = Parameter.continuous("speed", [0.0, 1.0, 2.0, 4.0])
        assert p.level_index(0.0) == 0
        assert p.level_index(0.999) == 0
        assert p.level_index(1.0) == 1
        assert p.level_index(1.999) == 1
        assert p.level_index(2.0) == 2
        assert p.level_index(4.0) == 2

    def test_categorical_known_answers(self) -> None:
        p = Parameter.categorical("light", ["sun", "dusk", "dark"])
        assert [p.level_index(v) for v in ("sun", "dusk", "dark")] == [0, 1, 2]

    def test_unknown_category_raises_with_the_declared_levels_listed(self) -> None:
        p = Parameter.categorical("light", ["sun", "dusk"])
        with pytest.raises(ValueError, match="not a declared level"):
            p.level_index("eclipse")

    @pytest.mark.parametrize("value", [-0.001, 4.001])
    def test_out_of_interval_values_raise(self, value: float) -> None:
        p = Parameter.continuous("speed", [0.0, 1.0, 2.0, 4.0])
        with pytest.raises(ValueError, match="outside the declared interval"):
            p.level_index(value)

    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
    def test_non_finite_values_raise(self, value: float) -> None:
        p = Parameter.continuous("speed", [0.0, 1.0, 2.0, 4.0])
        with pytest.raises(ValueError):
            p.level_index(value)


class TestScenarioSpace:
    def test_counts_are_exact_integers(self, tiny_space: ScenarioSpace) -> None:
        assert tiny_space.level_counts == (2, 2, 3)
        assert tiny_space.n_combinations == 12
        assert len(tiny_space) == 3
        assert tiny_space.names == ("mode", "speed", "light")

    def test_duplicate_parameter_names_are_rejected(self) -> None:
        p = Parameter.categorical("x", ["a", "b"])
        with pytest.raises(ValueError, match="unique"):
            ScenarioSpace([p, p])

    def test_empty_space_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one parameter"):
            ScenarioSpace([])

    def test_encode_known_answer(self, tiny_space: ScenarioSpace) -> None:
        # mode=nominal -> 1, speed=1.5 -> bin 1 of [0,1),[1,2], light=dark -> 2.
        assert tiny_space.encode({"mode": "nominal", "speed": 1.5, "light": "dark"}) == (1, 1, 2)

    def test_encode_reports_missing_parameters(self, tiny_space: ScenarioSpace) -> None:
        with pytest.raises(ValueError, match="missing parameter"):
            tiny_space.encode({"mode": "safe"})

    def test_encode_many_shape_and_dtype(self, tiny_space: ScenarioSpace) -> None:
        records = [{"mode": "safe", "speed": 0.5, "light": "sun"}] * 4
        encoded = tiny_space.encode_many(records)
        assert encoded.shape == (4, 3)
        assert np.issubdtype(encoded.dtype, np.integer)

    def test_encode_many_of_nothing_is_an_empty_array(self, tiny_space: ScenarioSpace) -> None:
        assert tiny_space.encode_many([]).shape == (0, 3)

    def test_decode_round_trips_level_labels(self, tiny_space: ScenarioSpace) -> None:
        assert tiny_space.decode((1, 0, 2)) == {"mode": "nominal", "speed": "[0,1)", "light": "dark"}

    def test_decode_rejects_the_wrong_length(self, tiny_space: ScenarioSpace) -> None:
        with pytest.raises(ValueError, match="expected 3 level indices"):
            tiny_space.decode((0, 0))

    def test_parameter_lookup_and_index(self, tiny_space: ScenarioSpace) -> None:
        assert tiny_space.parameter("speed").kind == "continuous"
        assert tiny_space.index_of("light") == 2
        with pytest.raises(KeyError):
            tiny_space.parameter("missing")
        with pytest.raises(KeyError):
            tiny_space.index_of("missing")

    def test_round_trip_through_dict(self, rendezvous_space: ScenarioSpace) -> None:
        clone = ScenarioSpace.from_dict(rendezvous_space.to_dict())
        assert clone.names == rendezvous_space.names
        assert clone.level_counts == rendezvous_space.level_counts

    def test_from_dict_rejects_junk(self) -> None:
        with pytest.raises(ValueError, match="non-empty 'parameters'"):
            ScenarioSpace.from_dict({"parameters": []})


class TestNormalisedEmbedding:
    def test_width_and_labels(self, tiny_space: ScenarioSpace) -> None:
        # 1 column for speed plus 2 for mode plus 3 for light.
        assert tiny_space.n_embedding_columns == 6
        assert tiny_space.embedding_columns == (
            "mode=safe", "mode=nominal", "speed", "light=sun", "light=dusk", "light=dark",
        )

    def test_continuous_column_is_linear_on_the_declared_interval(self) -> None:
        space = ScenarioSpace([Parameter.continuous("x", [0.0, 1.0, 4.0])])
        values = space.normalise([{"x": 0.0}, {"x": 2.0}, {"x": 4.0}])[:, 0]
        assert values == pytest.approx([0.0, 0.5, 1.0], abs=1e-12)

    def test_distinct_categories_are_exactly_one_apart_to_rounding(self) -> None:
        space = ScenarioSpace([Parameter.categorical("c", ["a", "b", "c"])])
        points = space.normalise([{"c": "a"}, {"c": "b"}])
        distance = float(np.linalg.norm(points[0] - points[1]))
        # Two one-hot columns scaled by 1/sqrt(2) give sqrt(2 * 0.5) = 1 in
        # exact arithmetic; the computed value agrees to rounding.
        assert distance == pytest.approx(1.0, abs=1e-12)

    def test_equal_categories_are_at_zero_distance(self) -> None:
        space = ScenarioSpace([Parameter.categorical("c", ["a", "b"])])
        points = space.normalise([{"c": "a"}, {"c": "a"}])
        assert float(np.linalg.norm(points[0] - points[1])) == pytest.approx(0.0, abs=1e-12)


class TestEdgeAmbiguities:
    def test_a_value_exactly_on_an_edge_is_reported(self) -> None:
        space = ScenarioSpace([Parameter.continuous("x", [0.0, 0.25, 1.0])])
        found = space.edge_ambiguities([{"x": 0.25}])
        assert len(found) == 1
        assert found[0].parameter == "x"
        assert found[0].distance == pytest.approx(0.0, abs=0.0)
        # The assignment this package made is reported; it is not promised.
        assert found[0].assigned_level == 1

    def test_a_value_far_from_every_edge_is_not_reported(self) -> None:
        space = ScenarioSpace([Parameter.continuous("x", [0.0, 0.25, 1.0])])
        assert space.edge_ambiguities([{"x": 0.6}]) == ()

    def test_tolerance_must_be_positive(self, tiny_space: ScenarioSpace) -> None:
        with pytest.raises(ValueError, match="rtol must be positive"):
            tiny_space.edge_ambiguities([], rtol=0.0)

    def test_the_committed_campaign_has_exactly_one_ambiguous_record(
        self, rendezvous_space, rendezvous_campaign
    ) -> None:
        # Known answer from the committed fixture: record 191 has
        # nav_position_error exactly 0.1, which is an internal bin edge.
        found = rendezvous_space.edge_ambiguities(rendezvous_campaign)
        assert len(found) == 1
        assert found[0].record_index == 191
        assert found[0].parameter == "nav_position_error"
        assert found[0].edge == pytest.approx(0.1, abs=1e-12)


def re_escape(text: str) -> str:
    """Escape a literal for use as a pytest.raises match pattern."""
    import re

    return re.escape(text)
