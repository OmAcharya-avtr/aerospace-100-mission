"""Rename scoring and matching, including hand-calculated similarity values."""

from __future__ import annotations

import math

import pytest

from tmtcdiff.model import Command, CommandArgument, Parameter
from tmtcdiff.rename import (
    AMBIGUOUS_BAND,
    DEFAULT_WEIGHTS,
    FIELD_WEIGHTS,
    STRATEGIES,
    RenameCandidate,
    match_renames,
    normalise_name,
    pair_score,
    score_matrix,
    signature_collision_bound,
    structural_similarity,
    text_similarity,
)

TOTAL_WEIGHT = sum(FIELD_WEIGHTS.values())


class TestNormaliseName:
    @pytest.mark.parametrize(
        ("written", "expected"),
        [
            ("t_bat_1", "T_BAT_1"),
            ("T.BAT.1", "T_BAT_1"),
            ("T-BAT-1", "T_BAT_1"),
            ("T/BAT/1", "T_BAT_1"),
            ("  T BAT 1  ", "T_BAT_1"),
        ],
    )
    def test_separators_and_case_collapse(self, written: str, expected: str) -> None:
        assert normalise_name(written) == expected


class TestTextSimilarity:
    def test_identical_strings_score_one(self) -> None:
        assert text_similarity("ABC", "ABC") == 1.0

    def test_known_answer_one_substitution(self) -> None:
        # Ratcliff/Obershelp: 2 * matching / total. "ABC" vs "ABD" matches "AB",
        # so 2 * 2 / (3 + 3) = 0.6666666666666666.
        assert text_similarity("ABC", "ABD") == pytest.approx(2 / 3, rel=1e-12, abs=0.0)

    def test_known_answer_prefix(self) -> None:
        # "AB" vs "ABCD": matching 2, total 6, 2 * 2 / 6 = 0.6666666666666666.
        assert text_similarity("AB", "ABCD") == pytest.approx(2 / 3, rel=1e-12, abs=0.0)

    @pytest.mark.parametrize(("left", "right"), [("", "A"), ("A", ""), ("", "")])
    def test_an_empty_string_scores_zero(self, left: str, right: str) -> None:
        assert text_similarity(left, right) == 0.0

    def test_symmetry_on_equal_lengths(self) -> None:
        assert text_similarity("ABCD", "ABDC") == text_similarity("ABDC", "ABCD")


class TestStructuralSimilarity:
    def test_identical_parameters_score_one(self, battery: Parameter) -> None:
        assert structural_similarity(battery, battery) == 1.0

    def test_known_answer_unit_only(self, battery: Parameter) -> None:
        # Only 'unit' differs. Weighted fraction = (14.0 - 2.0) / 14.0.
        other = battery.with_fields(unit="degC")
        expected = (TOTAL_WEIGHT - FIELD_WEIGHTS["unit"]) / TOTAL_WEIGHT
        assert structural_similarity(battery, other) == pytest.approx(
            expected, rel=1e-12, abs=0.0
        )

    def test_known_answer_packet_only(self, battery: Parameter) -> None:
        other = battery.with_fields(packet="OTHER_HK")
        expected = (TOTAL_WEIGHT - FIELD_WEIGHTS["packet"]) / TOTAL_WEIGHT
        assert structural_similarity(battery, other) == pytest.approx(
            expected, rel=1e-12, abs=0.0
        )

    def test_known_answer_limits_only(self, battery: Parameter) -> None:
        other = battery.with_fields(low_limit=0.0)
        expected = (TOTAL_WEIGHT - FIELD_WEIGHTS["limits"]) / TOTAL_WEIGHT
        assert structural_similarity(battery, other) == pytest.approx(
            expected, rel=1e-12, abs=0.0
        )

    def test_a_rename_does_not_move_the_structural_term(self, battery: Parameter) -> None:
        assert structural_similarity(battery, battery.with_fields(name="OTHER")) == 1.0

    def test_commands_compare_on_their_argument_signature(self, heater: Command) -> None:
        assert structural_similarity(heater, heater) == 1.0

    def test_a_command_and_a_parameter_cannot_be_compared(
        self, heater: Command, battery: Parameter
    ) -> None:
        with pytest.raises(TypeError, match="parameter with a parameter"):
            structural_similarity(heater, battery)

    def test_total_weight_is_what_the_tests_assume(self) -> None:
        assert TOTAL_WEIGHT == pytest.approx(14.0, rel=1e-12, abs=0.0)


class TestPairScore:
    def test_identical_items_score_one(self, battery: Parameter) -> None:
        score, structural, name, description = pair_score(battery, battery)
        assert (score, structural, name, description) == (1.0, 1.0, 1.0, 1.0)

    def test_known_answer_combination(self, battery: Parameter) -> None:
        # Structure identical, description identical, name "BATT_T_1" against
        # "BATT_T_2": Ratcliff/Obershelp matches 7 of 8 characters on each
        # side, so 2 * 7 / 16 = 0.875. Score = 0.55 + 0.30 * 0.875 + 0.15.
        other = battery.with_fields(name="BATT_T_2")
        score, structural, name, description = pair_score(battery, other)
        assert structural == 1.0
        assert description == 1.0
        assert name == pytest.approx(0.875, rel=1e-12, abs=0.0)
        assert score == pytest.approx(0.55 + 0.30 * 0.875 + 0.15, rel=1e-12, abs=0.0)

    def test_weights_must_sum_to_one(self, battery: Parameter) -> None:
        with pytest.raises(ValueError, match="must sum to 1.0"):
            pair_score(battery, battery, {"structural": 0.5, "name": 0.2, "description": 0.1})

    def test_weights_must_be_complete(self, battery: Parameter) -> None:
        with pytest.raises(ValueError, match="must define"):
            pair_score(battery, battery, {"structural": 1.0})

    def test_negative_weights_rejected(self, battery: Parameter) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            pair_score(battery, battery, {"structural": 1.5, "name": -0.5, "description": 0.0})

    def test_default_weights_sum_to_one(self) -> None:
        assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0, rel=1e-12, abs=0.0)

    def test_score_lies_in_the_unit_interval(self, battery: Parameter, mode: Parameter) -> None:
        score, _, _, _ = pair_score(battery, mode)
        assert 0.0 <= score <= 1.0


class TestMatchRenames:
    def test_an_obvious_rename_is_accepted(self, battery: Parameter) -> None:
        renamed = battery.with_fields(name="BATTERY_TEMPERATURE_1")
        candidates = match_renames([battery], [renamed], threshold=0.6)
        assert candidates[0].accepted
        assert candidates[0].new_name == "BATTERY_TEMPERATURE_1"

    def test_a_poor_match_is_produced_but_not_accepted(
        self, battery: Parameter, mode: Parameter
    ) -> None:
        candidates = match_renames([battery], [mode], threshold=0.9)
        assert len(candidates) == 1
        assert not candidates[0].accepted

    def test_empty_pools_produce_no_candidates(self, battery: Parameter) -> None:
        assert match_renames([], [battery], threshold=0.5) == ()
        assert match_renames([battery], [], threshold=0.5) == ()

    def test_candidates_are_sorted_by_descending_score(self, battery: Parameter) -> None:
        added = [
            battery.with_fields(name="BATT_T_2"),
            battery.with_fields(name="ZZZZZZZZ", unit="x"),
        ]
        removed = [battery, battery.with_fields(name="BATT_T_9")]
        candidates = match_renames(removed, added, threshold=0.0)
        scores = [candidate.score for candidate in candidates]
        assert scores == sorted(scores, reverse=True)

    def test_matching_is_one_to_one(self, battery: Parameter) -> None:
        removed = [battery, battery.with_fields(name="BATT_T_2")]
        added = [battery.with_fields(name="BATT_T_3")]
        candidates = match_renames(removed, added, threshold=0.0)
        assert len({c.new_name for c in candidates}) == len(candidates)

    @pytest.mark.parametrize("strategy", STRATEGIES)
    def test_both_strategies_run(self, battery: Parameter, strategy: str) -> None:
        renamed = battery.with_fields(name="BATT_T_2")
        candidates = match_renames([battery], [renamed], threshold=0.5, strategy=strategy)
        assert candidates[0].accepted

    @pytest.mark.parametrize("strategy", STRATEGIES)
    def test_both_strategies_agree_on_a_single_pair(
        self, battery: Parameter, strategy: str
    ) -> None:
        renamed = battery.with_fields(name="BATT_T_2")
        reference = match_renames([battery], [renamed], threshold=0.5, strategy="optimal")
        other = match_renames([battery], [renamed], threshold=0.5, strategy=strategy)
        assert other[0].score == pytest.approx(reference[0].score, rel=1e-12, abs=0.0)

    def test_unknown_strategy_rejected(self, battery: Parameter) -> None:
        with pytest.raises(ValueError, match="strategy must be one of"):
            match_renames([battery], [battery], threshold=0.5, strategy="magic")

    @pytest.mark.parametrize("threshold", [-0.1, 1.1])
    def test_threshold_outside_the_unit_interval_rejected(
        self, battery: Parameter, threshold: float
    ) -> None:
        with pytest.raises(ValueError, match=r"threshold must lie in \[0, 1\]"):
            match_renames([battery], [battery], threshold=threshold)

    def test_negative_ambiguous_band_rejected(self, battery: Parameter) -> None:
        with pytest.raises(ValueError, match="ambiguous_band must be >= 0"):
            match_renames([battery], [battery], threshold=0.5, ambiguous_band=-0.1)

    def test_a_pair_on_the_threshold_is_marked_ambiguous(self, battery: Parameter) -> None:
        renamed = battery.with_fields(name="BATT_T_2")
        score, _, _, _ = pair_score(battery, renamed)
        candidates = match_renames([battery], [renamed], threshold=score)
        assert candidates[0].ambiguous
        assert candidates[0].accepted

    def test_the_ambiguous_band_has_a_declared_width(self) -> None:
        assert 0.0 < AMBIGUOUS_BAND < 0.5

    def test_matching_is_deterministic(self, battery: Parameter) -> None:
        removed = [battery, battery.with_fields(name="BATT_T_2")]
        added = [battery.with_fields(name="BATT_T_3"), battery.with_fields(name="BATT_T_4")]
        first = match_renames(removed, added, threshold=0.3)
        second = match_renames(removed, added, threshold=0.3)
        assert first == second

    def test_commands_can_be_matched(self, heater: Command) -> None:
        renamed = Command(
            name="SET_HEATER_DUTY",
            path=heater.path,
            description=heater.description,
            arguments=heater.arguments,
        )
        candidates = match_renames([heater], [renamed], threshold=0.6)
        assert candidates[0].accepted


class TestRenameCandidate:
    def test_a_score_outside_the_unit_interval_is_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"must lie in \[0, 1\]"):
            RenameCandidate(
                old_name="A",
                new_name="B",
                score=1.5,
                structural=1.0,
                name_similarity=1.0,
                description_similarity=1.0,
                accepted=True,
                ambiguous=False,
            )

    def test_a_non_finite_score_is_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"must lie in \[0, 1\]"):
            RenameCandidate(
                old_name="A",
                new_name="B",
                score=math.nan,
                structural=1.0,
                name_similarity=1.0,
                description_similarity=1.0,
                accepted=True,
                ambiguous=False,
            )


class TestScoreMatrixAndBound:
    def test_matrix_shape(self, battery: Parameter, mode: Parameter) -> None:
        matrix = score_matrix([battery, mode], [battery])
        assert len(matrix) == 2
        assert len(matrix[0]) == 1

    def test_collision_bound_on_distinct_items(self, battery: Parameter, mode: Parameter) -> None:
        colliding, distinct, fraction = signature_collision_bound([battery, mode])
        assert (colliding, distinct) == (0, 2)
        assert fraction == 0.0

    def test_collision_bound_counts_identical_signatures(self, battery: Parameter) -> None:
        twin = battery.with_fields(name="OTHER", path="")
        colliding, distinct, fraction = signature_collision_bound([battery, twin])
        assert (colliding, distinct) == (2, 1)
        assert fraction == 1.0

    def test_collision_bound_on_an_empty_list(self) -> None:
        assert signature_collision_bound([]) == (0, 0, 0.0)

    def test_an_argument_rename_changes_the_command_signature(self, heater: Command) -> None:
        renamed_argument = Command(
            name=heater.name,
            arguments=(
                CommandArgument(name="HTR_ID", position=0, bit_width=8),
                CommandArgument(name="DUTY", position=1, bit_width=8, unit="percent"),
            ),
        )
        assert structural_similarity(heater, renamed_argument) < 1.0
