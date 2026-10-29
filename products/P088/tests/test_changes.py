"""The taxonomy: tiers, ranking, and the invariants the product's claim rests on."""

from __future__ import annotations

import pytest

from tmtcdiff.changes import (
    ADVISORY_CLASSES,
    CLASS_NOTES,
    CLASS_ORDER,
    CLASS_SEVERITY,
    LOUD_CLASSES,
    SILENT_CLASSES,
    Change,
    ChangeClass,
    Severity,
    rank_changes,
)


class TestTaxonomyCompleteness:
    def test_every_class_has_a_tier(self) -> None:
        assert set(CLASS_SEVERITY) == set(ChangeClass)

    def test_every_class_has_a_note(self) -> None:
        assert set(CLASS_NOTES) == set(ChangeClass)

    def test_the_declared_order_covers_every_class_once(self) -> None:
        assert sorted(CLASS_ORDER, key=lambda c: c.value) == sorted(
            ChangeClass, key=lambda c: c.value
        )
        assert len(CLASS_ORDER) == len(set(CLASS_ORDER))

    def test_the_three_tier_lists_partition_the_taxonomy(self) -> None:
        union = set(SILENT_CLASSES) | set(LOUD_CLASSES) | set(ADVISORY_CLASSES)
        assert union == set(ChangeClass)
        assert len(SILENT_CLASSES) + len(LOUD_CLASSES) + len(ADVISORY_CLASSES) == len(ChangeClass)

    def test_tier_lists_agree_with_the_tier_mapping(self) -> None:
        for change_class in SILENT_CLASSES:
            assert change_class.severity is Severity.SILENT
        for change_class in LOUD_CLASSES:
            assert change_class.severity is Severity.LOUD
        for change_class in ADVISORY_CLASSES:
            assert change_class.severity is Severity.ADVISORY

    def test_the_silent_tier_is_the_largest(self) -> None:
        # Not a coincidence worth hiding: a dictionary has more ways to change
        # meaning quietly than to break loudly.
        assert len(SILENT_CLASSES) > len(LOUD_CLASSES)
        assert len(SILENT_CLASSES) > len(ADVISORY_CLASSES)

    def test_every_note_is_a_sentence_fragment_not_a_placeholder(self) -> None:
        for note in CLASS_NOTES.values():
            assert len(note) > 20
            assert "TODO" not in note


class TestSeverityRank:
    def test_silent_outranks_loud(self) -> None:
        assert Severity.SILENT.rank < Severity.LOUD.rank

    def test_loud_outranks_advisory(self) -> None:
        assert Severity.LOUD.rank < Severity.ADVISORY.rank

    def test_ranks_are_distinct(self) -> None:
        assert len({severity.rank for severity in Severity}) == len(list(Severity))

    def test_is_silent_agrees_with_the_tier(self) -> None:
        for change_class in ChangeClass:
            assert change_class.is_silent == (change_class.severity is Severity.SILENT)


class TestChange:
    def test_kind_is_validated(self) -> None:
        with pytest.raises(ValueError, match="must be 'parameter' or 'command'"):
            Change(change_class=ChangeClass.RESCALED, kind="widget", name="A")

    def test_class_type_is_validated(self) -> None:
        with pytest.raises(TypeError, match="must be a ChangeClass"):
            Change(change_class="rescaled", kind="parameter", name="A")  # type: ignore[arg-type]

    def test_severity_comes_from_the_class(self) -> None:
        change = Change(change_class=ChangeClass.REMOVED, kind="parameter", name="A")
        assert change.severity is Severity.LOUD

    def test_one_line_names_the_class_and_the_item(self) -> None:
        change = Change(
            change_class=ChangeClass.RESCALED,
            kind="parameter",
            name="BATT_T_1",
            field="scale",
            old_value=1.0,
            new_value=2.0,
        )
        line = change.one_line()
        assert "rescaled" in line
        assert "BATT_T_1" in line
        assert "scale" in line

    def test_one_line_shows_the_new_name_for_a_rename(self) -> None:
        change = Change(
            change_class=ChangeClass.RENAMED,
            kind="parameter",
            name="OLD",
            new_name="NEW",
            similarity=0.9,
        )
        assert "OLD -> NEW" in change.one_line()
        assert "0.9000" in change.one_line()

    def test_sort_key_orders_by_tier_first(self) -> None:
        silent = Change(change_class=ChangeClass.RESCALED, kind="parameter", name="Z")
        loud = Change(change_class=ChangeClass.REMOVED, kind="parameter", name="A")
        assert silent.sort_key < loud.sort_key


class TestRanking:
    def test_silent_changes_come_first(self) -> None:
        changes = (
            Change(change_class=ChangeClass.ADDED, kind="parameter", name="C"),
            Change(change_class=ChangeClass.REMOVED, kind="parameter", name="B"),
            Change(change_class=ChangeClass.RESCALED, kind="parameter", name="A"),
        )
        ranked = rank_changes(changes)
        assert [change.change_class for change in ranked] == [
            ChangeClass.RESCALED,
            ChangeClass.REMOVED,
            ChangeClass.ADDED,
        ]

    def test_ranking_is_stable_across_names_within_a_class(self) -> None:
        changes = tuple(
            Change(change_class=ChangeClass.RESCALED, kind="parameter", name=name)
            for name in ("C", "A", "B")
        )
        assert [change.name for change in rank_changes(changes)] == ["A", "B", "C"]

    def test_ranking_is_idempotent(self) -> None:
        changes = (
            Change(change_class=ChangeClass.REMOVED, kind="parameter", name="B"),
            Change(change_class=ChangeClass.RESCALED, kind="parameter", name="A"),
        )
        once = rank_changes(changes)
        assert rank_changes(once) == once

    def test_empty_input(self) -> None:
        assert rank_changes(()) == ()

    def test_every_pair_of_tiers_is_ordered(self) -> None:
        changes = tuple(
            Change(change_class=change_class, kind="parameter", name="A")
            for change_class in ChangeClass
        )
        ranks = [change.severity.rank for change in rank_changes(changes)]
        assert ranks == sorted(ranks)
