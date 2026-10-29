"""The generator: determinism, ground truth, eligibility and the reordering control."""

from __future__ import annotations

import pytest

from tmtcdiff.changes import ChangeClass
from tmtcdiff.diff import DiffConfig, diff_dictionaries
from tmtcdiff.parse import loads
from tmtcdiff.rename import signature_collision_bound
from tmtcdiff.synth import (
    INJECTABLE,
    INJECTABLE_SILENT,
    REORDER_ALIASES,
    InjectedChange,
    baseline_dictionary,
    change_pair,
    inject_changes,
    rename_pair,
    reordering_control,
    synthetic_corpus,
)

EXACT = DiffConfig(numeric_rel_tol=0.0)


class TestBaselineDictionary:
    def test_size_is_honoured(self) -> None:
        assert len(baseline_dictionary(40, 1).parameters) == 40

    def test_command_count_floor(self) -> None:
        assert len(baseline_dictionary(4, 1).commands) == 3

    def test_command_count_scales(self) -> None:
        assert len(baseline_dictionary(80, 1).commands) == 20

    def test_determinism(self) -> None:
        assert baseline_dictionary(30, 7) == baseline_dictionary(30, 7)

    def test_different_seeds_differ(self) -> None:
        assert baseline_dictionary(30, 7) != baseline_dictionary(30, 8)

    def test_names_are_unique(self) -> None:
        dictionary = baseline_dictionary(200, 3)
        names = [parameter.name for parameter in dictionary.parameters]
        assert len(set(names)) == len(names)

    def test_some_parameters_carry_an_enumeration(self) -> None:
        dictionary = baseline_dictionary(60, 3)
        assert any(parameter.enum for parameter in dictionary.parameters)

    def test_paths_are_populated(self) -> None:
        dictionary = baseline_dictionary(20, 3)
        assert all(parameter.path.startswith("TM/") for parameter in dictionary.parameters)

    def test_too_small_rejected(self) -> None:
        with pytest.raises(ValueError, match="size must be >= 4"):
            baseline_dictionary(3, 1)

    def test_non_integer_size_rejected(self) -> None:
        with pytest.raises(TypeError, match="must be an int"):
            baseline_dictionary(30.0, 1)  # type: ignore[arg-type]

    def test_bit_offsets_wrap_so_signatures_can_collide(self) -> None:
        # Deliberate: a generator giving every parameter a globally unique bit
        # offset makes every structural signature unique, which would make the
        # measured rename false-match rate zero at every threshold and the
        # sweep vacuous.
        colliding, _, _ = signature_collision_bound(list(baseline_dictionary(400, 4100).parameters))
        assert colliding > 0


class TestInjection:
    @pytest.mark.parametrize("change_class", list(INJECTABLE))
    def test_every_injectable_class_produces_ground_truth(
        self, change_class: ChangeClass
    ) -> None:
        pair = inject_changes(baseline_dictionary(60, 5), (change_class,), 6)
        assert len(pair.truth) == 1
        assert pair.truth[0].change_class is change_class

    @pytest.mark.parametrize("change_class", list(INJECTABLE))
    def test_every_injectable_class_is_detected(self, change_class: ChangeClass) -> None:
        pair = inject_changes(baseline_dictionary(60, 5), (change_class,), 6)
        result = diff_dictionaries(pair.old, pair.new, EXACT)
        detected = {(change.change_class, change.name) for change in result.changes}
        assert pair.truth_keys <= detected

    def test_injection_is_deterministic(self) -> None:
        first = inject_changes(baseline_dictionary(40, 2), (ChangeClass.RESCALED,), 3)
        second = inject_changes(baseline_dictionary(40, 2), (ChangeClass.RESCALED,), 3)
        assert first.new == second.new
        assert first.truth == second.truth

    def test_per_class_repeats_the_injection(self) -> None:
        pair = inject_changes(baseline_dictionary(60, 2), (ChangeClass.RESCALED,), 3, per_class=4)
        assert len(pair.truth) == 4
        assert len({change.name for change in pair.truth}) == 4

    def test_running_out_of_eligible_items_raises(self) -> None:
        with pytest.raises(ValueError, match="no parameter left"):
            inject_changes(baseline_dictionary(8, 2), (ChangeClass.BIT_SHIFTED,), 3, per_class=20)

    def test_running_out_of_commands_raises(self) -> None:
        with pytest.raises(ValueError, match="no command left"):
            inject_changes(
                baseline_dictionary(8, 2), (ChangeClass.ARGUMENT_REORDERED,), 3, per_class=20
            )

    def test_every_change_class_is_injectable(self) -> None:
        # The taxonomy and the generator are kept in step on purpose: a class
        # the generator cannot produce is a class with no measured recall.
        assert set(INJECTABLE) == set(ChangeClass)

    def test_something_that_is_not_a_change_class_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="cannot be injected"):
            inject_changes(baseline_dictionary(20, 2), ("not_a_change_class",), 3)  # type: ignore[arg-type]

    def test_per_class_below_one_rejected(self) -> None:
        with pytest.raises(ValueError, match="per_class must be >= 1"):
            inject_changes(baseline_dictionary(20, 2), (ChangeClass.RESCALED,), 3, per_class=0)

    def test_type_is_checked(self) -> None:
        with pytest.raises(TypeError, match="expects a Dictionary"):
            inject_changes("not a dictionary", (ChangeClass.RESCALED,), 3)  # type: ignore[arg-type]

    def test_a_removal_shortens_the_dictionary(self) -> None:
        old = baseline_dictionary(40, 2)
        pair = inject_changes(old, (ChangeClass.REMOVED,), 3)
        assert len(pair.new.parameters) == len(old.parameters) - 1

    def test_an_addition_lengthens_the_dictionary(self) -> None:
        old = baseline_dictionary(40, 2)
        pair = inject_changes(old, (ChangeClass.ADDED,), 3)
        assert len(pair.new.parameters) == len(old.parameters) + 1

    def test_a_rename_keeps_the_length(self) -> None:
        old = baseline_dictionary(40, 2)
        pair = inject_changes(old, (ChangeClass.RENAMED,), 3)
        assert len(pair.new.parameters) == len(old.parameters)

    def test_a_rename_records_the_new_name(self) -> None:
        pair = inject_changes(baseline_dictionary(40, 2), (ChangeClass.RENAMED,), 3)
        assert pair.truth[0].new_name in pair.new.parameter_map

    def test_a_rename_preserves_the_structural_signature(self) -> None:
        pair = inject_changes(baseline_dictionary(40, 2), (ChangeClass.RENAMED,), 3)
        old_item = pair.old.parameter_map[pair.truth[0].name]
        new_item = pair.new.parameter_map[pair.truth[0].new_name]
        assert old_item.structural_signature() == new_item.structural_signature()

    def test_truth_keys_are_class_and_name(self) -> None:
        pair = inject_changes(baseline_dictionary(40, 2), (ChangeClass.RESCALED,), 3)
        assert pair.truth_keys == {(ChangeClass.RESCALED, pair.truth[0].name)}

    def test_truth_for_filters_by_class(self) -> None:
        pair = change_pair(60, 2)
        assert all(
            change.change_class is ChangeClass.RENAMED
            for change in pair.truth_for(ChangeClass.RENAMED)
        )

    def test_injected_change_key(self) -> None:
        change = InjectedChange(ChangeClass.RESCALED, "parameter", "A")
        assert change.key == (ChangeClass.RESCALED, "A")


class TestChangePair:
    def test_all_classes_at_the_documented_minimum_size(self) -> None:
        pair = change_pair(28, 11)
        assert {change.change_class for change in pair.truth} == set(INJECTABLE)

    def test_below_the_minimum_size_raises(self) -> None:
        with pytest.raises(ValueError, match="no command left"):
            change_pair(20, 11)

    def test_a_restricted_class_list_is_honoured(self) -> None:
        pair = change_pair(40, 11, classes=(ChangeClass.RESCALED, ChangeClass.REMOVED))
        assert {change.change_class for change in pair.truth} == {
            ChangeClass.RESCALED,
            ChangeClass.REMOVED,
        }


class TestRenamePair:
    def test_counts_are_honoured(self) -> None:
        pair = rename_pair(60, 11, renames=3, removals=2, additions=2)
        counts: dict[ChangeClass, int] = {}
        for change in pair.truth:
            counts[change.change_class] = counts.get(change.change_class, 0) + 1
        assert counts[ChangeClass.RENAMED] == 3
        assert counts[ChangeClass.REMOVED] == 2
        assert counts[ChangeClass.ADDED] == 2

    def test_unpaired_items_exist_so_a_false_match_is_possible(self) -> None:
        pair = rename_pair(60, 11, renames=3, removals=2, additions=2)
        removed = set(pair.old.parameter_map) - set(pair.new.parameter_map)
        added = set(pair.new.parameter_map) - set(pair.old.parameter_map)
        assert len(removed) == 5
        assert len(added) == 5


class TestReorderingControl:
    def test_the_reordered_text_re_reads_identically(self) -> None:
        dictionary = baseline_dictionary(40, 13)
        _, reordered = reordering_control(dictionary, 14)
        assert loads(reordered).parameters == dictionary.parameters

    def test_commands_survive_the_reordering(self) -> None:
        dictionary = baseline_dictionary(40, 13)
        _, reordered = reordering_control(dictionary, 14)
        assert loads(reordered).commands == dictionary.commands

    def test_the_semantic_diff_is_empty(self) -> None:
        dictionary = baseline_dictionary(60, 13)
        _, reordered = reordering_control(dictionary, 14)
        assert diff_dictionaries(dictionary, loads(reordered), EXACT).changes == ()

    def test_the_text_actually_differs(self) -> None:
        dictionary = baseline_dictionary(40, 13)
        canonical, reordered = reordering_control(dictionary, 14)
        assert canonical != reordered

    def test_aliases_can_be_switched_off(self) -> None:
        dictionary = baseline_dictionary(40, 13)
        _, reordered = reordering_control(dictionary, 14, aliases=False)
        assert '"mnemonic"' not in reordered
        assert diff_dictionaries(dictionary, loads(reordered), EXACT).changes == ()

    def test_aliases_are_used_when_asked_for(self) -> None:
        dictionary = baseline_dictionary(40, 13)
        _, reordered = reordering_control(dictionary, 14, aliases=True)
        assert '"mnemonic"' in reordered

    def test_the_control_is_deterministic(self) -> None:
        dictionary = baseline_dictionary(40, 13)
        assert reordering_control(dictionary, 14) == reordering_control(dictionary, 14)

    def test_every_alias_is_one_the_parser_accepts(self) -> None:
        from tmtcdiff.parse import ARGUMENT_ALIASES, COMMAND_ALIASES, PARAMETER_ALIASES

        known = set(PARAMETER_ALIASES) | set(COMMAND_ALIASES) | set(ARGUMENT_ALIASES)
        for alias in REORDER_ALIASES.values():
            assert alias in known or alias in ("parameters", "commands", "version")

    @pytest.mark.parametrize("change_class", list(INJECTABLE_SILENT))
    def test_the_reordering_does_not_mask_a_real_change(
        self, change_class: ChangeClass
    ) -> None:
        pair = inject_changes(baseline_dictionary(60, 13), (change_class,), 14)
        _, reordered = reordering_control(pair.new, 15)
        result = diff_dictionaries(pair.old, loads(reordered), EXACT)
        detected = {(change.change_class, change.name) for change in result.changes}
        assert pair.truth_keys <= detected


class TestSyntheticCorpus:
    def test_sources_are_produced(self) -> None:
        corpus = synthetic_corpus(baseline_dictionary(40, 17), 18)
        assert len(corpus) > 0

    def test_labels_are_relative(self) -> None:
        corpus = synthetic_corpus(baseline_dictionary(40, 17), 18)
        assert all(not label.startswith("/") for label in corpus.labels)

    def test_every_named_item_is_referenced(self) -> None:
        from tmtcdiff.corpus import extract_references

        dictionary = baseline_dictionary(40, 17)
        names = [parameter.name for parameter in dictionary.parameters[:5]]
        corpus = synthetic_corpus(dictionary, 18, names=names)
        targets = {reference.target for reference in extract_references(corpus)}
        assert set(names) <= targets

    def test_undecidable_references_are_produced(self) -> None:
        from tmtcdiff.corpus import extract_references

        corpus = synthetic_corpus(baseline_dictionary(40, 17), 18, undecidable_per_script=2)
        assert any(not reference.decidable for reference in extract_references(corpus))

    def test_the_unreadable_source_is_optional(self) -> None:
        dictionary = baseline_dictionary(40, 17)
        assert synthetic_corpus(dictionary, 18).unreadable == ()
        assert synthetic_corpus(dictionary, 18, include_unreadable=True).unreadable != ()

    def test_determinism(self) -> None:
        dictionary = baseline_dictionary(40, 17)
        assert synthetic_corpus(dictionary, 18) == synthetic_corpus(dictionary, 18)

    def test_all_three_source_kinds_are_present(self) -> None:
        corpus = synthetic_corpus(baseline_dictionary(40, 17), 18)
        assert {source.kind for source in corpus.sources} == {"procedure", "script", "config"}
