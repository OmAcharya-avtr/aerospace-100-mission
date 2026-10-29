"""Impact classification, the coverage verdict, and the statement it must not make."""

from __future__ import annotations

import re

import pytest

from tmtcdiff.changes import ChangeClass
from tmtcdiff.corpus import Corpus, CorpusSource
from tmtcdiff.diff import DiffConfig, diff_dictionaries
from tmtcdiff.impact import (
    IMPACT_NOTES,
    CoverageVerdict,
    ImpactClass,
    assess_impact,
)
from tmtcdiff.model import Command, CommandArgument, Dictionary, Parameter

EXACT = DiffConfig(numeric_rel_tol=0.0)


def _corpus(text: str, kind: str = "script") -> Corpus:
    return Corpus((CorpusSource(label="probe/one.py", kind=kind, text=text),))


def _assess(old: Dictionary, new: Dictionary, text: str, kind: str = "script"):
    result = diff_dictionaries(old, new, EXACT)
    return assess_impact(result, _corpus(text, kind), old, new)


def _replace(dictionary: Dictionary, parameter: Parameter) -> Dictionary:
    return Dictionary(
        version="1.1",
        parameters=tuple(
            parameter if p.name == parameter.name else p for p in dictionary.parameters
        ),
        commands=dictionary.commands,
    )


class TestImpactClassOrdering:
    def test_silently_changed_ranks_first(self) -> None:
        assert ImpactClass.SILENTLY_CHANGED.rank == 0

    def test_broken_ranks_below_silently_changed(self) -> None:
        assert ImpactClass.BROKEN.rank > ImpactClass.SILENTLY_CHANGED.rank

    def test_unaffected_ranks_last(self) -> None:
        assert ImpactClass.UNAFFECTED.rank == max(c.rank for c in ImpactClass)

    def test_every_class_has_a_note(self) -> None:
        assert set(IMPACT_NOTES) == set(ImpactClass)

    def test_ranks_are_distinct(self) -> None:
        assert len({c.rank for c in ImpactClass}) == len(list(ImpactClass))


class TestClassification:
    def test_a_silent_change_is_reported_as_silently_changed(
        self, dictionary: Dictionary, battery: Parameter
    ) -> None:
        new = _replace(dictionary, battery.with_fields(unit="degC"))
        report = _assess(dictionary, new, 'x = tm["BATT_T_1"]')
        assert report.findings[0].impact_class is ImpactClass.SILENTLY_CHANGED

    def test_the_change_classes_are_carried_into_the_finding(
        self, dictionary: Dictionary, battery: Parameter
    ) -> None:
        new = _replace(dictionary, battery.with_fields(unit="degC"))
        report = _assess(dictionary, new, 'x = tm["BATT_T_1"]')
        assert ChangeClass.UNIT_CHANGED in report.findings[0].change_classes

    def test_a_removal_is_reported_as_broken(self, dictionary: Dictionary) -> None:
        new = Dictionary(
            version="1.1",
            parameters=tuple(p for p in dictionary.parameters if p.name != "BATT_T_1"),
            commands=dictionary.commands,
        )
        report = _assess(dictionary, new, 'x = tm["BATT_T_1"]')
        assert report.findings[0].impact_class is ImpactClass.BROKEN

    def test_a_rename_is_reported_with_a_suggestion(
        self, dictionary: Dictionary, battery: Parameter
    ) -> None:
        renamed = battery.with_fields(name="BATTERY_TEMP_1", path="TM/PWR/BATTERY_TEMP_1")
        new = Dictionary(
            version="1.1",
            parameters=(renamed, dictionary.parameters[0]),
            commands=dictionary.commands,
        )
        report = _assess(dictionary, new, 'x = tm["BATT_T_1"]')
        assert report.findings[0].impact_class is ImpactClass.RENAMED_AWAY
        assert report.findings[0].suggestion == "BATTERY_TEMP_1"

    def test_an_unchanged_item_is_unaffected(self, dictionary: Dictionary) -> None:
        report = _assess(dictionary, dictionary, 'x = tm["BATT_T_1"]')
        assert report.findings[0].impact_class is ImpactClass.UNAFFECTED

    def test_a_run_time_reference_is_undecidable(self, dictionary: Dictionary) -> None:
        report = _assess(dictionary, dictionary, 'x = tm["BATT_T_%d" % index]')
        assert report.findings[0].impact_class is ImpactClass.UNDECIDABLE

    def test_a_reference_to_nothing_is_broken_and_says_so(self, dictionary: Dictionary) -> None:
        report = _assess(dictionary, dictionary, 'x = tm["NEVER_EXISTED"]')
        assert report.findings[0].impact_class is ImpactClass.BROKEN
        assert "already unresolved" in report.findings[0].detail

    def test_a_by_path_reference_breaks_when_the_path_moves(
        self, dictionary: Dictionary, battery: Parameter
    ) -> None:
        new = _replace(dictionary, battery.with_fields(path="TM/OTHER/BATT_T_1"))
        report = _assess(dictionary, new, 'x = tm["TM/PWR/BATT_T_1"]')
        assert report.findings[0].impact_class is ImpactClass.BROKEN

    def test_a_by_name_reference_survives_a_path_move(
        self, dictionary: Dictionary, battery: Parameter
    ) -> None:
        new = _replace(dictionary, battery.with_fields(path="TM/OTHER/BATT_T_1"))
        report = _assess(dictionary, new, 'x = tm["BATT_T_1"]')
        assert report.findings[0].impact_class is ImpactClass.UNAFFECTED


class TestEnumerationLabelImpact:
    def test_a_recoded_label_is_silently_changed(
        self, dictionary: Dictionary, mode: Parameter
    ) -> None:
        new = _replace(
            dictionary,
            mode.with_fields(enum=((0, "IDLE"), (1, "NOMINAL_MODE"), (2, "SAFE_MODE"))),
        )
        report = _assess(dictionary, new, 'if m == "SAFE_MODE":')
        assert report.findings[0].impact_class is ImpactClass.SILENTLY_CHANGED
        assert "moved from" in report.findings[0].detail

    def test_a_removed_label_says_the_test_never_fires(
        self, dictionary: Dictionary, mode: Parameter
    ) -> None:
        new = _replace(dictionary, mode.with_fields(enum=mode.enum[:-1]))
        report = _assess(dictionary, new, 'if m == "NOMINAL_MODE":')
        assert report.findings[0].impact_class is ImpactClass.SILENTLY_CHANGED
        assert "never fires" in report.findings[0].detail

    def test_a_new_label_is_unaffected(self, dictionary: Dictionary, mode: Parameter) -> None:
        new = _replace(dictionary, mode.with_fields(enum=(*mode.enum, (3, "SCIENCE_MODE"))))
        report = _assess(dictionary, new, 'if m == "SCIENCE_MODE":')
        assert report.findings[0].impact_class is ImpactClass.UNAFFECTED

    def test_a_label_that_is_not_in_either_version_is_unaffected(
        self, dictionary: Dictionary
    ) -> None:
        report = _assess(dictionary, dictionary, 'if m == "NOT_A_LABEL":')
        assert report.findings[0].impact_class is ImpactClass.UNAFFECTED


class TestCommandImpact:
    def test_reordered_arguments_are_silently_changed(self, dictionary: Dictionary) -> None:
        swapped = Command(
            name="SET_HEATER",
            path="TC/PWR/SET_HEATER",
            description="set a heater duty cycle",
            arguments=(
                CommandArgument(name="DUTY", position=0, bit_width=8, unit="percent"),
                CommandArgument(name="HEATER_ID", position=1, bit_width=8),
            ),
        )
        new = Dictionary(
            version="1.1", parameters=dictionary.parameters, commands=(swapped,)
        )
        report = _assess(dictionary, new, 'send_command("SET_HEATER", HEATER_ID=3, DUTY=50)')
        classes = {finding.impact_class for finding in report.findings}
        assert ImpactClass.SILENTLY_CHANGED in classes

    def test_a_missing_argument_is_broken(self, dictionary: Dictionary, heater: Command) -> None:
        trimmed = Command(
            name=heater.name,
            path=heater.path,
            description=heater.description,
            arguments=heater.arguments[:1],
        )
        new = Dictionary(version="1.1", parameters=dictionary.parameters, commands=(trimmed,))
        report = _assess(dictionary, new, 'send_command("SET_HEATER", DUTY=50)')
        assert any(
            finding.impact_class is ImpactClass.BROKEN
            and finding.reference.kind == "argument"
            for finding in report.findings
        )

    def test_a_removed_command_breaks_its_argument_references(
        self, dictionary: Dictionary
    ) -> None:
        new = Dictionary(version="1.1", parameters=dictionary.parameters, commands=())
        report = _assess(dictionary, new, 'send_command("SET_HEATER", DUTY=50)')
        assert all(
            finding.impact_class is ImpactClass.BROKEN for finding in report.findings
        )


class TestOrdering:
    def test_findings_are_ranked_by_impact_class(
        self, dictionary: Dictionary, battery: Parameter
    ) -> None:
        new = Dictionary(
            version="1.1",
            parameters=(battery.with_fields(unit="degC"),),
            commands=dictionary.commands,
        )
        text = '\n'.join(
            ['x = tm["AOCS_MODE"]', 'y = tm["BATT_T_1"]', 'z = tm["GONE"]']
        )
        report = _assess(dictionary, new, text)
        ranks = [finding.rank for finding in report.findings]
        assert ranks == sorted(ranks)

    def test_actionable_excludes_unaffected_and_undecidable(
        self, dictionary: Dictionary, battery: Parameter
    ) -> None:
        new = _replace(dictionary, battery.with_fields(unit="degC"))
        text = '\n'.join(['x = tm["BATT_T_1"]', 'y = tm["AOCS_MODE"]', "z = tm[name]"])
        report = _assess(dictionary, new, text)
        assert len(report.actionable) == 1

    def test_class_counts_lists_every_class(self, dictionary: Dictionary) -> None:
        report = _assess(dictionary, dictionary, 'x = tm["BATT_T_1"]')
        assert set(report.class_counts) == set(ImpactClass)


class TestCoverageVerdict:
    def test_full_coverage_with_no_change_is_determinable(self, dictionary: Dictionary) -> None:
        text = "\n".join(f'x = tm["{name}"]' for name in dictionary.item_names)
        report = _assess(dictionary, dictionary, text)
        assert report.verdict.determinable

    def test_an_unreferenced_item_is_a_reason(self, dictionary: Dictionary) -> None:
        report = _assess(dictionary, dictionary, 'x = tm["BATT_T_1"]')
        assert not report.verdict.determinable
        assert any("never referenced" in reason for reason in report.verdict.reasons)

    def test_an_undecidable_reference_is_a_reason(self, dictionary: Dictionary) -> None:
        text = "\n".join(
            [*[f'x = tm["{name}"]' for name in dictionary.item_names], "y = tm[name]"]
        )
        report = _assess(dictionary, dictionary, text)
        assert any("constructed at run time" in reason for reason in report.verdict.reasons)

    def test_an_unreadable_source_is_a_reason(self, dictionary: Dictionary) -> None:
        result = diff_dictionaries(dictionary, dictionary, EXACT)
        corpus = Corpus(sources=(), unreadable=("dump.bin (not recognised)",))
        report = assess_impact(result, corpus, dictionary, dictionary)
        assert any("could not be read" in reason for reason in report.verdict.reasons)

    def test_an_empty_corpus_is_a_reason(self, dictionary: Dictionary) -> None:
        result = diff_dictionaries(dictionary, dictionary, EXACT)
        report = assess_impact(result, Corpus(), dictionary, dictionary)
        assert any("corpus is empty" in reason for reason in report.verdict.reasons)

    def test_a_changed_item_outside_the_corpus_is_a_reason(
        self, dictionary: Dictionary, battery: Parameter
    ) -> None:
        new = _replace(dictionary, battery.with_fields(unit="degC"))
        text = "\n".join(
            f'x = tm["{name}"]' for name in dictionary.item_names if name != "BATT_T_1"
        )
        report = _assess(dictionary, new, text)
        assert any("changed item(s) are not referenced" in r for r in report.verdict.reasons)

    def test_coverage_fraction_is_counted(self, dictionary: Dictionary) -> None:
        report = _assess(dictionary, dictionary, 'x = tm["BATT_T_1"]')
        assert report.verdict.items_referenced == 1
        assert report.verdict.dictionary_items == 3
        assert report.verdict.coverage_fraction == pytest.approx(1 / 3, rel=1e-12, abs=0.0)

    def test_decidable_fraction_is_counted(self, dictionary: Dictionary) -> None:
        report = _assess(dictionary, dictionary, 'x = tm["BATT_T_1"]\ny = tm[name]')
        assert report.verdict.decidable_fraction == pytest.approx(0.5, rel=1e-12, abs=0.0)

    def test_an_empty_dictionary_has_zero_coverage_not_a_division_error(self) -> None:
        empty = Dictionary(version="1.0")
        result = diff_dictionaries(empty, empty, EXACT)
        report = assess_impact(result, Corpus(), empty, empty)
        assert report.verdict.coverage_fraction == 0.0

    def test_changed_item_coverage_is_one_when_nothing_changed(
        self, dictionary: Dictionary
    ) -> None:
        report = _assess(dictionary, dictionary, 'x = tm["BATT_T_1"]')
        assert report.verdict.changed_item_coverage == 1.0

    def test_decidable_fraction_of_no_references_is_zero(self, dictionary: Dictionary) -> None:
        result = diff_dictionaries(dictionary, dictionary, EXACT)
        report = assess_impact(result, Corpus(), dictionary, dictionary)
        assert report.verdict.decidable_fraction == 0.0

    def test_report_lines_name_every_count(self, dictionary: Dictionary) -> None:
        report = _assess(dictionary, dictionary, 'x = tm["BATT_T_1"]')
        text = "\n".join(report.verdict.report_lines())
        for fragment in ("references extracted", "items never referenced", "fully determinable"):
            assert fragment in text

    def test_an_ambiguous_label_is_only_reported_when_referenced(self) -> None:
        shared = (
            Parameter(name="A", data_type="enum", enum=((0, "ON"), (1, "OFF"))),
            Parameter(name="B", data_type="enum", enum=((0, "ON"), (1, "OFF"))),
        )
        dictionary = Dictionary(version="1.0", parameters=shared)
        result = diff_dictionaries(dictionary, dictionary, EXACT)
        quiet = assess_impact(
            result, _corpus('x = tm["A"]\ny = tm["B"]'), dictionary, dictionary
        )
        assert quiet.verdict.ambiguous_enum_labels == ()
        loud = assess_impact(
            result,
            _corpus('x = tm["A"]\ny = tm["B"]\nif s == "ON":'),
            dictionary,
            dictionary,
        )
        assert loud.verdict.ambiguous_enum_labels == ("ON",)


class TestInterpretation:
    def test_an_affected_report_counts_the_classes(
        self, dictionary: Dictionary, battery: Parameter
    ) -> None:
        new = _replace(dictionary, battery.with_fields(unit="degC"))
        text = "\n".join(f'x = tm["{name}"]' for name in dictionary.item_names)
        report = _assess(dictionary, new, text)
        assert "1 silently changed" in report.interpretation

    def test_a_complete_empty_report_is_scoped_to_this_corpus(
        self, dictionary: Dictionary
    ) -> None:
        text = "\n".join(f'x = tm["{name}"]' for name in dictionary.item_names)
        report = _assess(dictionary, dictionary, text)
        assert "this corpus and these two dictionary versions only" in report.interpretation

    def test_an_incomplete_empty_report_says_it_is_incomplete(
        self, dictionary: Dictionary
    ) -> None:
        report = _assess(dictionary, dictionary, 'x = tm["BATT_T_1"]')
        assert "NOT complete" in report.interpretation
        assert "absence of evidence" in report.interpretation

    @pytest.mark.parametrize("word", ["safe", "certified", "flight-qualified", "production-ready"])
    def test_no_interpretation_contains_a_safety_word(
        self, dictionary: Dictionary, battery: Parameter, word: str
    ) -> None:
        new = _replace(dictionary, battery.with_fields(unit="degC"))
        for text in (
            'x = tm["BATT_T_1"]',
            "\n".join(f'x = tm["{name}"]' for name in dictionary.item_names),
        ):
            for old, newer in ((dictionary, dictionary), (dictionary, new)):
                report = _assess(old, newer, text)
                assert (
                    re.search(
                        rf"\b{re.escape(word)}\b", report.interpretation, re.IGNORECASE
                    )
                    is None
                )


class TestAssessImpactValidation:
    def test_diff_type_is_checked(self, dictionary: Dictionary) -> None:
        with pytest.raises(TypeError, match="expects a DiffResult"):
            assess_impact("not a diff", Corpus(), dictionary, dictionary)  # type: ignore[arg-type]

    def test_corpus_type_is_checked(self, dictionary: Dictionary) -> None:
        result = diff_dictionaries(dictionary, dictionary, EXACT)
        with pytest.raises(TypeError, match="expects a Corpus"):
            assess_impact(result, "not a corpus", dictionary, dictionary)  # type: ignore[arg-type]

    def test_dictionary_types_are_checked(self, dictionary: Dictionary) -> None:
        result = diff_dictionaries(dictionary, dictionary, EXACT)
        with pytest.raises(TypeError, match="Dictionary for old"):
            assess_impact(result, Corpus(), "old", dictionary)  # type: ignore[arg-type]

    def test_references_can_be_supplied_directly(self, dictionary: Dictionary) -> None:
        from tmtcdiff.corpus import extract_references

        corpus = _corpus('x = tm["BATT_T_1"]')
        references = extract_references(corpus)
        result = diff_dictionaries(dictionary, dictionary, EXACT)
        report = assess_impact(result, corpus, dictionary, dictionary, references=references)
        assert len(report.findings) == 1


class TestCoverageVerdictRecord:
    def test_determinable_is_driven_by_the_reasons(self) -> None:
        verdict = CoverageVerdict(
            corpus_sources=1,
            unreadable_sources=(),
            references_total=1,
            references_undecidable=0,
            dictionary_items=1,
            items_referenced=1,
            items_unreferenced=0,
            changed_items_total=0,
            changed_items_referenced=0,
            changed_items_unreferenced=0,
            ambiguous_renames=0,
            incomparable_fields=0,
            ambiguous_enum_labels=(),
            reasons=(),
        )
        assert verdict.determinable
