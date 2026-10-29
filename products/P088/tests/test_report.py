"""Rendering: table layout, section order, and what must never appear in the text."""

from __future__ import annotations

import re

import pytest

from tmtcdiff.changes import ChangeClass
from tmtcdiff.corpus import Corpus, CorpusSource
from tmtcdiff.diff import DiffConfig, diff_dictionaries
from tmtcdiff.impact import assess_impact
from tmtcdiff.model import Dictionary, Parameter
from tmtcdiff.report import (
    format_table,
    impact_class_table,
    render_diff,
    render_impact,
    taxonomy_table,
)

EXACT = DiffConfig(numeric_rel_tol=0.0)


class TestFormatTable:
    def test_header_and_rule_and_rows(self) -> None:
        text = format_table(["a", "b"], [["1", "2"]])
        lines = text.splitlines()
        assert lines[0].startswith("a")
        assert set(lines[1]) <= {"-", " "}
        assert lines[2].startswith("1")

    def test_columns_are_as_wide_as_the_widest_cell(self) -> None:
        text = format_table(["a", "b"], [["longer", "2"]])
        assert "longer" in text.splitlines()[2]
        assert len(text.splitlines()[1].split("  ")[0]) == len("longer")

    def test_non_string_cells_are_rendered(self) -> None:
        assert "3" in format_table(["a"], [[3]])

    def test_a_short_row_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="expected 2"):
            format_table(["a", "b"], [["1"]])

    def test_no_row_still_renders_the_header(self) -> None:
        assert format_table(["a"], []).splitlines()[0] == "a"

    def test_trailing_whitespace_is_stripped(self) -> None:
        for line in format_table(["a", "b"], [["1", "2"]]).splitlines():
            assert line == line.rstrip()


class TestTaxonomyTable:
    def test_every_class_appears(self) -> None:
        text = taxonomy_table()
        for change_class in ChangeClass:
            assert change_class.value in text

    def test_the_silent_tier_is_printed_first(self) -> None:
        text = taxonomy_table()
        assert text.index("silent") < text.index("advisory")

    def test_impact_class_table_lists_every_class(self) -> None:
        from tmtcdiff.impact import ImpactClass

        text = impact_class_table()
        for impact_class in ImpactClass:
            assert impact_class.value in text


class TestRenderDiff:
    def test_the_silent_section_comes_first(self, dictionary: Dictionary) -> None:
        new = Dictionary(
            version="1.1",
            parameters=(
                dictionary.parameters[0],
                dictionary.parameters[1].with_fields(unit="degC"),
            ),
            commands=(),
        )
        text = render_diff(diff_dictionaries(dictionary, new, EXACT))
        assert text.index("-- silent (") < text.index("-- loud (")

    def test_the_versions_are_printed(self, dictionary: Dictionary) -> None:
        text = render_diff(diff_dictionaries(dictionary, dictionary, EXACT))
        assert "1.0 -> 1.0" in text

    def test_the_limits_section_is_always_present(self, dictionary: Dictionary) -> None:
        text = render_diff(diff_dictionaries(dictionary, dictionary, EXACT))
        assert "what this diff could not determine" in text

    def test_max_rows_truncates_and_says_so(self, dictionary: Dictionary) -> None:
        parameters = tuple(Parameter(name=f"P{index}") for index in range(10))
        old = Dictionary(version="1.0", parameters=parameters)
        new = Dictionary(version="1.1", parameters=())
        text = render_diff(diff_dictionaries(old, new, EXACT), max_rows=2)
        assert "more" in text

    def test_incomparable_fields_are_listed_when_present(self, dictionary: Dictionary) -> None:
        new = Dictionary(
            version="1.1",
            parameters=tuple(
                p.with_fields(extra=(("calib", "77"),)) for p in dictionary.parameters
            ),
            commands=dictionary.commands,
        )
        text = render_diff(diff_dictionaries(dictionary, new, EXACT))
        assert "incomparable fields, named" in text

    def test_the_within_tier_order_is_declared_as_a_convention(
        self, dictionary: Dictionary
    ) -> None:
        text = render_diff(diff_dictionaries(dictionary, dictionary, EXACT))
        assert "declared" in text and "convention" in text


class TestRenderImpact:
    def _report(self, dictionary: Dictionary):
        new = Dictionary(
            version="1.1",
            parameters=(
                dictionary.parameters[0],
                dictionary.parameters[1].with_fields(unit="degC"),
            ),
            commands=dictionary.commands,
        )
        result = diff_dictionaries(dictionary, new, EXACT)
        corpus = Corpus(
            (CorpusSource(label="a.py", kind="script", text='x = tm["BATT_T_1"]\ny = tm[n]'),)
        )
        return assess_impact(result, corpus, dictionary, new)

    def test_the_class_summary_is_printed(self, dictionary: Dictionary) -> None:
        assert "silently_changed" in render_impact(self._report(dictionary))

    def test_the_coverage_verdict_is_printed(self, dictionary: Dictionary) -> None:
        assert "coverage verdict" in render_impact(self._report(dictionary))

    def test_the_interpretation_is_printed(self, dictionary: Dictionary) -> None:
        assert "interpretation" in render_impact(self._report(dictionary))

    def test_silently_changed_is_rendered_before_broken(self, dictionary: Dictionary) -> None:
        text = render_impact(self._report(dictionary))
        assert text.index("-- silently_changed") < text.index("-- undecidable")

    @pytest.mark.parametrize(
        "word", ["safe", "certified", "flight-qualified", "production-ready", "guaranteed"]
    )
    def test_the_rendered_report_never_claims_safety(
        self, dictionary: Dictionary, word: str
    ) -> None:
        text = render_impact(self._report(dictionary))
        assert re.search(rf"\b{re.escape(word)}\b", text, re.IGNORECASE) is None

    def test_max_rows_truncates(self, dictionary: Dictionary) -> None:
        parameters = tuple(Parameter(name=f"P{index}") for index in range(10))
        old = Dictionary(version="1.0", parameters=parameters)
        new = Dictionary(version="1.1", parameters=())
        corpus = Corpus(
            (
                CorpusSource(
                    label="a.py",
                    kind="script",
                    text="\n".join(f'x = tm["P{index}"]' for index in range(10)),
                ),
            )
        )
        report = assess_impact(diff_dictionaries(old, new, EXACT), corpus, old, new)
        assert "more" in render_impact(report, max_rows=2)
