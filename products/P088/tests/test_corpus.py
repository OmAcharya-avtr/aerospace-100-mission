"""Reference extraction from procedures, scripts and configuration files."""

from __future__ import annotations

import pytest

from tmtcdiff.corpus import (
    DEFAULT_ACCESSORS,
    SOURCE_KINDS,
    Corpus,
    CorpusSource,
    Reference,
    extract_from_source,
    extract_references,
    load_corpus,
)


def _targets(text: str, kind: str = "script") -> list[str]:
    source = CorpusSource(label="probe", kind=kind, text=text)
    return [reference.target for reference in extract_from_source(source)]


def _references(text: str, kind: str = "script") -> tuple[Reference, ...]:
    return extract_from_source(CorpusSource(label="probe", kind=kind, text=text))


class TestCorpusSource:
    @pytest.mark.parametrize("kind", SOURCE_KINDS)
    def test_every_declared_kind_is_accepted(self, kind: str) -> None:
        assert CorpusSource(label="a", kind=kind, text="").kind == kind

    def test_unknown_kind_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be one of"):
            CorpusSource(label="a", kind="spreadsheet", text="")

    def test_empty_label_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            CorpusSource(label="", kind="script", text="")


class TestReference:
    def test_unknown_kind_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be one of"):
            Reference(source="a", line=1, kind="guess", text="", target="X")

    def test_zero_line_rejected(self) -> None:
        with pytest.raises(ValueError, match="1-based"):
            Reference(source="a", line=0, kind="name", text="", target="X")

    def test_an_undecidable_reference_needs_a_reason(self) -> None:
        with pytest.raises(ValueError, match="must carry a reason"):
            Reference(source="a", line=1, kind="name", text="", target="X", decidable=False)

    def test_one_line_marks_an_undecidable_reference(self) -> None:
        reference = Reference(
            source="a", line=3, kind="name", text="", target="X", decidable=False, reason="why"
        )
        assert "UNDECIDABLE" in reference.one_line()


class TestScriptExtraction:
    @pytest.mark.parametrize("quote", ['"', "'"])
    def test_subscript_with_either_quote(self, quote: str) -> None:
        assert _targets(f"x = tm[{quote}BATT_T_1{quote}]") == ["BATT_T_1"]

    def test_accessor_call(self) -> None:
        assert _targets('x = get_parameter("BATT_T_1")') == ["BATT_T_1"]

    def test_fetch_with_a_path(self) -> None:
        references = _references('x = fetch("TM/PWR/BATT_T_1")')
        assert references[0].kind == "path"

    def test_command_accessor_is_a_command_reference(self) -> None:
        references = _references('send_command("SET_HEATER")')
        assert references[0].kind == "command"

    def test_command_keyword_arguments_are_extracted(self) -> None:
        targets = _targets('send_command("SET_HEATER", HEATER_ID=3, DUTY=50)')
        assert targets == ["SET_HEATER", "SET_HEATER.HEATER_ID", "SET_HEATER.DUTY"]

    def test_enum_label_comparison(self) -> None:
        references = _references('if status == "SAFE_MODE":')
        assert references[0].kind == "enum_label"
        assert references[0].target == "SAFE_MODE"

    def test_inequality_comparison_is_also_a_label_reference(self) -> None:
        assert _targets('if status != "SAFE_MODE":') == ["SAFE_MODE"]

    def test_a_comment_is_ignored_in_a_script(self) -> None:
        assert _targets('# x = tm["BATT_T_1"]') == []

    def test_several_references_on_one_line(self) -> None:
        targets = _targets('y = tm["A"] + tm["B"]')
        assert targets == ["A", "B"]

    def test_duplicate_references_on_one_line_are_collapsed(self) -> None:
        assert _targets('y = tm["A"] + tm["A"]') == ["A"]


class TestUndecidableReferences:
    @pytest.mark.parametrize(
        ("line", "fragment"),
        [
            ('x = tm["T_%02d" % index]', "run-time expression"),
            ('x = tm[f"T_{index}"]', "f-string"),
            ("x = tm[name]", "variable"),
            ('x = tm["T_" + suffix]', "run-time expression"),
            ('x = tm["T_{}".format(index)]', "run-time expression"),
            ("x = tm[channels[k]]", "variable"),
        ],
    )
    def test_constructed_references_are_marked_undecidable(
        self, line: str, fragment: str
    ) -> None:
        reference = _references(line)[0]
        assert not reference.decidable
        assert fragment in reference.reason

    def test_an_undecidable_reference_keeps_the_source_expression(self) -> None:
        reference = _references("x = tm[name]")[0]
        assert reference.target == "name"

    def test_a_decidable_and_an_undecidable_reference_on_one_line(self) -> None:
        references = _references('y = tm["A"] + tm[name]')
        assert [r.decidable for r in references] == [True, False]


class TestProcedureExtraction:
    @pytest.mark.parametrize("verb", ["CHECK", "VERIFY", "MONITOR", "LOG", "REPORT", "LIMIT"])
    def test_every_verb_is_recognised(self, verb: str) -> None:
        assert _targets(f"{verb} BATT_T_1 > 10", kind="procedure") == ["BATT_T_1"]

    def test_a_path_argument_is_classified_as_a_path(self) -> None:
        references = _references("MONITOR TM/PWR/BATT_T_1", kind="procedure")
        assert references[0].kind == "path"

    def test_send_with_arguments(self) -> None:
        targets = _targets("SEND SET_HEATER(HEATER_ID=3, DUTY=50)", kind="procedure")
        assert targets == ["SET_HEATER", "SET_HEATER.HEATER_ID", "SET_HEATER.DUTY"]

    def test_execute_is_accepted_as_well_as_send(self) -> None:
        assert "SET_HEATER" in _targets("EXECUTE SET_HEATER(A=1)", kind="procedure")

    def test_a_label_comparison_in_a_procedure(self) -> None:
        assert _targets('WAIT UNTIL MODE == "SAFE_MODE"', kind="procedure") == ["SAFE_MODE"]

    def test_a_procedure_comment_character_is_not_stripped(self) -> None:
        # Procedures use '*' for comments; '#' is ordinary text there, so the
        # script comment rule must not apply.
        assert _targets("CHECK A > 1 # note", kind="procedure") == ["A"]


class TestConfigExtraction:
    def test_parameter_key(self) -> None:
        assert _targets("parameter: BATT_T_1", kind="config") == ["BATT_T_1"]

    def test_command_key_is_a_command_reference(self) -> None:
        references = _references("command: SET_HEATER", kind="config")
        assert references[0].kind == "command"

    def test_path_key_is_a_path_reference(self) -> None:
        references = _references("path: TM/PWR/BATT_T_1", kind="config")
        assert references[0].kind == "path"

    def test_json_style_quoting(self) -> None:
        assert _targets('  "parameter": "BATT_T_1",', kind="config") == ["BATT_T_1"]

    def test_equals_separator(self) -> None:
        assert _targets("parameter = BATT_T_1", kind="config") == ["BATT_T_1"]

    def test_key_matching_is_case_insensitive(self) -> None:
        assert _targets("Parameter: BATT_T_1", kind="config") == ["BATT_T_1"]


class TestCorpusLevel:
    def test_extract_references_walks_every_source(self) -> None:
        corpus = Corpus(
            (
                CorpusSource(label="a.py", kind="script", text='x = tm["A"]'),
                CorpusSource(label="b.cfg", kind="config", text="parameter: B"),
            )
        )
        assert [r.target for r in extract_references(corpus)] == ["A", "B"]

    def test_labels_are_carried_through(self) -> None:
        corpus = Corpus((CorpusSource(label="a.py", kind="script", text='x = tm["A"]'),))
        assert extract_references(corpus)[0].source == "a.py"

    def test_line_numbers_are_one_based(self) -> None:
        corpus = Corpus((CorpusSource(label="a.py", kind="script", text='\n\nx = tm["A"]'),))
        assert extract_references(corpus)[0].line == 3

    def test_type_is_checked(self) -> None:
        with pytest.raises(TypeError, match="expects a Corpus"):
            extract_references(["not a corpus"])  # type: ignore[arg-type]

    def test_corpus_length_and_labels(self) -> None:
        corpus = Corpus((CorpusSource(label="a.py", kind="script", text=""),))
        assert len(corpus) == 1
        assert corpus.labels == ("a.py",)

    def test_custom_accessors_are_honoured(self) -> None:
        source = CorpusSource(label="a.py", kind="script", text='x = channel("A")')
        assert extract_from_source(source) == ()
        assert extract_from_source(source, accessors=("channel",))[0].target == "A"

    def test_the_default_accessor_list_is_not_empty(self) -> None:
        assert len(DEFAULT_ACCESSORS) > 5


class TestLoadCorpus:
    def test_reads_recognised_suffixes(self, tmp_path) -> None:
        (tmp_path / "a.py").write_text('x = tm["A"]', encoding="utf-8")
        (tmp_path / "b.proc").write_text("CHECK B > 1", encoding="utf-8")
        corpus = load_corpus(tmp_path)
        assert sorted(corpus.labels) == ["a.py", "b.proc"]

    def test_labels_are_relative_paths(self, tmp_path) -> None:
        nested = tmp_path / "procedures"
        nested.mkdir()
        (nested / "a.proc").write_text("CHECK B > 1", encoding="utf-8")
        assert load_corpus(tmp_path).labels == ("procedures/a.proc",)

    def test_an_unrecognised_suffix_is_recorded_not_dropped(self, tmp_path) -> None:
        (tmp_path / "notes.bin").write_bytes(b"\x00\x01")
        corpus = load_corpus(tmp_path)
        assert corpus.sources == ()
        assert "notes.bin" in corpus.unreadable[0]

    def test_an_undecodable_file_is_recorded(self, tmp_path) -> None:
        (tmp_path / "a.py").write_bytes(b"\xff\xfe\x00bad")
        corpus = load_corpus(tmp_path)
        assert corpus.sources == ()
        assert "UnicodeDecodeError" in corpus.unreadable[0]

    def test_a_missing_directory_is_rejected(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="is not a directory"):
            load_corpus(tmp_path / "absent")

    def test_an_empty_directory_gives_an_empty_corpus(self, tmp_path) -> None:
        assert load_corpus(tmp_path) == Corpus()
