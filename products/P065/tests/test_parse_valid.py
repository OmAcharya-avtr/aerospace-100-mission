"""Parsing well-formed documents."""

from __future__ import annotations

from pathlib import Path

from hazardtrace import ComponentKind, EvidenceKind, UcaClass, load_analysis, parse_analysis


def test_minimal_document(minimal_yaml):
    analysis = parse_analysis(minimal_yaml, source="unit")
    assert analysis.title == "minimal"
    assert analysis.source == "unit"
    assert analysis.loss_ids == ("L-1",)
    assert analysis.hazard_ids == ("H-1",)
    assert analysis.constraint_ids == ("SC-1",)
    assert len(analysis.structure.components) == 4
    assert len(analysis.structure.control_actions) == 1
    assert len(analysis.structure.feedback) == 1
    assert analysis.ucas == ()


def test_line_numbers_are_recorded(minimal_yaml):
    analysis = parse_analysis(minimal_yaml)
    # `losses:` is on line 4 of MINIMAL (the string starts with a newline), so
    # the first loss mapping starts on line 5.
    assert analysis.losses[0].line == 5


def test_components_are_grouped_by_kind(minimal_yaml):
    structure = parse_analysis(minimal_yaml).structure
    assert [c.id for c in structure.by_kind(ComponentKind.CONTROLLER)] == ["CTRL-1"]
    assert [c.id for c in structure.by_kind(ComponentKind.ACTUATOR)] == ["ACT-1"]
    assert [c.id for c in structure.by_kind(ComponentKind.SENSOR)] == ["SNS-1"]
    assert [c.id for c in structure.by_kind(ComponentKind.PROCESS)] == ["PROC-1"]
    assert structure.component_ids == {"CTRL-1", "ACT-1", "SNS-1", "PROC-1"}


def test_continuous_defaults_to_true(minimal_yaml):
    assert parse_analysis(minimal_yaml).structure.control_actions[0].continuous is True


def test_continuous_can_be_declared_false(minimal_yaml):
    text = minimal_yaml.replace(
        "      process: PROC-1\n  feedback:", "      process: PROC-1\n      continuous: false\n  feedback:"
    )
    assert parse_analysis(text).structure.control_actions[0].continuous is False


def test_uca_classes_round_trip(minimal_yaml):
    extra = "\nucas:\n"
    for index, uca_class in enumerate(UcaClass, start=1):
        extra += (
            f"  - id: UCA-{index}\n"
            f"    control_action: CA-1\n"
            f"    uca_class: {uca_class.value}\n"
            f"    description: d\n"
            f"    hazards: [H-1]\n"
        )
    analysis = parse_analysis(minimal_yaml + extra)
    assert [u.uca_class for u in analysis.ucas] == list(UcaClass)


def test_shipped_cases_parse(cases_dir: Path):
    for path in sorted(cases_dir.glob("*.yaml")):
        analysis = load_analysis(path)
        assert analysis.title
        assert analysis.source == str(path)


def test_cited_identifiers_are_sorted_and_deduplicated(cases_dir: Path):
    analysis = load_analysis(cases_dir / "p024_bdot_detumble.yaml")
    requirements = analysis.cited_requirements()
    assert list(requirements) == sorted(set(requirements))
    tests = analysis.cited_tests()
    assert list(tests) == sorted(set(tests))
    assert len(tests) == 8


def test_evidence_kinds_parse(cases_dir: Path):
    analysis = load_analysis(cases_dir / "p024_bdot_detumble.yaml")
    kinds = {item.id: item.kind for item in analysis.evidence}
    assert kinds["EV-1"] is EvidenceKind.TEST
    assert kinds["EV-5"] is EvidenceKind.ANALYSIS


def test_process_model_is_parsed(cases_dir: Path):
    analysis = load_analysis(cases_dir / "p024_bdot_detumble.yaml")
    controller = analysis.component("CTRL-BDOT")
    assert controller is not None
    assert len(controller.process_model) == 3


def test_optional_sections_may_be_absent():
    analysis = parse_analysis("meta:\n  title: bare\n")
    assert analysis.losses == ()
    assert analysis.structure.control_actions == ()
    assert analysis.ucas == ()


def test_null_sections_are_treated_as_absent():
    analysis = parse_analysis("meta:\n  title: bare\nlosses:\nhazards:\nucas:\n")
    assert analysis.losses == ()
    assert analysis.hazards == ()
