"""Rejection paths. Every one must raise AnalysisFormatError and name the key.

The parser's contract is that a document it accepts has no dangling reference,
so every check downstream may assume references resolve. These tests are the
other half of that contract.
"""

from __future__ import annotations

import pytest

from hazardtrace import AnalysisFormatError, HazardTraceError, load_analysis, parse_analysis

# Each case: (label, YAML body appended to or replacing the base, expected key path
# fragment). The expected fragment is checked against AnalysisFormatError.key_path.
BASE = """
meta:
  title: t
losses:
  - id: L-1
    description: d
hazards:
  - id: H-1
    description: d
    losses: [L-1]
control_structure:
  controllers:
    - id: CTRL-1
      name: c
  actuators:
    - id: ACT-1
      name: a
  sensors:
    - id: SNS-1
      name: s
  controlled_processes:
    - id: PROC-1
      name: p
  control_actions:
    - id: CA-1
      name: ca
      controller: CTRL-1
      actuator: ACT-1
      process: PROC-1
"""

WHOLE_DOCUMENT_CASES = [
    ("empty document", "", ""),
    ("only whitespace", "   \n\n", ""),
    ("not a mapping", "- a\n- b", "<document>"),
    ("scalar document", "just a string", "<document>"),
    ("syntax error", "meta:\n  title: [unclosed", ""),
    ("unknown top key", "meta:\n  title: t\nhazzards: []\n", "<document>"),
    ("missing title", "meta:\n  system: s\n", "meta"),
    ("blank title", "meta:\n  title: '   '\n", "meta.title"),
    ("title is a list", "meta:\n  title: [a]\n", "meta.title"),
    ("unknown meta key", "meta:\n  title: t\n  owner: x\n", "meta"),
    ("meta is a list", "meta:\n  - title: t\n", "meta"),
]

SECTION_CASES = [
    ("losses not a list", "losses: 3\n", "losses"),
    ("loss not a mapping", "losses:\n  - just a string\n", "losses[0]"),
    ("loss missing id", "losses:\n  - description: d\n", "losses[0]"),
    ("loss blank id", "losses:\n  - id: ''\n    description: d\n", "losses[0].id"),
    ("loss missing description", "losses:\n  - id: L-1\n", "losses[0]"),
    ("loss unknown key", "losses:\n  - id: L-1\n    description: d\n    owner: x\n", "losses[0]"),
    (
        "loss stakeholders not a list",
        "losses:\n  - id: L-1\n    description: d\n    stakeholders: x\n",
        "losses[0].stakeholders",
    ),
    (
        "loss stakeholder blank",
        "losses:\n  - id: L-1\n    description: d\n    stakeholders: ['']\n",
        "losses[0].stakeholders[0]",
    ),
    (
        "duplicate loss id",
        "losses:\n  - id: L-1\n    description: d\n  - id: L-1\n    description: e\n",
        "losses[1].id",
    ),
    (
        "duplicate stakeholder",
        "losses:\n  - id: L-1\n    description: d\n    stakeholders: [a, a]\n",
        "losses[0].stakeholders[1]",
    ),
]


@pytest.mark.parametrize(
    ("label", "text", "key_path"), WHOLE_DOCUMENT_CASES, ids=[c[0] for c in WHOLE_DOCUMENT_CASES]
)
def test_whole_document_rejections(label, text, key_path):
    with pytest.raises(AnalysisFormatError) as excinfo:
        parse_analysis(text)
    assert key_path in excinfo.value.key_path


@pytest.mark.parametrize(
    ("label", "text", "key_path"), SECTION_CASES, ids=[c[0] for c in SECTION_CASES]
)
def test_section_rejections(label, text, key_path):
    with pytest.raises(AnalysisFormatError) as excinfo:
        parse_analysis("meta:\n  title: t\n" + text)
    assert excinfo.value.key_path == key_path


REFERENCE_CASES = [
    (
        "constraint requirements are not a list",
        "constraints:\n  - id: SC-1\n    description: d\n    requirements: x\n",
        "constraints[0].requirements",
    ),
    (
        "constraint names an undeclared hazard",
        "constraints:\n  - id: SC-1\n    description: d\n    hazards: [H-9]\n",
        "constraints[0].hazards[0]",
    ),
    (
        "uca names an undeclared control action",
        "ucas:\n  - id: U-1\n    control_action: CA-9\n    uca_class: provided\n"
        "    description: d\n",
        "ucas[0].control_action",
    ),
    (
        "uca names an undeclared hazard",
        "ucas:\n  - id: U-1\n    control_action: CA-1\n    uca_class: provided\n"
        "    description: d\n    hazards: [H-9]\n",
        "ucas[0].hazards[0]",
    ),
    (
        "exclusion names an undeclared control action",
        "exclusions:\n  - control_action: CA-9\n    uca_class: provided\n"
        "    justification: a justification long enough to pass\n",
        "exclusions[0].control_action",
    ),
    (
        "scenario names an undeclared uca",
        "loss_scenarios:\n  - id: LS-1\n    description: d\n    ucas: [U-9]\n",
        "loss_scenarios[0].ucas[0]",
    ),
    (
        "scenario names an undeclared mitigation",
        "loss_scenarios:\n  - id: LS-1\n    description: d\n    mitigations: [M-9]\n",
        "loss_scenarios[0].mitigations[0]",
    ),
    (
        "mitigation names an undeclared evidence artifact",
        "mitigations:\n  - id: M-1\n    description: d\n    evidence: [EV-9]\n",
        "mitigations[0].evidence[0]",
    ),
]


@pytest.mark.parametrize(
    ("label", "text", "key_path"),
    REFERENCE_CASES,
    ids=[c[0] for c in REFERENCE_CASES],
)
def test_reference_rejections(label, text, key_path):
    with pytest.raises(AnalysisFormatError) as excinfo:
        parse_analysis(BASE + text)
    assert excinfo.value.key_path == key_path


def test_hazard_naming_an_undeclared_loss_is_rejected():
    with pytest.raises(AnalysisFormatError) as excinfo:
        parse_analysis(BASE.replace("    losses: [L-1]\n", "    losses: [L-9]\n"))
    assert excinfo.value.key_path == "hazards[0].losses[0]"


STRUCTURE_CASES = [
    (
        "control action names an undeclared controller",
        "    - id: CA-2\n      name: x\n      controller: CTRL-9\n"
        "      actuator: ACT-1\n      process: PROC-1\n",
        "control_structure.control_actions[1].controller",
    ),
    (
        "control action names a sensor as its actuator",
        "    - id: CA-2\n      name: x\n      controller: CTRL-1\n"
        "      actuator: SNS-1\n      process: PROC-1\n",
        "control_structure.control_actions[1].actuator",
    ),
    (
        "control action names a controller as its process",
        "    - id: CA-2\n      name: x\n      controller: CTRL-1\n"
        "      actuator: ACT-1\n      process: CTRL-1\n",
        "control_structure.control_actions[1].process",
    ),
    (
        "duplicate control action id",
        "    - id: CA-1\n      name: x\n      controller: CTRL-1\n"
        "      actuator: ACT-1\n      process: PROC-1\n",
        "control_structure.control_actions[1].id",
    ),
    (
        "continuous is not a boolean",
        "    - id: CA-2\n      name: x\n      controller: CTRL-1\n"
        "      actuator: ACT-1\n      process: PROC-1\n      continuous: maybe\n",
        "control_structure.control_actions[1].continuous",
    ),
]


@pytest.mark.parametrize(
    ("label", "text", "key_path"), STRUCTURE_CASES, ids=[c[0] for c in STRUCTURE_CASES]
)
def test_structure_rejections(label, text, key_path):
    with pytest.raises(AnalysisFormatError) as excinfo:
        parse_analysis(BASE + text)
    assert excinfo.value.key_path == key_path


def test_feedback_names_an_actuator_as_its_sensor():
    text = BASE + (
        "  feedback:\n    - id: FB-1\n      name: f\n      sensor: ACT-1\n"
        "      controller: CTRL-1\n      process: PROC-1\n"
    )
    with pytest.raises(AnalysisFormatError) as excinfo:
        parse_analysis(text)
    assert excinfo.value.key_path == "control_structure.feedback[0].sensor"


def test_unknown_uca_class_names_the_allowed_values():
    text = BASE + (
        "ucas:\n  - id: U-1\n    control_action: CA-1\n    uca_class: provided_late\n"
        "    description: d\n"
    )
    with pytest.raises(AnalysisFormatError) as excinfo:
        parse_analysis(text)
    assert excinfo.value.key_path == "ucas[0].uca_class"
    assert "not_provided" in str(excinfo.value)


def test_unknown_evidence_kind_names_the_allowed_values():
    with pytest.raises(AnalysisFormatError) as excinfo:
        parse_analysis("meta:\n  title: t\nevidence:\n  - id: E\n    kind: proof\n    description: d\n")
    assert excinfo.value.key_path == "evidence[0].kind"
    assert "analysis" in str(excinfo.value)


def test_tests_on_non_test_evidence_are_rejected():
    text = (
        "meta:\n  title: t\nevidence:\n  - id: E\n    kind: analysis\n"
        "    description: d\n    tests: [a::b]\n"
    )
    with pytest.raises(AnalysisFormatError) as excinfo:
        parse_analysis(text)
    assert excinfo.value.key_path == "evidence[0].tests"


def test_process_model_on_an_actuator_is_rejected():
    text = (
        "meta:\n  title: t\ncontrol_structure:\n  actuators:\n    - id: A\n"
        "      name: a\n      process_model: [x]\n"
    )
    with pytest.raises(AnalysisFormatError) as excinfo:
        parse_analysis(text)
    assert excinfo.value.key_path == "control_structure.actuators[0].process_model"


def test_duplicate_exclusion_of_the_same_cell_is_rejected():
    text = BASE + (
        "exclusions:\n  - control_action: CA-1\n    uca_class: provided\n"
        "    justification: a justification long enough to pass the policy\n"
        "  - control_action: CA-1\n    uca_class: provided\n"
        "    justification: a second justification long enough to pass\n"
    )
    with pytest.raises(AnalysisFormatError) as excinfo:
        parse_analysis(text)
    assert excinfo.value.key_path == "exclusions[1]"


def test_exclusion_without_a_justification_is_rejected_at_parse_time():
    # A missing justification is a malformed document; a SHORT one is a finding
    # (HT002). The distinction is deliberate: an author who wrote nothing has
    # not declared an exclusion at all.
    text = BASE + "exclusions:\n  - control_action: CA-1\n    uca_class: provided\n"
    with pytest.raises(AnalysisFormatError) as excinfo:
        parse_analysis(text)
    assert excinfo.value.key_path == "exclusions[0]"


def test_unknown_control_structure_section_is_rejected():
    with pytest.raises(AnalysisFormatError) as excinfo:
        parse_analysis("meta:\n  title: t\ncontrol_structure:\n  humans: []\n")
    assert excinfo.value.key_path == "control_structure"


def test_component_id_collides_with_a_control_action_id():
    text = BASE.replace("    - id: CA-1\n", "    - id: CTRL-1\n")
    with pytest.raises(AnalysisFormatError) as excinfo:
        parse_analysis(text)
    assert excinfo.value.key_path.endswith(".id")


def test_missing_file(tmp_path):
    with pytest.raises(AnalysisFormatError) as excinfo:
        load_analysis(tmp_path / "absent.yaml")
    assert "no such file" in str(excinfo.value)


def test_directory_instead_of_file(tmp_path):
    with pytest.raises(AnalysisFormatError) as excinfo:
        load_analysis(tmp_path)
    assert "not a file" in str(excinfo.value)


def test_not_utf8(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_bytes(b"meta:\n  title: \xff\xfe\n")
    with pytest.raises(AnalysisFormatError) as excinfo:
        load_analysis(path)
    assert "UTF-8" in str(excinfo.value)


def test_format_error_is_a_hazardtrace_error():
    assert issubclass(AnalysisFormatError, HazardTraceError)


def test_format_error_without_a_key_path_has_a_plain_message():
    with pytest.raises(AnalysisFormatError) as excinfo:
        parse_analysis("")
    assert excinfo.value.key_path == ""
    assert str(excinfo.value) == excinfo.value.message


def test_a_duplicate_top_level_key_is_rejected():
    # PyYAML's default is last-one-wins, silently. For a gap-reporting tool
    # that is the worst failure mode available: a second `ucas:` section would
    # drop the first and the report would look MORE complete than the document.
    # Found while writing validation/validate_parser_rejections.py.
    text = (
        "meta:\n  title: t\nlosses:\n  - id: L-1\n    description: d\n"
        "losses:\n  - id: L-2\n    description: e\n"
    )
    with pytest.raises(AnalysisFormatError) as excinfo:
        parse_analysis(text)
    assert "declared twice" in str(excinfo.value)
    assert excinfo.value.key_path == "line 6"


def test_a_duplicate_nested_key_is_rejected():
    text = "meta:\n  title: t\n  title: u\n"
    with pytest.raises(AnalysisFormatError) as excinfo:
        parse_analysis(text)
    assert "declared twice" in str(excinfo.value)


def test_the_same_key_in_two_different_mappings_is_fine():
    analysis = parse_analysis(
        "meta:\n  title: t\nlosses:\n  - id: L-1\n    description: d\n"
        "  - id: L-2\n    description: e\n"
    )
    assert analysis.loss_ids == ("L-1", "L-2")
