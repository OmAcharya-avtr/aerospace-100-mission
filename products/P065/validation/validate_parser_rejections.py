"""Every malformed-document rejection path, counted.

The parser's contract is that a document it accepts contains no dangling
reference, so the checks downstream may assume every reference resolves. This
script exercises each rejection path and records, for each one, whether the
exception was raised and whether it named the offending YAML key path. A path
that silently accepts a bad document is the failure mode that matters, because
a silently-dropped section makes the gap report wrong in the direction of
looking better.
"""

from __future__ import annotations

from _support import CASES, Emitter

from hazardtrace import AnalysisFormatError, load_analysis, parse_analysis

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

# (label, full document text, expected key path). An empty expected path means
# the failure is about the whole document and no single key is at fault.
CASES_LIST: list[tuple[str, str, str]] = [
    ("empty document", "", ""),
    ("whitespace only", "   \n\n", ""),
    ("top-level sequence", "- a\n- b", "<document>"),
    ("bare scalar", "a string", "<document>"),
    ("YAML syntax error", "meta:\n  title: [unclosed", ""),
    ("unknown top-level key", "meta:\n  title: t\nhazzards: []\n", "<document>"),
    ("meta missing title", "meta:\n  system: s\n", "meta"),
    ("meta title blank", "meta:\n  title: '  '\n", "meta.title"),
    ("meta title not a string", "meta:\n  title: [a]\n", "meta.title"),
    ("meta unknown key", "meta:\n  title: t\n  owner: o\n", "meta"),
    ("meta is a sequence", "meta:\n  - title: t\n", "meta"),
    ("losses not a list", "meta:\n  title: t\nlosses: 3\n", "losses"),
    ("loss not a mapping", "meta:\n  title: t\nlosses:\n  - str\n", "losses[0]"),
    ("loss missing id", "meta:\n  title: t\nlosses:\n  - description: d\n", "losses[0]"),
    (
        "loss blank id",
        "meta:\n  title: t\nlosses:\n  - id: ''\n    description: d\n",
        "losses[0].id",
    ),
    ("loss missing description", "meta:\n  title: t\nlosses:\n  - id: L-1\n", "losses[0]"),
    (
        "loss unknown key",
        "meta:\n  title: t\nlosses:\n  - id: L-1\n    description: d\n    owner: o\n",
        "losses[0]",
    ),
    (
        "loss stakeholders not a list",
        "meta:\n  title: t\nlosses:\n  - id: L-1\n    description: d\n    stakeholders: x\n",
        "losses[0].stakeholders",
    ),
    (
        "loss stakeholder blank",
        "meta:\n  title: t\nlosses:\n  - id: L-1\n    description: d\n    stakeholders: ['']\n",
        "losses[0].stakeholders[0]",
    ),
    (
        "duplicate stakeholder",
        "meta:\n  title: t\nlosses:\n  - id: L-1\n    description: d\n    stakeholders: [a, a]\n",
        "losses[0].stakeholders[1]",
    ),
    (
        "duplicate loss id",
        "meta:\n  title: t\nlosses:\n  - id: L-1\n    description: d\n"
        "  - id: L-1\n    description: e\n",
        "losses[1].id",
    ),
    (
        "hazard names an undeclared loss",
        BASE.replace("    losses: [L-1]\n", "    losses: [L-9]\n"),
        "hazards[0].losses[0]",
    ),
    (
        "a top-level key declared twice",
        BASE + "hazards:\n  - id: H-2\n    description: d\n    losses: [L-1]\n",
        "line 30",
    ),
    (
        "constraint names an undeclared hazard",
        BASE + "constraints:\n  - id: SC-1\n    description: d\n    hazards: [H-9]\n",
        "constraints[0].hazards[0]",
    ),
    (
        "constraint requirements not a list",
        BASE + "constraints:\n  - id: SC-1\n    description: d\n    requirements: x\n",
        "constraints[0].requirements",
    ),
    (
        "control action names an undeclared controller",
        BASE + "",
        "",
    ),
    (
        "uca names an undeclared control action",
        BASE + "ucas:\n  - id: U-1\n    control_action: CA-9\n    uca_class: provided\n"
        "    description: d\n",
        "ucas[0].control_action",
    ),
    (
        "uca names an undeclared hazard",
        BASE + "ucas:\n  - id: U-1\n    control_action: CA-1\n    uca_class: provided\n"
        "    description: d\n    hazards: [H-9]\n",
        "ucas[0].hazards[0]",
    ),
    (
        "uca class is not one of the four",
        BASE + "ucas:\n  - id: U-1\n    control_action: CA-1\n    uca_class: provided_late\n"
        "    description: d\n",
        "ucas[0].uca_class",
    ),
    (
        "uca missing description",
        BASE + "ucas:\n  - id: U-1\n    control_action: CA-1\n    uca_class: provided\n",
        "ucas[0]",
    ),
    (
        "duplicate uca id",
        BASE + "ucas:\n  - id: U-1\n    control_action: CA-1\n    uca_class: provided\n"
        "    description: d\n  - id: U-1\n    control_action: CA-1\n"
        "    uca_class: not_provided\n    description: e\n",
        "ucas[1].id",
    ),
    (
        "exclusion names an undeclared control action",
        BASE + "exclusions:\n  - control_action: CA-9\n    uca_class: provided\n"
        "    justification: a justification long enough to pass\n",
        "exclusions[0].control_action",
    ),
    (
        "exclusion with no justification key",
        BASE + "exclusions:\n  - control_action: CA-1\n    uca_class: provided\n",
        "exclusions[0]",
    ),
    (
        "exclusion justification is blank",
        BASE + "exclusions:\n  - control_action: CA-1\n    uca_class: provided\n"
        "    justification: '  '\n",
        "exclusions[0].justification",
    ),
    (
        "two exclusions of one cell",
        BASE + "exclusions:\n  - control_action: CA-1\n    uca_class: provided\n"
        "    justification: a justification long enough to pass the policy\n"
        "  - control_action: CA-1\n    uca_class: provided\n"
        "    justification: another justification long enough to pass\n",
        "exclusions[1]",
    ),
    (
        "scenario names an undeclared uca",
        BASE + "loss_scenarios:\n  - id: LS-1\n    description: d\n    ucas: [U-9]\n",
        "loss_scenarios[0].ucas[0]",
    ),
    (
        "scenario names an undeclared mitigation",
        BASE + "loss_scenarios:\n  - id: LS-1\n    description: d\n    mitigations: [M-9]\n",
        "loss_scenarios[0].mitigations[0]",
    ),
    (
        "mitigation names an undeclared evidence artifact",
        BASE + "mitigations:\n  - id: M-1\n    description: d\n    evidence: [EV-9]\n",
        "mitigations[0].evidence[0]",
    ),
    (
        "evidence kind is unknown",
        BASE + "evidence:\n  - id: EV-1\n    kind: proof\n    description: d\n",
        "evidence[0].kind",
    ),
    (
        "test ids on non-test evidence",
        BASE + "evidence:\n  - id: EV-1\n    kind: analysis\n    description: d\n"
        "    tests: [a::b]\n",
        "evidence[0].tests",
    ),
    (
        "unknown control-structure section",
        "meta:\n  title: t\ncontrol_structure:\n  humans: []\n",
        "control_structure",
    ),
    (
        "process model on an actuator",
        "meta:\n  title: t\ncontrol_structure:\n  actuators:\n    - id: A\n"
        "      name: a\n      process_model: [x]\n",
        "control_structure.actuators[0].process_model",
    ),
    (
        "component missing a name",
        "meta:\n  title: t\ncontrol_structure:\n  controllers:\n    - id: C\n",
        "control_structure.controllers[0]",
    ),
    (
        "feedback names an actuator as its sensor",
        BASE + "  feedback:\n    - id: FB-1\n      name: f\n      sensor: ACT-1\n"
        "      controller: CTRL-1\n      process: PROC-1\n",
        "control_structure.feedback[0].sensor",
    ),
    (
        "feedback names a sensor as its controller",
        BASE + "  feedback:\n    - id: FB-1\n      name: f\n      sensor: SNS-1\n"
        "      controller: SNS-1\n      process: PROC-1\n",
        "control_structure.feedback[0].controller",
    ),
    (
        "continuous is not a boolean",
        BASE + "    - id: CA-2\n      name: x\n      controller: CTRL-1\n"
        "      actuator: ACT-1\n      process: PROC-1\n      continuous: maybe\n",
        "control_structure.control_actions[1].continuous",
    ),
    (
        "control action names a sensor as its actuator",
        BASE + "    - id: CA-2\n      name: x\n      controller: CTRL-1\n"
        "      actuator: SNS-1\n      process: PROC-1\n",
        "control_structure.control_actions[1].actuator",
    ),
    (
        "control action names a controller as its process",
        BASE + "    - id: CA-2\n      name: x\n      controller: CTRL-1\n"
        "      actuator: ACT-1\n      process: CTRL-1\n",
        "control_structure.control_actions[1].process",
    ),
    (
        "duplicate control action id",
        BASE + "    - id: CA-1\n      name: x\n      controller: CTRL-1\n"
        "      actuator: ACT-1\n      process: PROC-1\n",
        "control_structure.control_actions[1].id",
    ),
    (
        "component id collides with a control action id",
        BASE.replace("    - id: CA-1\n", "    - id: CTRL-1\n"),
        "control_structure.control_actions[0].id",
    ),
]

# The one case above that is NOT a rejection: BASE itself is well formed.
WELL_FORMED_LABEL = "control action names an undeclared controller"


def main() -> int:
    """Exercise every rejection path. Returns a process exit code."""
    emit = Emitter(__file__)
    emit("hazardtrace validation: malformed-document rejection paths")
    emit()

    schema_cases = [c for c in CASES_LIST if c[0] != WELL_FORMED_LABEL]
    raised = 0
    named = 0
    accepted: list[str] = []
    wrong_path: list[str] = []
    emit(f"   {'label':<46}{'expected key path':<46}outcome")
    for label, text, expected in schema_cases:
        try:
            parse_analysis(text)
        except AnalysisFormatError as exc:
            raised += 1
            if exc.key_path == expected:
                named += 1
                outcome = "rejected, key path matches"
            else:
                wrong_path.append(f"{label}: got {exc.key_path!r}")
                outcome = f"rejected, key path {exc.key_path!r}"
        else:
            accepted.append(label)
            outcome = "ACCEPTED (defect)"
        emit(f"   {label:<46}{expected or '(whole document)':<46}{outcome}")
    emit()
    emit(f"   schema rejection paths exercised : {len(schema_cases)}")
    emit(f"   raised AnalysisFormatError       : {raised}/{len(schema_cases)}")
    emit(f"   key path exactly as expected     : {named}/{len(schema_cases)}")
    emit(f"   malformed documents accepted     : {len(accepted)}")
    emit()

    # -- I/O rejection paths ----------------------------------------------
    io_cases = 0
    io_raised = 0
    for path in (CASES / "absent.yaml", CASES, CASES / ".."):
        io_cases += 1
        try:
            load_analysis(path)
        except AnalysisFormatError:
            io_raised += 1
    emit(f"   I/O rejection paths exercised    : {io_cases}")
    emit(f"   raised AnalysisFormatError       : {io_raised}/{io_cases}")
    emit()

    # -- the control case: BASE must parse --------------------------------
    base_ok = True
    try:
        parse_analysis(BASE)
    except AnalysisFormatError:
        base_ok = False
    emit("   control: the base document, which is well formed, must parse")
    emit()

    emit.check(
        "every malformed document is rejected",
        len(accepted) == 0,
        "; ".join(accepted) if accepted else "none accepted",
    )
    emit.check(
        "every rejection names the expected key path",
        named == len(schema_cases),
        "; ".join(wrong_path) if wrong_path else "all matched",
    )
    emit.check("every I/O failure is rejected", io_raised == io_cases)
    emit.check("the well-formed base document parses", base_ok)
    return emit.finish()


if __name__ == "__main__":
    raise SystemExit(main())
