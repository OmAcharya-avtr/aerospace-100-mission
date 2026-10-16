"""Every rejection path of the parser, each checked to name its key path.

A rejection that does not say which key is wrong is nearly as unhelpful as no
rejection at all, so each case asserts the key path as well as the exception.
"""

from __future__ import annotations

import pytest

from watchdogspec import SpecFormatError, load_spec, parse_spec

BASE = """
spec_version: 1
system: "s"
supervisor:
  poll_period_ms: 10.0
  poll_jitter_ms: 1.0
  declaration_overhead_ms: 0.5
ladders:
  - id: L
    rungs:
      - action: log
        latency_ms: 1.0
        recovery_cost: 0.0
      - action: reset
        latency_ms: 2.0
tasks:
  - id: T
    heartbeat_period_ms: 20.0
    heartbeat_jitter_ms: 2.0
    jitter_sigma_ms: 0.5
    missed_beats_tolerated: 1
    liveness_deadline_ms: 100.0
    escalation: L
    false_trip_budget_per_hour: 1.0
"""

# (label, document, expected key path fragment)
CASES: list[tuple[str, str, str]] = [
    ("empty document", "", "document is empty"),
    ("only a comment", "# nothing here\n", "document is empty"),
    ("top level is a list", "- a\n- b\n", "<document>"),
    ("top level is a scalar", "42\n", "<document>"),
    ("unparsable yaml", "a: [1, 2\n", "not valid YAML"),
    ("unknown top-level key", BASE + "extra: 1\n", "<document>"),
    ("missing spec_version", BASE.replace("spec_version: 1\n", ""), "<document>"),
    ("spec_version not an int", BASE.replace("spec_version: 1", "spec_version: '1'"),
     "spec_version"),
    ("spec_version zero", BASE.replace("spec_version: 1", "spec_version: 0"), "spec_version"),
    ("spec_version unsupported", BASE.replace("spec_version: 1", "spec_version: 2"),
     "spec_version"),
    ("missing system", BASE.replace('system: "s"\n', ""), "<document>"),
    ("empty system", BASE.replace('system: "s"', 'system: "  "'), "system"),
    ("system not a string", BASE.replace('system: "s"', "system: 3"), "system"),
    ("notes not a string", BASE + "notes: 7\n", "notes"),
    ("missing supervisor", BASE.replace("supervisor:\n  poll_period_ms: 10.0\n"
                                        "  poll_jitter_ms: 1.0\n"
                                        "  declaration_overhead_ms: 0.5\n", ""), "<document>"),
    ("supervisor not a mapping", BASE.replace(
        "supervisor:\n  poll_period_ms: 10.0\n  poll_jitter_ms: 1.0\n"
        "  declaration_overhead_ms: 0.5\n", "supervisor: 3\n"), "supervisor"),
    ("supervisor unknown key", BASE.replace("  poll_jitter_ms: 1.0", "  pol_jitter_ms: 1.0"),
     "supervisor"),
    ("missing poll_period_ms", BASE.replace("  poll_period_ms: 10.0\n", ""), "supervisor"),
    ("poll_period_ms zero", BASE.replace("poll_period_ms: 10.0", "poll_period_ms: 0"),
     "supervisor.poll_period_ms"),
    ("poll_period_ms negative", BASE.replace("poll_period_ms: 10.0", "poll_period_ms: -1"),
     "supervisor.poll_period_ms"),
    ("poll_period_ms not a number", BASE.replace("poll_period_ms: 10.0", "poll_period_ms: fast"),
     "supervisor.poll_period_ms"),
    ("poll_period_ms is a bool", BASE.replace("poll_period_ms: 10.0", "poll_period_ms: true"),
     "supervisor.poll_period_ms"),
    ("poll_period_ms infinite", BASE.replace("poll_period_ms: 10.0", "poll_period_ms: .inf"),
     "supervisor.poll_period_ms"),
    ("poll_period_ms nan", BASE.replace("poll_period_ms: 10.0", "poll_period_ms: .nan"),
     "supervisor.poll_period_ms"),
    ("poll_jitter_ms negative", BASE.replace("poll_jitter_ms: 1.0", "poll_jitter_ms: -0.1"),
     "supervisor.poll_jitter_ms"),
    ("declaration_overhead_ms negative",
     BASE.replace("declaration_overhead_ms: 0.5", "declaration_overhead_ms: -2"),
     "supervisor.declaration_overhead_ms"),
    ("supervisor name empty", BASE.replace("supervisor:\n", "supervisor:\n  name: ''\n"),
     "supervisor.name"),
    ("missing ladders", BASE.replace(
        "ladders:\n  - id: L\n    rungs:\n      - action: log\n        latency_ms: 1.0\n"
        "        recovery_cost: 0.0\n      - action: reset\n        latency_ms: 2.0\n", ""),
     "<document>"),
    ("ladders not a list", BASE[: BASE.index("ladders:")] + "ladders: 3\n"
     + BASE[BASE.index("tasks:") :], "ladders"),
    ("ladders empty", BASE.replace(
        "ladders:\n  - id: L\n    rungs:\n      - action: log\n        latency_ms: 1.0\n"
        "        recovery_cost: 0.0\n      - action: reset\n        latency_ms: 2.0\n",
        "ladders: []\n"), "ladders"),
    ("ladder not a mapping", BASE[: BASE.index("ladders:")] + "ladders:\n  - 3\n"
     + BASE[BASE.index("tasks:") :], "ladders[0]"),
    ("ladder unknown key", BASE.replace("  - id: L\n", "  - id: L\n    descriptionn: x\n"),
     "ladders[0]"),
    ("ladder missing id", BASE.replace("  - id: L\n    rungs:", "  - rungs:"), "ladders[0]"),
    ("ladder id empty", BASE.replace("  - id: L\n", "  - id: ''\n"), "ladders[0].id"),
    ("ladder description not a string",
     BASE.replace("  - id: L\n", "  - id: L\n    description: 9\n"), "ladders[0].description"),
    ("ladder rungs missing", BASE.replace(
        "    rungs:\n      - action: log\n        latency_ms: 1.0\n        recovery_cost: 0.0\n"
        "      - action: reset\n        latency_ms: 2.0\n", "    description: x\n"),
     "ladders[0]"),
    ("ladder rungs empty", BASE.replace(
        "    rungs:\n      - action: log\n        latency_ms: 1.0\n        recovery_cost: 0.0\n"
        "      - action: reset\n        latency_ms: 2.0\n", "    rungs: []\n"),
     "ladders[0].rungs"),
    ("rung not a mapping", BASE.replace("      - action: log\n        latency_ms: 1.0\n"
                                        "        recovery_cost: 0.0\n", "      - 4\n"),
     "ladders[0].rungs[0]"),
    ("rung unknown key", BASE.replace("        recovery_cost: 0.0", "        cost: 0.0"),
     "ladders[0].rungs[0]"),
    ("rung missing action", BASE.replace("      - action: log\n", "      - \n"),
     "ladders[0].rungs[0]"),
    ("rung unknown action", BASE.replace("action: log", "action: reboot"),
     "ladders[0].rungs[0].action"),
    ("rung action not a string", BASE.replace("action: log", "action: 5"),
     "ladders[0].rungs[0].action"),
    ("rung missing latency", BASE.replace("        latency_ms: 1.0\n", "", 1),
     "ladders[0].rungs[0]"),
    ("rung negative latency", BASE.replace("latency_ms: 1.0", "latency_ms: -1"),
     "ladders[0].rungs[0].latency_ms"),
    ("rung negative cost", BASE.replace("recovery_cost: 0.0", "recovery_cost: -3"),
     "ladders[0].rungs[0].recovery_cost"),
    ("rung note not a string",
     BASE.replace("        recovery_cost: 0.0", "        recovery_cost: 0.0\n        note: 2"),
     "ladders[0].rungs[0].note"),
    ("duplicate ladder id", BASE.replace(
        "tasks:", "  - id: L\n    rungs:\n      - action: reset\n        latency_ms: 1.0\ntasks:"),
     "ladders[1].id"),
    ("missing tasks", BASE[: BASE.index("tasks:")], "<document>"),
    ("tasks empty", BASE[: BASE.index("tasks:")] + "tasks: []\n", "tasks"),
    ("tasks not a list", BASE[: BASE.index("tasks:")] + "tasks: 3\n", "tasks"),
    ("task not a mapping", BASE[: BASE.index("tasks:")] + "tasks:\n  - 7\n", "tasks[0]"),
    ("task unknown key", BASE.replace("    escalation: L", "    escalationn: L"), "tasks[0]"),
    ("task missing id", BASE.replace("  - id: T\n", "  - \n"), "tasks[0]"),
    ("task id not a string", BASE.replace("  - id: T\n", "  - id: 4\n"), "tasks[0].id"),
    ("task missing period", BASE.replace("    heartbeat_period_ms: 20.0\n", ""), "tasks[0]"),
    ("task period zero", BASE.replace("heartbeat_period_ms: 20.0", "heartbeat_period_ms: 0"),
     "tasks[0].heartbeat_period_ms"),
    ("task period negative", BASE.replace("heartbeat_period_ms: 20.0",
                                          "heartbeat_period_ms: -20"),
     "tasks[0].heartbeat_period_ms"),
    ("task jitter negative", BASE.replace("heartbeat_jitter_ms: 2.0",
                                          "heartbeat_jitter_ms: -2"),
     "tasks[0].heartbeat_jitter_ms"),
    ("task sigma negative", BASE.replace("jitter_sigma_ms: 0.5", "jitter_sigma_ms: -0.5"),
     "tasks[0].jitter_sigma_ms"),
    ("task tolerance missing", BASE.replace("    missed_beats_tolerated: 1\n", ""), "tasks[0]"),
    ("task tolerance negative", BASE.replace("missed_beats_tolerated: 1",
                                             "missed_beats_tolerated: -1"),
     "tasks[0].missed_beats_tolerated"),
    ("task tolerance not an int", BASE.replace("missed_beats_tolerated: 1",
                                               "missed_beats_tolerated: 1.5"),
     "tasks[0].missed_beats_tolerated"),
    ("task tolerance is a bool", BASE.replace("missed_beats_tolerated: 1",
                                              "missed_beats_tolerated: true"),
     "tasks[0].missed_beats_tolerated"),
    ("task deadline missing", BASE.replace("    liveness_deadline_ms: 100.0\n", ""), "tasks[0]"),
    ("task deadline zero", BASE.replace("liveness_deadline_ms: 100.0",
                                        "liveness_deadline_ms: 0"),
     "tasks[0].liveness_deadline_ms"),
    ("task deadline negative", BASE.replace("liveness_deadline_ms: 100.0",
                                            "liveness_deadline_ms: -5"),
     "tasks[0].liveness_deadline_ms"),
    ("task escalation missing", BASE.replace("    escalation: L\n", ""), "tasks[0]"),
    ("task escalation empty", BASE.replace("escalation: L", "escalation: ''"),
     "tasks[0].escalation"),
    ("task budget negative", BASE.replace("false_trip_budget_per_hour: 1.0",
                                          "false_trip_budget_per_hour: -1"),
     "tasks[0].false_trip_budget_per_hour"),
    ("task budget not a number", BASE.replace("false_trip_budget_per_hour: 1.0",
                                              "false_trip_budget_per_hour: lots"),
     "tasks[0].false_trip_budget_per_hour"),
    ("task name not a string", BASE.replace("  - id: T\n", "  - id: T\n    name: 3\n"),
     "tasks[0].name"),
    ("duplicate task id", BASE + (
        "  - id: T\n    heartbeat_period_ms: 20.0\n    missed_beats_tolerated: 1\n"
        "    liveness_deadline_ms: 100.0\n    escalation: L\n"), "tasks[1].id"),
    ("non-string mapping key", BASE + "? [1, 2]\n: 3\n", "not valid YAML"),
]


@pytest.mark.parametrize(("label", "document", "path"), CASES, ids=[case[0] for case in CASES])
def test_malformed_documents_are_rejected_by_key_path(label, document, path):
    with pytest.raises(SpecFormatError) as excinfo:
        parse_spec(document)
    assert path in str(excinfo.value), f"{label}: {excinfo.value}"


def test_every_shipped_case_is_distinct():
    assert len({case[1] for case in CASES}) == len(CASES)


def test_parse_spec_rejects_a_non_string():
    with pytest.raises(TypeError, match="text must be a str"):
        parse_spec(b"spec_version: 1")  # type: ignore[arg-type]


def test_load_spec_reports_a_missing_file(tmp_path):
    with pytest.raises(SpecFormatError, match="no such specification file"):
        load_spec(tmp_path / "absent.yaml")


def test_load_spec_reports_a_directory(tmp_path):
    with pytest.raises(SpecFormatError, match="is a directory"):
        load_spec(tmp_path)


def test_load_spec_reports_invalid_utf8(tmp_path):
    target = tmp_path / "bad.yaml"
    target.write_bytes(b"\xff\xfe\x00spec_version: 1")
    with pytest.raises(SpecFormatError, match="not valid UTF-8"):
        load_spec(target)
