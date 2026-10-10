"""Every finding code, against an input built so that it must fire.

The expected sets below were enumerated by hand from the fixture sources.
Exits 0; the failure count is printed.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from traceaudit import (  # noqa: E402
    CODE_DESCRIPTIONS,
    TraceConfig,
    audit,
    parse_collection_report,
    parse_junit_xml,
    parse_requirements_file,
)

SAMPLE = ROOT / "fixtures" / "sample_project"
FAILURES: list[str] = []


def check(label: str, got, expected) -> None:
    ok = got == expected
    if not ok:
        FAILURES.append(label)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label:<56s} got {got!r}   expected {expected!r}")


def main() -> int:
    document = parse_requirements_file(SAMPLE / "docs" / "REQUIREMENTS.md", relative_to=SAMPLE)
    report = parse_junit_xml(SAMPLE / "junit.xml", relative_to=SAMPLE)
    result = audit(document, report, test_root=SAMPLE / "tests")

    print("1. the bundled sample project, junit XML, heuristic on")
    print("   hand enumeration: TA001 REQ-005, TA002 REQ-042, TA003 REQ-007,")
    print("   TA004 REQ-003 and REQ-004, TA005 REQ-008, TA006 test_request_counter")
    check("counts by code", result.counts_by_code(),
          {"TA001": 1, "TA002": 1, "TA003": 1, "TA004": 2, "TA005": 1, "TA006": 1})
    check("total findings", len(result.findings), 7)
    check("blocking findings", len(result.blocking), 7)
    check("exit code", result.exit_code, 1)
    check("subjects in order", [(f.code, f.subject) for f in result.findings],
          [("TA001", "REQ-005"), ("TA002", "REQ-042"), ("TA003", "REQ-007"),
           ("TA004", "REQ-003"), ("TA004", "REQ-004"), ("TA005", "REQ-008"),
           ("TA006", "tests/test_monitor.py::test_request_counter")])
    check("only TA006 is marked heuristic",
          {f.code for f in result.findings if f.heuristic}, {"TA006"})

    print()
    print("2. the finding messages name the evidence")
    by_subject = {(f.code, f.subject): f for f in result.findings}
    ta003 = by_subject[("TA003", "REQ-007")]
    check("TA003 names both declaration lines",
          ("docs/REQUIREMENTS.md:40" in ta003.message
           and "docs/REQUIREMENTS.md:54" in ta003.message), True)
    check("TA002 names the claiming test",
          "test_claims_an_undeclared_requirement"
          in by_subject[("TA002", "REQ-042")].message, True)
    check("TA004 names the skipped outcome for REQ-003",
          "[skipped]" in by_subject[("TA004", "REQ-003")].message, True)
    check("TA004 names the xfailed outcome for REQ-004",
          "[xfailed]" in by_subject[("TA004", "REQ-004")].message, True)
    check("TA005 names the failed outcome",
          "[failed]" in by_subject[("TA005", "REQ-008")].message, True)

    print()
    print("3. the skipped and xfailed requirements are the illusory ones")
    print("   REQ-003 and REQ-004 appear covered in nominal coverage and are not")
    check("nominal counts them", "REQ-003" in result.matrix.traced(), True)
    check("executed does not", "REQ-003" in result.matrix.executed_traced(), False)
    check("illusory set", list(result.matrix.illusory_traced()), ["REQ-003", "REQ-004"])
    check("nominal minus executed", len(result.matrix.traced())
          - len(result.matrix.executed_traced()), 2)

    print()
    print("4. withholding the test source root removes TA006 and nothing else")
    no_root = audit(document, report)
    check("counts by code", no_root.counts_by_code(),
          {"TA001": 1, "TA002": 1, "TA003": 1, "TA004": 2, "TA005": 1, "TA006": 0})
    check("a note says why", any("no test source root" in n for n in no_root.notes), True)

    print()
    print("5. ignoring a code keeps it reported and drops it from the exit status")
    ignored = audit(document, report, config=TraceConfig(ignored_codes=("TA006",)),
                    test_root=SAMPLE / "tests")
    check("findings unchanged", len(ignored.findings), 7)
    check("blocking reduced", len(ignored.blocking), 6)
    check("exit still 1 because others remain", ignored.exit_code, 1)
    all_ignored = audit(document, report,
                        config=TraceConfig(ignored_codes=tuple(CODE_DESCRIPTIONS)),
                        test_root=SAMPLE / "tests")
    check("ignoring every code gives exit 0", all_ignored.exit_code, 0)
    check("but the findings are still printed", len(all_ignored.findings), 7)

    print()
    print("6. the same project through a collection report loses two checks")
    print("   and gains six TA001, because a collection report carries no claims")
    collection = parse_collection_report(SAMPLE / "collect_only.txt")
    coll = audit(document, collection, test_root=SAMPLE / "tests")
    check("counts by code", coll.counts_by_code(),
          {"TA001": 8, "TA002": 0, "TA003": 1, "TA004": 0, "TA005": 0, "TA006": 0})
    check("a note says TA004 and TA005 were not computed",
          any("TA004 and TA005 were not computed" in n for n in coll.notes), True)
    check("only nominal coverage is defined", [f.label for f in coll.coverage],
          ["nominal coverage"])

    print()
    print("7. this package's own requirements document against a clean report")
    print("   see validate_self_trace.py for the live run; here the committed")
    print("   sample is used so the check does not depend on a pytest invocation")
    print("   (no numbers are produced in this section)")

    print()
    print(f"FAILED CHECKS: {len(FAILURES)}")
    for name in FAILURES:
        print(f"  - {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
