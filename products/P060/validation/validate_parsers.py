"""Known-answer checks on the two parsers.

Every expected value was obtained by reading the fixture, not by running the
code. A failed check is printed as FAIL and counted; the script still exits 0,
because its job is to produce evidence, not to gate a build. The gate command
is the CLI, which does exit non-zero on findings.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from traceaudit import (  # noqa: E402
    TraceConfig,
    parse_collection_report,
    parse_junit_xml,
    parse_requirements_file,
    parse_requirements_text,
)

SAMPLE = ROOT / "fixtures" / "sample_project"
FAILURES: list[str] = []


def check(label: str, got, expected) -> None:
    ok = got == expected
    if not ok:
        FAILURES.append(label)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label:<58s} got {got!r}   expected {expected!r}")


def main() -> int:
    print("1. requirement declarations in the bundled sample document")
    print("   hand count: REQ-001..REQ-008 as headings, plus one repeat of REQ-007")
    doc = parse_requirements_file(SAMPLE / "docs" / "REQUIREMENTS.md", relative_to=SAMPLE)
    check("declarations", len(doc.declarations), 9)
    check("unique ids", len(doc.ids), 8)
    check("duplicated ids", sorted(doc.duplicates), ["REQ-007"])
    check("REQ-007 declared twice", len(doc.duplicates["REQ-007"]), 2)
    check("first id", doc.declarations[0].id, "REQ-001")
    check("first title", doc.declarations[0].title, "Wheel speed input validation")
    check(
        "mid-sentence mention of REQ-002 not declared",
        sum(1 for r in doc.declarations if r.id == "REQ-002"),
        1,
    )

    print()
    print("2. requirement declarations in this package's own document")
    print("   hand count: REQ-001..REQ-026, each a level-3 heading, no duplicates")
    own = parse_requirements_file(ROOT / "docs" / "REQUIREMENTS.md", relative_to=ROOT)
    check("declarations", len(own.declarations), 26)
    check("unique ids", len(own.ids), 26)
    check("duplicates", own.duplicates, {})
    check("ids are contiguous", list(own.ids),
          [f"REQ-{n:03d}" for n in range(1, 27)])

    print()
    print("3. declaration forms, on a document written for this check")
    text = ("## REQ-001 Heading\n- REQ-002 Bullet\n* **REQ-003** Bold bullet\n"
            "+ REQ-004\nprose mentioning REQ-005 mid-line\n"
            "```\n## REQ-006 In a fence\n```\n")
    parsed = parse_requirements_text(text, source="forms.md")
    check("heading, bullet, bold, bare", [r.id for r in parsed.declarations],
          ["REQ-001", "REQ-002", "REQ-003", "REQ-004"])
    check("mid-line mention excluded", "REQ-005" in parsed.ids, False)
    check("fenced declaration excluded", "REQ-006" in parsed.ids, False)
    unfenced = parse_requirements_text(text, source="forms.md",
                                       config=TraceConfig(skip_code_fences=False))
    check("fenced declaration included when configured", "REQ-006" in unfenced.ids, True)

    print()
    print("4. junit XML outcomes in the bundled sample report")
    print("   hand count from the fixture suite: 5 passed, 1 failed, 1 skipped, 1 xfailed")
    report = parse_junit_xml(SAMPLE / "junit.xml", relative_to=SAMPLE)
    check("cases parsed", len(report.cases), 8)
    check("outcome counts", report.outcome_counts,
          {"passed": 5, "failed": 1, "errored": 0, "skipped": 1, "xfailed": 1, "unknown": 0})
    check("header tests attribute", report.declared_totals["tests"], 8)
    check("header skipped attribute counts the xfail too",
          report.declared_totals["skipped"], 2)
    check("counted skipped + xfailed equals the header",
          report.outcome_counts["skipped"] + report.outcome_counts["xfailed"], 2)
    check("claimed ids", list(report.claimed_ids),
          ["REQ-001", "REQ-002", "REQ-003", "REQ-004",
           "REQ-006", "REQ-007", "REQ-008", "REQ-042"])
    skip_case = next(c for c in report.cases if c.name == "test_desaturation_threshold")
    check("skip reason preserved", skip_case.message,
          "simulator fixture not available in this environment")
    xfail_case = next(c for c in report.cases if c.name == "test_hysteresis_band")
    check("xfail distinguished from skip", xfail_case.outcome.value, "xfailed")
    check("source file recovered from the xunit1 file attribute",
          skip_case.file, "tests/test_monitor.py")

    print()
    print("5. the same suite through a collection report")
    print("   hand count: the same 8 node ids, no outcomes, no properties")
    collection = parse_collection_report(SAMPLE / "collect_only.txt")
    check("cases parsed", len(collection.cases), 8)
    check("every outcome unknown",
          {c.outcome.value for c in collection.cases}, {"unknown"})
    check("no claims at all", collection.claimed_ids, ())
    check("node ids identical to the junit ones",
          [c.node_id for c in collection.cases],
          [c.node_id for c in report.cases])

    print()
    print(f"FAILED CHECKS: {len(FAILURES)}")
    for name in FAILURES:
        print(f"  - {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
