"""Known-answer check of the three shipped example cases.

The expected finding set for each case was worked out by hand from the YAML
before the script was run, and is written into EXPECTED below. The script
compares the computed set against it and prints both. A mismatch is a FAIL and
makes the script exit nonzero; the committed output is the run the README and
VALIDATION.md tables were taken from.

Run: ``python validation/validate_known_answers.py``
Output: ``validation/validate_known_answers_output.txt``
"""

from __future__ import annotations

import os

import _bootstrap

from assuregraph import format_report, load_case, run_checks

Expectation = tuple[str, str, tuple[str, ...]]

#: Hand-worked expectations. (check, severity, node_ids).
EXPECTED: dict[str, list[Expectation]] = {
    "complete_case.yaml": [],
    "incomplete_case.yaml": [
        ("unsupported_claims", "error", ("G5",)),
        ("unsupported_claims", "info", ("G7",)),
        ("unsupported_claims", "info", ("G8",)),
        ("missing_evidence", "error", ("Sn3",)),
        ("stale_evidence", "warning", ("Sn1",)),
        ("stale_evidence", "error", ("Sn2",)),
        ("undischarged_assumptions", "error", ("A1",)),
        ("orphan_nodes", "error", ("G8",)),
        ("orphan_nodes", "error", ("C3",)),
    ],
    # The cycle walk's node ids are compared as a set, because which node the
    # walk starts at is an implementation detail of the traversal.
    "cyclic_case.yaml": [("cycles", "error", ("G1", "S1", "G2", "S2"))],
}

EXPECTED_EXIT = {"complete_case.yaml": 0, "incomplete_case.yaml": 1, "cyclic_case.yaml": 1}


def main() -> int:
    lines = [
        "Known-answer check of the shipped example cases -- assuregraph 0.1.0",
        "",
        "Expected finding sets were written by hand from the YAML before this",
        "script ran. Cycle findings are compared as a set of node ids because",
        "the walk's starting node is a traversal detail.",
        "",
    ]
    failures = 0
    for filename, expected in EXPECTED.items():
        path = os.path.join(_bootstrap.EXAMPLES, filename)
        report = run_checks(load_case(path))
        computed: list[Expectation] = [
            (f.check, f.severity.value, tuple(sorted(set(f.node_ids))) if f.check == "cycles"
             else f.node_ids)
            for f in report.findings
        ]
        normalised_expected = [
            (c, s, tuple(sorted(set(ids))) if c == "cycles" else ids) for c, s, ids in expected
        ]
        ok_findings = sorted(computed) == sorted(normalised_expected)
        ok_exit = report.exit_code == EXPECTED_EXIT[filename]
        if not (ok_findings and ok_exit):
            failures += 1

        lines.append(f"--- {filename} ---")
        lines.append(f"  nodes {len(load_case(path).nodes)}, edges {len(load_case(path).edges)}")
        lines.append(f"  expected findings ({len(normalised_expected)}):")
        for item in sorted(normalised_expected):
            lines.append(f"    {item[1]:<8} {item[0]:<26} {','.join(item[2])}")
        lines.append(f"  computed findings ({len(computed)}):")
        for item in sorted(computed):
            lines.append(f"    {item[1]:<8} {item[0]:<26} {','.join(item[2])}")
        lines.append(f"  finding sets match: {'yes' if ok_findings else 'NO -- FAIL'}")
        lines.append(
            f"  exit code expected {EXPECTED_EXIT[filename]}, computed {report.exit_code}: "
            f"{'match' if ok_exit else 'NO -- FAIL'}"
        )
        lines.append(
            f"  severity totals: {report.error_count} error, "
            f"{report.warning_count} warning, {report.info_count} info"
        )
        cov = report.coverage
        lines.append(
            "  coverage: claims argued or declared "
            f"{cov.claims_supported + cov.claims_undeveloped}/{cov.claims_total}; "
            f"artifacts present {cov.evidence_total - cov.evidence_missing}/{cov.evidence_total}; "
            f"fresh of checkable {cov.evidence_fresh}/"
            f"{cov.evidence_fresh + cov.evidence_stale}; "
            f"assumptions discharged {cov.assumptions_discharged}/{cov.assumptions_total}; "
            f"unverifiable {cov.evidence_unverifiable}; orphans {cov.orphan_count}; "
            f"cyclic regions {cov.cycle_regions}"
        )
        lines.append("")

    lines.append(
        "Full text report for incomplete_case.yaml, verbatim, with the repository"
    )
    lines.append("root replaced by <repo> so the committed output is byte-stable:")
    lines.append("")
    # Rendered wide so no filesystem path is wrapped across a line break, then
    # the repository root is replaced, so the committed output is byte-stable
    # wherever the repository is checked out.
    verbatim = format_report(
        run_checks(load_case(os.path.join(_bootstrap.EXAMPLES, "incomplete_case.yaml"))),
        width=400,
    ).rstrip("\n")
    lines.append(verbatim.replace(_bootstrap.REPO_ROOT, "<repo>"))
    lines.append("")
    lines.append(f"FAILURES: {failures}")

    target = __file__.replace(".py", "_output.txt")
    with open(target, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
