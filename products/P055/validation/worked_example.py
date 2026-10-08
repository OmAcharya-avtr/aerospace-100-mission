"""The worked example printed in the README, run so the README's output is real.

The snippet in README.md under "A worked example" is the body of :func:`main`
below with ``emit`` replaced by ``print``; its output block is
``worked_example_output.txt``, pasted verbatim. Re-running this script is how
that section stays true.

Run: ``python validation/worked_example.py``
Output: ``validation/worked_example_output.txt``
"""

from __future__ import annotations

import os

import _bootstrap

from assuregraph import format_coverage_table, load_case, render_mermaid, run_checks

LINES: list[str] = []


def emit(text: str = "") -> None:
    """Collect a line. The README snippet uses ``print`` here instead."""
    LINES.append(text)


def main() -> int:
    case = load_case(os.path.join(_bootstrap.EXAMPLES, "incomplete_case.yaml"))
    report = run_checks(case)

    emit(f"nodes / edges : {len(case.nodes)} / {len(case.edges)}")
    emit(f"top goals     : {', '.join(case.resolved_top_goals())}")
    emit(f"complete      : {report.is_complete}    exit code: {report.exit_code}")
    emit()

    for finding in report.findings:
        emit(f"  {finding.severity.value:<8} {finding.check:<26} {','.join(finding.node_ids)}")
    emit()

    for item in report.evidence:
        recorded = (item.recorded_sha256 or "none")[:12]
        actual = (item.actual_sha256 or "none")[:12]
        emit(
            f"  {item.node_id:<4} {item.status.value:<13} recorded {recorded:<13} "
            f"found {actual:<13} {item.cited_path}"
        )
    emit()

    emit(format_coverage_table(report.coverage).rstrip("\n"))
    emit()
    emit(render_mermaid(case, report=report, max_label_chars=44).splitlines()[1])

    target = __file__.replace(".py", "_output.txt")
    with open(target, "w", encoding="utf-8") as handle:
        handle.write("\n".join(LINES) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
