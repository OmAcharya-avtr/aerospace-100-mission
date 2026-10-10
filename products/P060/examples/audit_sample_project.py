"""Audit the bundled sample project and plot its trace matrix.

Writes ../screenshots/trace_matrix.png. Prints only paths relative to the
repository root, so the output can be pasted into a document.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from traceaudit import audit, parse_junit_xml, parse_requirements_file, render_text  # noqa: E402
from traceaudit.plotting import plot_trace_matrix  # noqa: E402

SAMPLE = ROOT / "fixtures" / "sample_project"


def main() -> int:
    document = parse_requirements_file(SAMPLE / "docs" / "REQUIREMENTS.md", relative_to=SAMPLE)
    report = parse_junit_xml(SAMPLE / "junit.xml", relative_to=SAMPLE)
    result = audit(document, report, test_root=SAMPLE / "tests")
    print(render_text(result, show_matrix=True))
    print()
    out = plot_trace_matrix(
        result,
        ROOT / "screenshots" / "trace_matrix.png",
        title="sample project: requirement-to-test trace matrix",
    )
    print(f"wrote {out.relative_to(ROOT)}")
    print(f"the CLI would exit {result.exit_code} on this input")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
