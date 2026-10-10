"""The worked example reproduced verbatim in README.md.

Everything below the import block is the code the README shows, unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# ---- README code block starts here -----------------------------------------
from traceaudit import (  # noqa: E402
    audit,
    parse_junit_xml,
    parse_requirements_file,
    render_markdown_matrix,
)

project = ROOT / "fixtures" / "sample_project"

document = parse_requirements_file(project / "docs" / "REQUIREMENTS.md",
                                   relative_to=project)
report = parse_junit_xml(project / "junit.xml", relative_to=project)
result = audit(document, report, test_root=project / "tests")

print(f"declared ids      : {result.matrix.n_requirements}")
print(f"test cases        : {len(result.report.cases)}")
for figure in result.coverage:
    print(f"{figure.label:<18s}: {figure.numerator}/{figure.denominator} "
          f"= {figure.percent:.4f} %")
print()
for finding in result.findings:
    print(finding.render())
print()
print(f"exit status the CLI would use: {result.exit_code}")
print()
print(render_markdown_matrix(result))
# ---- README code block ends here -------------------------------------------
