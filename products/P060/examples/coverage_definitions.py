"""The same requirements and the same tests, three definitions of covered.

Also shows what a pytest collection report costs: it carries no outcomes and
no properties, so nothing is claimed and nothing is traced.

Writes ../screenshots/coverage_definitions.png.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from traceaudit import (  # noqa: E402
    TraceConfig,
    audit,
    parse_collection_report,
    parse_junit_xml,
    parse_requirements_file,
)
from traceaudit.plotting import plot_coverage_figures  # noqa: E402

SAMPLE = ROOT / "fixtures" / "sample_project"


def main() -> int:
    document = parse_requirements_file(SAMPLE / "docs" / "REQUIREMENTS.md", relative_to=SAMPLE)
    results = {}

    junit = parse_junit_xml(SAMPLE / "junit.xml", relative_to=SAMPLE)
    results["junit XML"] = audit(document, junit, test_root=SAMPLE / "tests")

    # Claims from the node id alone: the sample project encodes nothing in its
    # test names, so every requirement reads as untraced.
    node_only = TraceConfig(claim_property_names=("no_such_property",))
    junit_node_only = parse_junit_xml(SAMPLE / "junit.xml", config=node_only, relative_to=SAMPLE)
    results["node ids only"] = audit(
        document, junit_node_only, config=node_only, test_root=SAMPLE / "tests"
    )

    collection = parse_collection_report(SAMPLE / "collect_only.txt")
    results["collection report"] = audit(document, collection, test_root=SAMPLE / "tests")

    for label, result in results.items():
        print(f"{label}:")
        for figure in result.coverage:
            print(f"  {figure.render()}")
        if not result.matrix.outcomes_known:
            print("  executed coverage: undefined (report carries no outcomes)")
            print("  passing coverage : undefined (report carries no outcomes)")
        print(f"  findings: {len(result.findings)}, exit status {result.exit_code}")

    out = plot_coverage_figures(
        results,
        ROOT / "screenshots" / "coverage_definitions.png",
        title="coverage depends on the definition and on the report you feed it",
    )
    print()
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
