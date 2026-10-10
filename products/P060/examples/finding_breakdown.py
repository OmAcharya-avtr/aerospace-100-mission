"""Findings by code, for the same project read three different ways.

Writes ../screenshots/finding_breakdown.png.
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
from traceaudit.plotting import plot_finding_breakdown  # noqa: E402

SAMPLE = ROOT / "fixtures" / "sample_project"


def main() -> int:
    document = parse_requirements_file(SAMPLE / "docs" / "REQUIREMENTS.md", relative_to=SAMPLE)
    junit = parse_junit_xml(SAMPLE / "junit.xml", relative_to=SAMPLE)
    collection = parse_collection_report(SAMPLE / "collect_only.txt")

    results = {
        "junit,\nheuristic on": audit(document, junit, test_root=SAMPLE / "tests"),
        "junit,\nno test root": audit(document, junit),
        "collection\nreport": audit(document, collection, test_root=SAMPLE / "tests"),
    }
    results["junit,\nTA006 ignored"] = audit(
        document, junit,
        config=TraceConfig(ignored_codes=("TA006",)),
        test_root=SAMPLE / "tests",
    )

    header = "  ".join(f"{code}" for code in CODE_DESCRIPTIONS)
    print(f"{'input':<26s} {header}  total  blocking  exit")
    for label, result in results.items():
        counts = result.counts_by_code()
        row = "  ".join(f"{counts[code]:>5d}" for code in CODE_DESCRIPTIONS)
        flat = label.replace("\n", " ")
        print(f"{flat:<26s} {row}  {len(result.findings):>5d}  "
              f"{len(result.blocking):>8d}  {result.exit_code:>4d}")

    out = plot_finding_breakdown(
        results,
        ROOT / "screenshots" / "finding_breakdown.png",
        title="the same project read four ways: findings by code",
    )
    print()
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
