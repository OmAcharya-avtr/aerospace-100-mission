"""Findings per check code on the planted and worked cases.

Writes ``screenshots/findings_by_code.png``.

    python examples/findings_by_code.py
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import _plotstyle
from _bootstrap import CASES, FIXTURES, SCREENSHOTS
from _plotstyle import BAD, NEUTRAL, plt

from hazardtrace import load_analysis, load_junit, load_requirements, run_checks
from hazardtrace.checks import CHECKS


def main() -> int:
    """Draw the per-code bar chart. Returns a process exit code."""
    planted = run_checks(
        load_analysis(CASES / "planted_gaps.yaml"),
        requirements=load_requirements(FIXTURES / "planted" / "docs" / "REQUIREMENTS.md"),
        tests=load_junit(FIXTURES / "planted" / "junit.xml"),
    )
    worked = run_checks(
        load_analysis(CASES / "p024_bdot_detumble.yaml"),
        requirements=load_requirements(FIXTURES / "p024_bdot" / "docs" / "REQUIREMENTS.md"),
        tests=load_junit(FIXTURES / "p024_bdot" / "junit.xml"),
    )

    codes = [spec.code for spec in CHECKS]
    planted_counts = [planted.count_by_code()[code] for code in codes]
    worked_counts = [worked.count_by_code()[code] for code in codes]

    fig, axis = plt.subplots(figsize=(12.0, 4.6))
    positions = range(len(codes))
    width = 0.4
    axis.bar(
        [p - width / 2 for p in positions], planted_counts, width,
        label="planted_gaps.yaml (one per code, by construction)",
        color=NEUTRAL, alpha=0.8,
    )
    axis.bar(
        [p + width / 2 for p in positions], worked_counts, width,
        label="p024_bdot_detumble.yaml (the real gaps)", color=BAD, alpha=0.8,
    )
    axis.set_xticks(list(positions))
    axis.set_xticklabels(codes, rotation=45, ha="right", fontsize=8)
    axis.set_ylabel("findings")
    axis.set_yticks(range(0, max(max(planted_counts), max(worked_counts)) + 2))
    axis.legend(fontsize=8, loc="upper right")
    axis.set_title(
        f"Findings per check. Planted case total {len(planted.findings)}, "
        f"worked example total {len(worked.findings)}",
        fontsize=10,
    )
    out = _plotstyle.finish(fig, SCREENSHOTS / "findings_by_code.png")

    for spec, planted_n, worked_n in zip(CHECKS, planted_counts, worked_counts, strict=True):
        print(f"{spec.code}  planted {planted_n}  worked {worked_n}  {spec.title}")
    print(f"wrote {out.relative_to(SCREENSHOTS.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
