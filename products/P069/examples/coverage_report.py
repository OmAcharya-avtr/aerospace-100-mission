"""Report what a finished campaign missed, and draw it.

    python examples/coverage_report.py

Writes ../screenshots/coverage_report.png and prints the report this package
exists to produce.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scenariocov import (  # noqa: E402
    build_report,
    load_campaign,
    load_class_document,
    load_space,
)
from scenariocov.plotting import plot_report_summary  # noqa: E402


def main() -> int:
    """Build the report, print it and save the figure."""
    space = load_space(ROOT / "fixtures" / "space.json")
    campaign = load_campaign(ROOT / "fixtures" / "campaign_executed.json")
    classes, boundaries = load_class_document(ROOT / "fixtures" / "classes.json")
    report = build_report(space, campaign, strength=2, classes=classes, boundaries=boundaries)
    print(report.to_text())
    out = plot_report_summary(report, ROOT / "screenshots" / "coverage_report.png")
    print(f"\nwrote screenshots/{out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
