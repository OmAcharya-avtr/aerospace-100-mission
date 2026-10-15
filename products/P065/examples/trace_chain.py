"""The six link ratios of the worked example, each with its denominator.

Writes ``screenshots/trace_chain.png``.

    python examples/trace_chain.py
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import _plotstyle
from _bootstrap import CASES, FIXTURES, SCREENSHOTS
from _plotstyle import BAD, GOOD, NEUTRAL, plt

from hazardtrace import load_analysis, load_junit, load_requirements, run_checks


def main() -> int:
    """Draw the trace chain. Returns a process exit code."""
    report = run_checks(
        load_analysis(CASES / "p024_bdot_detumble.yaml"),
        requirements=load_requirements(FIXTURES / "p024_bdot" / "docs" / "REQUIREMENTS.md"),
        tests=load_junit(FIXTURES / "p024_bdot" / "junit.xml"),
    )
    items = [(label, ratio) for label, ratio in report.coverage.as_items() if ratio.defined]

    labels = [f"{label}\n{ratio.numerator}/{ratio.denominator}" for label, ratio in items]
    values = [ratio.percent for _label, ratio in items]
    colours = [GOOD if value >= 100.0 else BAD for value in values]

    fig, axis = plt.subplots(figsize=(11.0, 5.0))
    bars = axis.barh(range(len(items)), values, color=colours, alpha=0.75)
    axis.set_yticks(range(len(items)))
    axis.set_yticklabels(labels, fontsize=8)
    axis.invert_yaxis()
    axis.set_xlim(0, 108)
    axis.set_xlabel("percent of the denominator printed beside each label")
    axis.axvline(100.0, color=NEUTRAL, linestyle="--", linewidth=1.0)
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            bar.get_width() + 1.0, bar.get_y() + bar.get_height() / 2,
            f"{value:.2f} %", va="center", fontsize=8,
        )
    axis.set_title(
        "P024 B-dot worked example: every link in the chain, computed from the "
        "repository\nA bar short of the dashed line is a gap that is reported, not "
        "tidied away",
        fontsize=10,
    )
    out = _plotstyle.finish(fig, SCREENSHOTS / "trace_chain.png")

    for label, ratio in items:
        print(f"{label:<32}{ratio}")
    print(f"findings            : {len(report.findings)}; exit code {report.exit_code}")
    print(f"wrote {out.relative_to(SCREENSHOTS.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
