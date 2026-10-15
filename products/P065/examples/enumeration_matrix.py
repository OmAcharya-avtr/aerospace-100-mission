"""Enumeration coverage of the three shipped cases, side by side.

Writes ``screenshots/enumeration_matrix.png``.

    python examples/enumeration_matrix.py
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import _plotstyle
from _bootstrap import CASES, SCREENSHOTS
from _plotstyle import BAD, GOOD, MUTED, plt

from hazardtrace import UCA_CLASSES, CellState, load_analysis, run_checks

CASE_FILES = ("complete_minimal.yaml", "p024_bdot_detumble.yaml", "planted_gaps.yaml")
FILL = {
    CellState.UCA: GOOD,
    CellState.EXCLUDED: MUTED,
    CellState.CONFLICT: "#d08a00",
    CellState.GAP: BAD,
}
MARK = {
    CellState.UCA: "UCA",
    CellState.EXCLUDED: "excl",
    CellState.CONFLICT: "both",
    CellState.GAP: "GAP",
}


def main() -> int:
    """Draw the three grids. Returns a process exit code."""
    reports = []
    for name in CASE_FILES:
        analysis = load_analysis(CASES / name)
        reports.append((name, analysis, run_checks(analysis)))

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))
    for axis, (name, analysis, report) in zip(axes, reports, strict=True):
        actions = list(analysis.structure.control_actions)
        axis.set_xlim(-0.5, 3.5)
        axis.set_ylim(-0.5, max(1, len(actions)) - 0.5)
        axis.set_xticks(range(4))
        axis.set_xticklabels(
            ["not\nprovided", "provided", "wrong\ntiming", "stopped /\ntoo long"],
            fontsize=7,
        )
        axis.set_yticks(range(len(actions)))
        axis.set_yticklabels([a.id for a in actions], fontsize=8)
        axis.invert_yaxis()
        covered, total = report.grid.completeness()
        axis.set_title(f"{name}\n{covered}/{total} cells covered", fontsize=9)
        for row in range(len(actions)):
            for col in range(4):
                cell = report.grid.cells[row * 4 + col]
                axis.add_patch(
                    plt.Rectangle(
                        (col - 0.45, row - 0.45), 0.9, 0.9,
                        facecolor=FILL[cell.state], alpha=0.25,
                        edgecolor=FILL[cell.state], linewidth=1.0,
                    )
                )
                axis.text(
                    col, row, MARK[cell.state], ha="center", va="center",
                    fontsize=7, color=FILL[cell.state],
                )
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.tick_params(length=0)

    fig.suptitle(
        "UCA enumeration: every control action against all four STPA classes "
        "(Leveson and Thomas, 2018, step 3)",
        fontsize=10,
    )
    out = _plotstyle.finish(fig, SCREENSHOTS / "enumeration_matrix.png")

    for name, _analysis, report in reports:
        covered, total = report.grid.completeness()
        counts = {state.value: report.grid.count(state) for state in CellState}
        print(f"{name:<30}{covered}/{total} covered  {counts}")
    print(f"classes per action  : {len(UCA_CLASSES)}")
    print(f"wrote {out.relative_to(SCREENSHOTS.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
