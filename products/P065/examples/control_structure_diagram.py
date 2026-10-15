"""Render the P024 B-dot control structure and its enumeration grid.

Writes ``screenshots/bdot_control_structure.png`` and
``screenshots/bdot_control_structure.mmd``.

    python examples/control_structure_diagram.py
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
from _bootstrap import CASES, SCREENSHOTS

from hazardtrace import load_analysis, render_control_structure_png, render_mermaid, run_checks


def main() -> int:
    """Render the diagram. Returns a process exit code."""
    analysis = load_analysis(CASES / "p024_bdot_detumble.yaml")
    report = run_checks(analysis)

    mermaid_path = SCREENSHOTS / "bdot_control_structure.mmd"
    SCREENSHOTS.mkdir(parents=True, exist_ok=True)
    mermaid_path.write_text(render_mermaid(analysis, report=report), encoding="utf-8")
    png_path = render_control_structure_png(
        report, SCREENSHOTS / "bdot_control_structure.png"
    )

    structure = analysis.structure
    covered, total = report.grid.completeness()
    print(f"analysis            : {analysis.title}")
    print(f"components          : {len(structure.components)}")
    print(f"control actions     : {len(structure.control_actions)}")
    print(f"feedback paths      : {len(structure.feedback)}")
    print(f"enumeration cells   : {covered}/{total} covered, {len(report.grid.gaps)} gap(s)")
    for cell in report.grid.gaps:
        print(f"  gap: {cell.control_action} x {cell.uca_class.value}")
    print(f"wrote {mermaid_path.relative_to(SCREENSHOTS.parent)}")
    print(f"wrote {png_path.relative_to(SCREENSHOTS.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
