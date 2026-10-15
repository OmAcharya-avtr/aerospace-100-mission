"""The planted broken declaration, and the findings the checker reports on it.

Writes ../screenshots/planted_case.png.

Run:
    PYTHONPATH=src python examples/planted_case.py
"""

from __future__ import annotations

from pathlib import Path

from safemodetree import analyse, load_spec, packaged_spec_path, reachability_report
from safemodetree.plotting import plot_reachability

OUTPUT = Path(__file__).resolve().parent.parent / "screenshots" / "planted_case.png"


def main() -> None:
    """Analyse the planted declaration, print the findings and save the figure."""
    spec = load_spec(packaged_spec_path("planted_unreachable"))
    report = analyse(spec)
    print(report.report())
    figure = plot_reachability(reachability_report(spec), spec)
    OUTPUT.parent.mkdir(exist_ok=True)
    figure.savefig(OUTPUT, dpi=140)
    print(f"wrote {OUTPUT.relative_to(OUTPUT.parent.parent)}")


if __name__ == "__main__":
    main()
