"""Safe-mode reachability for every fault state of the shipped declaration.

Writes ../screenshots/reachability_map.png and prints the table it plots.

Run:
    PYTHONPATH=src python examples/reachability_map.py
"""

from __future__ import annotations

from pathlib import Path

from safemodetree import (
    DetectionAssumption,
    load_spec,
    packaged_spec_path,
    reachability_report,
)
from safemodetree.plotting import plot_reachability

OUTPUT = Path(__file__).resolve().parent.parent / "screenshots" / "reachability_map.png"


def main() -> None:
    """Build the report, print it and save the figure."""
    spec = load_spec(packaged_spec_path("reaction_wheel_fmea"))
    report = reachability_report(spec)
    print(report.table())
    print(f"all fault states reach a declared safe mode: {report.all_reachable}")
    print(f"worst cheapest-path cost: {report.worst_cost_min:.3f} min")
    tight = reachability_report(spec, DetectionAssumption(latency_budget_s=60.0))
    print(
        f"with a 60 s detection latency budget: all reachable {tight.all_reachable}, "
        f"worst {tight.worst_cost_min:.3f} min"
    )
    figure = plot_reachability(report, spec)
    OUTPUT.parent.mkdir(exist_ok=True)
    figure.savefig(OUTPUT, dpi=140)
    print(f"wrote {OUTPUT.relative_to(OUTPUT.parent.parent)}")


if __name__ == "__main__":
    main()
