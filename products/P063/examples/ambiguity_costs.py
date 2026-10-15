"""The ambiguity classes of the shipped declaration and what they cost.

Writes ../screenshots/ambiguity_costs.png.

Run:
    PYTHONPATH=src python examples/ambiguity_costs.py
"""

from __future__ import annotations

from pathlib import Path

from safemodetree import analyse, load_spec, packaged_spec_path
from safemodetree.plotting import plot_ambiguity_costs

OUTPUT = Path(__file__).resolve().parent.parent / "screenshots" / "ambiguity_costs.png"


def main() -> None:
    """Print every ambiguity class with its costs and save the figure."""
    spec = load_spec(packaged_spec_path("reaction_wheel_fmea"))
    report = analyse(spec)
    header = f"{'class':<36} {'mode':<16} {'conflict':<9} {'cons':<18} {'cons_min':>9} {'cost':>8}"
    print(header)
    print("-" * len(header))
    for item in report.ambiguity:
        print(
            f"{'+'.join(item.faults):<36} {item.mode:<16} "
            f"{'YES' if item.conflicting else 'no':<9} "
            f"{item.conservative_response or 'NONE':<18} "
            f"{item.conservative_cost_min:>9.3f} {item.conservatism_cost_min:>8.3f}"
        )
    print()
    print(f"conflicting classes: {len(report.conflicting_ambiguity)} of {len(report.ambiguity)}")
    figure = plot_ambiguity_costs(
        report.ambiguity, title=f"{spec.name}: conservative against fault-specific cost"
    )
    OUTPUT.parent.mkdir(exist_ok=True)
    figure.savefig(OUTPUT, dpi=140)
    print(f"wrote {OUTPUT.relative_to(OUTPUT.parent.parent)}")


if __name__ == "__main__":
    main()
