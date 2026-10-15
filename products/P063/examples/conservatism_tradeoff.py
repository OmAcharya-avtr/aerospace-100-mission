"""What routing on confidence costs, and what it buys.

Writes ../screenshots/conservatism_tradeoff.png.

Run:
    PYTHONPATH=src python examples/conservatism_tradeoff.py
"""

from __future__ import annotations

from pathlib import Path

from safemodetree import (
    compare_detectors,
    compare_policies,
    conservatism_sweep,
    load_spec,
    packaged_spec_path,
)
from safemodetree.plotting import plot_conservatism_sweep

OUTPUT = Path(__file__).resolve().parent.parent / "screenshots" / "conservatism_tradeoff.png"


def main() -> None:
    """Print the four policies and the threshold sweep, then save the figure."""
    spec = load_spec(packaged_spec_path("reaction_wheel_fmea"))
    comparison = compare_detectors(spec, n_per_fault=400, noise_scale=2.0, seed=63001)
    scores = compare_policies(
        spec, comparison, mode_id="fine_pointing", confidence_threshold=0.7
    )
    header = (
        f"{'policy':<22} {'thr':>6} {'cmded':>7} {'missed':>7} {'cost_min':>10} "
        f"{'misrout':>8} {'cons_fr':>8}"
    )
    print(header)
    print("-" * len(header))
    for name in ("oracle", "always_specific", "confidence_routed", "always_conservative"):
        score = scores[name]
        print(
            f"{score.name:<22} {score.confidence_threshold:>6.3f} {score.n_commanded:>7} "
            f"{score.n_missed:>7} {score.mean_cost_min:>10.4f} {score.n_misrouted:>8} "
            f"{score.conservative_fraction:>8.4f}"
        )
    print()
    print(
        f"cost of conservatism against the oracle:      "
        f"{float(scores['cost_of_conservatism_vs_oracle']):+.4f} min per commanded event"
    )
    print(
        f"cost of conservatism against always-specific: "
        f"{float(scores['cost_of_conservatism_vs_specific']):+.4f} min per commanded event"
    )
    print(f"mis-routes avoided: {scores['misroutes_avoided']}")
    rows = conservatism_sweep(spec, comparison, mode_id="fine_pointing")
    figure = plot_conservatism_sweep(rows)
    OUTPUT.parent.mkdir(exist_ok=True)
    figure.savefig(OUTPUT, dpi=140)
    print(f"wrote {OUTPUT.relative_to(OUTPUT.parent.parent)}")


if __name__ == "__main__":
    main()
