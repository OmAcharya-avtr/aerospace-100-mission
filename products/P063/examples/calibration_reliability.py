"""Reliability curves for the learned classifier and the threshold baseline.

Writes ../screenshots/calibration_reliability.png.

Run:
    PYTHONPATH=src python examples/calibration_reliability.py
"""

from __future__ import annotations

from pathlib import Path

from safemodetree import compare_detectors, load_spec, packaged_spec_path
from safemodetree.plotting import plot_reliability

OUTPUT = Path(__file__).resolve().parent.parent / "screenshots" / "calibration_reliability.png"


def main() -> None:
    """Score both detectors, print both reliability tables and save the figure."""
    spec = load_spec(packaged_spec_path("reaction_wheel_fmea"))
    result = compare_detectors(spec, n_per_fault=400, noise_scale=2.0, seed=63001)
    for key in ("baseline", "learned"):
        score = result[key]
        print(f"== {score.name} (operating point {score.operating_point:.6f})")
        print(score.calibration.table())
        print()
    figure = plot_reliability(
        {
            "threshold logic": result["baseline"].calibration,
            "learned, matched FAR": result["learned"].calibration,
        },
        title=f"noise scale {result['noise_scale']:g}, equal false-alarm rate",
    )
    OUTPUT.parent.mkdir(exist_ok=True)
    figure.savefig(OUTPUT, dpi=140)
    print(f"wrote {OUTPUT.relative_to(OUTPUT.parent.parent)}")


if __name__ == "__main__":
    main()
