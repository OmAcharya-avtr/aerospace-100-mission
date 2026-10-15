"""The learned classifier against the declared threshold logic, over noise.

Both detectors are scored at an equal false-alarm rate at each declared noise
scale. Writes ../screenshots/detector_vs_threshold.png.

Run:
    PYTHONPATH=src python examples/detector_vs_threshold.py
"""

from __future__ import annotations

from pathlib import Path

from safemodetree import load_spec, noise_sweep, packaged_spec_path
from safemodetree.plotting import plot_detector_sweep

OUTPUT = Path(__file__).resolve().parent.parent / "screenshots" / "detector_vs_threshold.png"
SCALES = (1.0, 1.5, 2.0, 2.5, 3.0)


def main() -> None:
    """Run the sweep, print the table and save the figure."""
    spec = load_spec(packaged_spec_path("reaction_wheel_fmea"))
    results = noise_sweep(spec, noise_scales=SCALES, n_per_fault=400, seed=63001)
    header = (
        f"{'scale':>6} {'detector':<22} {'op_point':>9} {'FAR':>8} {'ident_acc':>10} "
        f"{'ECE':>8} {'brier':>8}"
    )
    print(header)
    print("-" * len(header))
    for scale, result in zip(SCALES, results, strict=True):
        for key in ("baseline", "learned"):
            score = result[key]
            brier = "-" if score.brier is None else f"{score.brier:.5f}"
            print(
                f"{scale:>6.1f} {score.name:<22} {score.operating_point:>9.6f} "
                f"{score.false_alarm_rate:>8.5f} {score.identification_accuracy:>10.5f} "
                f"{score.calibration.ece:>8.5f} {brier:>8}"
            )
    print()
    wins = sum(
        1
        for result in results
        if result["learned"].identification_accuracy
        > result["baseline"].identification_accuracy + 1e-9
    )
    print(f"the learned model beats the threshold baseline at {wins} of {len(SCALES)} scales")
    figure = plot_detector_sweep(
        SCALES,
        tuple(result["baseline"] for result in results),
        tuple(result["learned"] for result in results),
        float(results[0]["ceiling"]),
    )
    OUTPUT.parent.mkdir(exist_ok=True)
    figure.savefig(OUTPUT, dpi=140)
    print(f"wrote {OUTPUT.relative_to(OUTPUT.parent.parent)}")


if __name__ == "__main__":
    main()
