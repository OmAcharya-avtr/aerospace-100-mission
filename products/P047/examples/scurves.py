"""S-curves of the three detectors, and what each one's shape costs you.

Writes ``../screenshots/scurves.png``.  Runtime about 3 s.

Left column: the S-curve of each detector on a truncated Nyquist raised cosine,
with the straight line of slope K_d, the 10 % linear range, the flattening point
and the reversal marked.  Right column: the same detector across four pulse
shapes, which is where the gain collapses or the S-curve stops being
differentiable at the origin.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from slotsync.pulses import (  # noqa: E402
    half_sine,
    nyquist_raised_cosine,
    raised_cosine_time,
    rectangular,
    triangular,
)
from slotsync.scurve import default_offsets, scurve  # noqa: E402
from slotsync.ted import TedConfig  # noqa: E402

CONFIGS = (
    TedConfig("early-late", "antipodal", 0.25, "dd"),
    TedConfig("gardner", "antipodal"),
    TedConfig("mueller-muller", "antipodal"),
)
PULSES = (
    rectangular(1.0),
    triangular(1.0),
    raised_cosine_time(1.0),
    half_sine(1.0),
    nyquist_raised_cosine(0.5, 4.0),
)
REFERENCE = nyquist_raised_cosine(0.5, 4.0)
OFFSETS = default_offsets(0.75, 301)


def main() -> None:
    figure, axes = plt.subplots(3, 2, figsize=(12.5, 11.0), sharex=True)
    for row, config in enumerate(CONFIGS):
        curve = scurve(config, REFERENCE, OFFSETS, max_exact_symbols=16)
        left = axes[row, 0]
        left.axhline(0.0, color="0.75", linewidth=0.8)
        left.axvline(0.0, color="0.75", linewidth=0.8)
        left.plot(curve.offsets, curve.values, color="#1f4e79", linewidth=2.0, label="S(eps)")
        line = curve.gain * curve.offsets
        left.plot(
            curve.offsets,
            line,
            color="#c00000",
            linestyle="--",
            linewidth=1.2,
            label=f"slope K_d = {curve.gain:.4f}",
        )
        half = curve.linear_halfwidth
        left.axvspan(-half, half, color="#2e7d32", alpha=0.12, label=f"linear to {half:.3f}")
        left.axvline(curve.peak_offset, color="#ef6c00", linestyle=":", linewidth=1.3)
        left.annotate(
            f"flattens\n{curve.peak_offset:.3f}",
            xy=(curve.peak_offset, curve.peak_value),
            xytext=(curve.peak_offset + 0.03, curve.peak_value * 0.55),
            fontsize=8,
            color="#ef6c00",
        )
        reversal = curve.reversal_offset
        if reversal is not None:
            left.axvline(reversal, color="#6a1b9a", linestyle=":", linewidth=1.3)
            left.annotate(
                f"reverses\n{reversal:.3f}",
                xy=(reversal, 0.0),
                xytext=(reversal - 0.22, curve.peak_value * 0.25),
                fontsize=8,
                color="#6a1b9a",
            )
        left.set_ylim(-1.35 * abs(curve.peak_value), 1.35 * abs(curve.peak_value))
        left.set_ylabel("mean detector output")
        left.set_title(f"{config.label} on {REFERENCE.name}", fontsize=10)
        left.legend(fontsize=8, loc="upper left")
        left.grid(alpha=0.25)

        right = axes[row, 1]
        right.axhline(0.0, color="0.75", linewidth=0.8)
        right.axvline(0.0, color="0.75", linewidth=0.8)
        for pulse in PULSES:
            other = scurve(config, pulse, OFFSETS, max_exact_symbols=16)
            style = "-" if other.gain != 0.0 else "--"
            right.plot(
                other.offsets,
                other.values,
                style,
                linewidth=1.6,
                label=f"{pulse.name}: K_d = {other.gain:.3f}",
            )
        right.set_title(f"{config.detector} across pulse shapes", fontsize=10)
        right.legend(fontsize=8, loc="upper left")
        right.grid(alpha=0.25)
        finite = [
            scurve(config, pulse, OFFSETS, max_exact_symbols=16).peak_value for pulse in PULSES
        ]
        limit = max(abs(value) for value in finite if np.isfinite(value))
        right.set_ylim(-1.3 * limit, 1.3 * limit)

    for column in range(2):
        axes[2, column].set_xlabel("timing offset (symbol periods)")
    figure.suptitle(
        "Detector S-curves: the slope at the origin is K_d, and everything the loop "
        "analysis needs comes from it",
        fontsize=12,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    figure.savefig("../screenshots/scurves.png", dpi=130)
    plt.close(figure)
    print("wrote ../screenshots/scurves.png")
    for config in CONFIGS:
        for pulse in PULSES:
            curve = scurve(config, pulse, OFFSETS, max_exact_symbols=16)
            print(
                f"  {config.label:<26} {pulse.name:<12} K_d = {curve.gain:9.5f}  "
                f"linear {curve.linear_halfwidth:.3f}  peak {curve.peak_offset:.3f}"
            )


if __name__ == "__main__":
    main()
