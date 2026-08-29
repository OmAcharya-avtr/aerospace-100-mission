"""Example 1: the scintillation saturation curve and its multi-valued band.

Plots the heuristic weak-to-saturated scintillation index model against the
weak (Rytov) theory line, marking the asymptote, the "focusing" overshoot
peak, and one concrete multi-valued inversion (two Cn2 candidates for one
measurement).

    python examples/saturation_curve.py

Writes ``screenshots/saturation_curve.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from turbscope.scintillometer import (  # noqa: E402
    SATURATION_ASYMPTOTE,
    invert_cn2_all_roots,
    saturation_peak,
    scintillation_index_full,
)
from turbscope.synthetic import SCINT_WAVELENGTH_M, WAVE_TYPE  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "screenshots" / "saturation_curve.png"


def main() -> int:
    x = np.geomspace(1e-3, 30.0, 2000)
    y_full = scintillation_index_full(x)
    x_peak, val_peak = saturation_peak()

    example_target = 0.5 * (SATURATION_ASYMPTOTE + val_peak)
    result = invert_cn2_all_roots(example_target, 1000.0, SCINT_WAVELENGTH_M, WAVE_TYPE)

    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    ax.plot(x, y_full, color="tab:blue", lw=2.2, label=r"$\sigma_I^2$ (full curve, this product)")
    ax.plot(x, x, color="k", lw=1.2, ls="--", label=r"weak theory: $\sigma_I^2 = \sigma_R^2$")
    ax.axhline(SATURATION_ASYMPTOTE, color="tab:gray", lw=1.0, ls=":", label=r"asymptote $\to 1$")
    ax.axhspan(
        SATURATION_ASYMPTOTE, val_peak, color="tab:red", alpha=0.12,
        label="multi-valued band (2+ roots for one measurement)",
    )
    ax.plot([x_peak], [val_peak], marker="o", color="tab:red", ms=7, zorder=5)
    ax.annotate(
        f"focusing peak\n$\\sigma_R^2$={x_peak:.2f}, $\\sigma_I^2$={val_peak:.3f}",
        xy=(x_peak, val_peak), xytext=(0.5, 1.45),
        arrowprops={"arrowstyle": "->", "color": "tab:red"}, fontsize=9, color="tab:red",
    )
    for rv in result.rytov_roots:
        ax.plot([rv], [example_target], marker="x", color="tab:purple", ms=10, mew=2.5, zorder=6)
    ax.axhline(example_target, color="tab:purple", lw=0.8, ls="-.", alpha=0.7)
    ax.annotate(
        f"one measurement, {len(result.rytov_roots)} Cn2 candidates\n"
        f"ratio {result.cn2_roots[1] / result.cn2_roots[0]:.2f}x"
        if len(result.cn2_roots) >= 2
        else "one root",
        xy=(result.rytov_roots[-1], example_target), xytext=(8.0, example_target - 0.35),
        arrowprops={"arrowstyle": "->", "color": "tab:purple"}, fontsize=9, color="tab:purple",
    )

    ax.set_xscale("log")
    ax.set_xlabel(r"Rytov variance $\sigma_R^2$ (weak-theory prediction)")
    ax.set_ylabel(r"scintillation index $\sigma_I^2$")
    ax.set_ylim(0.0, 1.65)
    ax.set_title(
        "Scintillation saturation: weak theory vs the full curve\n"
        "Heuristic bridging model built for this product -- see "
        "turbscope.scintillometer module docstring",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=130)
    plt.close(fig)

    sys.stdout.write(
        f"wrote {OUT}\n"
        f"saturation peak: sigma_R^2={x_peak:.4f}, sigma_I^2={val_peak:.4f}\n"
        f"multi-valued band: [{SATURATION_ASYMPTOTE:.4f}, {val_peak:.4f}]\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
