"""The nine design-matrix columns, as functions on the sky.

What to notice: over a narrow elevation band the three cross-elevation columns
cos(E), 1 and sin(E) are nearly the same constant, and the four elevation-side
columns 1, cos(E), sin(E) and rho_hat(E) are too. That is the whole source of
the degeneracies this package refuses to split.
"""

from __future__ import annotations

import numpy as np

from _common import GRID, MARGINAL, REFUSED, bootstrap, save, style

ROOT = bootstrap()

import matplotlib.pyplot as plt  # noqa: E402

from pointmodel import terms  # noqa: E402


def main() -> int:
    style()
    elevation = np.radians(np.linspace(2.0, 89.0, 400))
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    cross, elev = terms.design_matrix(np.zeros_like(elevation), elevation)
    degrees = np.rad2deg(elevation)

    for name, style_kwargs in (
        ("IA", {"color": "#1f4e79"}),
        ("CA", {"color": "#3f8f4f"}),
        ("NPAE", {"color": "#6f3f8f"}),
    ):
        axes[0].plot(
            degrees, cross[:, terms.TERM_INDEX[name]], label=name, lw=1.6, **style_kwargs
        )
    axes[0].axvspan(43.0, 47.0, color=REFUSED, alpha=0.15)
    axes[0].text(
        45.0, 1.05, "narrow_el\nband", ha="center", va="bottom", fontsize=8, color=REFUSED
    )
    axes[0].set_title("cross-elevation columns at azimuth 0")
    axes[0].set_xlabel("elevation, degrees")
    axes[0].set_ylabel("dimensionless column value")
    axes[0].set_ylim(-0.05, 1.25)
    axes[0].legend(loc="center left")

    for name, colour in (
        ("IE", "#1f4e79"),
        ("TF", "#3f8f4f"),
        ("TS", "#6f3f8f"),
        ("RF", MARGINAL),
    ):
        axes[1].plot(
            degrees, elev[:, terms.TERM_INDEX[name]], label=name, lw=1.6, color=colour
        )
    axes[1].axvspan(43.0, 47.0, color=REFUSED, alpha=0.15)
    axes[1].set_title("elevation columns (RF is the refraction shape)")
    axes[1].set_xlabel("elevation, degrees")
    axes[1].set_ylabel("dimensionless column value")
    axes[1].set_yscale("log")
    axes[1].set_ylim(0.005, 40.0)
    axes[1].grid(which="both", color=GRID, lw=0.4)
    axes[1].legend(loc="upper right")

    figure.suptitle(
        "pointing-model design-matrix columns: where the degeneracies come from",
        fontsize=11,
    )
    figure.subplots_adjust(top=0.86, wspace=0.26)
    path = save(figure, "term_shapes")
    print(f"written: {path}")
    print("")
    print("Column values at the centre of the narrow_el band (45 degrees):")
    mid_cross, mid_elev = terms.design_matrix(
        np.array([0.0]), np.array([np.radians(45.0)])
    )
    for name in ("IA", "CA", "NPAE"):
        print(f"  dX {name:<4s} = {float(mid_cross[0, terms.TERM_INDEX[name]]):.6f}")
    for name in ("IE", "TF", "TS", "RF"):
        print(f"  dE {name:<4s} = {float(mid_elev[0, terms.TERM_INDEX[name]]):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
