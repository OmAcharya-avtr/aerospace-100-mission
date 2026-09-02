"""Map the singular surfaces of a four-CMG pyramid array.

Writes ``screenshots/singularity_surfaces.png``.  Runtime: a few seconds.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cmgsteer.arrays import pyramid_array  # noqa: E402
from cmgsteer.singularity import (  # noqa: E402
    classify_singularity,
    momentum_envelope,
    singular_surface,
    singularity_measure,
)

OUT = Path(__file__).resolve().parents[1] / "screenshots" / "singularity_surfaces.png"


def main() -> int:
    array = pyramid_array()
    fig = plt.figure(figsize=(15.0, 4.6))

    ax = fig.add_subplot(1, 3, 1, projection="3d")
    momenta, _ = momentum_envelope(array, n_points=3000)
    radii = np.linalg.norm(momenta, axis=1)
    scatter = ax.scatter(
        momenta[:, 0], momenta[:, 1], momenta[:, 2], c=radii, s=3, cmap="viridis"
    )
    ax.set_xlabel("$h_x$ [N m s]")
    ax.set_ylabel("$h_y$ [N m s]")
    ax.set_zlabel("$h_z$ [N m s]")
    ax.set_title(
        f"outer (saturation) envelope\nradius {radii.min():.3f} to {radii.max():.3f} N m s"
    )
    fig.colorbar(scatter, ax=ax, shrink=0.65, label="$|h|$ [N m s]")

    ax = fig.add_subplot(1, 3, 2)
    slab = 0.04
    for signs in itertools.product((1.0, -1.0), repeat=4):
        arr_signs = np.array(signs, dtype=float)
        pts, angles = singular_surface(array, signs=arr_signs, n_points=6000)
        near = np.abs(pts[:, 1]) < slab
        if not np.any(near):
            continue
        external = bool(np.all(arr_signs == arr_signs[0]))
        ax.plot(
            pts[near, 0],
            pts[near, 2],
            ".",
            markersize=2.0,
            color="#b2182b" if external else "#2166ac",
            alpha=0.9 if external else 0.35,
        )
    ax.plot([], [], ".", color="#b2182b", label="external (saturation)")
    ax.plot([], [], ".", color="#2166ac", label="internal")
    ax.set_xlabel("$h_x$ [N m s]")
    ax.set_ylabel("$h_z$ [N m s]")
    ax.set_title(f"singular set, slice $|h_y| < {slab}$ N m s\nall 16 sign vectors")
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(1, 3, 3)
    frac = np.linspace(0.0, 1.0, 400)
    for target, label, colour in (
        (np.full(4, np.pi / 2), "to (90, 90, 90, 90) deg: external", "#b2182b"),
        (
            np.array([1.0, 1.0, 1.0, -1.0]) * np.pi / 2,
            "to (90, 90, 90, -90) deg: internal",
            "#2166ac",
        ),
        (
            np.array([1.0, 1.0, -1.0, -1.0]) * np.pi / 2,
            "to (90, 90, -90, -90) deg: internal",
            "#4d9221",
        ),
    ):
        measures = [singularity_measure(array.jacobian(f * target)) for f in frac]
        info = classify_singularity(array, target)
        ax.plot(frac, measures, color=colour, label=f"{label} ({info.passability})")
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.set_xlabel("fraction of the way from $\\delta = 0$ to the singular configuration")
    ax.set_ylabel("singularity measure $m$ [(N m s/rad)$^3$]")
    ax.set_title("$m$ along straight lines in gimbal space")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=130)
    plt.close(fig)
    print(f"saved {OUT}")
    print(f"envelope radius min {radii.min():.9f} max {radii.max():.9f} N*m*s")
    print(f"capacity sum(h0) {array.total_momentum_capacity:.6f} N*m*s")
    print(f"m at delta = 0 {singularity_measure(array.jacobian(np.zeros(4))):.9f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
