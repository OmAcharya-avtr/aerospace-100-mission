"""Attainable moment sets, nominal and after effector failures.

Draws the AMS of the eight-thruster reference cluster and of the four-wheel
pyramid array, and overlays the shrunken set left after one and two thruster
failures. Saves ``screenshots/attainable_moment_set.png``.

Run: ``python examples/ams_demo.py``
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

from alloclab.ams import attainable_moment_set, zonotope_volume  # noqa: E402
from alloclab.dataset import reference_thruster_cluster  # noqa: E402
from alloclab.effectors import pyramid_reaction_wheels  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "screenshots" / "attainable_moment_set.png"


def draw(ax, ams, color, alpha, label):
    """Draw an AMS as a translucent triangulated surface."""
    tris = [ams.hull.points[s] for s in ams.hull.simplices]
    coll = Poly3DCollection(tris, alpha=alpha, facecolor=color, edgecolor=color, linewidths=0.25)
    ax.add_collection3d(coll)
    ax.scatter([], [], [], color=color, label=label)


def set_equal(ax, pts):
    span = np.max(np.abs(pts)) * 1.05
    ax.set_xlim(-span, span)
    ax.set_ylim(-span, span)
    ax.set_zlim(-span, span)
    ax.set_xlabel("$L_x$ [N$\\cdot$m]")
    ax.set_ylabel("$L_y$ [N$\\cdot$m]")
    ax.set_zlabel("$L_z$ [N$\\cdot$m]")


def main() -> None:
    cluster = reference_thruster_cluster(max_thrust=1.0, arm=0.5)
    wheels = pyramid_reaction_wheels(max_torque=0.1)

    nominal = attainable_moment_set(cluster)
    one_out = attainable_moment_set(cluster.with_failures([0]))
    two_out = attainable_moment_set(cluster.with_failures([0, 1]))
    wheel_ams = attainable_moment_set(wheels)

    fig = plt.figure(figsize=(15.8, 5.4))

    ax = fig.add_subplot(1, 3, 1, projection="3d")
    draw(ax, nominal, "#3b6ea5", 0.20, f"nominal, V={nominal.volume:.3f}")
    set_equal(ax, nominal.vertices)
    ax.set_title(
        f"8-thruster cluster\n{nominal.n_vertices} vertices, "
        f"closed-form V={zonotope_volume(cluster):.6f}",
        fontsize=10,
    )
    ax.legend(loc="upper left", fontsize=8)

    ax = fig.add_subplot(1, 3, 2, projection="3d")
    draw(ax, nominal, "#b0b7bf", 0.10, f"nominal, V={nominal.volume:.3f}")
    draw(ax, one_out, "#d98324", 0.25, f"t1 failed, V={one_out.volume:.3f}")
    draw(ax, two_out, "#a63a3a", 0.35, f"t1+t2 failed, V={two_out.volume:.3f}")
    set_equal(ax, nominal.vertices)
    ax.set_title(
        "Same cluster after failures\n"
        f"volume ratios {one_out.volume / nominal.volume:.4f} and "
        f"{two_out.volume / nominal.volume:.4f}",
        fontsize=10,
    )
    ax.legend(loc="upper left", fontsize=8)

    ax = fig.add_subplot(1, 3, 3, projection="3d")
    draw(ax, wheel_ams, "#3f7d54", 0.22, f"4-wheel pyramid, V={wheel_ams.volume:.6f}")
    set_equal(ax, wheel_ams.vertices)
    ax.set_title(
        f"Pyramid wheel array, 0.1 N$\\cdot$m each\n{wheel_ams.n_vertices} vertices, "
        "54.7356 deg half angle",
        fontsize=10,
    )
    ax.legend(loc="upper left", fontsize=8)

    fig.suptitle(
        "Attainable moment sets: every body torque the effector set can produce "
        "within its command bounds",
        fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.0, 0.985, 0.97))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)
    plt.close(fig)

    print(f"saved {OUT}")
    print(
        f"cluster nominal : {nominal.n_vertices} vertices, "
        f"volume {nominal.volume:.9f} (N*m)^3, closed form {zonotope_volume(cluster):.9f}"
    )
    print(
        f"one thruster out: volume {one_out.volume:.9f}, "
        f"ratio {one_out.volume / nominal.volume:.6f}"
    )
    print(
        f"two thrusters out: volume {two_out.volume:.9f}, "
        f"ratio {two_out.volume / nominal.volume:.6f}"
    )
    print(
        f"pyramid wheels  : {wheel_ams.n_vertices} vertices, "
        f"volume {wheel_ams.volume:.9f}, closed form {zonotope_volume(wheels):.9f}"
    )


if __name__ == "__main__":
    main()
