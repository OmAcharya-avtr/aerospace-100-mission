"""Example: SLERP attitude interpolation visualization.

Interpolates between two attitudes with quatkit.quat_slerp (Shoemake 1985)
and shows the two properties that make SLERP the standard attitude
interpolant:

  1. the rotation angle from the start attitude grows LINEARLY in t
     (constant angular velocity along the geodesic), unlike normalized
     linear interpolation (nlerp);
  2. a body axis rotated by the interpolated attitudes traces a great-circle
     arc on the unit sphere.

Run from products/P007/:  python examples/slerp_demo.py
Output: ../screenshots/slerp_interpolation.png
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quatkit import (  # noqa: E402
    Quaternion,
    angle_between,
    quat_normalize,
    quat_rotate,
    quat_slerp,
)


def main() -> None:
    q0 = Quaternion.from_euler_zyx(np.radians(10.0), np.radians(-20.0), np.radians(5.0))
    q1 = Quaternion.from_euler_zyx(np.radians(140.0), np.radians(45.0), np.radians(-60.0))
    total = float(angle_between(q1.as_array(), q0.as_array()))

    t = np.linspace(0.0, 1.0, 101)
    q_slerp = quat_slerp(q0.as_array(), q1.as_array(), t)
    # Naive normalized linear interpolation for comparison.
    a, b = q0.as_array(), q1.as_array()
    if np.dot(a, b) < 0:
        b = -b
    q_nlerp = quat_normalize((1 - t)[:, None] * a + t[:, None] * b)

    ang_slerp = angle_between(q_slerp, np.tile(a, (t.size, 1)))
    ang_nlerp = angle_between(q_nlerp, np.tile(a, (t.size, 1)))

    # Trace of the body x-axis under the interpolated attitudes.
    trace = quat_rotate(q_slerp, np.array([1.0, 0.0, 0.0]))

    fig = plt.figure(figsize=(11, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(t, np.degrees(ang_slerp), label="SLERP (geodesic)", lw=2)
    ax1.plot(t, np.degrees(ang_nlerp), "--", label="nlerp (for comparison)", lw=1.5)
    ax1.plot([0, 1], [0, np.degrees(total)], ":", color="k", lw=1,
             label=f"ideal linear ({np.degrees(total):.1f}° total)")
    ax1.set_xlabel("interpolation parameter t [-]")
    ax1.set_ylabel("angle from start attitude [deg]")
    ax1.set_title("SLERP: rotation angle is linear in t\n(constant angular rate)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    u, v = np.mgrid[0 : 2 * np.pi : 40j, 0 : np.pi : 20j]
    ax2.plot_wireframe(
        np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v),
        color="lightgray", lw=0.3, alpha=0.6,
    )
    ax2.plot(trace[:, 0], trace[:, 1], trace[:, 2], color="C0", lw=2.5,
             label="body x̂ under SLERP")
    ax2.scatter(*trace[0], color="green", s=50, label="t = 0")
    ax2.scatter(*trace[-1], color="red", s=50, label="t = 1")
    ax2.set_title("Body x-axis traces a great-circle arc")
    ax2.set_box_aspect([1, 1, 1])
    ax2.legend(loc="upper left", fontsize=8)

    out = Path(__file__).resolve().parents[1] / "screenshots" / "slerp_interpolation.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"saved {out}")
    dev = float(np.max(np.abs(ang_slerp - t * total)))
    print(f"max deviation of SLERP angle from linear ramp: {dev:.2e} rad")


if __name__ == "__main__":
    main()
