"""Example: propagate a tumbling rigid body and plot ZYX Euler angles vs time.

A torque-free asymmetric rigid body obeys Euler's rotational equations
(Goldstein, *Classical Mechanics*, 3rd ed., Ch. 5; Markley & Crassidis 2014,
Sec. 3.3):

    I ω̇ = -ω × (I ω)          (body frame, N·m torque-free)

The body angular velocity ω(t) is integrated here with a fixed-step RK4 on a
fine grid, then fed to quatkit.propagate (quaternion RK4, q̇ = ½ q ⊗ [0, ω])
via linear interpolation. Attitude is converted to aerospace ZYX Euler angles
for plotting; gimbal-lock passages (pitch near ±90°) are expected for a
tumbling body and show up as yaw/roll jumps — that is a property of the Euler
chart, not of the quaternion propagation.

Run from products/P007/:  python examples/tumbling_body.py
Output: ../screenshots/tumbling_euler_angles.png
"""

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quatkit import GimbalLockWarning, propagate, quat_to_euler_zyx  # noqa: E402

# Principal moments of inertia [kg m^2] — asymmetric body (I1 < I2 < I3).
INERTIA = np.array([1.0, 2.0, 3.0])
OMEGA0 = np.array([0.05, 0.55, 0.03])  # rad/s — near the unstable intermediate axis
T_END = 120.0  # s
DT = 0.02  # s integration step (|ω| dt ≈ 0.011 rad)


def euler_eqs(omega: np.ndarray) -> np.ndarray:
    """ω̇ = -I⁻¹ (ω × I ω), torque-free rigid body [rad/s²]."""
    return -np.cross(omega, INERTIA * omega) / INERTIA


def integrate_omega(times: np.ndarray) -> np.ndarray:
    """RK4 on Euler's equations over the given grid."""
    out = np.empty((times.size, 3))
    w = OMEGA0.copy()
    out[0] = w
    for i in range(times.size - 1):
        dt = times[i + 1] - times[i]
        k1 = euler_eqs(w)
        k2 = euler_eqs(w + 0.5 * dt * k1)
        k3 = euler_eqs(w + 0.5 * dt * k2)
        k4 = euler_eqs(w + dt * k3)
        w = w + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        out[i + 1] = w
    return out


def main() -> None:
    times = np.arange(0.0, T_END + 1e-9, DT)
    omegas = integrate_omega(times)

    def omega_fn(t: float) -> np.ndarray:
        return np.array([np.interp(t, times, omegas[:, k]) for k in range(3)])

    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    qs = propagate(q0, omega_fn, times)

    with warnings.catch_warnings():
        # A tumbling body legitimately crosses pitch = ±90°; the warning is
        # informative, not an error, for this plot.
        warnings.simplefilter("ignore", GimbalLockWarning)
        euler = quat_to_euler_zyx(qs)

    # Conservation diagnostics (torque-free: H and T are constants of motion).
    h = INERTIA * omegas
    h_norm = np.linalg.norm(h, axis=1)
    t_kin = 0.5 * np.sum(omegas * h, axis=1)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    labels = ["yaw ψ", "pitch θ", "roll φ"]
    ax0, ax1, ax2 = axes
    for k, lab in enumerate(labels):
        ax0.plot(times, np.degrees(euler[:, k]), lw=0.9, label=lab)
    ax0.set_ylabel("Euler angle [deg]")
    ax0.set_title(
        "Torque-free tumbling body — ZYX Euler angles from quaternion RK4 propagation\n"
        "(I = diag(1, 2, 3) kg·m², ω₀ ≈ intermediate axis, dt = 0.02 s; "
        "jumps are Euler-chart wrap/gimbal artifacts)"
    )
    ax0.legend(loc="upper right", ncol=3)
    ax0.grid(alpha=0.3)

    for k, lab in enumerate(["ω₁", "ω₂", "ω₃"]):
        ax1.plot(times, omegas[:, k], lw=0.9, label=lab)
    ax1.set_ylabel("body rate [rad/s]")
    ax1.legend(loc="upper right", ncol=3)
    ax1.grid(alpha=0.3)

    ax2.plot(times, h_norm - h_norm[0], lw=0.9, label="|H| − |H₀|  [kg·m²/s]")
    ax2.plot(times, t_kin - t_kin[0], lw=0.9, label="T − T₀  [J]")
    ax2.plot(times, np.linalg.norm(qs, axis=1) - 1.0, lw=0.9, label="|q| − 1")
    ax2.set_ylabel("conservation drift")
    ax2.set_xlabel("time [s]")
    ax2.legend(loc="upper right")
    ax2.grid(alpha=0.3)

    out = Path(__file__).resolve().parents[1] / "screenshots" / "tumbling_euler_angles.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"saved {out}")
    print(f"max |q|-1 during propagation: {np.max(np.abs(np.linalg.norm(qs, axis=1)-1)):.2e}")
    print(f"|H| drift: {np.max(np.abs(h_norm - h_norm[0])):.2e} kg m^2/s, "
          f"T drift: {np.max(np.abs(t_kin - t_kin[0])):.2e} J")


if __name__ == "__main__":
    main()
