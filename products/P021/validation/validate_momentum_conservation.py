"""Angular momentum conservation, integrator order, and the second invariant.

PART A  torque-free rigid body: |L| is conserved; the direction of L is not
        conserved to the same precision, and the difference is explained
PART B  RK4 order by step halving
PART C  rotational kinetic energy, the second torque-free invariant
PART D  wheels exchange momentum with the body without changing the total
PART E  both invariants over 120 random torque-free initial states

Run: ``python validation/validate_momentum_conservation.py``  (about 100 s)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from slewforge.attitude import quat_angle, quat_normalize  # noqa: E402
from slewforge.dynamics import RigidBody, inertial_momentum, propagate  # noqa: E402
from slewforge.wheels import pyramid_wheels  # noqa: E402


def rule(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def main() -> int:
    failures = 0
    inertia = np.diag([120.0, 100.0, 80.0])
    body = RigidBody(inertia)
    rng = np.random.default_rng(20260902)

    cases = {
        "spin about max axis": np.array([0.0, 0.0, 0.05]),
        "spin about int axis": np.array([0.0, 0.05, 0.0]),
        "spin about min axis": np.array([0.05, 0.0, 0.0]),
        "tumble (0.03,0.04,0.02)": np.array([0.03, 0.04, 0.02]),
        "near-separatrix": np.array([1e-4, 0.05, 1e-4]),
    }

    # ---------------------------------------------------------------- PART A
    rule("PART A -- torque-free conservation of the inertial angular momentum")
    print("Euler's equation with tau_ext = 0 conserves L = R(q) J omega exactly.")
    print("300 s at dt = 0.02 s, RK4, quaternion renormalised every step.")
    print()
    print(f"{'case':<26}{'|L| drift [N*m*s]':>20}{'relative':>14}{'dir drift [rad]':>18}")
    worst_rel = worst_dir = 0.0
    for name, w0 in cases.items():
        q0 = quat_normalize(rng.normal(size=4))
        sim = propagate(body, q0, w0, 300.0, 0.02)
        mag = float(np.linalg.norm(sim.momentum[0]))
        drift = sim.momentum_drift()
        rel = drift / mag
        ddir = sim.momentum_direction_drift()
        worst_rel = max(worst_rel, rel)
        worst_dir = max(worst_dir, ddir)
        print(f"{name:<26}{drift:>20.6e}{rel:>14.3e}{ddir:>18.6e}")
    print()
    print(f"worst relative |L| drift  : {worst_rel:.6e}   tolerance 1e-12")
    failures += worst_rel > 1e-12
    print(f"worst direction drift     : {worst_dir:.6e} rad")
    print()
    print("The two columns are not the same quantity and do not have the same")
    print("error. |L| = |J omega| is an invariant of Euler's equation on its own,")
    print("independent of the attitude, and RK4 preserves it to round-off. The")
    print("*direction* of L in inertial space is L = R(q) J omega, so it inherits")
    print("the attitude truncation error, which is O(dt^4) and shown in PART B.")
    print("Reporting only the first number would be flattering and wrong.")
    print()
    print("Direction drift against step size, tumble case, 300 s:")
    print(f"{'dt [s]':>10}{'dir drift [rad]':>20}{'ratio':>10}")
    prev = None
    for dt in (0.32, 0.16, 0.08, 0.04):
        sim = propagate(body, quat_normalize(np.array([1.0, 0.2, -0.1, 0.05])),
                        cases["tumble (0.03,0.04,0.02)"], 300.0, dt)
        d = sim.momentum_direction_drift()
        ratio = "" if prev is None or d <= 0.0 else f"{prev / d:.2f}"
        print(f"{dt:>10.4f}{d:>20.6e}{ratio:>10}")
        prev = d

    # ---------------------------------------------------------------- PART B
    rule("PART B -- RK4 order by step halving")
    print("Reference: the same tumble integrated at dt = 2.5e-3 s. Error is the")
    print("eigenaxis angle between the final attitudes, via quat_angle(), which")
    print("uses atan2 rather than arccos and so resolves angles below 1e-8 rad.")
    w0 = np.array([0.03, 0.04, 0.02])
    q0 = quat_normalize(np.array([1.0, 0.2, -0.1, 0.05]))
    ref = propagate(body, q0, w0, 100.0, 2.5e-3)
    q_ref = ref.quat[-1]
    print()
    print(f"{'dt [s]':>10}{'attitude error [rad]':>24}{'ratio':>10}")
    prev = None
    for dt in (0.32, 0.16, 0.08, 0.04):
        sim = propagate(body, q0, w0, 100.0, dt)
        err = quat_angle(sim.quat[-1], q_ref)
        ratio = "" if prev is None or err <= 0.0 else f"{prev / err:.2f}"
        print(f"{dt:>10.4f}{err:>24.6e}{ratio:>10}")
        prev = err
    print("expected ratio 16 for a fourth-order method")

    # ---------------------------------------------------------------- PART C
    rule("PART C -- rotational kinetic energy, the second torque-free invariant")
    print("2 T = omega . J omega is conserved independently of L.")
    print()
    print(f"{'case':<26}{'relative energy drift':>24}")
    worst_e = 0.0
    for name, w0 in cases.items():
        q0 = quat_normalize(rng.normal(size=4))
        sim = propagate(body, q0, w0, 300.0, 0.02)
        energy = np.einsum("ij,ij->i", sim.omega, sim.omega @ inertia.T)
        rel = float(np.max(np.abs(energy - energy[0])) / energy[0])
        worst_e = max(worst_e, rel)
        print(f"{name:<26}{rel:>24.6e}")
    print(f"\nworst relative energy drift: {worst_e:.6e}   tolerance 1e-11")
    failures += worst_e > 1e-11

    # ---------------------------------------------------------------- PART D
    rule("PART D -- wheels move momentum between body and wheels, not out of it")
    wheeled = RigidBody(inertia, pyramid_wheels(0.15, 12.0))
    q0 = quat_normalize(np.array([1.0, 0.0, 0.0, 0.0]))
    torque = np.array([0.05, -0.03, 0.02])

    def tau(t: float) -> np.ndarray:
        return torque if t < 40.0 else -torque

    sim = propagate(wheeled, q0, np.zeros(3), 80.0, 0.01, torque_fn=tau)
    body_only = np.array(
        [inertial_momentum(wheeled, sim.quat[i], sim.omega[i]) for i in range(len(sim.time))]
    )
    total = sim.momentum
    print("80 s manoeuvre driven by the wheels alone, no external torque,")
    print("starting from rest with unspun wheels, so the exact total L is zero.")
    print()
    print(f"  peak |body momentum J omega|  : {float(np.max(np.linalg.norm(body_only, axis=1))):.9f} N*m*s")
    print(f"  peak per-wheel momentum       : {float(np.max(np.abs(sim.wheel_momentum))):.9f} N*m*s")
    print(f"  peak |A h| (wheel momentum in body) : "
          f"{float(np.max(np.linalg.norm(sim.wheel_momentum @ wheeled.wheels.distribution.T, axis=1))):.9f} N*m*s")
    worst_total = float(np.max(np.linalg.norm(total, axis=1)))
    print(f"  worst |total L| over the run  : {worst_total:.6e} N*m*s   tolerance 1e-12")
    failures += worst_total > 1e-12
    print(f"  final body rate |omega|       : {float(np.linalg.norm(sim.omega[-1])):.6e} rad/s")
    print("  (equal and opposite torque pulses return the body to rest)")
    print(f"  wheel torque saturation flagged   : {bool(np.any(sim.saturated_torque))}")
    print(f"  wheel momentum saturation flagged : {bool(np.any(sim.saturated_momentum))}")

    # ---------------------------------------------------------------- PART E
    rule("PART E -- both invariants over 120 random torque-free initial states")
    worst_l = worst_t = 0.0
    for _ in range(120):
        q0 = quat_normalize(rng.normal(size=4))
        w0 = rng.normal(size=3) * 0.02
        sim = propagate(body, q0, w0, 60.0, 0.02)
        mag = np.linalg.norm(sim.momentum, axis=1)
        energy = np.einsum("ij,ij->i", sim.omega, sim.omega @ inertia.T)
        worst_l = max(worst_l, float(np.max(np.abs(mag - mag[0])) / mag[0]))
        worst_t = max(worst_t, float(np.max(np.abs(energy - energy[0])) / energy[0]))
    print(f"worst relative |L| drift over 120 runs      : {worst_l:.6e}   tolerance 1e-12")
    print(f"worst relative energy drift over 120 runs   : {worst_t:.6e}   tolerance 1e-11")
    failures += worst_l > 1e-12 or worst_t > 1e-11

    rule("SUMMARY")
    print(f"failed checks: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
