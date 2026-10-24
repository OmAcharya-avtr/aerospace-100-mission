"""Check the universal-variable two-body propagator against an independent integration.

The propagator in :mod:`pointahead.kepler` is the only dynamics in this
package, and the light-time reference queries it at the retarded and advanced
epochs. It is checked three ways:

1. Against ``scipy.integrate.solve_ivp`` with DOP853 at a tight tolerance,
   which is an independent numerical integration of the same equation of
   motion and shares no code with the universal-variable formulation.
2. Against the two exact invariants of the two-body problem, the specific
   orbital energy and the specific angular momentum vector.
3. Against itself, by propagating forward and back.

Conserved quantities are compared with a relative tolerance on both sides.
Nothing here asserts a bit-exact equality over a floating-point pipeline.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from _bootstrap import Tee, Timer, bootstrap, header

bootstrap()

from pointahead.kepler import MU_EARTH, propagate_two_body  # noqa: E402


def two_body_rhs(_t: float, state: np.ndarray, mu: float) -> np.ndarray:
    """``[v, -mu r / |r|^3]`` for ``scipy.integrate.solve_ivp``."""
    r = state[:3]
    distance = float(np.linalg.norm(r))
    return np.concatenate([state[3:], -mu * r / distance**3])


def main() -> int:
    tee = Tee("validate_kepler_output.txt")
    header(tee, "two-body propagation: universal variables vs an independent integration")

    rng = np.random.default_rng(19760)
    mu = MU_EARTH
    cases = []
    for _ in range(40):
        altitude = float(rng.uniform(400.0e3, 20000.0e3))
        radius = 6378137.0 + altitude
        circular = float(np.sqrt(mu / radius))
        speed_factor = float(rng.uniform(0.85, 1.25))
        inclination = float(rng.uniform(0.0, 0.5 * np.pi))
        r0 = np.array([radius, 0.0, 0.0])
        v0 = speed_factor * circular * np.array(
            [0.0, float(np.cos(inclination)), float(np.sin(inclination))]
        )
        cases.append((r0, v0, float(rng.uniform(60.0, 3600.0))))

    tee.rule("1. universal variables vs scipy.integrate.solve_ivp (DOP853, rtol 1e-13)")
    worst_position = 0.0
    worst_velocity = 0.0
    kepler_timer = Timer()
    ode_timer = Timer()
    for r0, v0, dt in cases:
        with kepler_timer:
            step = propagate_two_body(r0, v0, dt, mu)
        with ode_timer:
            solution = solve_ivp(
                two_body_rhs,
                (0.0, dt),
                np.concatenate([r0, v0]),
                method="DOP853",
                rtol=1.0e-13,
                atol=1.0e-9,
                args=(mu,),
            )
        reference = solution.y[:, -1]
        position_error = float(np.linalg.norm(step.position_m - reference[:3]))
        velocity_error = float(np.linalg.norm(step.velocity_mps - reference[3:]))
        scale = float(np.linalg.norm(reference[:3]))
        worst_position = max(worst_position, position_error / scale)
        worst_velocity = max(worst_velocity, velocity_error / float(np.linalg.norm(reference[3:])))
    tee(f"  cases                              : {len(cases)}")
    tee("  altitudes                          : 400 to 20000 km")
    tee("  speed factors                      : 0.85 to 1.25 of circular")
    tee("  intervals                          : 60 to 3600 s")
    tee(f"  worst relative position difference : {worst_position:.6e}")
    tee(f"  worst relative velocity difference : {worst_velocity:.6e}")
    tee.check(
        "universal variables agree with DOP853 to 1e-10 relative",
        worst_position < 1.0e-10 and worst_velocity < 1.0e-10,
        f"position {worst_position:.3e}, velocity {worst_velocity:.3e}",
    )

    tee.rule("2. exact two-body invariants")
    worst_energy = 0.0
    worst_momentum = 0.0
    worst_fg = 0.0
    for r0, v0, dt in cases:
        step = propagate_two_body(r0, v0, dt, mu)
        before = 0.5 * float(np.dot(v0, v0)) - mu / float(np.linalg.norm(r0))
        after = 0.5 * float(np.dot(step.velocity_mps, step.velocity_mps)) - mu / float(
            np.linalg.norm(step.position_m)
        )
        worst_energy = max(worst_energy, abs(after - before) / abs(before))
        h0 = np.cross(r0, v0)
        h1 = np.cross(step.position_m, step.velocity_mps)
        worst_momentum = max(
            worst_momentum, float(np.linalg.norm(h1 - h0)) / float(np.linalg.norm(h0))
        )
        worst_fg = max(worst_fg, step.fg_residual)
    tee(f"  worst relative energy drift        : {worst_energy:.6e}")
    tee(f"  worst relative momentum drift      : {worst_momentum:.6e}")
    tee(f"  worst |f gdot - fdot g - 1|        : {worst_fg:.6e}")
    tee.check("specific energy conserved to 1e-12 relative", worst_energy < 1.0e-12)
    tee.check("angular momentum conserved to 1e-12 relative", worst_momentum < 1.0e-12)
    tee.check("f and g series identity holds to 1e-12", worst_fg < 1.0e-12)

    tee.rule("3. forward-and-back round trip")
    worst_round_trip = 0.0
    for r0, v0, dt in cases:
        forward = propagate_two_body(r0, v0, dt, mu)
        back = propagate_two_body(forward.position_m, forward.velocity_mps, -dt, mu)
        worst_round_trip = max(
            worst_round_trip,
            float(np.linalg.norm(back.position_m - r0)) / float(np.linalg.norm(r0)),
        )
    tee(f"  worst relative round-trip error    : {worst_round_trip:.6e}")
    tee.check("round trip returns to 1e-12 relative", worst_round_trip < 1.0e-12)

    tee.rule("4. circular-orbit period, computed from the elements")
    for altitude in (400.0e3, 600.0e3, 1200.0e3, 35786.0e3):
        radius = 6378137.0 + altitude
        speed = float(np.sqrt(mu / radius))
        period = float(2.0 * np.pi * np.sqrt(radius**3 / mu))
        step = propagate_two_body([radius, 0.0, 0.0], [0.0, speed, 0.0], period, mu)
        error = float(np.linalg.norm(step.position_m - np.array([radius, 0.0, 0.0])))
        tee(
            f"  altitude {altitude / 1.0e3:>8.1f} km  period {period:>12.4f} s  "
            f"return error {error:.6e} m"
        )
        tee.check(
            f"one period at {altitude / 1.0e3:.0f} km returns within 1 micrometre",
            error < 1.0e-6,
            f"{error:.3e} m",
        )

    tee.rule("5. timings -- UNSTABLE, for order of magnitude only")
    tee("  These numbers move between runs on shared hardware and are never")
    tee("  used in a prose conclusion anywhere in this repository.")
    tee("")
    tee(f"  {'operation':<42s} {'calls':>7s} {'mean / ms':>12s}")
    tee(f"  {'-' * 42} {'-' * 7} {'-' * 12}")
    tee(f"  {'propagate_two_body':<42s} {kepler_timer.calls:>7d} {kepler_timer.mean_ms:>12.4f}")
    tee(f"  {'solve_ivp DOP853 rtol 1e-13':<42s} {ode_timer.calls:>7d} {ode_timer.mean_ms:>12.4f}")

    tee.summarise_failures()
    tee("")
    tee(f"  written: {tee.save()}")
    return 1 if tee.blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())
