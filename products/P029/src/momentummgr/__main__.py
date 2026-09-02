"""Command-line interface: ``python -m momentummgr <command>``.

Three commands, all deterministic and all reporting units:

``budget``
    Per-source disturbance-torque and momentum budget over one orbit, plus what the
    result implies for wheel sizing and desaturation cadence.
``controllability``
    The magnetic controllability of one orbit: the time-averaged Gramian, its
    eigenvalues, and the worst instantaneous uncontrollable fraction.
``schedule``
    Run the tuned fixed-threshold scheduler on one sampled episode and report its duty
    and saturation metrics. The learned scheduler is not exposed here because it must be
    trained first; see ``validation/learned_vs_fixed_ci.py``.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from .accumulation import SOURCES, momentum_budget
from .desaturation import averaged_controllability, thruster_dump, uncontrollable_fraction
from .environment import (
    CircularOrbit,
    circular_state,
    dipole_field_eci,
    eclipse_fraction,
    reference_smallsat,
    sun_direction_for_beta,
)
from .episodes import rollout, sample_episode
from .policies import FixedThresholdScheduler
from .wheels import pyramid_four


def _orbit(args: argparse.Namespace) -> CircularOrbit:
    return CircularOrbit(
        altitude_m=args.altitude_km * 1000.0,
        inclination_rad=np.radians(args.inclination_deg),
        raan_rad=np.radians(args.raan_deg),
        pitch_rad=np.radians(args.pitch_deg),
        roll_rad=np.radians(args.roll_deg),
    )


def _cmd_budget(args: argparse.Namespace) -> int:
    orbit = _orbit(args)
    sc = reference_smallsat()
    sun = sun_direction_for_beta(orbit.inclination_rad, orbit.raan_rad, np.radians(args.beta_deg))
    budget = momentum_budget(sc, orbit, sun)
    wheels = pyramid_four(max_momentum_nms=args.wheel_nms)
    env = wheels.guaranteed_body_envelope_nms
    secular = budget["total"]["secular_per_orbit_nms"]
    orbits_to_fill = env / secular if secular > 0.0 else float("inf")
    dump = thruster_dump(secular, args.moment_arm_m, args.isp_s)
    if args.json:
        print(
            json.dumps(
                {
                    "period_s": orbit.period_s,
                    "eclipse_fraction": eclipse_fraction(
                        orbit.radius_m, orbit.inclination_rad, orbit.raan_rad, sun
                    ),
                    "budget_nms": budget,
                    "wheel_envelope_nms": env,
                    "orbits_to_fill": orbits_to_fill,
                    "thruster_propellant_kg_per_orbit": dump.propellant_kg,
                },
                indent=2,
                default=float,
            )
        )
        return 0
    print(f"Reference smallsat, {args.altitude_km:.0f} km, i = {args.inclination_deg:.1f} deg, "
          f"beta = {args.beta_deg:.1f} deg")
    print(f"period {orbit.period_s:.1f} s   eclipse fraction "
          f"{eclipse_fraction(orbit.radius_m, orbit.inclination_rad, orbit.raan_rad, sun):.4f}")
    print(f"\n{'source':<18}{'|dh|/orbit [N m s]':>20}{'cyclic peak [N m s]':>22}"
          f"{'peak T [N m]':>16}")
    print("-" * 76)
    for name in (*SOURCES, "total"):
        b = budget[name]
        print(f"{name:<18}{b['secular_per_orbit_nms']:>20.6e}{b['cyclic_peak_nms']:>22.6e}"
              f"{b['peak_torque_nm']:>16.6e}")
    print(f"\nwheel array envelope        {env:.6e} N m s "
          f"({wheels.n_wheels} wheels at {args.wheel_nms:.4g} N m s)")
    print(f"orbits to fill on secular   {orbits_to_fill:.2f}")
    print(f"thruster propellant/orbit   {dump.propellant_kg:.4e} kg "
          f"(couple, arm {args.moment_arm_m:.2f} m, Isp {args.isp_s:.0f} s)")
    return 0


def _cmd_controllability(args: argparse.Namespace) -> int:
    orbit = _orbit(args)
    u = np.linspace(0.0, 2.0 * np.pi, args.samples)
    r, _ = circular_state(orbit.radius_m, orbit.inclination_rad, orbit.raan_rad, u)
    b = dipole_field_eci(r)
    t = u / (2.0 * np.pi) * orbit.period_s
    gram, eig, vecs = averaged_controllability(b, t)
    h_dir = np.array([1.0, 0.0, 0.0])
    frac = uncontrollable_fraction(np.repeat(h_dir[None, :], b.shape[0], axis=0), b)
    if args.json:
        print(json.dumps({"gramian": gram.tolist(), "eigenvalues": eig.tolist(),
                          "eigenvectors": vecs.tolist(),
                          "max_uncontrollable_fraction_x": float(frac.max())}, indent=2))
        return 0
    print(f"Magnetic controllability over one orbit at {args.altitude_km:.0f} km, "
          f"i = {args.inclination_deg:.1f} deg")
    print("\ntime-averaged Gramian <I - B_hat B_hat^T>:")
    for row in gram:
        print("   " + "  ".join(f"{v: .6f}" for v in row))
    print(f"\ntrace {gram.trace():.12f} (exactly 2 for any field history)")
    print("eigenvalues (fraction of the orbit each direction is dumpable):")
    for value, vec in zip(eig, vecs.T, strict=True):
        print(f"   {value: .6f}   along [{vec[0]: .4f} {vec[1]: .4f} {vec[2]: .4f}]")
    print(f"\nworst instantaneous uncontrollable fraction for a momentum along ECI x: "
          f"{frac.max():.6f}")
    return 0


def _cmd_schedule(args: argparse.Namespace) -> int:
    episode = sample_episode(args.seed)
    sched = FixedThresholdScheduler(args.on_fraction, args.off_fraction)
    roll = rollout(episode, sched.decider())
    m = roll.metrics
    if args.json:
        print(json.dumps({"seed": args.seed, "n_windows": episode.n_windows,
                          "envelope_nms": episode.envelope_nms,
                          "dipole_cost_am2s": m.dipole_cost_am2s,
                          "duty_fraction": m.duty_fraction,
                          "near_saturation_fraction": m.near_saturation_fraction,
                          "max_h_fraction": m.max_h_fraction, "violated": m.violated,
                          "cost": m.cost}, indent=2))
        return 0
    print(f"Episode seed {args.seed}: {episode.n_windows} windows of {episode.window_s:.0f} s, "
          f"envelope {episode.envelope_nms:.5f} N m s")
    print(f"fixed-threshold scheduler on={args.on_fraction} off={args.off_fraction}")
    print(f"  dipole cost              {m.dipole_cost_am2s:.4e} A m^2 s")
    print(f"  magnetorquer duty        {m.duty_fraction:.4f}")
    print(f"  time near saturation     {m.near_saturation_fraction:.4f}")
    print(f"  peak |h| / envelope      {m.max_h_fraction:.4f}")
    print(f"  envelope exceeded        {m.violated}")
    print(f"  windows actuated         {int(roll.actions.sum())} of {episode.n_windows}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for ``python -m momentummgr``."""
    p = argparse.ArgumentParser(prog="momentummgr", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)

    def add_orbit(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--altitude-km", type=float, default=500.0)
        sp.add_argument("--inclination-deg", type=float, default=51.6)
        sp.add_argument("--raan-deg", type=float, default=0.0)
        sp.add_argument("--pitch-deg", type=float, default=5.0)
        sp.add_argument("--roll-deg", type=float, default=5.0)
        sp.add_argument("--json", action="store_true", help="machine-readable output")

    b = sub.add_parser("budget", help="disturbance torque and momentum budget over one orbit")
    add_orbit(b)
    b.add_argument("--beta-deg", type=float, default=20.0)
    b.add_argument("--wheel-nms", type=float, default=0.05, help="per-wheel limit [N m s]")
    b.add_argument("--moment-arm-m", type=float, default=0.5)
    b.add_argument("--isp-s", type=float, default=220.0)
    b.set_defaults(func=_cmd_budget)

    c = sub.add_parser("controllability", help="magnetic controllability over one orbit")
    add_orbit(c)
    c.add_argument("--samples", type=int, default=721)
    c.set_defaults(func=_cmd_controllability)

    s = sub.add_parser("schedule", help="run the fixed-threshold scheduler on one episode")
    s.add_argument("--seed", type=int, default=5000)
    s.add_argument("--on-fraction", type=float, default=0.5)
    s.add_argument("--off-fraction", type=float, default=0.4)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=_cmd_schedule)
    return p


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns a process exit code."""
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
