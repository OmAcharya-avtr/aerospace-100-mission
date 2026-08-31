"""Command-line interface: ``python -m disturbtorque``.

Two subcommands:

``budget``
    Per-source torque and momentum summary over one orbit for a spacecraft given on the
    command line (defaults are the reference smallsat of :mod:`disturbtorque.presets`).
``sweep``
    The same summary at a series of altitudes, which is how the aerodynamic/solar
    crossover altitude is located.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

import numpy as np

from .frames import sun_direction_for_beta
from .presets import reference_orbit, reference_smallsat
from .profile import SOURCES, budget, compute_profile
from .spacecraft import Orbit, Spacecraft


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--inclination-deg", type=float, default=51.6, help="orbit inclination [deg]")
    p.add_argument("--raan-deg", type=float, default=0.0, help="RAAN [deg]")
    p.add_argument("--beta-deg", type=float, default=0.0, help="Sun beta angle [deg]")
    p.add_argument("--pitch-deg", type=float, default=5.0, help="body pitch offset from LVLH [deg]")
    p.add_argument("--roll-deg", type=float, default=5.0, help="body roll offset from LVLH [deg]")
    p.add_argument("--yaw-deg", type=float, default=0.0, help="body yaw offset from LVLH [deg]")
    p.add_argument("--inertia", type=float, nargs=3, default=None, help="Ixx Iyy Izz [kg m^2]")
    p.add_argument("--drag-area", type=float, default=None, help="projected drag area [m^2]")
    p.add_argument("--cd", type=float, default=None, help="drag coefficient [-]")
    p.add_argument("--srp-area", type=float, default=None, help="projected sunlit area [m^2]")
    p.add_argument("--reflectance", type=float, default=None, help="reflectance factor q [-]")
    p.add_argument("--cp-offset", type=float, nargs=3, default=None, help="cp-cm offset [m]")
    p.add_argument("--dipole", type=float, nargs=3, default=None, help="residual dipole [A m^2]")
    p.add_argument("--samples", type=int, default=721, help="samples per orbit")
    p.add_argument("--frame", choices=("body", "eci"), default="body", help="reporting frame")
    p.add_argument("--json", action="store_true", help="emit JSON instead of a table")


def _spacecraft_from_args(args: argparse.Namespace) -> Spacecraft:
    ref = reference_smallsat()
    inertia = np.diag(args.inertia) if args.inertia else ref.inertia
    offset = np.asarray(args.cp_offset, float) if args.cp_offset else ref.cp_aero_offset_m
    return Spacecraft(
        inertia=inertia,
        drag_area_m2=ref.drag_area_m2 if args.drag_area is None else args.drag_area,
        drag_coefficient=ref.drag_coefficient if args.cd is None else args.cd,
        cp_aero_offset_m=offset,
        srp_area_m2=ref.srp_area_m2 if args.srp_area is None else args.srp_area,
        srp_reflectance=ref.srp_reflectance if args.reflectance is None else args.reflectance,
        cp_srp_offset_m=offset,
        residual_dipole_am2=(
            np.asarray(args.dipole, float) if args.dipole else ref.residual_dipole_am2
        ),
        mass_kg=ref.mass_kg,
    )


def _orbit_from_args(args: argparse.Namespace, altitude_km: float) -> Orbit:
    base = reference_orbit(altitude_km)
    return Orbit(
        altitude_m=base.altitude_m,
        inclination_rad=np.radians(args.inclination_deg),
        raan_rad=np.radians(args.raan_deg),
        yaw_rad=np.radians(args.yaw_deg),
        pitch_rad=np.radians(args.pitch_deg),
        roll_rad=np.radians(args.roll_deg),
    )


def _one_budget(args: argparse.Namespace, altitude_km: float) -> dict:
    sc = _spacecraft_from_args(args)
    orb = _orbit_from_args(args, altitude_km)
    sun = sun_direction_for_beta(orb.inclination_rad, orb.raan_rad, np.radians(args.beta_deg))
    prof = compute_profile(sc, orb, sun, n_samples=args.samples)
    b = budget(prof, frame=args.frame)
    return {
        "altitude_km": altitude_km,
        "period_s": prof.period_s,
        "eclipse_fraction": prof.eclipse_fraction,
        "frame": args.frame,
        "sources": {
            name: {
                "peak_nm": b[name]["peak_nm"],
                "rms_nm": b[name]["rms_nm"],
                "secular_magnitude_nm": b[name]["secular_magnitude_nm"],
                "cyclic_peak_nm": b[name]["cyclic_peak_nm"],
                "secular_momentum_per_orbit_nms": b[name]["secular_momentum_per_orbit_nms"],
                "cyclic_momentum_peak_nms": b[name]["cyclic_momentum_peak_nms"],
            }
            for name in (*SOURCES, "total")
        },
    }


def _print_budget(res: dict, out) -> None:
    print(
        f"altitude {res['altitude_km']:.1f} km   period {res['period_s']:.1f} s   "
        f"eclipse fraction {res['eclipse_fraction']:.4f}   frame {res['frame']}",
        file=out,
    )
    header = (
        f"{'source':<18}{'peak [N m]':>13}{'rms [N m]':>13}{'secular [N m]':>15}"
        f"{'cyclic [N m]':>14}{'dh_sec/orbit':>15}{'h_cyc peak':>13}"
    )
    print(header, file=out)
    print("-" * len(header), file=out)
    for name, v in res["sources"].items():
        print(
            f"{name:<18}{v['peak_nm']:>13.4e}{v['rms_nm']:>13.4e}"
            f"{v['secular_magnitude_nm']:>15.4e}{v['cyclic_peak_nm']:>14.4e}"
            f"{v['secular_momentum_per_orbit_nms']:>15.4e}"
            f"{v['cyclic_momentum_peak_nms']:>13.4e}",
            file=out,
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        prog="python -m disturbtorque",
        description=(
            "Environmental disturbance torque budget for a circular Earth orbit. "
            "Research-grade; not flight-qualified."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_budget = sub.add_parser("budget", help="torque and momentum budget for one orbit")
    p_budget.add_argument("--altitude-km", type=float, default=500.0)
    _add_common(p_budget)

    p_sweep = sub.add_parser("sweep", help="budget over a range of altitudes")
    p_sweep.add_argument("--altitude-km", type=float, nargs="+", default=[400.0, 500.0, 600.0])
    _add_common(p_sweep)

    args = parser.parse_args(argv)
    out = sys.stdout
    if args.command == "budget":
        res = _one_budget(args, args.altitude_km)
        if args.json:
            print(json.dumps(res, indent=2), file=out)
        else:
            _print_budget(res, out)
        return 0

    results = [_one_budget(args, h) for h in args.altitude_km]
    if args.json:
        print(json.dumps(results, indent=2), file=out)
        return 0
    header = (
        f"{'alt [km]':>9}" + "".join(f"{n[:12]:>14}" for n in (*SOURCES, "total"))
    )
    print(f"peak torque magnitude [N m], frame {args.frame}", file=out)
    print(header, file=out)
    print("-" * len(header), file=out)
    for res in results:
        row = f"{res['altitude_km']:>9.1f}" + "".join(
            f"{res['sources'][n]['peak_nm']:>14.4e}" for n in (*SOURCES, "total")
        )
        print(row, file=out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
