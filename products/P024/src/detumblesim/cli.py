"""Command-line interface: ``python -m detumblesim <command>``.

Commands
--------
``field``            dipole field magnitudes at reference points, pole location
``detumble``         one B-dot detumble run, summary numbers
``sweep``            detumble time against B-dot gain
``controllability``  orbit-averaged controllability report
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import numpy as np

from .analytic import (
    detumble_time_first_order,
    geometry_factors,
    orbit_field_moments,
    saturation_time_bound_s,
)
from .controllability import controllability_report
from .magfield import (
    B0_NT,
    dipole_tilt_deg,
    field_magnitude_nt,
    geomagnetic_north_pole_deg,
)
from .orbit import CircularOrbit
from .policies import FixedGainPolicy
from .simulate import DetumbleConfig, simulate_detumble
from .spacecraft import Magnetorquer, inertia_from_diagonal


def _orbit_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--altitude-km", type=float, default=500.0)
    p.add_argument("--inclination-deg", type=float, default=97.4)
    p.add_argument("--raan-deg", type=float, default=0.0)
    p.add_argument("--arg-lat-deg", type=float, default=0.0)
    p.add_argument("--gmst0-rad", type=float, default=0.0)


def _vehicle_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--inertia", type=float, nargs=3, default=[0.05, 0.06, 0.04],
        metavar=("IXX", "IYY", "IZZ"), help="principal moments [kg m^2]",
    )
    p.add_argument("--max-dipole", type=float, default=0.2, help="[A m^2] per axis")
    p.add_argument(
        "--rate0-deg-s", type=float, nargs=3, default=[8.0, -6.0, 5.0],
        metavar=("WX", "WY", "WZ"), help="initial body rate [deg/s]",
    )
    p.add_argument("--target-deg-s", type=float, default=1.0)


def _make_orbit(a: argparse.Namespace) -> CircularOrbit:
    return CircularOrbit(
        altitude_km=a.altitude_km,
        inclination_deg=a.inclination_deg,
        raan_deg=a.raan_deg,
        arg_lat0_deg=a.arg_lat_deg,
        gmst0_rad=a.gmst0_rad,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for ``python -m detumblesim``."""
    p = argparse.ArgumentParser(
        prog="detumblesim",
        description=(
            "Magnetorquer detumbling simulation (research-grade, not "
            "flight-qualified)."
        ),
    )
    sub = p.add_subparsers(dest="command", required=True)

    f = sub.add_parser("field", help="tilted-dipole field magnitudes")
    f.add_argument("--alt-km", type=float, default=500.0)

    d = sub.add_parser("detumble", help="one B-dot detumble run")
    _orbit_args(d)
    _vehicle_args(d)
    d.add_argument("--gain", type=float, default=1.0e5, help="[A m^2 s / T]")
    d.add_argument("--duration-s", type=float, default=23000.0)
    d.add_argument("--control-dt-s", type=float, default=2.0)

    s = sub.add_parser("sweep", help="detumble time against gain")
    _orbit_args(s)
    _vehicle_args(s)
    s.add_argument("--gain-min", type=float, default=1.0e4)
    s.add_argument("--gain-max", type=float, default=1.0e6)
    s.add_argument("--n-gains", type=int, default=8)
    s.add_argument("--duration-s", type=float, default=23000.0)

    c = sub.add_parser("controllability", help="orbit-averaged controllability")
    _orbit_args(c)
    c.add_argument("--orbits", type=float, default=10.0)

    return p


def _cmd_field(a: argparse.Namespace, out) -> int:
    lat, lon = geomagnetic_north_pole_deg()
    print("Tilted centred-dipole model, IGRF-14 degree-1 terms, epoch 2025.0", file=out)
    print(f"  B0 (equatorial surface field)   = {B0_NT:.1f} nT", file=out)
    print(f"  geomagnetic north pole          = {lat:.3f} N, {lon:.3f} E", file=out)
    print(f"  dipole tilt from rotation axis  = {dipole_tilt_deg():.3f} deg", file=out)
    print(f"  field magnitudes at {a.alt_km:.0f} km altitude:", file=out)
    for latitude in (-80.0, -45.0, 0.0, 45.0, 80.0):
        vals = [field_magnitude_nt(latitude, lo, a.alt_km) for lo in (0.0, 90.0, 180.0, 270.0)]
        cols = "  ".join(f"{v:8.1f}" for v in vals)
        print(f"    lat {latitude:+6.1f} deg, lon 0/90/180/270 E: {cols} nT", file=out)
    return 0


def _cmd_detumble(a: argparse.Namespace, out) -> int:
    orbit = _make_orbit(a)
    cfg = DetumbleConfig(
        inertia=inertia_from_diagonal(*a.inertia),
        orbit=orbit,
        magnetorquer=Magnetorquer.isotropic(a.max_dipole),
        omega0_rad_s=np.radians(a.rate0_deg_s),
        duration_s=a.duration_s,
        control_dt_s=a.control_dt_s,
        substeps=2,
        target_rate_rad_s=np.radians(a.target_deg_s),
        stop_when_detumbled=True,
    )
    res = simulate_detumble(cfg, FixedGainPolicy(a.gain))
    mom = orbit_field_moments(orbit, 2000, 10.0 * orbit.period_s)
    j = float(np.mean(a.inertia))
    print(f"orbit period                 = {orbit.period_s:.1f} s", file=out)
    print(f"RMS field over 10 orbits     = {mom.rms_b_t * 1e6:.2f} uT", file=out)
    rate0 = float(np.degrees(np.linalg.norm(cfg.omega0_rad_s)))
    print(f"initial rate                 = {rate0:.3f} deg/s", file=out)
    if res.detumbled:
        print(
            f"detumble time (simulated)    = {res.detumble_time_s:.1f} s "
            f"({res.detumble_time_s / orbit.period_s:.2f} orbits)",
            file=out,
        )
    else:
        print(
            f"detumble time (simulated)    = NOT REACHED within {a.duration_s:.0f} s",
            file=out,
        )
    t_iso = detumble_time_first_order(
        j, a.gain, mom, float(np.linalg.norm(cfg.omega0_rad_s)),
        np.radians(a.target_deg_s), "isotropic",
    )
    print(f"first-order model (no sat.)  = {t_iso:.1f} s", file=out)
    bound = saturation_time_bound_s(
        cfg.magnetorquer, mom, cfg.inertia, cfg.omega0_rad_s, np.radians(a.target_deg_s)
    )
    print(f"dipole-limit lower bound     = {bound:.1f} s", file=out)
    print(f"saturated control steps      = {100.0 * res.saturated_fraction:.1f} %", file=out)
    print(f"actuation cost int|m|^2 dt   = {res.actuation_cost_a2m4s:.3f} A^2 m^4 s", file=out)
    return 0


def _cmd_sweep(a: argparse.Namespace, out) -> int:
    orbit = _make_orbit(a)
    inertia = inertia_from_diagonal(*a.inertia)
    mtq = Magnetorquer.isotropic(a.max_dipole)
    w0 = np.radians(a.rate0_deg_s)
    target = np.radians(a.target_deg_s)
    gains = np.geomspace(a.gain_min, a.gain_max, int(a.n_gains))
    print(f"{'gain [A m^2 s/T]':>18}  {'t_detumble [s]':>14}  {'orbits':>7}  "
          f"{'sat [%]':>8}  {'int|m|^2 dt':>12}", file=out)
    for k in gains:
        cfg = DetumbleConfig(
            inertia=inertia, orbit=orbit, magnetorquer=mtq, omega0_rad_s=w0,
            duration_s=a.duration_s, control_dt_s=2.0, substeps=1,
            target_rate_rad_s=target, stop_when_detumbled=True,
        )
        r = simulate_detumble(cfg, FixedGainPolicy(float(k)))
        t = f"{r.detumble_time_s:14.1f}" if r.detumbled else f"{'not reached':>14}"
        orb = (
            f"{r.detumble_time_s / orbit.period_s:7.2f}" if r.detumbled else f"{'-':>7}"
        )
        print(f"{k:18.4e}  {t}  {orb}  {100 * r.saturated_fraction:8.1f}  "
              f"{r.actuation_cost_a2m4s:12.3f}", file=out)
    return 0


def _cmd_controllability(a: argparse.Namespace, out) -> int:
    orbit = _make_orbit(a)
    span = a.orbits * orbit.period_s
    rep = controllability_report(orbit, 4000, span)
    mom = orbit_field_moments(orbit, 4000, span)
    print(f"averaging span              = {span:.0f} s ({a.orbits:.1f} orbits)", file=out)
    print(f"RMS field                   = {rep.rms_field_t * 1e6:.3f} uT", file=out)
    print("weighted geometry factors   = "
          f"{np.array2string(rep.weighted_eigenvalues, precision=4)}  "
          f"(isotropic value {rep.isotropic_reference:.4f}, sum is exactly 2)", file=out)
    print(f"anisotropy (max/min)        = {rep.anisotropy:.3f}", file=out)
    print("weakest inertial direction  = "
          f"{np.array2string(rep.weakest_direction_eci, precision=4)}", file=out)
    print("direction-only eigenvalues  = "
          f"{np.array2string(rep.direction_eigenvalues, precision=4)}", file=out)
    print("check: geometry_factors()   = "
          f"{np.array2string(geometry_factors(mom), precision=4)}", file=out)
    return 0


def main(argv: Sequence[str] | None = None, out=None) -> int:
    """Entry point; returns a process exit code."""
    stream = sys.stdout if out is None else out
    args = build_parser().parse_args(argv)
    handlers = {
        "field": _cmd_field,
        "detumble": _cmd_detumble,
        "sweep": _cmd_sweep,
        "controllability": _cmd_controllability,
    }
    try:
        return handlers[args.command](args, stream)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
