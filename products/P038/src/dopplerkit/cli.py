"""Command-line interface: ``python -m dopplerkit``.

Four subcommands:

``convention``
    Print the sign conventions verbatim and exit.  Deliberately the first
    subcommand in the help text, because every other output depends on it.
``analytic``
    Closed-form circular overhead pass profile (:mod:`dopplerkit.analytic`).
``tle``
    Pass profile from a TLE through ``sgp4`` and a real ground station with
    its rotation velocity (:mod:`dopplerkit.frames`).
``relativistic``
    Report the O(beta^2) and gravitational terms, in Hz, for a stated pass and
    carrier, alongside the classical peak so the reader can see the ratio.

Exit codes: 0 on success, 2 on invalid input (argparse convention, and
explicitly for a ``ValueError`` from the library, with the offending field
named).  No ``print`` appears in the library -- all output is produced here.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime, timedelta

import numpy as np

from .analytic import CircularOverheadPass
from .constants import C_M_S, WGS84_A_M
from .doppler import (
    DOPPLER_CONVENTION,
    LinkDirection,
    gravitational_shift_fraction,
    one_way_doppler_hz,
    relativistic_correction_hz,
    two_way_doppler_hz,
)
from .frames import julian_date, satellite_state_teme, station_state_teme
from .geometry import RANGE_RATE_CONVENTION, range_rate_mps, slant_range_m
from .passprofile import compute_profile, doppler_zero_crossing_s, time_of_closest_approach_s

CONVENTION_TEXT = f"""dopplerkit sign conventions
===========================
range-rate   {RANGE_RATE_CONVENTION}
doppler      {DOPPLER_CONVENTION}

In words:
  rho_dot > 0  means RECEDING (the range is increasing)
  Delta_f > 0  means UP-SHIFT, which means APPROACHING
  Delta_f      = -f_carrier * rho_dot / c            (one-way)
  Delta_f      = -2 G f_uplink * rho_dot / c         (two-way, coherent)
  precomp      = -Delta_f                            (transmit offset)

Range-rate is INVARIANT under reversing the link direction. Uplink and
downlink share one range-rate; only the carrier differs.

Order kept: O(beta^1), beta = rho_dot/c. The O(beta^2) relativistic and
classical-cascade terms are reported by the 'relativistic' subcommand, never
applied.
"""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dopplerkit",
        description=(
            "Doppler and range-rate geometry for a satellite link. "
            "One-way and two-way kept separate; sign convention stated at every "
            "interface. Run 'dopplerkit convention' first."
        ),
        epilog="Research-grade and educational. Not flight-qualified, not certified.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("convention", help="print the sign conventions and exit")

    ana = sub.add_parser("analytic", help="closed-form circular overhead pass profile")
    ana.add_argument("--altitude-km", type=float, default=500.0,
                     help="circular orbit altitude above the WGS-84 equatorial radius [km]")
    ana.add_argument("--carrier-hz", type=float, default=2.2e9,
                     help="carrier frequency [Hz]; for two-way this is the uplink carrier")
    ana.add_argument("--two-way", action="store_true",
                     help="report two-way coherent Doppler instead of one-way")
    ana.add_argument("--turnaround-ratio", type=float, default=1.0,
                     help="coherent transponder turnaround ratio G [-], two-way only")
    ana.add_argument("--samples", type=int, default=21, help="number of profile samples")
    ana.add_argument("--json", action="store_true", help="emit JSON instead of a table")

    tle = sub.add_parser("tle", help="pass profile from a TLE via sgp4 and a ground station")
    tle.add_argument("--line1", required=True, help="TLE line 1 (69 characters)")
    tle.add_argument("--line2", required=True, help="TLE line 2 (69 characters)")
    tle.add_argument("--lat-deg", type=float, required=True, help="station geodetic latitude [deg]")
    tle.add_argument("--lon-deg", type=float, required=True, help="station east longitude [deg]")
    tle.add_argument("--alt-m", type=float, default=0.0, help="station altitude above ellipsoid [m]")
    tle.add_argument("--epoch", required=True,
                     help="ISO-8601 UTC start epoch, e.g. 2019-12-09T16:30:00")
    tle.add_argument("--duration-s", type=float, default=600.0, help="profile duration [s]")
    tle.add_argument("--step-s", type=float, default=30.0, help="profile sample step [s]")
    tle.add_argument("--carrier-hz", type=float, default=2.2e9, help="carrier frequency [Hz]")
    tle.add_argument("--json", action="store_true", help="emit JSON instead of a table")

    rel = sub.add_parser("relativistic", help="report the O(beta^2) and gravitational terms")
    rel.add_argument("--altitude-km", type=float, default=500.0, help="circular orbit altitude [km]")
    rel.add_argument("--carrier-hz", type=float, default=2.2e9, help="carrier frequency [Hz]")
    return parser


def _format_profile_table(times: np.ndarray, rng: np.ndarray, rate: np.ndarray,
                          doppler: np.ndarray, d_rate: np.ndarray,
                          precomp: np.ndarray | None) -> str:
    header = (
        f"{'t [s]':>10} {'range [km]':>12} {'rho_dot [m/s]':>14} "
        f"{'Doppler [Hz]':>14} {'rate [Hz/s]':>12}"
    )
    if precomp is not None:
        header += f" {'precomp [Hz]':>14}"
    lines = [header, "-" * len(header)]
    for i in range(times.size):
        row = (
            f"{times[i]:10.2f} {rng[i] / 1e3:12.3f} {rate[i]:14.3f} "
            f"{doppler[i]:14.2f} {d_rate[i]:12.3f}"
        )
        if precomp is not None:
            row += f" {precomp[i]:14.2f}"
        lines.append(row)
    return "\n".join(lines)


def _cmd_analytic(args: argparse.Namespace) -> int:
    orbit = CircularOverheadPass(orbit_radius_m=WGS84_A_M + args.altitude_km * 1e3)
    horizon = orbit.horizon_time_s()
    if args.samples < 2:
        raise ValueError(f"--samples must be >= 2, got {args.samples}")
    times = np.linspace(-horizon, horizon, int(args.samples))
    direction = LinkDirection.TWO_WAY if args.two_way else LinkDirection.DOWNLINK
    profile = compute_profile(
        orbit.observer_state, orbit.satellite_state, times, args.carrier_hz,
        direction=direction, turnaround_ratio=args.turnaround_ratio,
    )
    tca = time_of_closest_approach_s(orbit.observer_state, orbit.satellite_state,
                                     -horizon, horizon)
    zero = doppler_zero_crossing_s(orbit.observer_state, orbit.satellite_state,
                                   -horizon, horizon)
    summary = {
        "model": "circular overhead pass, non-rotating Earth, two-body",
        "altitude_km": args.altitude_km,
        "orbit_radius_m": orbit.orbit_radius_m,
        "orbital_period_s": orbit.orbital_period_s,
        "orbital_speed_mps": orbit.orbital_speed_mps,
        "horizon_to_horizon_s": 2.0 * horizon,
        "reference_carrier_hz": profile.carrier_hz,
        "direction": profile.direction.value,
        "ways": profile.ways,
        "peak_doppler_hz": profile.peak_doppler_hz(),
        "doppler_span_hz": profile.doppler_span_hz(),
        "doppler_rate_at_tca_hz_per_s": orbit.max_doppler_rate_hz_per_s(args.carrier_hz)
        * profile.ways
        * (args.turnaround_ratio if profile.ways == 2 else 1.0),
        "time_of_closest_approach_s": tca,
        "doppler_zero_crossing_s": zero,
        "range_rate_convention": RANGE_RATE_CONVENTION,
        "doppler_convention": DOPPLER_CONVENTION,
    }
    if args.json:
        payload = dict(summary)
        payload["times_s"] = times.tolist()
        payload["range_m"] = profile.range_m.tolist()
        payload["range_rate_mps"] = profile.range_rate_mps.tolist()
        payload["doppler_hz"] = profile.doppler_hz.tolist()
        payload["doppler_rate_hz_per_s"] = profile.doppler_rate_hz_per_s.tolist()
        if profile.precomp_offset_hz is not None:
            payload["precomp_offset_hz"] = profile.precomp_offset_hz.tolist()
        print(json.dumps(payload, indent=2))
        return 0
    print("Circular overhead pass, closed form (non-rotating Earth)")
    print(f"  altitude                {args.altitude_km:.1f} km")
    print(f"  orbital period          {orbit.orbital_period_s / 60.0:.3f} min")
    print(f"  orbital speed           {orbit.orbital_speed_mps:.1f} m/s")
    print(f"  horizon to horizon      {2.0 * horizon:.1f} s")
    print(f"  link                    {profile.direction.value} ({profile.ways}-way)")
    print(f"  reference carrier       {profile.carrier_hz / 1e9:.6f} GHz")
    print(f"  peak Doppler            {profile.peak_doppler_hz():+.2f} Hz")
    print(f"  Doppler span            {profile.doppler_span_hz():.2f} Hz")
    print(f"  Doppler rate at TCA     {summary['doppler_rate_at_tca_hz_per_s']:+.3f} Hz/s")
    print(f"  closest approach at     {tca:+.6f} s")
    print(f"  Doppler zero crossing   {zero:+.6f} s")
    print()
    print(_format_profile_table(times, profile.range_m, profile.range_rate_mps,
                                profile.doppler_hz, profile.doppler_rate_hz_per_s,
                                profile.precomp_offset_hz))
    print()
    print(f"convention: {RANGE_RATE_CONVENTION}")
    print(f"convention: {DOPPLER_CONVENTION}")
    return 0


def _cmd_tle(args: argparse.Namespace) -> int:
    from sgp4.api import Satrec

    for name, line in (("--line1", args.line1), ("--line2", args.line2)):
        if len(line) != 69:
            raise ValueError(f"{name} must be 69 characters, got {len(line)}")
    satrec = Satrec.twoline2rv(args.line1, args.line2)
    try:
        start = datetime.fromisoformat(args.epoch)
    except ValueError as exc:
        raise ValueError(f"--epoch must be ISO-8601, got {args.epoch!r}: {exc}") from exc
    if start.tzinfo is None:
        start = start.replace(tzinfo=UTC)
    if args.step_s <= 0.0:
        raise ValueError(f"--step-s must be > 0, got {args.step_s}")
    if args.duration_s <= args.step_s:
        raise ValueError(
            f"--duration-s ({args.duration_s}) must exceed --step-s ({args.step_s})"
        )
    n = int(args.duration_s // args.step_s) + 1
    times = np.arange(n, dtype=float) * args.step_s
    rng = np.empty(n)
    rate = np.empty(n)
    doppler = np.empty(n)
    precomp = np.empty(n)
    for i, dt_s in enumerate(times):
        t = start + timedelta(seconds=float(dt_s))
        sat = satellite_state_teme(satrec, t)
        sta = station_state_teme(args.lat_deg, args.lon_deg, args.alt_m, t)
        rng[i] = slant_range_m(sta, sat)
        rate[i] = range_rate_mps(sta, sat)
        doppler[i] = one_way_doppler_hz(rate[i], args.carrier_hz)
        precomp[i] = -doppler[i]
    jd, fr = julian_date(start)
    payload = {
        "source": "sgp4 TEME propagation; station in TEME with omega_E x r velocity",
        "epoch_utc": start.isoformat(),
        "julian_date": jd + fr,
        "station_lat_deg": args.lat_deg,
        "station_lon_deg": args.lon_deg,
        "station_alt_m": args.alt_m,
        "carrier_hz": args.carrier_hz,
        "direction": "downlink",
        "ways": 1,
        "times_s": times.tolist(),
        "range_m": rng.tolist(),
        "range_rate_mps": rate.tolist(),
        "doppler_hz": doppler.tolist(),
        "precomp_offset_hz": precomp.tolist(),
        "range_rate_convention": RANGE_RATE_CONVENTION,
        "doppler_convention": DOPPLER_CONVENTION,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0
    print(f"TLE pass profile, downlink one-way, carrier {args.carrier_hz / 1e9:.6f} GHz")
    print(f"  station  lat {args.lat_deg:+.4f} deg  lon {args.lon_deg:+.4f} deg  "
          f"alt {args.alt_m:.1f} m")
    print(f"  epoch    {start.isoformat()}")
    print()
    print(_format_profile_table(times, rng, rate, doppler, np.zeros(n), precomp))
    print()
    print("Doppler rate column is zero: the sgp4 path carries no acceleration (see")
    print("frames.satellite_state_teme). Use the 'analytic' subcommand for Doppler rate.")
    print(f"convention: {RANGE_RATE_CONVENTION}")
    print(f"convention: {DOPPLER_CONVENTION}")
    return 0


def _cmd_relativistic(args: argparse.Namespace) -> int:
    orbit = CircularOverheadPass(orbit_radius_m=WGS84_A_M + args.altitude_km * 1e3)
    horizon = orbit.horizon_time_s()
    carrier = args.carrier_hz
    speed = orbit.orbital_speed_mps

    rate_tca = orbit.range_rate_mps(0.0)
    rate_horizon = orbit.range_rate_mps(horizon)
    classical_tca = one_way_doppler_hz(rate_tca, carrier)
    classical_peak = one_way_doppler_hz(rate_horizon, carrier)
    rel_tca = relativistic_correction_hz(rate_tca, speed, carrier)
    rel_peak = relativistic_correction_hz(rate_horizon, speed, carrier)
    grav = gravitational_shift_fraction(orbit.orbit_radius_m, orbit.observer_radius_m) * carrier
    cross = two_way_doppler_hz(rate_horizon, carrier) - 2.0 * classical_peak
    cascade = carrier * rate_horizon**2 / C_M_S**2

    print("Relativistic and second-order terms, one-way downlink")
    print(f"  model                        circular overhead pass, {args.altitude_km:.1f} km")
    print(f"  carrier                      {carrier / 1e9:.6f} GHz")
    print(f"  orbital speed                {speed:.1f} m/s   (beta = {speed / C_M_S:.4e})")
    print()
    print(f"  classical Doppler at horizon {classical_peak:+15.6f} Hz   O(beta^1), KEPT")
    print(f"  classical Doppler at TCA     {classical_tca:+15.6f} Hz   O(beta^1), KEPT")
    print(f"  O(beta^2) SR term at horizon {rel_peak:+15.6f} Hz   REPORTED, not applied")
    print(f"  O(beta^2) SR term at TCA     {rel_tca:+15.6f} Hz   REPORTED, not applied")
    print(f"  two-way cascade term         {cascade:+15.6f} Hz   O(beta^2), dropped")
    print(f"  gravitational (static, est.) {grav:+15.6f} Hz   order of magnitude only")
    print()
    denom = abs(classical_peak)
    print(f"  |SR at TCA| / |classical peak|      {abs(rel_tca) / denom:.3e}")
    print(f"  |cascade|   / |classical peak|      {abs(cascade) / denom:.3e}")
    print(f"  |grav|      / |classical peak|      {abs(grav) / denom:.3e}")
    print(f"  two-way minus exactly-2x one-way    {cross:+.6e} Hz (expected 0 by construction)")
    print()
    print("The SR term does not vanish at closest approach: it is dominated there by")
    print("transmitter time dilation -v^2/(2c^2), which is independent of range-rate.")
    print("Dropped entirely: Shapiro delay, troposphere and ionosphere and their rates,")
    print("transponder group delay, oscillator drift, higher relativistic orders.")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "convention":
            print(CONVENTION_TEXT, end="")
            return 0
        if args.command == "analytic":
            return _cmd_analytic(args)
        if args.command == "tle":
            return _cmd_tle(args)
        if args.command == "relativistic":
            return _cmd_relativistic(args)
    except (ValueError, TypeError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    parser.error(f"unknown command {args.command!r}")
    return 2


__all__ = ["CONVENTION_TEXT", "main"]
