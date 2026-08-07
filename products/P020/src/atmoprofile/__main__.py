"""Command-line interface: ``python -m atmoprofile``.

Sub-commands
------------
``summary``
    Print every turbulence metric for one standard profile, wavelength and
    zenith angle (or a sweep of zenith angles).
``profile``
    Print Cn^2(h) samples and the provenance of a standard model.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence

import numpy as np

from . import __version__
from .metrics import summarize
from .profiles import STANDARD_PROFILES, standard_profile
from .wind import bufton_wind


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m atmoprofile",
        description=(
            "Atmospheric turbulence integrals (r0, theta0, Greenwood frequency, "
            "Rytov variance, scintillation index) from a Cn^2 profile."
        ),
    )
    parser.add_argument("--version", action="version", version=f"atmoprofile {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    s = sub.add_parser("summary", help="all metrics for one case or a zenith sweep")
    s.add_argument(
        "--profile",
        default="hv57",
        choices=sorted(STANDARD_PROFILES),
        help="standard Cn^2 model (default: hv57)",
    )
    s.add_argument(
        "--wavelength-nm", type=float, default=500.0, help="wavelength in nm (default: 500)"
    )
    s.add_argument(
        "--zenith-deg",
        type=float,
        nargs="+",
        default=[0.0],
        help="one or more zenith angles in degrees (default: 0)",
    )
    s.add_argument(
        "--ground-wind",
        type=float,
        default=5.0,
        help="Bufton ground wind speed in m/s for the Greenwood frequency (default: 5)",
    )
    s.add_argument("--h-ground", type=float, default=0.0, help="observer altitude, m")
    s.add_argument("--h-top", type=float, default=None, help="path top altitude, m")
    s.add_argument("--json", action="store_true", help="emit JSON instead of a table")

    p = sub.add_parser("profile", help="print Cn^2 samples and provenance")
    p.add_argument("--profile", default="hv57", choices=sorted(STANDARD_PROFILES))
    p.add_argument("--n", type=int, default=12, help="number of log-spaced samples")
    return parser


def _cmd_summary(args: argparse.Namespace, stream) -> int:
    profile = standard_profile(args.profile)
    wind = bufton_wind(args.ground_wind)
    lam = args.wavelength_nm * 1e-9
    rows = [
        summarize(
            profile,
            lam,
            zenith_rad=math.radians(z),
            wind=wind,
            h_ground=args.h_ground,
            h_top=args.h_top,
        )
        for z in args.zenith_deg
    ]
    if args.json:
        json.dump([r.as_dict() for r in rows], stream, indent=2)
        stream.write("\n")
        return 0

    print(f"AtmoProfile {__version__} - {profile.name} at {args.wavelength_nm:g} nm", file=stream)
    print(f"  Cn^2 source : {profile.reference}", file=stream)
    print(f"  wind source : {wind.reference}", file=stream)
    print(
        f"  path        : {args.h_ground:g} m to "
        f"{profile.h_max if args.h_top is None else args.h_top:g} m",
        file=stream,
    )
    seeing_label = 'seeing["]'
    header = (
        f"{'zen[deg]':>8} {'r0[cm]':>9} {'r0sph[cm]':>10} {'th0[urad]':>10} "
        f"{'fG[Hz]':>9} {'sig2R,pl':>10} {'sig2R,sp':>10} {seeing_label:>10} {'weak?':>6}"
    )
    print(header, file=stream)
    print("-" * len(header), file=stream)
    for r in rows:
        fg = "n/a" if r.f_greenwood_hz is None else f"{r.f_greenwood_hz:9.2f}"
        print(
            f"{r.zenith_deg:8.1f} {r.r0_m * 100:9.3f} {r.r0_spherical_m * 100:10.3f} "
            f"{r.theta0_urad:10.3f} {fg:>9} {r.rytov_plane:10.4f} "
            f"{r.rytov_spherical:10.4f} {r.seeing_arcsec:10.3f} "
            f"{str(r.weak_fluctuation_valid):>6}",
            file=stream,
        )
    print(
        "\nsigma_I^2 (weak regime, point receiver) = sigma_R^2 in the columns above.",
        file=stream,
    )
    return 0


def _cmd_profile(args: argparse.Namespace, stream) -> int:
    profile = standard_profile(args.profile)
    print(profile.describe(), file=stream)
    heights = np.unique(
        np.concatenate(
            [
                np.array([profile.h_min]),
                np.geomspace(max(profile.h_min, 1.0), profile.h_max, max(args.n, 2)),
            ]
        )
    )
    print(f"\n{'h [m]':>12} {'Cn^2 [m^-2/3]':>16}", file=stream)
    for h in heights:
        print(f"{h:12.1f} {float(profile(h)):16.4e}", file=stream)
    return 0


def main(argv: Sequence[str] | None = None, stream=None) -> int:
    """Entry point. Returns a process exit code."""
    out = sys.stdout if stream is None else stream
    args = _build_parser().parse_args(argv)
    if args.command == "summary":
        return _cmd_summary(args, out)
    return _cmd_profile(args, out)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
