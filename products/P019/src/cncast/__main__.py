"""Command-line interface: ``python -m cncast``.

Two subcommands:

``baseline``  print a published baseline profile and its integrated seeing
              quantities at a chosen wavelength and zenith angle;
``predict``   train the default learned model (seeded, ~30 s on 2 cores) and
              print a predicted profile with its prediction interval, next to
              the HV 5/7 baseline.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import numpy as np

from .baselines import bufton_wind, hv57, slc_day, slc_night
from .dataset import default_altitude_grid
from .model import Hv57Baseline, train_default_model
from .seeing import (
    fried_parameter,
    greenwood_frequency,
    isoplanatic_angle,
    seeing_fwhm_arcsec,
)

_BASELINES = {"hv57": hv57, "slc_day": slc_day, "slc_night": slc_night}


def _seeing_block(h: np.ndarray, cn2: np.ndarray, lam_m: float, zenith: float, wind: float) -> str:
    r0 = fried_parameter(h, cn2, lam_m, zenith)
    th0 = isoplanatic_angle(h, cn2, lam_m, zenith)
    fg = greenwood_frequency(h, cn2, bufton_wind(h, wind), lam_m, zenith)
    return (
        f"  integrated Cn^2 : {np.trapezoid(cn2, h):.4e} m^(1/3)\n"
        f"  r0              : {r0 * 100:.3f} cm\n"
        f"  theta0          : {th0 * 1e6:.3f} urad\n"
        f"  Greenwood f_G   : {fg:.2f} Hz  (Bufton wind, ground {wind:g} m/s)\n"
        f"  seeing FWHM     : {seeing_fwhm_arcsec(r0, lam_m):.3f} arcsec\n"
    )


def _cmd_baseline(args: argparse.Namespace) -> int:
    h = np.linspace(0.0, 20000.0, args.n_points)
    cn2 = _BASELINES[args.model](h)
    lam = args.wavelength_nm * 1e-9
    out = [
        f"# {args.model} at lambda = {args.wavelength_nm:g} nm, "
        f"zenith = {args.zenith_deg:g} deg"
    ]
    out.append("# CLIMATOLOGICAL MODEL - an average condition, not a forecast.")
    out.append(f"{'h [m]':>10}  {'Cn2 [m^-2/3]':>14}")
    for hi in np.geomspace(10.0, 20000.0, 12):
        out.append(f"{hi:10.1f}  {float(_BASELINES[args.model](np.array([hi]))[0]):14.4e}")
    out.append("")
    out.append(_seeing_block(h, cn2, lam, args.zenith_deg, args.ground_wind_m_s))
    sys.stdout.write("\n".join(out))
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    model, _ = train_default_model(n_scenarios=args.n_scenarios)
    h = default_altitude_grid(args.n_points)
    pred = model.predict(
        args.temp_c, args.wind_m_s, args.rh_pct, args.hour, args.day_of_year, h
    )
    base = 10.0 ** Hv57Baseline().predict_log10_cn2(
        np.column_stack([np.zeros((h.size, 7)), np.log10(h)])
    )
    lam = args.wavelength_nm * 1e-9
    out = [
        f"# CnCast prediction  T={args.temp_c:g} C  wind={args.wind_m_s:g} m/s  "
        f"RH={args.rh_pct:g} %  hour={args.hour:g}  doy={args.day_of_year}",
        f"# nominal interval coverage {pred.coverage:.0%};  "
        f"extrapolating={pred.extrapolating}",
        "# Trained on SYNTHETIC data (DATASET_CARD.md). Not certified for operational "
        "flight use.",
        f"{'h [m]':>10}  {'lower':>12}  {'median':>12}  {'upper':>12}  {'HV 5/7':>12}",
    ]
    for i, hi in enumerate(h):
        out.append(
            f"{hi:10.1f}  {pred.cn2_lower[i]:12.4e}  {pred.cn2[i]:12.4e}  "
            f"{pred.cn2_upper[i]:12.4e}  {base[i]:12.4e}"
        )
    out.append("")
    out.append("Predicted-profile seeing quantities:")
    out.append(_seeing_block(h, pred.cn2, lam, args.zenith_deg, args.wind_m_s))
    sys.stdout.write("\n".join(out))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for ``python -m cncast``."""
    p = argparse.ArgumentParser(prog="cncast", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("baseline", help="print a published baseline profile + seeing quantities")
    b.add_argument("--model", choices=sorted(_BASELINES), default="hv57")
    b.add_argument("--wavelength-nm", type=float, default=500.0)
    b.add_argument("--zenith-deg", type=float, default=0.0)
    b.add_argument("--ground-wind-m-s", type=float, default=5.0)
    b.add_argument("--n-points", type=int, default=20001)
    b.set_defaults(func=_cmd_baseline)

    q = sub.add_parser("predict", help="train the learned model and predict one profile")
    q.add_argument("--temp-c", type=float, required=True)
    q.add_argument("--wind-m-s", type=float, required=True)
    q.add_argument("--rh-pct", type=float, required=True)
    q.add_argument("--hour", type=float, required=True)
    q.add_argument("--day-of-year", type=int, required=True)
    q.add_argument("--wavelength-nm", type=float, default=500.0)
    q.add_argument("--zenith-deg", type=float, default=0.0)
    q.add_argument("--n-points", type=int, default=24)
    q.add_argument("--n-scenarios", type=int, default=700)
    q.set_defaults(func=_cmd_predict)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.  Returns a process exit code."""
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, RuntimeError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
