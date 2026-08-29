"""Command-line interface: ``python -m turbscope <command>``.

Commands
--------
``forward``     known Cn^2 -> predicted sensor readings
``invert``      a scintillometer or DIMM reading -> path-averaged Cn^2 + interval
``saturation``  where the analytic scintillation inversion becomes multi-valued
``predict``     multi-sensor learned estimate (fits the model first; ~10 s)
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import numpy as np

from .dataset import features_from_measurement
from .dimm import dimm_variance, fried_from_average, seeing_fwhm_rad
from .geometry import PathGeometry
from .inversion import invert_dimm, invert_scintillation, saturation_report
from .measurements import Measurement, SensorSuite
from .model import train_default_model
from .scintillation import aperture_parameter_sq, rytov_variance_from_average, scintillation_index

ARCSEC = 180.0 * 3600.0 / np.pi


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--length-m", type=float, default=1000.0, help="path length, m")
    parser.add_argument(
        "--wavelength-nm", type=float, default=1550.0, help="wavelength, nm (300-3000)"
    )
    parser.add_argument(
        "--geometry", choices=("spherical", "plane"), default="spherical", help="wave geometry"
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="turbscope", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    fwd = sub.add_parser("forward", help="known Cn2 -> sensor readings")
    _common(fwd)
    fwd.add_argument("--cn2", type=float, required=True, help="uniform Cn2, m^-2/3")
    fwd.add_argument("--receiver-diameter-m", type=float, default=0.10)
    fwd.add_argument("--dimm-subaperture-m", type=float, default=0.06)
    fwd.add_argument("--dimm-baseline-m", type=float, default=0.20)

    inv = sub.add_parser("invert", help="sensor reading -> path-averaged Cn2")
    _common(inv)
    inv.add_argument("--sigma-i2", type=float, help="measured scintillation index")
    inv.add_argument("--receiver-diameter-m", type=float, default=0.0)
    inv.add_argument("--n-samples", type=int, default=1000)
    inv.add_argument("--method", choices=("weak", "saturation"), default="saturation")
    inv.add_argument("--dimm-variance-rad2", type=float, help="measured differential variance")
    inv.add_argument("--dimm-subaperture-m", type=float, default=0.06)
    inv.add_argument("--dimm-baseline-m", type=float, default=0.20)
    inv.add_argument("--dimm-noise-arcsec", type=float, default=0.05)
    inv.add_argument("--n-frames", type=int, default=500)
    inv.add_argument("--coverage", type=float, default=0.90)

    sat = sub.add_parser("saturation", help="saturation / multi-valued range report")
    sat.add_argument("--aperture-d-sq", type=float, default=0.0)

    pred = sub.add_parser("predict", help="learned multi-sensor estimate")
    _common(pred)
    pred.add_argument("--sigma-i2-point", type=float, required=True)
    pred.add_argument("--sigma-i2-aperture", type=float, required=True)
    pred.add_argument("--sigma-l2-rad2", type=float, required=True)
    pred.add_argument("--sigma-t2-rad2", type=float, required=True)
    pred.add_argument("--receiver-diameter-m", type=float, default=0.10)
    pred.add_argument("--n-samples", type=int, default=1000)
    pred.add_argument("--n-frames", type=int, default=500)
    pred.add_argument("--dimm-noise-arcsec", type=float, default=0.05)
    pred.add_argument("--n-scenarios", type=int, default=3000, help="training set size")
    return p


def _cmd_forward(args: argparse.Namespace, out) -> int:
    path = PathGeometry(args.length_m, args.wavelength_nm * 1e-9, args.geometry)
    beta = rytov_variance_from_average(args.cn2, path)
    d_sq = aperture_parameter_sq(args.receiver_diameter_m, path)
    r0 = fried_from_average(args.cn2, path)
    print(f"path              L = {path.length_m:.1f} m, lambda = {path.wavelength_m * 1e9:.1f} nm,"
          f" {path.geometry}", file=out)
    print(f"Fresnel scale     {path.fresnel_scale_m() * 1000:.2f} mm", file=out)
    print(f"Cn2 (uniform)     {args.cn2:.4g} m^-2/3", file=out)
    print(f"beta_0^2          {beta:.6g}", file=out)
    print(f"sigma_I^2 point   {float(scintillation_index(beta, 0.0)):.6g}", file=out)
    print(f"aperture d^2      {d_sq:.4g}  (D = {args.receiver_diameter_m:.3f} m)", file=out)
    print(f"sigma_I^2 apert.  {float(scintillation_index(beta, d_sq)):.6g}", file=out)
    print(f"r_0               {r0 * 100:.4g} cm", file=out)
    print(f"seeing FWHM       {seeing_fwhm_rad(r0, path.wavelength_m) * ARCSEC:.4g} arcsec",
          file=out)
    for comp in ("longitudinal", "transverse"):
        var = dimm_variance(
            r0, path.wavelength_m, args.dimm_subaperture_m, args.dimm_baseline_m, comp
        )
        print(f"DIMM {comp:<13s}{var:.6g} rad^2  ({np.sqrt(var) * ARCSEC:.4g} arcsec rms)",
              file=out)
    return 0


def _report(est, out) -> None:
    print(f"kernel            {est.kernel} ({est.method})", file=out)
    if not est.valid:
        print("estimate          INVALID", file=out)
    else:
        print(f"Cn2               {est.cn2:.4g} m^-2/3", file=out)
        print(f"{est.coverage:.0%} interval      [{est.cn2_lower:.4g}, {est.cn2_upper:.4g}]"
              f"  (rel. sigma {est.relative_sigma:.3g})", file=out)
        if est.ambiguous:
            print("branches          " + ", ".join(f"{b:.4g}" for b in est.branches), file=out)
    for note in est.notes:
        print(f"note              {note}", file=out)


def _cmd_invert(args: argparse.Namespace, out) -> int:
    path = PathGeometry(args.length_m, args.wavelength_nm * 1e-9, args.geometry)
    if args.sigma_i2 is None and args.dimm_variance_rad2 is None:
        raise ValueError("supply --sigma-i2 and/or --dimm-variance-rad2")
    if args.sigma_i2 is not None:
        _report(
            invert_scintillation(
                args.sigma_i2,
                path,
                n_samples=args.n_samples,
                receiver_diameter_m=args.receiver_diameter_m,
                method=args.method,
                coverage=args.coverage,
            ),
            out,
        )
    if args.dimm_variance_rad2 is not None:
        noise = (args.dimm_noise_arcsec / ARCSEC) ** 2
        _report(
            invert_dimm(
                args.dimm_variance_rad2,
                path,
                subaperture_m=args.dimm_subaperture_m,
                baseline_m=args.dimm_baseline_m,
                n_frames=args.n_frames,
                noise_variance_rad2=noise,
                coverage=args.coverage,
            ),
            out,
        )
    return 0


def _cmd_saturation(args: argparse.Namespace, out) -> int:
    rep = saturation_report(args.aperture_d_sq)
    print(f"aperture d^2              {rep.aperture_d_sq:.6g}", file=out)
    print(f"weak-regime limit         beta_0^2 < {rep.beta0_sq_weak_limit}", file=out)
    print(f"peak                      beta_0^2 = {rep.beta0_sq_peak:.6g} -> "
          f"sigma_I^2 = {rep.sigma_i2_peak:.6g}", file=out)
    print(f"asymptote (beta_0^2=1e4)  sigma_I^2 = {rep.sigma_i2_asymptote:.6g}", file=out)
    lo, hi = rep.ambiguous_sigma_i2_range
    print(f"multi-valued readings     {lo:.6g} < sigma_I^2 < {hi:.6g}", file=out)
    print(f"unexplainable readings    sigma_I^2 > {rep.sigma_i2_peak:.6g}", file=out)
    return 0


def _cmd_predict(args: argparse.Namespace, out) -> int:
    path = PathGeometry(args.length_m, args.wavelength_nm * 1e-9, args.geometry)
    suite = SensorSuite(
        receiver_diameter_m=args.receiver_diameter_m,
        n_irradiance_samples=args.n_samples,
        n_dimm_frames=args.n_frames,
        dimm_noise_arcsec=args.dimm_noise_arcsec,
    )
    meas = Measurement(
        sigma_i2_point=args.sigma_i2_point,
        sigma_i2_aperture=args.sigma_i2_aperture,
        sigma_l2_rad2=args.sigma_l2_rad2,
        sigma_t2_rad2=args.sigma_t2_rad2,
        true_sigma_i2_point=np.nan,
        true_sigma_i2_aperture=np.nan,
        true_sigma_l2_rad2=np.nan,
        true_sigma_t2_rad2=np.nan,
        true_beta0_sq=np.nan,
        true_r0_m=np.nan,
        aperture_d_sq=aperture_parameter_sq(args.receiver_diameter_m, path),
    )
    x = features_from_measurement(path, suite, meas).reshape(1, -1)
    print(f"fitting on {args.n_scenarios} synthetic scenarios ...", file=out)
    model, _ = train_default_model(n_scenarios=args.n_scenarios)
    pred = model.predict(x)
    print(f"Cn2 (scintillation-weighted path average)  {pred.cn2[0]:.4g} m^-2/3", file=out)
    print(f"{pred.coverage:.0%} prediction interval  "
          f"[{pred.cn2_lower[0]:.4g}, {pred.cn2_upper[0]:.4g}] m^-2/3", file=out)
    print(f"extrapolating outside training domain      {bool(pred.extrapolating[0])}", file=out)
    print("NOTE: trained on synthetic data; not certified for operational flight use.", file=out)
    return 0


def main(argv: Sequence[str] | None = None, stream=None) -> int:
    """Entry point; returns a process exit code."""
    out = sys.stdout if stream is None else stream
    args = _build_parser().parse_args(argv)
    handlers = {
        "forward": _cmd_forward,
        "invert": _cmd_invert,
        "saturation": _cmd_saturation,
        "predict": _cmd_predict,
    }
    return handlers[args.command](args, out)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
