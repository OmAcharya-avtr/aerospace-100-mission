"""Command-line interface: ``python -m turbscope``.

Subcommands
-----------
``forward``    compute scintillometer + DIMM forward-model measurements from
                a known Cn2_path and path length.
``invert``     classical closed-form inversion: weak-regime single-sensor
                baseline, plus a full-curve multi-root inversion that
                demonstrates the saturation failure mode.
``predict``    train the default learned model (seeded, well under a minute
                on 2 cores) and predict Cn2_path with a prediction interval
                from raw sensor readings, next to the classical baselines.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import numpy as np

from .dimm import differential_variance, fried_parameter_from_cn2_path
from .inversion import invert_scintillometer_weak_with_uncertainty
from .model import DimmOnlyBaseline, ScintillometerWeakBaseline, train_default_model
from .scintillometer import (
    invert_cn2_all_roots,
    invert_cn2_weak,
    rytov_variance,
    scintillation_index_full,
)
from .synthetic import (
    APERTURE_DIAM_M,
    DIMM_WAVELENGTH_M,
    SCINT_WAVELENGTH_M,
    SEPARATION_M,
    WAVE_TYPE,
)


def _cmd_forward(args: argparse.Namespace) -> int:
    lam_s = args.scint_wavelength_nm * 1e-9
    lam_d = args.dimm_wavelength_nm * 1e-9
    r_var = float(rytov_variance(args.cn2, args.path_length_m, lam_s, args.wave_type))
    sigma_i2 = float(scintillation_index_full(r_var))
    r0 = float(fried_parameter_from_cn2_path(args.cn2, args.path_length_m, lam_d))
    var_l = float(
        differential_variance(
            args.cn2, args.path_length_m, lam_d, args.aperture_m, args.separation_m,
            "longitudinal",
        )
    )
    var_t = float(
        differential_variance(
            args.cn2, args.path_length_m, lam_d, args.aperture_m, args.separation_m, "transverse"
        )
    )
    out = [
        f"# forward model: Cn2_path = {args.cn2:.4e} m^-2/3, L = {args.path_length_m:g} m",
        f"  Rytov variance (scintillometer, {args.wave_type} wave) : {r_var:.6e}",
        f"  scintillation index sigma_I^2 (full curve)             : {sigma_i2:.6e}",
        f"  Fried parameter r0 (DIMM wavelength)                   : {r0 * 100:.4f} cm",
        f"  DIMM differential variance, longitudinal               : {var_l:.6e} rad^2",
        f"  DIMM differential variance, transverse                 : {var_t:.6e} rad^2",
    ]
    sys.stdout.write("\n".join(out) + "\n")
    return 0


def _cmd_invert(args: argparse.Namespace) -> int:
    lam_s = args.scint_wavelength_nm * 1e-9
    weak = invert_cn2_weak(args.sigma_i2, args.path_length_m, lam_s, args.wave_type)
    est = invert_scintillometer_weak_with_uncertainty(
        args.sigma_i2, args.relative_std, args.path_length_m, lam_s, args.wave_type
    )
    roots = invert_cn2_all_roots(args.sigma_i2, args.path_length_m, lam_s, args.wave_type)
    out = [
        f"# closed-form inversion of sigma_I^2 = {args.sigma_i2:g}, L = {args.path_length_m:g} m",
        f"  weak-regime baseline Cn2 (single-valued, always returned): {weak:.4e} m^-2/3",
        f"    +/- {est.cn2_std:.4e} (relative std {args.relative_std:.0%} on the measurement)",
        f"  full-curve inversion: {len(roots.cn2_roots)} root(s) found on the search bracket",
    ]
    for i, (rr, cc) in enumerate(zip(roots.rytov_roots, roots.cn2_roots, strict=True)):
        out.append(f"    root {i}: sigma_R^2 = {rr:.4f}  ->  Cn2 = {cc:.4e} m^-2/3")
    if roots.is_multivalued:
        out.append(
            "  WARNING: multiple roots -- this measurement is in the saturation regime and "
            "the inversion is genuinely ill-posed from sigma_I^2 alone. See "
            "validation/saturation_regime.py."
        )
    sys.stdout.write("\n".join(out) + "\n")
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    model, _ = train_default_model(n_scenarios=args.n_scenarios)
    pred = model.predict(
        args.sigma_i2, args.var_long, args.var_trans, args.path_length_m
    )
    scint_base = ScintillometerWeakBaseline()
    dimm_base = DimmOnlyBaseline()
    x = np.array(
        [[
            np.log10(args.sigma_i2),
            np.log10(args.var_long),
            np.log10(args.var_trans),
            np.log10(args.path_length_m),
        ]]
    )
    scint_cn2 = 10.0 ** scint_base.predict_log10_cn2(x)[0]
    dimm_cn2 = 10.0 ** dimm_base.predict_log10_cn2(x)[0]
    out = [
        f"# TurbScope prediction  sigma_I2={args.sigma_i2:g}  var_long={args.var_long:g}  "
        f"var_trans={args.var_trans:g}  L={args.path_length_m:g} m",
        f"# nominal interval coverage {pred.coverage:.0%};  extrapolating={pred.extrapolating}",
        "# Trained on SYNTHETIC data (DATASET_CARD.md). Not certified for operational "
        "flight use.",
        f"  learned model : {pred.cn2_path:.4e}  [{pred.cn2_lower:.4e}, {pred.cn2_upper:.4e}]"
        f"  m^-2/3  (width {pred.interval_width_dex:.3f} dex)",
        f"  scintillometer weak baseline : {scint_cn2:.4e} m^-2/3",
        f"  DIMM-only baseline           : {dimm_cn2:.4e} m^-2/3",
    ]
    sys.stdout.write("\n".join(out) + "\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for ``python -m turbscope``."""
    p = argparse.ArgumentParser(prog="turbscope", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)

    f = sub.add_parser("forward", help="compute forward-model measurements from a known Cn2")
    f.add_argument("--cn2", type=float, required=True, help="path-averaged Cn2, m^-2/3")
    f.add_argument("--path-length-m", type=float, required=True)
    f.add_argument("--scint-wavelength-nm", type=float, default=SCINT_WAVELENGTH_M * 1e9)
    f.add_argument("--dimm-wavelength-nm", type=float, default=DIMM_WAVELENGTH_M * 1e9)
    f.add_argument("--wave-type", choices=["plane", "spherical"], default=WAVE_TYPE)
    f.add_argument("--aperture-m", type=float, default=APERTURE_DIAM_M)
    f.add_argument("--separation-m", type=float, default=SEPARATION_M)
    f.set_defaults(func=_cmd_forward)

    i = sub.add_parser("invert", help="classical closed-form inversion of a measurement")
    i.add_argument("--sigma-i2", type=float, required=True, help="measured scintillation index")
    i.add_argument("--path-length-m", type=float, required=True)
    i.add_argument("--scint-wavelength-nm", type=float, default=SCINT_WAVELENGTH_M * 1e9)
    i.add_argument("--wave-type", choices=["plane", "spherical"], default=WAVE_TYPE)
    i.add_argument("--relative-std", type=float, default=0.08)
    i.set_defaults(func=_cmd_invert)

    q = sub.add_parser("predict", help="train the learned model and predict from sensor readings")
    q.add_argument("--sigma-i2", type=float, required=True)
    q.add_argument(
        "--var-long", type=float, required=True, help="DIMM longitudinal variance, rad^2"
    )
    q.add_argument(
        "--var-trans", type=float, required=True, help="DIMM transverse variance, rad^2"
    )
    q.add_argument("--path-length-m", type=float, required=True)
    q.add_argument("--n-scenarios", type=int, default=300)
    q.set_defaults(func=_cmd_predict)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, RuntimeError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
