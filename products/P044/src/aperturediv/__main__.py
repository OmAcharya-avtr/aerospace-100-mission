"""Command-line interface: ``python -m aperturediv <command>``.

Five subcommands, each printing a small table to stdout:

``channel``    irradiance statistics at a stated scintillation index or Rytov
               variance
``aperture``   aperture averaging factor against diameter, with the derived
               asymptotes
``outage``     outage probability and finite-window diversity order against
               the number of apertures, independent and correlated
``combiner``   the learned combiner against its four non-learned references
``ber``        sample BPSK BER over lognormal fading, with its binomial
               standard error

Research-grade output. Not flight-qualified, not certified.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import numpy as np

from . import __version__
from .aperture import (
    aperture_averaging_factor,
    aperture_averaging_large_d,
    aperture_averaging_small_d,
    fresnel_scale,
)
from .ber import bpsk_ber_lognormal_gauss_hermite, bpsk_ber_lognormal_sample
from .channel import (
    gamma_gamma_params_from_rytov,
    gamma_gamma_scintillation_index,
    lognormal_moment,
    lognormal_quantile,
    lognormal_sigma_log,
)
from .combining import (
    COMBINERS,
    branch_mean_snr,
    combined_gain,
    diversity_order,
    outage_probability,
)
from .correlation import correlation_matrix, equispaced_positions, sample_correlated_lognormal
from .datasets import make_combiner_dataset, split_dataset
from .estimation import log_error_sigma_db
from .learned import (
    LearnedCombiner,
    egc_weights,
    fit_shrinkage_exponent,
    mrc_estimated_weights,
    mrc_true_weights,
    score_weights,
    shrinkage_weights,
)

_SAFETY = (
    "research-grade output; not flight-qualified, not certified, "
    "not approved for operational aerospace use"
)


def _cmd_channel(args: argparse.Namespace) -> int:
    si = args.si
    if args.rytov is not None:
        alpha, beta = gamma_gamma_params_from_rytov(args.rytov)
        si_gg = gamma_gamma_scintillation_index(alpha, beta)
        print(f"rytov variance            {args.rytov:.6f}")
        print(f"gamma-gamma alpha         {alpha:.6f}")
        print(f"gamma-gamma beta          {beta:.6f}")
        print(f"gamma-gamma si            {si_gg:.6f}")
        if si is None:
            si = si_gg
    if si is None:
        si = 0.6
    print(f"scintillation index si    {si:.6f}")
    print(f"lognormal sigma_log       {lognormal_sigma_log(si):.6f} (natural log units)")
    print(f"E[I]                      {lognormal_moment(1, si):.6f}")
    print(f"E[I^2]                    {lognormal_moment(2, si):.6f}")
    for p in (0.01, 0.1, 0.5, 0.9, 0.99):
        q = float(lognormal_quantile(p, si))
        print(f"lognormal quantile p={p:<5.2f} I={q:.6f}  ({10 * np.log10(q):+.3f} dB)")
    print(_SAFETY)
    return 0


def _cmd_aperture(args: argparse.Namespace) -> int:
    rho_c = args.correlation_scale
    if rho_c is None:
        rho_c = fresnel_scale(args.wavelength, args.path_length)
        print(
            f"correlation scale taken as the Fresnel scale sqrt(lambda L) = {rho_c:.6f} m "
            f"(lambda={args.wavelength:.3e} m, L={args.path_length:.1f} m)"
        )
    print(f"correlation scale rho_c   {rho_c:.6f} m, model {args.model}")
    print(f"point scintillation index {args.si:.6f}")
    print()
    print("  D (m)      D/rho_c        A(D)   A small-D   A large-D      si(D)")
    for d in args.diameter:
        a = aperture_averaging_factor(d, rho_c, args.model)
        small = aperture_averaging_small_d(d, rho_c) if args.model == "gaussian" else float("nan")
        large = aperture_averaging_large_d(d, rho_c) if args.model == "gaussian" else float("nan")
        print(
            f"{d:8.4f} {d / rho_c:12.4f} {a:11.6f} {small:11.6f} {large:11.6f} "
            f"{a * args.si:10.6f}"
        )
    print()
    print("small-D and large-D columns are the derived Gaussian-covariance limits;")
    print("they are not valid outside their own regime (see the module docstring).")
    print(_SAFETY)
    return 0


def _cmd_outage(args: argparse.Namespace) -> int:
    rng = np.random.default_rng(args.seed)
    pos = equispaced_positions(args.max_apertures, args.spacing)
    r_corr = correlation_matrix(pos, args.correlation_scale, "gaussian")
    snr_db = np.arange(args.snr_min, args.snr_max + 1e-9, args.snr_step)
    print(f"si={args.si}  n_samples={args.samples}  seed={args.seed}")
    print(f"threshold={args.threshold} dB  spacing={args.spacing} m")
    print(f"rho_c={args.correlation_scale} m  window={args.window}")
    adjacent = f"{r_corr[0, 1]:.6f}" if r_corr.shape[0] > 1 else "n/a (single aperture)"
    print(f"log-correlation of adjacent apertures: {adjacent}")
    print()
    for label, matrix in (("independent", np.eye(args.max_apertures)), ("correlated", r_corr)):
        irr = sample_correlated_lognormal(args.samples, args.si, matrix, rng)
        print(f"-- {label}")
        print(
            f"  L  scheme   P_out @ {args.report_snr:.0f} dB    diversity order"
            "  fit points  SNR window"
        )
        for n_ap in range(1, args.max_apertures + 1):
            for scheme in COMBINERS:
                gain = combined_gain(irr[:, :n_ap], scheme)
                branch_db = 10.0 * np.log10(
                    branch_mean_snr(10.0 ** (args.report_snr / 10.0), n_ap, args.normalisation)
                )
                p_at = float(outage_probability(gain, branch_db, args.threshold))
                p_curve = outage_probability(gain, snr_db, args.threshold)
                try:
                    res = diversity_order(snr_db, p_curve, window=args.window)
                    order = f"{res.order:15.3f}  {res.n_points:10d}  " \
                            f"{res.snr_db_range[0]:.2f}-{res.snr_db_range[1]:.2f} dB"
                except ValueError as exc:
                    order = f"  not measurable: {exc}"
                print(f"{n_ap:3d}  {scheme:6s}  {p_at:14.3e}  {order}")
        print()
    print("diversity order is a least-squares log-log slope over the stated outage window,")
    print("not an asymptotic limit; the window is part of the number.")
    print(_SAFETY)
    return 0


def _cmd_combiner(args: argparse.Namespace) -> int:
    data = make_combiner_dataset(
        args.samples,
        n_apertures=args.apertures,
        si=args.si,
        aperture_spacing_m=args.spacing,
        correlation_scale_m=args.correlation_scale,
        sigma_e_range=(0.0, args.sigma_e_max),
        seed=args.seed,
    )
    n_train = int(0.6 * args.samples)
    n_val = int(0.2 * args.samples)
    train, val, test = split_dataset(data, n_train=n_train, n_validation=n_val)
    print(f"L={args.apertures}  si={args.si}  seed={args.seed}")
    print(f"rows train/val/test = {len(train)}/{len(val)}/{len(test)}")
    print(
        f"sigma_e in [0, {args.sigma_e_max}] -> "
        f"[0, {log_error_sigma_db(args.sigma_e_max):.2f}] dB"
    )
    print(f"branch Eb/N0 for the BER column: {args.ebn0} dB")
    print()
    p_star = fit_shrinkage_exponent(val)
    model = LearnedCombiner(random_state=args.seed).fit(train)
    h = test.amplitude_true
    rows = [
        score_weights("mrc_true", mrc_true_weights(test.irradiance_true), h, args.ebn0),
        score_weights(
            "mrc_estimated", mrc_estimated_weights(test.irradiance_estimated), h, args.ebn0
        ),
        score_weights("egc", egc_weights(len(test), test.n_apertures), h, args.ebn0),
        score_weights(
            f"shrinkage p={p_star:.2f}",
            shrinkage_weights(test.irradiance_estimated, p_star),
            h,
            args.ebn0,
        ),
        score_weights(
            "learned", model.combine(test.irradiance_estimated, test.sigma_e)[0], h, args.ebn0
        ),
    ]
    print("combiner             mean dB   median dB    p90 dB    max dB      mean BER")
    for row in rows:
        print(
            f"{row.name:20s} {row.mean_penalty_db:8.4f} {row.median_penalty_db:11.4f} "
            f"{row.p90_penalty_db:9.4f} {row.max_penalty_db:9.4f} {row.mean_ber:13.6e}"
        )
    print()
    print("penalty is SNR loss in dB against MRC with the true state, which is optimal")
    print("by Cauchy-Schwarz; mrc_true is therefore 0 dB by construction, not by merit.")
    print(_SAFETY)
    return 0


def _cmd_ber(args: argparse.Namespace) -> int:
    res = bpsk_ber_lognormal_sample(
        args.ebn0, args.si, args.bits, args.seed, chunk_size=args.chunk
    )
    gh = bpsk_ber_lognormal_gauss_hermite(args.ebn0, args.si)
    print("coherent BPSK over unit-mean lognormal fading, single aperture")
    print(f"Eb/N0                 {args.ebn0:.4f} dB")
    print(f"scintillation index   {args.si:.6f}")
    print(f"bits = realisations   {res.n_bits}")
    print(f"seed                  {args.seed}  chunk_size {args.chunk}")
    print(f"sample BER            {res.ber:.6e}")
    print(f"bit errors observed   {res.n_errors}")
    print(f"binomial SE           {res.binomial_se:.6e}  ({res.relative_se * 100:.3f} % of BER)")
    print(f"Gauss-Hermite (diag)  {gh:.6e}")
    if res.binomial_se > 0.0:
        print(f"z = (sample - GH)/SE  {(res.ber - gh) / res.binomial_se:+.4f}")
    print(_SAFETY)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for ``python -m aperturediv``."""
    parser = argparse.ArgumentParser(
        prog="aperturediv",
        description=(
            "Multi-aperture receive diversity for free-space optical links: "
            "irradiance statistics, aperture averaging, inter-aperture correlation, "
            "MRC/EGC/SC combining, and a learned combiner for imperfect CSI. "
            "Research-grade; not flight-qualified, not certified."
        ),
    )
    parser.add_argument("--version", action="version", version=f"aperturediv {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ch = sub.add_parser("channel", help="irradiance statistics")
    p_ch.add_argument("--si", type=float, default=None, help="scintillation index (default 0.6)")
    p_ch.add_argument("--rytov", type=float, default=None, help="Rytov variance for gamma-gamma")
    p_ch.set_defaults(func=_cmd_channel)

    p_ap = sub.add_parser("aperture", help="aperture averaging factor")
    p_ap.add_argument("--diameter", type=float, nargs="+", default=[0.01, 0.02, 0.05, 0.1, 0.2])
    p_ap.add_argument("--correlation-scale", type=float, default=None, help="rho_c in m")
    p_ap.add_argument("--wavelength", type=float, default=1.55e-6, help="m, for the Fresnel scale")
    p_ap.add_argument("--path-length", type=float, default=2000.0, help="m")
    p_ap.add_argument("--si", type=float, default=0.6, help="point scintillation index")
    p_ap.add_argument("--model", choices=("gaussian", "exponential"), default="gaussian")
    p_ap.set_defaults(func=_cmd_aperture)

    p_out = sub.add_parser("outage", help="outage probability and diversity order")
    p_out.add_argument("--si", type=float, default=0.9)
    p_out.add_argument("--max-apertures", type=int, default=4)
    p_out.add_argument("--samples", type=int, default=500_000)
    p_out.add_argument("--spacing", type=float, default=0.05, help="aperture pitch in m")
    p_out.add_argument("--correlation-scale", type=float, default=0.10, help="rho_c in m")
    p_out.add_argument("--threshold", type=float, default=5.0, help="outage threshold in dB")
    p_out.add_argument("--snr-min", type=float, default=0.0)
    p_out.add_argument("--snr-max", type=float, default=50.0)
    p_out.add_argument("--snr-step", type=float, default=0.25)
    p_out.add_argument("--report-snr", type=float, default=25.0, help="total mean SNR in dB")
    p_out.add_argument(
        "--normalisation", choices=("fixed_total", "fixed_branch"), default="fixed_total"
    )
    p_out.add_argument("--window", type=float, nargs=2, default=(1e-4, 1e-2))
    p_out.add_argument("--seed", type=int, default=44044)
    p_out.set_defaults(func=_cmd_outage)

    p_cmb = sub.add_parser("combiner", help="learned combiner against its baselines")
    p_cmb.add_argument("--samples", type=int, default=60_000)
    p_cmb.add_argument("--apertures", type=int, default=4)
    p_cmb.add_argument("--si", type=float, default=0.9)
    p_cmb.add_argument("--spacing", type=float, default=0.15)
    p_cmb.add_argument("--correlation-scale", type=float, default=0.10)
    p_cmb.add_argument("--sigma-e-max", type=float, default=3.0)
    p_cmb.add_argument("--ebn0", type=float, default=10.0, help="branch mean Eb/N0 in dB")
    p_cmb.add_argument("--seed", type=int, default=44044)
    p_cmb.set_defaults(func=_cmd_combiner)

    p_ber = sub.add_parser("ber", help="sample BPSK BER over lognormal fading")
    p_ber.add_argument("--ebn0", type=float, default=10.0)
    p_ber.add_argument("--si", type=float, default=0.3)
    p_ber.add_argument("--bits", type=int, default=2_000_000)
    p_ber.add_argument("--seed", type=int, default=44044)
    p_ber.add_argument("--chunk", type=int, default=2_000_000)
    p_ber.set_defaults(func=_cmd_ber)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit status."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
