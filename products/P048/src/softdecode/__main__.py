"""Command-line interface: ``python -m softdecode <subcommand>``.

Subcommands
-----------
``llr``       print exact, max-log and known-CSI LLRs over a grid of samples
``maxlog``    measure the max-log LLR error and its decoded cost
``clip``      sweep the LLR clip level and report both costs
``mismatch``  the mismatched-CSI penalty, both directions
``ppm``       M-ary PPM bit LLRs for one slot vector
``crosscheck`` the X3 soft-versus-hard ordering at one Eb/N0
``ldpc``      report the shipped LDPC code's parameters
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from .channel import GammaGammaFading, LognormalFading, amplitude_quadrature
from .codes import EXTENDED_HAMMING_84
from .csi import MultiplicativeCsiError
from .detection import DetectionModel
from .ldpc import make_regular_ldpc
from .llr import (
    clip_llr,
    gray_labels,
    llr_ook_known_csi,
    llr_ook_marginal,
    llr_ook_maxlog,
    llr_ppm_known_csi,
    llr_ppm_marginal,
    llr_ppm_maxlog,
)
from .metrics import bit_error_rate, generalised_mutual_information, llr_error
from .simulate import decode_ber, demap_ook, simulate_ook


def _fading(args):
    if args.fading == "lognormal":
        return LognormalFading(args.sigma_i2)
    return GammaGammaFading(args.alpha, args.beta)


def _add_fading(p):
    p.add_argument("--fading", choices=("lognormal", "gammagamma"), default="lognormal")
    p.add_argument("--sigma-i2", type=float, default=0.3, help="scintillation index, lognormal")
    p.add_argument("--alpha", type=float, default=4.0, help="gamma-gamma alpha")
    p.add_argument("--beta", type=float, default=2.0, help="gamma-gamma beta")


def cmd_llr(args) -> int:
    det = DetectionModel()
    fading = _fading(args)
    a = det.ook_amplitude(args.ebn0)
    hq, wq = amplitude_quadrature(fading)
    y = np.linspace(-0.5 * a, 1.6 * a, args.points)
    exact = llr_ook_marginal(y, a, hq, wq, det.sigma)
    maxlog = llr_ook_maxlog(y, a, hq, wq, det.sigma)
    known = llr_ook_known_csi(y, a, 1.0, det.sigma)
    print(f"fading                {fading}")
    print(f"scintillation index   {fading.scintillation_index:.6f}")
    print(f"Eb/N0                 {args.ebn0:.2f} dB   amplitude a = {a:.6f}, sigma = 1")
    print(f"quadrature nodes      {hq.size}")
    print(f"{'y':>12} {'exact':>14} {'max-log':>14} {'known h=1':>14}")
    for yi, e, m, k in zip(y, exact, maxlog, known, strict=True):
        print(f"{yi:12.5f} {e:14.6f} {m:14.6f} {k:14.6f}")
    err = llr_error(maxlog, exact)
    print(f"\nmax-log error vs exact   {err}")
    print(f"max-log >= exact everywhere: {bool(np.all(maxlog >= exact - 1e-9))}")
    return 0


def _coded_setup(args):
    code = make_regular_ldpc()
    fading = _fading(args)
    error = MultiplicativeCsiError(args.bias_db, args.jitter_db)
    realisation = simulate_ook(code, args.ebn0, fading, error, args.blocks, args.seed)
    return code, fading, error, realisation


def cmd_maxlog(args) -> int:
    code, fading, error, r = _coded_setup(args)
    exact = demap_ook(r, "no_csi_exact", fading)
    maxlog = demap_ook(r, "no_csi_maxlog", fading)
    print(f"code                  n={code.length} k={code.dimension} rate={code.rate:.4f}")
    print(f"Eb/N0 {args.ebn0:.2f} dB, blocks {args.blocks}, seed {args.seed}")
    print(f"LLR error (max-log vs exact, no CSI)  {llr_error(maxlog, exact)}")
    for name, llr in (("exact", exact), ("max-log", maxlog)):
        ber = decode_ber(code, r, llr)
        gmi = generalised_mutual_information(llr, r.codeword)
        print(f"  {name:<10} decoded BER {ber}  GMI {gmi:.6f}")
    return 0


def cmd_clip(args) -> int:
    code, fading, error, r = _coded_setup(args)
    exact = demap_ook(r, "no_csi_exact", fading)
    base = decode_ber(code, r, exact)
    print(f"unclipped decoded BER {base}")
    print(f"{'L_max':>8} {'LLR rmse':>14} {'clipped frac':>13} {'decoded BER':>14} {'GMI':>10}")
    for limit in args.limits:
        c = clip_llr(exact, limit)
        frac = float(np.mean(np.abs(exact) > limit))
        print(
            f"{limit:8.2f} {llr_error(c, exact).rmse:14.6f} {frac:13.6f} "
            f"{decode_ber(code, r, c).rate:14.6e} "
            f"{generalised_mutual_information(c, r.codeword):10.5f}"
        )
    return 0


def cmd_mismatch(args) -> int:
    code = make_regular_ldpc()
    fading = _fading(args)
    print(f"code n={code.length} k={code.dimension} rate={code.rate:.4f}, "
          f"Eb/N0 {args.ebn0:.2f} dB, blocks {args.blocks}, seed {args.seed}")
    print(f"{'bias dB':>9} {'jitter dB':>10} {'plug-in BER':>14} {'plug-in GMI':>12} "
          f"{'csi-aware BER':>14} {'known-CSI BER':>14}")
    for bias in args.bias_db:
        error = MultiplicativeCsiError(bias, args.jitter_db)
        r = simulate_ook(code, args.ebn0, fading, error, args.blocks, args.seed)
        pl = demap_ook(r, "plugin", fading, error)
        ca = demap_ook(r, "csi_aware_exact", fading, error)
        kn = demap_ook(r, "known_csi", fading, error)
        print(
            f"{bias:9.2f} {args.jitter_db:10.2f} {decode_ber(code, r, pl).rate:14.6e} "
            f"{generalised_mutual_information(pl, r.codeword):12.5f} "
            f"{decode_ber(code, r, ca).rate:14.6e} {decode_ber(code, r, kn).rate:14.6e}"
        )
    return 0


def cmd_ppm(args) -> int:
    det = DetectionModel()
    fading = _fading(args)
    a = det.ppm_amplitude(args.ebn0, args.order)
    hq, wq = amplitude_quadrature(fading)
    rng = np.random.default_rng(args.seed)
    h = fading.sample(1, rng)
    slot = args.slot % args.order
    y = np.zeros((1, args.order))
    y[0, slot] = a * h[0]
    y += rng.standard_normal(y.shape)
    labels = gray_labels(args.order)
    print(f"M = {args.order}, Gray labels:\n{labels}")
    print(f"a = {a:.6f}, true h = {h[0]:.6f}, pulsed slot = {slot}")
    print(f"y = {np.array2string(y[0], precision=5)}")
    print(f"known-CSI bit LLRs  {np.array2string(llr_ppm_known_csi(y, a, h)[0], precision=6)}")
    print(f"marginal bit LLRs   "
          f"{np.array2string(llr_ppm_marginal(y, a, hq, wq)[0], precision=6)}")
    print(f"max-log bit LLRs    "
          f"{np.array2string(llr_ppm_maxlog(y, a, hq, wq)[0], precision=6)}")
    return 0


def cmd_crosscheck(args) -> int:
    code = EXTENDED_HAMMING_84
    det = DetectionModel()
    fading = _fading(args)
    rng = np.random.default_rng(args.seed)
    a = det.ook_amplitude(args.ebn0, code.rate)
    messages = rng.integers(0, 2, size=(args.blocks, code.dimension)).astype(np.int8)
    codeword = code.encode(messages)
    h = fading.sample(codeword.shape, rng)
    y = a * h * codeword + det.sigma * rng.standard_normal(codeword.shape)
    llr = llr_ook_known_csi(y, a, h, det.sigma)
    soft = bit_error_rate(code.decode_soft(llr), messages)
    hard = bit_error_rate(code.decode_hard(llr), messages)
    print(f"extended Hamming (8,4), Eb/N0 {args.ebn0:.2f} dB, blocks {args.blocks}, "
          f"seed {args.seed}")
    print(f"  soft {soft}")
    print(f"  hard {hard}")
    print(f"  soft <= hard: {soft.rate <= hard.rate}")
    return 0


def cmd_ldpc(args) -> int:
    code = make_regular_ldpc(args.n, args.dv, args.dc, args.seed)
    print(f"n                 {code.length}")
    print(f"checks m          {code.n_checks}")
    print(f"dimension k       {code.dimension}")
    print(f"rate              {code.rate:.6f}")
    print(f"row weights       {sorted(set(code.parity_check.sum(axis=1).tolist()))}")
    print(f"column weights    {sorted(set(code.parity_check.sum(axis=0).tolist()))}")
    print(f"length-4 cycles   {code.cycles4}")
    print(f"seed              {args.seed}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m softdecode",
        description=(
            "Soft-decision demapping over fading optical channels. Research-grade; "
            "not flight-qualified, not certified, not approved for operational "
            "aerospace use."
        ),
    )
    sub = p.add_subparsers(dest="command", required=True)

    q = sub.add_parser("llr", help="LLR values over a grid of samples")
    _add_fading(q)
    q.add_argument("--ebn0", type=float, default=10.0, help="Eb/N0 in dB")
    q.add_argument("--points", type=int, default=11)
    q.set_defaults(func=cmd_llr)

    q = sub.add_parser("maxlog", help="max-log LLR error and its decoded cost")
    _add_fading(q)
    q.add_argument("--ebn0", type=float, default=8.0)
    q.add_argument("--blocks", type=int, default=400)
    q.add_argument("--seed", type=int, default=20261006)
    q.add_argument("--bias-db", type=float, default=0.0)
    q.add_argument("--jitter-db", type=float, default=0.0)
    q.set_defaults(func=cmd_maxlog)

    q = sub.add_parser("clip", help="LLR clipping sweep")
    _add_fading(q)
    q.add_argument("--ebn0", type=float, default=8.0)
    q.add_argument("--blocks", type=int, default=400)
    q.add_argument("--seed", type=int, default=20261006)
    q.add_argument("--bias-db", type=float, default=0.0)
    q.add_argument("--jitter-db", type=float, default=0.0)
    q.add_argument("--limits", type=float, nargs="+",
                   default=[0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    q.set_defaults(func=cmd_clip)

    q = sub.add_parser("mismatch", help="mismatched-CSI penalty, both directions")
    _add_fading(q)
    q.add_argument("--ebn0", type=float, default=8.0)
    q.add_argument("--blocks", type=int, default=400)
    q.add_argument("--seed", type=int, default=20261006)
    q.add_argument("--jitter-db", type=float, default=0.0)
    q.add_argument("--bias-db", type=float, nargs="+",
                   default=[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    q.set_defaults(func=cmd_mismatch)

    q = sub.add_parser("ppm", help="M-ary PPM bit LLRs for one random symbol")
    _add_fading(q)
    q.add_argument("--ebn0", type=float, default=10.0)
    q.add_argument("--order", type=int, default=4)
    q.add_argument("--slot", type=int, default=1)
    q.add_argument("--seed", type=int, default=20261006)
    q.set_defaults(func=cmd_ppm)

    q = sub.add_parser("crosscheck", help="X3 soft-versus-hard ordering at one Eb/N0")
    _add_fading(q)
    q.add_argument("--ebn0", type=float, default=8.0)
    q.add_argument("--blocks", type=int, default=20000)
    q.add_argument("--seed", type=int, default=20261006)
    q.set_defaults(func=cmd_crosscheck)

    q = sub.add_parser("ldpc", help="shipped LDPC code parameters")
    q.add_argument("--n", type=int, default=96)
    q.add_argument("--dv", type=int, default=3)
    q.add_argument("--dc", type=int, default=6)
    q.add_argument("--seed", type=int, default=20261006)
    q.set_defaults(func=cmd_ldpc)
    return p


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit status."""
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
