"""Command-line interface: ``python -m berbench sweep ...``.

Example
-------
python -m berbench sweep --mod ook bpsk ppm --snr 0:20:2 --channel awgn
python -m berbench sweep --mod ook --snr 0:16:2 --channel lognormal \
    --sigma-i2 0.3 --mc --n 200000 --png sweep.png
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from .analytic import MODULATIONS, analytic_ber
from .channels import CHANNELS
from .montecarlo import mc_ber


def _parse_snr(spec: str) -> np.ndarray:
    """Parse 'start:stop:step' (inclusive stop, dB) or a comma list '0,4,8'."""
    try:
        if ":" in spec:
            parts = spec.split(":")
            if len(parts) != 3:
                raise ValueError
            start, stop, step = (float(p) for p in parts)
            if step <= 0 or stop < start:
                raise ValueError
            n_pts = int(round((stop - start) / step)) + 1
            return start + step * np.arange(n_pts)
        return np.array([float(p) for p in spec.split(",")])
    except ValueError:
        raise SystemExit(
            f"error: cannot parse --snr {spec!r}; use start:stop:step (e.g. 0:20:2) "
            "or a comma list (e.g. 0,4,8)"
        ) from None


def _fmt(x: float) -> str:
    return f"{x:.3e}" if np.isfinite(x) else "nan"


def _run_sweep(args: argparse.Namespace) -> int:
    snr = _parse_snr(args.snr)
    kw = dict(channel=args.channel, M=args.M, threshold=args.threshold)
    if args.channel == "lognormal":
        if args.sigma_i2 is None:
            raise SystemExit("error: --channel lognormal requires --sigma-i2")
        kw["sigma_i2"] = args.sigma_i2

    rows = []
    curves = {}
    for mod in args.mod:
        ana = analytic_ber(mod, snr, **kw)
        mc = None
        if args.mc:
            mc = mc_ber(mod, snr, n=args.n, seed=args.seed, ci_level=args.ci,
                        max_seconds=args.max_seconds, **kw)
        curves[mod] = (ana, mc)
        for j, s in enumerate(snr):
            row = [mod, f"{s:g}", _fmt(ana.ber[j])]
            if mc is not None:
                row += [_fmt(mc.ber[j]), f"[{_fmt(mc.ci_low[j])}, {_fmt(mc.ci_high[j])}]",
                        str(int(mc.n_errors[j])), str(int(mc.n_bits[j]))]
            rows.append(row)

    header = ["mod", "Eb/N0 dB", "BER analytic"]
    if args.mc:
        header += [f"BER MC (n={args.n})", f"Wilson {args.ci:.0%} CI", "errors", "bits"]
    widths = [max(len(header[c]), *(len(r[c]) for r in rows)) for c in range(len(header))]
    line = "  ".join(h.ljust(w) for h, w in zip(header, widths))
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(v.ljust(w) for v, w in zip(r, widths)))
    if args.channel == "lognormal":
        print(f"\nchannel: lognormal, sigma_I^2 = {args.sigma_i2:g} "
              "(weak-fluctuation model, valid for sigma_I^2 < ~1)")

    if args.png:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 5))
        for mod, (ana, mc) in curves.items():
            lbl = mod.upper() + (f" (M={args.M})" if mod == "ppm" else "")
            (ln,) = ax.semilogy(ana.snr_db, np.maximum(ana.ber, 1e-16), label=f"{lbl} analytic")
            if mc is not None:
                ok = mc.n_errors > 0
                ax.semilogy(mc.snr_db[ok], mc.ber[ok], "o", color=ln.get_color(),
                            mfc="none", label=f"{lbl} MC")
        ax.set_xlabel("Eb/N0 [dB]")
        ax.set_ylabel("BER")
        ax.set_ylim(bottom=max(1e-12, ax.get_ylim()[0]))
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        title = f"BER over {args.channel}"
        if args.channel == "lognormal":
            title += rf" ($\sigma_I^2$={args.sigma_i2:g})"
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(args.png, dpi=150)
        print(f"\nsaved plot: {args.png}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m berbench",
                                description="BER benchmarking for OOK/BPSK/M-PPM over "
                                            "AWGN and lognormal fading.")
    sub = p.add_subparsers(dest="cmd", required=True)
    sw = sub.add_parser("sweep", help="analytic (and optional Monte Carlo) BER vs Eb/N0 sweep")
    sw.add_argument("--mod", nargs="+", choices=MODULATIONS, required=True)
    sw.add_argument("--snr", required=True,
                    help="Eb/N0 sweep in dB: start:stop:step (inclusive) or comma list")
    sw.add_argument("--channel", choices=CHANNELS, default="awgn")
    sw.add_argument("--sigma-i2", type=float, default=None, dest="sigma_i2",
                    help="scintillation index sigma_I^2 (lognormal channel only)")
    sw.add_argument("--M", type=int, default=4, help="PPM alphabet size (power of two)")
    sw.add_argument("--threshold", default="optimal",
                    help="OOK threshold: 'optimal' or fixed fraction in (0,1)")
    sw.add_argument("--mc", action="store_true", help="add Monte Carlo estimates")
    sw.add_argument("--n", type=int, default=200_000, help="MC bits per SNR point")
    sw.add_argument("--seed", type=int, default=1, help="MC seed")
    sw.add_argument("--ci", type=float, default=0.95, help="Wilson CI level")
    sw.add_argument("--max-seconds", type=float, default=120.0,
                    help="hard MC wall-clock budget per modulation (s)")
    sw.add_argument("--png", default=None, help="save a BER waterfall plot to this path")
    sw.set_defaults(func=_run_sweep)

    args = p.parse_args(argv)
    if args.cmd == "sweep" and args.threshold != "optimal":
        try:
            args.threshold = float(args.threshold)
        except ValueError:
            raise SystemExit(f"error: --threshold must be 'optimal' or a float, "
                             f"got {args.threshold!r}") from None
    try:
        return args.func(args)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from None


if __name__ == "__main__":
    sys.exit(main())
