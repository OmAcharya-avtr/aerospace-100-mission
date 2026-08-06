"""Command-line interface: ``python -m jitterscope analyze ...``.

Reads single-channel telemetry from CSV, prints PSD/jitter statistics
and an anomaly report. The leading ``--train-frac`` fraction of the
record is ASSUMED NOMINAL and used to fit the detector — a standard
condition-monitoring convention that the operator must verify.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .detect import BandZScoreBaseline, FeatureExtractor, NominalModel, detect
from .psd import band_rms, cumulative_rms, psd


def _load_csv(path: Path) -> np.ndarray:
    """Load telemetry from CSV: last column of numeric rows, header ok."""
    if not path.exists():
        raise SystemExit(f"error: input file not found: {path}")
    try:
        data = np.genfromtxt(path, delimiter=",", comments="#")
    except Exception as exc:  # pragma: no cover - genfromtxt rarely raises
        raise SystemExit(f"error: could not parse {path}: {exc}") from exc
    if data.ndim == 2:
        data = data[:, -1]
    data = data[np.isfinite(data)] if np.isnan(data[:1]).all() else data
    # Drop a NaN header row artifact if present at the top only.
    while data.size and np.isnan(data[0]):
        data = data[1:]
    if data.size < 64:
        raise SystemExit(f"error: {path} yielded {data.size} samples; need >= 64")
    if not np.all(np.isfinite(data)):
        n_bad = int(np.sum(~np.isfinite(data)))
        raise SystemExit(
            f"error: telemetry contains {n_bad} non-finite samples (NaN/Inf). "
            "jitterscope does not impute gaps; clean the data first."
        )
    return data


def _parse_bands(spec: str, fs: float) -> list[tuple[float, float]]:
    bands = []
    for part in spec.split(","):
        lo_s, _, hi_s = part.partition("-")
        try:
            lo, hi = float(lo_s), float(hi_s)
        except ValueError:
            raise SystemExit(f"error: bad band spec {part!r}; use e.g. 0-10,10-100") from None
        bands.append((lo, min(hi, fs / 2)))
    return bands


def _cmd_analyze(args: argparse.Namespace) -> int:
    x = _load_csv(Path(args.input))
    fs = args.fs
    print(f"jitterscope analyze: {args.input}")
    print(f"  samples: {x.size}  fs: {fs:g} Hz  duration: {x.size / fs:.2f} s")
    print(f"  mean: {np.mean(x):.4e}  std: {np.std(x):.4e} (signal units)")

    f, pxx = psd(x, fs, nperseg=min(x.size, args.nperseg))
    _, cum = cumulative_rms((f, pxx))
    print(f"\nPSD (Welch, hann, nperseg={min(x.size, args.nperseg)}, 50% overlap)")
    print(f"  resolution: {f[1] - f[0]:.3f} Hz   total RMS (Parseval): {cum[-1]:.4e}")
    ipk = int(np.argmax(pxx[1:]) + 1)
    print(f"  dominant peak: {f[ipk]:.2f} Hz at {pxx[ipk]:.3e} u^2/Hz")

    bands = _parse_bands(args.bands, fs)
    rms = band_rms((f, pxx), bands)
    print("\nBand-limited RMS (sigma = sqrt(int PSD df)):")
    print("  band [Hz]          RMS")
    for (lo, hi), r in zip(bands, rms):
        print(f"  {lo:7.1f}-{hi:7.1f}   {r:.4e}")

    ext = FeatureExtractor(fs=fs, window_s=args.window_s)
    n_train = int(x.size * args.train_frac)
    feats_train, _ = ext.transform(x[:n_train])
    if args.detector == "mlp":
        model: NominalModel | BandZScoreBaseline = NominalModel(quantile=args.quantile, seed=0)
    else:
        model = BandZScoreBaseline(quantile=args.quantile)
    model.fit(feats_train)
    result = detect(x, model=model, extractor=ext)

    print(f"\nAnomaly report ({args.detector} detector, threshold = "
          f"q{args.quantile:g} of nominal scores = {result.threshold:.4g})")
    print(f"  training on first {args.train_frac:.0%} of record (ASSUMED nominal)")
    print(f"  windows: {result.scores.size}   flagged: {result.n_anomalous}")
    if result.n_anomalous:
        print("  t_center [s]    score        confidence")
        for tc, s, c in zip(
            result.window_centers_s[result.flags],
            result.scores[result.flags],
            result.confidence[result.flags],
        ):
            print(f"  {tc:10.2f}    {s:.4g}       {c:.3f}")
    else:
        print("  no windows exceeded the threshold")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m jitterscope``."""
    parser = argparse.ArgumentParser(
        prog="python -m jitterscope",
        description="Platform jitter/vibration characterization for optical pointing "
        "(research-grade; not for operational flight use).",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    pa = sub.add_parser("analyze", help="PSD statistics + anomaly report from a CSV record")
    pa.add_argument("--input", required=True, help="CSV file; last column is telemetry")
    pa.add_argument("--fs", type=float, required=True, help="sample rate [Hz]")
    pa.add_argument("--bands", default="0-10,10-50,50-200,200-500",
                    help="comma-separated RMS bands in Hz (default: 0-10,10-50,50-200,200-500)")
    pa.add_argument("--nperseg", type=int, default=1024, help="Welch segment length")
    pa.add_argument("--detector", choices=["baseline", "mlp"], default="baseline",
                    help="anomaly detector (default: baseline z-score)")
    pa.add_argument("--train-frac", type=float, default=0.5,
                    help="leading fraction of record assumed nominal for fitting (default 0.5)")
    pa.add_argument("--window-s", type=float, default=1.0, help="analysis window length [s]")
    pa.add_argument("--quantile", type=float, default=0.995,
                    help="nominal-score quantile for the threshold (default 0.995)")
    pa.set_defaults(func=_cmd_analyze)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
