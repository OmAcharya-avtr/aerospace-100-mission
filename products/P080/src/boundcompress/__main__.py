"""Command-line interface: ``python -m boundcompress``.

Subcommands
-----------

``info``
    Environment, predictors, bound modes and signal names.
``certify``
    Encode one signal at one declared bound and print the certificate.
``sweep``
    Rate-distortion sweep for one signal across every predictor and the
    standard-library baselines.
``compare``
    Predictor benchmark across every built-in signal, with the learned
    predictors fitted on the disjoint training realisations.
``negative-control``
    Inject a deliberate bound violation into a reconstruction and show the
    certificate catching it.

Every subcommand reads its data from the built-in generator in
:mod:`boundcompress.dataset` or from a single-column text file given with
``--input``, so the CLI has no dependency on a data file that might not be
there.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from . import __version__, baselines, benchmark, dataset, learned, ratedistortion
from . import quantise as quantise_module
from .certificate import certify
from .codec import encode
from .predictors import CLASSICAL_NAMES, noise_gain
from .report import format_certificate, format_table


def _load(args: argparse.Namespace) -> tuple[str, np.ndarray]:
    if args.input is not None:
        path = Path(args.input)
        values = np.loadtxt(path, dtype=np.float64, ndmin=1)
        return path.name, np.ascontiguousarray(values, dtype=np.float64)
    signal = dataset.generate(args.signal, samples=args.samples, seed=args.seed)
    return signal.name, signal.values


def _fit_models(samples: int) -> dict[str, learned.LearnedPredictor]:
    arrays: list[np.ndarray] = []
    for signal in dataset.training_signals(samples=samples).values():
        span = float(signal.values.max() - signal.values.min())
        if span <= 0.0:  # pragma: no cover - no built-in signal is constant
            continue
        for fraction in (1e-3, 1e-4):
            spec = quantise_module.make_spec(signal.values, "abs", fraction * span)
            symbols, _, _ = quantise_module.quantise(signal.values, spec)
            arrays.append(symbols)
    return {
        kind: learned.LearnedPredictor(kind=kind).fit(arrays) for kind in learned.KINDS
    }


def _cmd_info(_: argparse.Namespace) -> int:
    rows = [
        ["version", __version__],
        ["bound modes", ", ".join(quantise_module.MODES)],
        ["classical predictors", ", ".join(CLASSICAL_NAMES)],
        ["predictor noise gains", ", ".join(f"{n}={noise_gain(n)}" for n in CLASSICAL_NAMES)],
        ["learned kinds", ", ".join(learned.KINDS)],
        ["signals", ", ".join(dataset.SIGNAL_NAMES)],
        ["baseline coders", ", ".join(baselines.LOSSY_METHODS)],
        ["lossless references", ", ".join(baselines.LOSSLESS_METHODS)],
        ["numcodecs / zfpy", "not installed; no number here comes from either"],
    ]
    print(format_table(["item", "value"], rows))
    return 0


def _cmd_certify(args: argparse.Namespace) -> int:
    name, values = _load(args)
    model = None
    if args.predictor == "learned":
        model = _fit_models(args.samples)[args.learned_kind]
    result = encode(values, args.mode, args.bound, predictor=args.predictor, model=model)
    print(f"signal                 : {name}")
    print(f"samples                : {values.size}")
    print(format_certificate(result.certificate))
    return 0 if result.certificate.bound_holds else 1


def _cmd_sweep(args: argparse.Namespace) -> int:
    name, values = _load(args)
    bounds = ratedistortion.geometric_bounds(values, count=args.points)
    rows = ratedistortion.sweep_signal(
        name, values, bounds, mode="abs", predictors=CLASSICAL_NAMES
    )
    table = [
        [
            f"{row.bound:.6e}",
            row.coder,
            str(row.encoded_bytes),
            f"{row.bits_per_sample:.4f}",
            f"{row.max_error:.6e}",
            "yes" if row.bound_holds else "NO",
            "yes" if row.roundtrip_ok else "NO",
        ]
        for row in rows
    ]
    print(
        format_table(
            ["bound", "coder", "bytes", "bits/sample", "max_error", "holds", "roundtrip"],
            table,
        )
    )
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    models = _fit_models(args.samples)
    signals = {
        name: signal.values
        for name, signal in dataset.generate_all(samples=args.samples).items()
    }
    bounds = {
        name: ratedistortion.geometric_bounds(values, count=args.points)
        for name, values in signals.items()
    }
    cells = benchmark.run(signals, bounds, dict(models))
    verdict = benchmark.summarise(cells)
    rows = [
        ["classical wins", str(verdict.classical_wins)],
        ["learned wins", str(verdict.learned_wins)],
        ["median learned gain %", f"{verdict.median_learned_gain_pct:.4f}"],
        ["max learned gain %", f"{verdict.max_learned_gain_pct:.4f}"],
        ["worst learned loss %", f"{verdict.worst_learned_loss_pct:.4f}"],
        ["best classical overall", verdict.best_classical_overall],
        ["best learned overall", verdict.best_learned_overall],
    ]
    for label, model in models.items():
        rows.append([f"model bytes ({label})", str(model.model_bytes)])
        rows.append(
            [
                f"break-even arrays ({label})",
                f"{benchmark.break_even_arrays(cells, label, model.model_bytes):.2f}",
            ]
        )
    print(format_table(["item", "value"], rows))
    return 0


def _cmd_negative_control(args: argparse.Namespace) -> int:
    name, values = _load(args)
    result = encode(values, args.mode, args.bound, predictor="linear")
    corrupted = result.reconstruction.copy()
    index = int(args.index % corrupted.size)
    if args.mode == "rel":
        corrupted[index] = values[index] * (1.0 + 3.0 * args.bound)
    else:
        corrupted[index] = corrupted[index] + 3.0 * args.bound * max(
            1.0, float(np.abs(values).max())
        )
    certificate = certify(
        values,
        corrupted,
        args.mode,
        args.bound,
        predictor="injected-violation",
        notes=(f"sample {index} moved deliberately outside the declared bound",),
    )
    print(f"signal                 : {name}")
    print(f"injected at index      : {index}")
    print(format_certificate(certificate))
    caught = (not certificate.bound_holds) and index in certificate.violation_indices
    print(f"negative control caught: {caught}")
    return 0 if caught else 1


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser. Exposed so ``tests/test_cli.py`` can drive it."""
    parser = argparse.ArgumentParser(
        prog="boundcompress",
        description=(
            "Certify that a declared per-sample error bound held, and report what the "
            "guarantee cost in bytes. Research grade; not flight-qualified."
        ),
    )
    parser.add_argument("--version", action="version", version=f"boundcompress {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_source(target: argparse.ArgumentParser) -> None:
        target.add_argument(
            "--signal", default="battery_voltage", choices=list(dataset.SIGNAL_NAMES)
        )
        target.add_argument("--input", default=None, help="single-column text file of samples")
        target.add_argument("--samples", type=int, default=dataset.DEFAULT_SAMPLES)
        target.add_argument("--seed", type=int, default=0)

    info = subparsers.add_parser("info", help="environment, predictors and signals")
    info.set_defaults(func=_cmd_info)

    certify_parser = subparsers.add_parser("certify", help="encode and print the certificate")
    add_source(certify_parser)
    certify_parser.add_argument("--mode", default="abs", choices=list(quantise_module.MODES))
    certify_parser.add_argument("--bound", type=float, default=1e-3)
    certify_parser.add_argument(
        "--predictor", default="linear", choices=[*CLASSICAL_NAMES, "learned"]
    )
    certify_parser.add_argument("--learned-kind", default="linear", choices=list(learned.KINDS))
    certify_parser.set_defaults(func=_cmd_certify)

    sweep = subparsers.add_parser("sweep", help="rate-distortion sweep for one signal")
    add_source(sweep)
    sweep.add_argument("--points", type=int, default=5)
    sweep.set_defaults(func=_cmd_sweep)

    compare = subparsers.add_parser("compare", help="learned against classical on rate")
    compare.add_argument("--samples", type=int, default=2048)
    compare.add_argument("--points", type=int, default=4)
    compare.set_defaults(func=_cmd_compare)

    negative = subparsers.add_parser(
        "negative-control", help="inject a violation and show the certificate catching it"
    )
    add_source(negative)
    negative.add_argument("--mode", default="abs", choices=list(quantise_module.MODES))
    negative.add_argument("--bound", type=float, default=1e-3)
    negative.add_argument("--index", type=int, default=1234)
    negative.set_defaults(func=_cmd_negative_control)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit status.

    ``certify`` exits non-zero when the bound does not hold, and
    ``negative-control`` exits non-zero when the injected violation is *not*
    caught, so both are usable as checks in a shell script.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover - exercised through the module path
    sys.exit(main())
