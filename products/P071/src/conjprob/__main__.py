"""Command-line interface: ``python -m conjprob <subcommand>``.

Four subcommands:

``assess``
    Assess a declared conjunction case and print the full report, including the
    disagreement between methods and the dilution diagnosis.
``dilution``
    Print the Pc-against-covariance-scale table for a case, which is the
    dilution curve in text form.
``regimes``
    Print the measured validity boundary of each analytic method: the value of
    ``R / sigma_eff`` at which its relative error against the quadrature
    reference first exceeds a stated tolerance.
``screen``
    Train the learned screening classifier and print its recall at fixed
    false-positive rate beside every analytic screen's.

Nothing in the library prints. All output is produced here.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from collections.abc import Sequence

import numpy as np

from . import __version__
from .cases import load_case
from .dataset import generate_population
from .dilution import DilutionWarning, dilution_curve
from .regimes import METHOD_NAMES, regime_sweep
from .report import assess_conjunction
from .screening import ScreeningClassifier, compare_screens


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="conjprob",
        description=(
            "Collision probability for a close approach by several independent "
            "methods, with their disagreement and validity regimes reported. "
            "Research grade; no output is suitable as an input to a manoeuvre "
            "decision."
        ),
    )
    parser.add_argument("--version", action="version", version=f"conjprob {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    assess = subparsers.add_parser("assess", help="assess a declared conjunction case")
    assess.add_argument("case", help="path to a YAML conjunction case file")
    assess.add_argument(
        "--monte-carlo",
        type=int,
        default=None,
        metavar="N",
        help="also run the 3-D Monte Carlo reference with N samples",
    )
    assess.add_argument("--seed", type=int, default=0, help="Monte Carlo seed")
    assess.add_argument(
        "--chan-order",
        type=int,
        default=None,
        help="Chan series order; the default is the recommended order for the geometry",
    )
    assess.add_argument("--alfano-nodes", type=int, default=201, help="Alfano Simpson nodes")

    dilution = subparsers.add_parser("dilution", help="print the dilution curve for a case")
    dilution.add_argument("case", help="path to a YAML conjunction case file")
    dilution.add_argument(
        "--decades", type=float, default=2.0, help="half-width of the scale sweep in decades"
    )
    dilution.add_argument("--points", type=int, default=21, help="number of scale points")

    regimes = subparsers.add_parser("regimes", help="print measured method validity boundaries")
    regimes.add_argument(
        "--tolerance", type=float, default=1e-3, help="relative-error tolerance"
    )
    regimes.add_argument(
        "--mahalanobis", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0]
    )
    regimes.add_argument("--aspects", type=float, nargs="+", default=[1.0, 2.0, 5.0, 10.0])
    regimes.add_argument("--points", type=int, default=40, help="points per rho axis")

    screen = subparsers.add_parser("screen", help="score the learned screen against baselines")
    screen.add_argument("--n-train", type=int, default=20000)
    screen.add_argument("--n-test", type=int, default=8000)
    screen.add_argument("--seed", type=int, default=0)
    screen.add_argument("--regime", default="nominal", help="test-set generator regime")
    return parser


def _assess(args: argparse.Namespace, stream) -> int:
    case = load_case(args.case)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = assess_conjunction(
            case.primary,
            case.secondary,
            case.radius_m,
            chan_order=args.chan_order,
            alfano_nodes=args.alfano_nodes,
            monte_carlo_samples=args.monte_carlo,
            monte_carlo_seed=args.seed,
        )
    print(f"case: {case.name}", file=stream)
    print(report.summary(), file=stream)
    for warning in caught:
        if issubclass(warning.category, DilutionWarning):
            print(f"\nDilutionWarning: {warning.message}", file=stream)
    return 0


def _dilution(args: argparse.Namespace, stream) -> int:
    case = load_case(args.case)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DilutionWarning)
        report = assess_conjunction(case.primary, case.secondary, case.radius_m)
    frame = report.geometry.frame
    scales = np.logspace(-args.decades, args.decades, args.points)
    exact, approx = dilution_curve(
        frame.sigma_x_m, frame.sigma_y_m, frame.x_bar_m, frame.y_bar_m, case.radius_m, scales
    )
    print(f"case: {case.name}", file=stream)
    print(
        f"maximum Pc {report.dilution.pc_max_numerical:.6e} at covariance scale "
        f"{report.dilution.scale_at_max_numerical:.6f}; present scale is 1.0",
        file=stream,
    )
    print(f"{'scale':>12s} {'pc_foster':>14s} {'pc_small_body':>14s}", file=stream)
    for scale, value, small in zip(scales, exact, approx, strict=True):
        print(f"{scale:12.6g} {value:14.6e} {small:14.6e}", file=stream)
    return 0


def _regimes(args: argparse.Namespace, stream) -> int:
    rho = np.logspace(-3.0, 0.8, args.points)
    sweep = regime_sweep(
        rho, np.asarray(args.aspects, dtype=float), np.asarray(args.mahalanobis, dtype=float)
    )
    print(
        f"measured R / sigma_eff at which the relative error first exceeds "
        f"{args.tolerance:.3e}",
        file=stream,
    )
    print("nan means the tolerance was never exceeded over the swept range", file=stream)
    for method in METHOD_NAMES:
        boundary = sweep.boundary(method, args.tolerance)
        print(f"\n{method}  (worst error on grid {sweep.worst_error(method):.3e})", file=stream)
        header = "  aspect " + " ".join(f"M={m:<8.3g}" for m in sweep.mahalanobis_values)
        print(header, file=stream)
        for i, aspect in enumerate(sweep.aspect_ratios):
            row = " ".join(f"{boundary[i, j]:<10.4g}" for j in range(boundary.shape[1]))
            print(f"  {aspect:<7.3g} {row}", file=stream)
    return 0


def _screen(args: argparse.Namespace, stream) -> int:
    train = generate_population(args.n_train, seed=args.seed, regime="nominal")
    test = generate_population(args.n_test, seed=args.seed + 1_000, regime=args.regime)
    model = ScreeningClassifier(random_state=args.seed).fit(train.features, train.labels)
    comparison = compare_screens(
        test.labels, model.score_samples(test.features), test.features
    )
    print(
        f"train regime nominal n={len(train)} positives={train.labels.sum()}; "
        f"test regime {args.regime} n={len(test)} positives={test.labels.sum()}",
        file=stream,
    )
    print(
        f"{'screen':<16s} {'fpr target':>11s} {'fpr achieved':>13s} {'recall':>9s} "
        f"{'precision':>10s}",
        file=stream,
    )
    for name in ("miss_distance", "mahalanobis", "small_body", "learned"):
        for point in comparison[name]:
            print(
                f"{name:<16s} {point.fpr_target:11.4f} {point.fpr_achieved:13.5f} "
                f"{point.recall:9.5f} {point.precision:10.5f}",
                file=stream,
            )
    return 0


def main(argv: Sequence[str] | None = None, stream=None) -> int:
    """Entry point. Returns a process exit status; 2 on a usage or input error."""
    target = sys.stdout if stream is None else stream
    parser = _build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "assess": _assess,
        "dilution": _dilution,
        "regimes": _regimes,
        "screen": _screen,
    }
    try:
        return handlers[args.command](args, target)
    except (FileNotFoundError, ValueError, KeyError) as error:
        print(f"conjprob: error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
