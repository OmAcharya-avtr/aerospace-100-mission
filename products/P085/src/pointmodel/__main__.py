"""Command line entry point: ``python -m pointmodel <command>``.

Commands
--------
``terms``           print the declared term set and its design-matrix columns
``decompose``       fit a synthetic residual set and print the whole budget
``identify``        print the rank verdict for one declared geometry
``compare``         print the rank verdict for every declared geometry
``refraction``      compare Bennett's formula with the independent ray trace
``outliers``        fit a contaminated set three ways and print the diagnostic

Every command takes its data from :mod:`pointmodel.synthetic`, because this
package has no flight data and will not pretend to. A caller with real
residuals uses :func:`pointmodel.decompose` directly; the shape of the call is
in the ``decompose`` command's source below and in ``README.md``.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from . import __version__
from .budget import decompose
from .identifiability import DEFAULT_MARGINAL_THRESHOLD, DEFAULT_NULL_THRESHOLD
from .refraction import K_OPTICAL_550NM, K_RADIO_DRY, compare_refraction
from .report import (
    format_table,
    render_budget,
    render_disagreement,
    render_identifiability,
    render_robust,
    render_terms,
)
from .synthetic import GEOMETRIES, generate


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pointmodel",
        description=(
            "Decompose a pointing error budget from measured residuals, with a "
            "confidence interval on every term and an explicit statement of which "
            "terms the observed geometry cannot separate."
        ),
    )
    parser.add_argument("--version", action="version", version=f"pointmodel {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("terms", help="print the declared term set")

    def add_common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "--geometry",
            default="full_sky",
            choices=sorted(GEOMETRIES),
            help="declared observing geometry (default: full_sky)",
        )
        sub.add_argument(
            "--observations", type=int, default=300, help="number of pointings (default: 300)"
        )
        sub.add_argument(
            "--sigma", type=float, default=2.0, help="residual sigma, arcsec (default: 2.0)"
        )
        sub.add_argument("--seed", type=int, default=20260101, help="geometry and outlier seed")
        sub.add_argument(
            "--null-threshold",
            type=float,
            default=DEFAULT_NULL_THRESHOLD,
            help=f"singular-value ratio below which a direction is refused "
            f"(default: {DEFAULT_NULL_THRESHOLD:g})",
        )
        sub.add_argument(
            "--marginal-threshold",
            type=float,
            default=DEFAULT_MARGINAL_THRESHOLD,
            help=f"ratio below which a direction is flagged weak "
            f"(default: {DEFAULT_MARGINAL_THRESHOLD:g})",
        )

    budget = subparsers.add_parser("decompose", help="fit a synthetic set and print the budget")
    add_common(budget)
    budget.add_argument(
        "--outlier-fraction",
        type=float,
        default=0.0,
        help="fraction of acquisitions given a failure offset (default: 0.0)",
    )
    budget.add_argument(
        "--outlier-arcsec", type=float, default=300.0, help="failure magnitude, arcsec"
    )
    budget.add_argument(
        "--no-robust", action="store_true", help="skip the Huber and RANSAC fits"
    )

    identify = subparsers.add_parser("identify", help="rank verdict for one geometry")
    add_common(identify)

    compare = subparsers.add_parser("compare", help="rank verdict for every declared geometry")
    add_common(compare)

    refraction = subparsers.add_parser(
        "refraction", help="Bennett's formula against the independent ray trace"
    )
    refraction.add_argument(
        "--low", type=float, default=10.0, help="lowest true elevation, deg (default: 10)"
    )
    refraction.add_argument(
        "--high", type=float, default=85.0, help="highest true elevation, deg (default: 85)"
    )
    refraction.add_argument(
        "--steps", type=int, default=16, help="grid points (default: 16)"
    )
    refraction.add_argument(
        "--pressure", type=float, default=1010.0, help="site pressure, hPa (default: 1010)"
    )
    refraction.add_argument(
        "--temperature", type=float, default=10.0, help="site temperature, C (default: 10)"
    )

    outliers = subparsers.add_parser("outliers", help="contaminated fit, three ways")
    add_common(outliers)
    outliers.add_argument(
        "--outlier-fraction", type=float, default=0.06, help="failure fraction (default: 0.06)"
    )
    outliers.add_argument(
        "--outlier-arcsec", type=float, default=300.0, help="failure magnitude, arcsec"
    )
    return parser


def _decompose(args: argparse.Namespace, robust: bool) -> None:
    residuals = generate(
        args.geometry,
        args.observations,
        sigma_arcsec=args.sigma,
        outlier_fraction=getattr(args, "outlier_fraction", 0.0),
        outlier_arcsec=getattr(args, "outlier_arcsec", 300.0),
        seed=args.seed,
    )
    budget = decompose(
        residuals.design,
        residuals.observed,
        sigma_arcsec=args.sigma,
        null_threshold=args.null_threshold,
        marginal_threshold=args.marginal_threshold,
        robust=robust,
    )
    print(f"geometry            : {residuals.name}")
    print(f"observations        : {residuals.n_observations}  rows {residuals.observed.size}")
    print(f"injected failures   : {int(residuals.outlier_mask.sum())} acquisitions at "
          f"{residuals.outlier_arcsec:g} arcsec")
    source = "design matrix" if residuals.linear_truth else "exact rotation model"
    print(f"truth generated by  : {source}")
    print("")
    print(render_budget(budget))
    print("")
    print("TRUTH, for reference. This is a synthetic set; the generating coefficients")
    print("are declared test values in pointmodel.synthetic and are not a measurement")
    print("of any real instrument.")
    print("")
    truth = residuals.truth_mapping()
    print(
        format_table(
            ["term", "truth arcsec", "fitted arcsec", "error arcsec", "separable"],
            [
                [
                    row.name,
                    f"{truth[row.name]:.4f}",
                    f"{row.coefficient:.4f}",
                    f"{row.coefficient - truth[row.name]:+.4f}",
                    "yes" if row.separable else "REFUSED",
                ]
                for row in budget.terms
            ],
        )
    )


def _identify(args: argparse.Namespace) -> None:
    residuals = generate(
        args.geometry, args.observations, sigma_arcsec=args.sigma, seed=args.seed
    )
    budget = decompose(
        residuals.design,
        residuals.observed,
        sigma_arcsec=args.sigma,
        null_threshold=args.null_threshold,
        marginal_threshold=args.marginal_threshold,
        robust=False,
    )
    print(f"geometry            : {residuals.name}")
    print(f"observations        : {residuals.n_observations}")
    print("")
    print(render_identifiability(budget.identifiability))


def _compare(args: argparse.Namespace) -> None:
    rows = []
    for name in sorted(GEOMETRIES):
        residuals = generate(name, args.observations, sigma_arcsec=args.sigma, seed=args.seed)
        budget = decompose(
            residuals.design,
            residuals.observed,
            sigma_arcsec=args.sigma,
            null_threshold=args.null_threshold,
            marginal_threshold=args.marginal_threshold,
            robust=False,
        )
        report = budget.identifiability
        rows.append(
            [
                name,
                f"{report.condition_number:.4g}",
                str(report.numerical_rank),
                report.verdict,
                ",".join(report.unidentifiable_terms) or "-",
                f"{float(np.max(report.vif)):.4g}",
            ]
        )
    print("RANK VERDICT BY DECLARED GEOMETRY")
    print("")
    print(f"observations per geometry : {args.observations}")
    print(f"declared sigma            : {args.sigma:g} arcsec")
    print(f"thresholds                : null <= {args.null_threshold:g}, "
          f"marginal <= {args.marginal_threshold:g}")
    print("")
    print(
        format_table(
            ["geometry", "condition", "rank", "verdict", "terms refused", "max VIF"], rows
        )
    )


def _refraction(args: argparse.Namespace) -> None:
    grid = np.linspace(args.low, args.high, args.steps)
    print("REFRACTION CROSS-CHECK")
    print("")
    print("Bennett (1982) as published by Meeus (1998) ch.16 eq (16.4), against a")
    print("numerical integration of the refraction integral through a US Standard")
    print("Atmosphere 1976 profile, written in this repository. No refraction package")
    print("is installed in the measured environment and none is called.")
    print("")
    print(f"site conditions : {args.pressure:g} hPa, {args.temperature:g} C")
    print("")
    for label, constant in (("radio dry (77.6 K/hPa)", K_RADIO_DRY),
                            ("optical 550 nm", K_OPTICAL_550NM)):
        result = compare_refraction(grid, args.pressure, args.temperature, constant)
        shape_relative = (
            result.shape_bennett - result.shape_raytrace
        ) / result.shape_raytrace
        print(f"ray-trace refractivity constant : {label} = {constant:.4f} K/hPa")
        print(
            format_table(
                ["el deg", "bennett arcsec", "raytrace arcsec", "diff arcsec",
                 "shape bennett", "shape raytrace", "shape rel diff"],
                [
                    [
                        f"{result.elevation_deg[i]:.2f}",
                        f"{result.bennett_arcsec[i]:.4f}",
                        f"{result.raytrace_arcsec[i]:.4f}",
                        f"{result.bennett_arcsec[i] - result.raytrace_arcsec[i]:+.4f}",
                        f"{result.shape_bennett[i]:.6f}",
                        f"{result.shape_raytrace[i]:.6f}",
                        f"{shape_relative[i]:+.3e}",
                    ]
                    for i in range(result.elevation_deg.size)
                ],
            )
        )
        print("")
        print(f"  max |absolute difference| : {result.max_absolute_arcsec:.4f} arcsec")
        print(f"  mean bennett / raytrace   : {result.mean_ratio:.6f}")
        print(f"  max relative shape diff   : {result.max_shape_relative:.4e}")
        print("")
    print("The shape is what the pointing model uses: the fitted RF coefficient carries")
    print("the scale, so the scale difference is absorbed and the shape difference is")
    print("not. The shape comparison is independent of the refractivity constant.")


def _outliers(args: argparse.Namespace) -> None:
    residuals = generate(
        args.geometry,
        args.observations,
        sigma_arcsec=args.sigma,
        outlier_fraction=args.outlier_fraction,
        outlier_arcsec=args.outlier_arcsec,
        seed=args.seed,
    )
    budget = decompose(
        residuals.design,
        residuals.observed,
        sigma_arcsec=args.sigma,
        null_threshold=args.null_threshold,
        marginal_threshold=args.marginal_threshold,
        robust=True,
    )
    truth = residuals.coefficients
    print(f"geometry            : {residuals.name}")
    print(f"observations        : {residuals.n_observations}")
    print(f"injected failures   : {int(residuals.outlier_mask.sum())} acquisitions at "
          f"{residuals.outlier_arcsec:g} arcsec")
    print("")
    assert budget.huber is not None and budget.ransac is not None
    assert budget.huber_disagreement is not None and budget.ransac_disagreement is not None
    print(
        format_table(
            ["term", "truth", "least squares", "huber", "ransac", "ls error", "huber error",
             "ransac error"],
            [
                [
                    name,
                    f"{truth[i]:.4f}",
                    f"{budget.least_squares.coefficients[i]:.4f}",
                    f"{budget.huber.coefficients[i]:.4f}",
                    f"{budget.ransac.coefficients[i]:.4f}",
                    f"{budget.least_squares.coefficients[i] - truth[i]:+.4f}",
                    f"{budget.huber.coefficients[i] - truth[i]:+.4f}",
                    f"{budget.ransac.coefficients[i] - truth[i]:+.4f}",
                ]
                for i, name in enumerate(row.name for row in budget.terms)
            ],
        )
    )
    print("")
    print(render_robust(budget.huber))
    print("")
    print(render_disagreement(budget.huber_disagreement, "Huber"))
    print("")
    print(render_robust(budget.ransac))
    print("")
    print(render_disagreement(budget.ransac_disagreement, "RANSAC"))


def main(argv: list[str] | None = None) -> int:
    """Run the command line. Returns a process exit status."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "terms":
            print(render_terms())
        elif args.command == "decompose":
            _decompose(args, robust=not args.no_robust)
        elif args.command == "identify":
            _identify(args)
        elif args.command == "compare":
            _compare(args)
        elif args.command == "refraction":
            _refraction(args)
        elif args.command == "outliers":
            _outliers(args)
        else:  # pragma: no cover - argparse enforces the choices
            parser.error(f"unknown command {args.command!r}")
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
