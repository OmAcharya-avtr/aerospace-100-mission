"""Command line for refractpath. ``python -m refractpath --help``.

Six subcommands, each printing a table and nothing else:

``info``          the environment, the models, the equations and the references
``refract``       one elevation: the verdict first, then the ensemble
``curve``         the refraction and the disagreement over an elevation grid
``floor``         the measured validity floor for a declared tolerance
``dispersion``    the four refractivity formulations across the optical band
``references``    every published reference, checked
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import numpy as np

from . import (
    DEFAULT_TOLERANCE_ARCSEC,
    PUBLISHED_REFERENCES,
    REFRACTIVITY_MODELS,
    REFRACTIVITY_VALIDITY_UM,
    AtmosphericState,
    Observer,
    __version__,
    default_floor_grid,
    format_spread_point,
    format_table,
    format_validity_floor,
    format_verdict,
    refraction,
    refractivity_spread,
    spread_curve,
    validity_floor,
)
from .disagreement import default_ensemble
from .models import CLOSED_FORM_MODELS, plane_parallel_floor_deg
from .verdict import UNDETERMINABLE


def _state_from_args(args: argparse.Namespace) -> AtmosphericState:
    return AtmosphericState(
        pressure_hpa=args.pressure,
        temperature_c=args.temperature,
        relative_humidity=args.humidity,
        wavelength_um=args.wavelength,
        co2_ppm=args.co2,
        surface_lapse_rate_k_per_m=args.lapse / 1000.0,
        surface_layer_top_m=args.inversion_depth,
    )


def _add_state_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pressure", type=float, default=1013.25, help="station pressure, hPa")
    parser.add_argument(
        "--temperature", type=float, default=10.0, help="air temperature, degrees C"
    )
    parser.add_argument(
        "--humidity", type=float, default=0.0, help="relative humidity as a fraction, 0 to 1"
    )
    parser.add_argument(
        "--wavelength", type=float, default=0.55, help="vacuum wavelength, micrometres"
    )
    parser.add_argument("--co2", type=float, default=450.0, help="CO2 mole fraction, ppm")
    parser.add_argument(
        "--lapse",
        type=float,
        default=-6.5,
        help="surface lapse rate, K/km (US Standard Atmosphere 1976 is -6.5)",
    )
    parser.add_argument(
        "--inversion-depth",
        type=float,
        default=None,
        help="depth in metres over which --lapse applies; above it the 1976 "
        "tropospheric rate resumes. Omit to apply --lapse to the whole troposphere.",
    )
    parser.add_argument(
        "--observer-height", type=float, default=0.0, help="observer height, metres"
    )
    parser.add_argument(
        "--refractivity",
        choices=sorted(REFRACTIVITY_MODELS),
        default="ciddor_1996",
        help="refractivity formulation for the methods that take one",
    )


def _cmd_info(args: argparse.Namespace) -> int:
    del args
    import platform

    import scipy

    print(f"refractpath {__version__}")
    print(f"python {platform.python_version()}  numpy {np.__version__}  scipy {scipy.__version__}")
    print()
    print("Refraction methods in the disagreement ensemble (equation 22):")
    rows = [
        [member.name, member.kind, f"{member.validity_floor_deg:g}", member.citation]
        for member in default_ensemble()
    ]
    print(format_table(["method", "kind", "floor (deg)", "source"], rows))
    print()
    print("Declared or published validity source, per closed form:")
    print(
        format_table(
            ["method", "equation", "source of the floor"],
            [[m.name, m.equation, m.validity_source] for m in CLOSED_FORM_MODELS],
        )
    )
    print()
    print("Refractivity formulations (equations 5 to 12):")
    print(
        format_table(
            ["formulation", "published range (um)", "source of the range"],
            [
                [name, f"{low:g} to {high:g}", source]
                for name, (low, high, source) in REFRACTIVITY_VALIDITY_UM.items()
            ],
        )
    )
    print()
    print("Published references this package is scored against:")
    print(
        format_table(
            ["key", "kind", "quantity", "value", "tolerance"],
            [
                [r.key, r.kind, r.quantity, f"{r.value:.6g}", f"{r.tolerance:.3g}"]
                for r in PUBLISHED_REFERENCES
            ],
        )
    )
    print()
    print("Cannot be determined from these inputs at any elevation:")
    for item in UNDETERMINABLE:
        print(f"  - {item}")
    print()
    print(
        "Research grade. Not flight-qualified, not certified, not approved for\n"
        "operational aerospace use."
    )
    return 0


def _cmd_refract(args: argparse.Namespace) -> int:
    state = _state_from_args(args)
    observer = Observer(height_m=args.observer_height)
    result = refraction(
        args.elevation,
        state,
        observer,
        tolerance_arcsec=args.tolerance,
        refractivity_model=args.refractivity,
    )
    print(format_verdict(result.verdict))
    print()
    print(
        f"reference method            : {result.reference_method}, "
        f"{result.reference_refraction_arcsec:.5f} arcsec"
    )
    print(f"true elevation              : {result.true_elevation_deg:.6f} deg")
    print(
        "profile envelope members    : "
        + ", ".join(
            f"{name} {value:.4f}" for name, value in result.profile_envelope_members.items()
        )
    )
    print()
    from .disagreement import ensemble_at

    print(
        format_spread_point(
            ensemble_at(
                args.elevation,
                state,
                observer,
                default_ensemble(),
                args.refractivity,
            )
        )
    )
    print()
    print("Refractivity at the observer, n - 1:")
    print(
        format_table(
            ["formulation", "n - 1"],
            [
                [name, f"{value:.8e}"]
                for name, value in result.refractivity_at_observer.items()
                if name not in {"spread", "relative_spread"}
            ]
            + [
                ["spread", f"{result.refractivity_at_observer['spread']:.3e}"],
                [
                    "relative spread",
                    f"{result.refractivity_at_observer['relative_spread']:.3e}",
                ],
            ],
        )
    )
    return 0


def _cmd_curve(args: argparse.Namespace) -> int:
    state = _state_from_args(args)
    observer = Observer(height_m=args.observer_height)
    grid = default_floor_grid(args.low, args.high, args.points)
    points = spread_curve(grid, state, observer, default_ensemble(), args.refractivity)
    names = [member.name for member in default_ensemble()]
    rows = []
    for point in points:
        row = [f"{point.elevation_deg:8.4f}"]
        for name in names:
            value = point.values_arcsec.get(name)
            row.append("undefined" if value is None else f"{value:12.4f}")
        row.append(f"{point.spread_arcsec:11.4f}")
        rows.append(row)
    print(format_table(["elevation", *names, "spread"], rows))
    return 0


def _cmd_floor(args: argparse.Namespace) -> int:
    state = _state_from_args(args)
    observer = Observer(height_m=args.observer_height)
    floor = validity_floor(
        state,
        args.tolerance,
        observer,
        default_ensemble(),
        args.refractivity,
    )
    print(format_validity_floor(floor))
    print()
    print(
        f"plane-parallel analytic floor: "
        f"{plane_parallel_floor_deg(state, args.refractivity):.5f} deg"
    )
    print()
    print(
        "No correction from this package is suitable for operational pointing below\n"
        "the measured validity floor above."
    )
    return 0


def _cmd_dispersion(args: argparse.Namespace) -> int:
    state = _state_from_args(args)
    wavelengths = np.linspace(args.low_um, args.high_um, args.points)
    names = list(REFRACTIVITY_MODELS)
    rows = []
    for wavelength in wavelengths:
        values = refractivity_spread(
            float(wavelength),
            state.pressure_pa,
            state.temperature_c,
            state.vapour_pressure_pa,
            state.co2_ppm,
        )
        rows.append(
            [f"{wavelength:7.4f}"]
            + [f"{values[name]:.8e}" for name in names]
            + [f"{values['spread']:.3e}", f"{values['relative_spread']:.3e}"]
        )
    print(format_table(["lambda (um)", *names, "spread", "rel spread"], rows))
    return 0


def _cmd_references(args: argparse.Namespace) -> int:
    del args
    from .reference_checks import run_reference_checks

    comparisons = run_reference_checks()
    rows = []
    for comparison in comparisons:
        reference = comparison.reference
        rows.append(
            [
                reference.key,
                reference.kind,
                f"{reference.value:.6g}",
                f"{comparison.measured:.6g}",
                f"{comparison.difference:+.4g}",
                f"{reference.tolerance:.3g}",
                "PASS" if comparison.within_tolerance else "FAIL",
            ]
        )
    print(
        format_table(
            ["reference", "kind", "published", "measured", "difference", "tolerance", "result"],
            rows,
        )
    )
    failures = [c for c in comparisons if not c.within_tolerance]
    print()
    print(f"checks: {len(comparisons)}   failures: {len(failures)}")
    for comparison in failures:
        print(f"  FAILED {comparison.reference.key}: {comparison.detail}")
        print(f"    note: {comparison.reference.note}")
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        The parser for ``python -m refractpath``.
    """
    parser = argparse.ArgumentParser(
        prog="refractpath",
        description=(
            "Slant-path atmospheric refraction by several independent standard "
            "methods, with their disagreement measured and an explicit validity "
            "floor. Research grade; not flight-qualified, not certified, not "
            "approved for operational aerospace use."
        ),
    )
    parser.add_argument("--version", action="version", version=f"refractpath {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="models, equations, references and limits").set_defaults(
        handler=_cmd_info
    )

    refract = sub.add_parser("refract", help="one elevation, verdict first")
    refract.add_argument("--elevation", type=float, default=5.0, help="apparent elevation, deg")
    refract.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE_ARCSEC,
        help="declared pointing tolerance, arcsec",
    )
    _add_state_arguments(refract)
    refract.set_defaults(handler=_cmd_refract)

    curve = sub.add_parser("curve", help="refraction and disagreement over elevation")
    curve.add_argument("--low", type=float, default=0.0, help="lowest elevation, deg")
    curve.add_argument("--high", type=float, default=90.0, help="highest elevation, deg")
    curve.add_argument("--points", type=int, default=25, help="grid points")
    _add_state_arguments(curve)
    curve.set_defaults(handler=_cmd_curve)

    floor = sub.add_parser("floor", help="measure the validity floor for a tolerance")
    floor.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE_ARCSEC,
        help="declared pointing tolerance, arcsec",
    )
    _add_state_arguments(floor)
    floor.set_defaults(handler=_cmd_floor)

    dispersion = sub.add_parser("dispersion", help="the four refractivity formulations")
    dispersion.add_argument("--low-um", type=float, default=0.35, help="lowest wavelength, um")
    dispersion.add_argument("--high-um", type=float, default=1.0, help="highest wavelength, um")
    dispersion.add_argument("--points", type=int, default=14, help="grid points")
    _add_state_arguments(dispersion)
    dispersion.set_defaults(handler=_cmd_dispersion)

    sub.add_parser("references", help="check every published reference").set_defaults(
        handler=_cmd_references
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument vector. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit status. Non-zero when a reference check fails or an input
        is rejected.
    """
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return int(args.handler(args))
    except (ValueError, TypeError, KeyError) as error:
        print(f"refractpath: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
