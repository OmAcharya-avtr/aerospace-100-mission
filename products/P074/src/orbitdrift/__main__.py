"""Command-line interface: ``python -m orbitdrift``.

Five subcommands, each of which prints numbers and nothing else:

``version``
    Version, the environment the numbers were measured in, and the safety
    statement.
``vectors``
    Reproduce the published SGP4 verification vectors shipped inside the
    installed ``sgp4`` distribution and print the worst disagreement.
``drift``
    Take two element-set lines on the command line or from a file and print a
    divergence curve against epoch age, with the analytic growth forms fitted.
``stale``
    The product's question. Same input plus a position tolerance, and
    optionally the age of the element set in hand, and print the staleness
    verdict.
``atmosphere``
    Print the declared atmosphere table so a user can see what drag model
    their divergence number rests on.

A sixth thing the CLI deliberately does not have is any way to ask for a
covariance; :mod:`orbitdrift.guard` says why, and ``stale --purpose`` is
screened.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sgp4.api import Satrec

from . import SAFETY_STATEMENT, __version__
from .divergence import model_divergence
from .errors import CovarianceRefused, OrbitDriftError
from .features import features_from_satrec
from .forces import ATMOSPHERE_MODEL_NAME, ForceModel, atmospheric_density_kg_m3
from .growth import best_form, fit_all_forms
from .guard import check_purpose
from .report import build_report
from .staleness import staleness_verdict
from .vectors import compare_to_published, published_cases, sgp4_distribution_version

DEMO_TLE = (
    "1 25544U 98067A   26015.50000000  .00016717  00000-0  10270-3 0  9003",
    "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537",
)
"""An element set used when none is supplied.

The catalogue number and the orbit are those of a real object, but the element
values are a plausible made-up set and are **not** a real published element
set for it; nothing in this repository claims to have read the catalogue. It
is here so that ``python -m orbitdrift drift`` runs with no arguments.
"""


def _read_tle(args: argparse.Namespace) -> tuple[str, str]:
    if args.tle_file:
        lines = [
            line.rstrip("\n")
            for line in Path(args.tle_file).read_text().splitlines()
            if line[:2] in ("1 ", "2 ")
        ]
        if len(lines) < 2:
            raise OrbitDriftError(
                f"{args.tle_file} does not contain a line starting '1 ' and a line "
                "starting '2 '"
            )
        return lines[0], lines[1]
    if args.line1 and args.line2:
        return args.line1, args.line2
    return DEMO_TLE


def _command_version(_args: argparse.Namespace) -> int:
    import matplotlib
    import numpy
    import scipy
    import sklearn

    print(f"orbitdrift {__version__}")
    print(f"python       {sys.version.split()[0]}")
    print(f"numpy        {numpy.__version__}")
    print(f"scipy        {scipy.__version__}")
    print(f"scikit-learn {sklearn.__version__}")
    print(f"matplotlib   {matplotlib.__version__}")
    print(f"sgp4         {sgp4_distribution_version()}")
    print("astropy      not installed; no number here is cross-checked against it")
    print("torch        not installed; the learned model is scikit-learn")
    print()
    print(SAFETY_STATEMENT)
    return 0


def _command_vectors(args: argparse.Namespace) -> int:
    cases = published_cases()
    worst_r = 0.0
    worst_v = 0.0
    compared = 0
    skipped = 0
    print(
        f"published SGP4 verification vectors from the installed sgp4 "
        f"{sgp4_distribution_version()} distribution"
    )
    print("  files SGP4-VER.TLE and tcppver.out, published with AIAA 2006-6753")
    print()
    print("  satnum   rows  skipped   max |dr| km   max |dv| km/s")
    for case in cases:
        result = compare_to_published(case)
        compared += result.rows_compared
        skipped += result.rows_skipped_error_code
        worst_r = max(worst_r, result.max_position_error_km)
        worst_v = max(worst_v, result.max_velocity_error_km_s)
        if args.verbose:
            print(
                f"  {result.satnum:6d} {result.rows_compared:6d} "
                f"{result.rows_skipped_error_code:8d} "
                f"{result.max_position_error_km:13.3e} "
                f"{result.max_velocity_error_km_s:15.3e}"
            )
    print()
    print(
        f"  {len(cases)} cases, {compared} rows compared, "
        f"{skipped} rows skipped on an error code"
    )
    print(f"  worst position disagreement {worst_r:.6e} km")
    print(f"  worst velocity disagreement {worst_v:.6e} km/s")
    return 0


def _command_atmosphere(_args: argparse.Namespace) -> int:
    print(f"atmosphere model: {ATMOSPHERE_MODEL_NAME}")
    print("source: Vallado 2013, Fundamentals of Astrodynamics and Applications, Table 8-4")
    print()
    print("  altitude km      density kg/m^3")
    for altitude in (200.0, 300.0, 400.0, 500.0, 600.0, 800.0, 1000.0, 1400.0):
        print(f"  {altitude:11.1f}      {float(atmospheric_density_kg_m3(altitude)):.6e}")
    print()
    print(
        "This model is static: no solar index, no diurnal bulge, no geomagnetic "
        "term. In high solar activity the true density at 400 km can exceed it "
        "by a factor of several, and every drag-driven divergence number in this "
        "repository inherits that."
    )
    return 0


def _command_drift(args: argparse.Namespace) -> int:
    line1, line2 = _read_tle(args)
    satrec = Satrec.twoline2rv(line1, line2)
    curve = model_divergence(
        satrec,
        span_days=args.span_days,
        steps_per_rev=args.steps_per_rev,
        report_every_hours=args.report_every_hours,
        force=ForceModel(drag=not args.no_drag, third_body=not args.no_third_body),
    )
    print(build_report(satrec, curve).summary())
    return 0


def _command_stale(args: argparse.Namespace) -> int:
    # The purpose is screened BEFORE any propagation. A refusal that arrived
    # after two minutes of integration would be a refusal the user had already
    # paid for, and -- worse -- a short span that cannot support a growth fit
    # would raise a fitting error first and hide the refusal entirely, which is
    # what happened before this line was moved to the top.
    check_purpose(args.purpose)
    line1, line2 = _read_tle(args)
    satrec = Satrec.twoline2rv(line1, line2)
    curve = model_divergence(
        satrec,
        span_days=args.span_days,
        steps_per_rev=args.steps_per_rev,
        report_every_hours=args.report_every_hours,
    )
    fits = fit_all_forms(curve.ages_days, curve.drift_km)
    name = best_form(fits)
    fit = fits[name]
    print(f"this object's own divergence curve is best fitted by the {name} form")
    print(f"  amplitude {fit.amplitude_km:.6f} km, exponent {fit.exponent:.4f}, "
          f"log-residual sd {fit.log_residual_std:.5f}")
    print()

    # A single object cannot calibrate a population interval, so the verdict
    # below uses this object's own fit as its own predictor: one object, one
    # curve, no held-out coverage. The measured_coverage is deliberately left
    # as not-a-number so that StalenessVerdict.describe says the horizon has
    # no measured backing, which is the truth for a single element set.
    predictor = _SingleCurvePredictor(fit, name)
    verdict = staleness_verdict(
        predictor,
        features_from_satrec(satrec),
        args.tolerance_km,
        age_days=args.age_days,
        purpose=args.purpose,
        coverage=args.coverage,
    )
    print(verdict.describe())
    return 0


class _SingleCurvePredictor:
    """Adapt one :class:`orbitdrift.growth.GrowthFit` to the predictor protocol."""

    def __init__(self, fit, form_name: str) -> None:
        self._fit = fit
        self.name = f"own_curve_{form_name}"

    def predict_median_km(self, x: np.ndarray) -> np.ndarray:
        return self._fit.predict_km(np.atleast_2d(np.asarray(x, dtype=float))[:, -1])

    def predict_interval_km(
        self, x: np.ndarray, coverage: float = 0.90
    ) -> tuple[np.ndarray, np.ndarray]:
        ages = np.atleast_2d(np.asarray(x, dtype=float))[:, -1]
        return self._fit.interval_km(ages, coverage=coverage)


def _add_tle_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--line1", default=None, help="first element-set line")
    parser.add_argument("--line2", default=None, help="second element-set line")
    parser.add_argument("--tle-file", default=None, help="file holding the two lines")
    parser.add_argument("--span-days", type=float, default=7.0, help="propagation span, days")
    parser.add_argument(
        "--steps-per-rev", type=int, default=360, help="integration steps per revolution"
    )
    parser.add_argument(
        "--report-every-hours", type=float, default=12.0, help="spacing of reported ages"
    )


def build_parser() -> argparse.ArgumentParser:
    """The argument parser, exposed so ``tests/test_cli.py`` can exercise it."""
    parser = argparse.ArgumentParser(
        prog="python -m orbitdrift",
        description=(
            "Measure how fast two propagations of one element set separate, and "
            "turn that into an epoch-age budget."
        ),
        epilog=(
            "orbitdrift measures model divergence between two propagators. It does "
            "not measure ephemeris error against truth, because it has no precision "
            "ephemerides and no element-set archive. No output is a covariance and "
            "no output may be used as the input covariance of a conjunction "
            "assessment; see P071 ConjProb for that computation. Research grade; not "
            "flight-qualified, not certified, not approved for operational "
            "aerospace use."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("version", help="versions and the safety statement").set_defaults(
        handler=_command_version
    )

    vectors = sub.add_parser("vectors", help="reproduce the published SGP4 test vectors")
    vectors.add_argument("--verbose", action="store_true", help="one line per case")
    vectors.set_defaults(handler=_command_vectors)

    sub.add_parser("atmosphere", help="print the declared atmosphere model").set_defaults(
        handler=_command_atmosphere
    )

    drift = sub.add_parser("drift", help="divergence curve for one element set")
    _add_tle_arguments(drift)
    drift.add_argument("--no-drag", action="store_true", help="switch the drag term off")
    drift.add_argument(
        "--no-third-body", action="store_true", help="switch the luni-solar term off"
    )
    drift.set_defaults(handler=_command_drift)

    stale = sub.add_parser("stale", help="is this element set still good enough")
    _add_tle_arguments(stale)
    stale.add_argument(
        "--tolerance-km", type=float, default=5.0, help="position tolerance the decision needs"
    )
    stale.add_argument(
        "--age-days", type=float, default=None, help="age of the element set in hand"
    )
    stale.add_argument("--coverage", type=float, default=0.90, help="nominal interval level")
    stale.add_argument(
        "--purpose",
        default="ephemeris staleness check",
        help="declared purpose; a conjunction or covariance purpose is refused",
    )
    stale.set_defaults(handler=_command_stale)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit status rather than raising."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except CovarianceRefused as error:
        print(f"refused: {error}", file=sys.stderr)
        return 3
    except OrbitDriftError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
