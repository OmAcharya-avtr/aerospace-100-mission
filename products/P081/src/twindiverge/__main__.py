"""Command line for twindiverge: ``python -m twindiverge <subcommand>``.

Every subcommand prints the standing of its numbers beside them, and the
subcommand order below is the order of operations the product exists to
enforce: ``telemetry`` then ``identify`` and ``nullspace`` (baseline 1), then
``attribute`` (baseline 2), then ``profile`` (baseline 3), then ``crb`` and
``learned``. Running ``attribute`` prints baseline 1's verdict above its own
output because an attribution read without that verdict is the mistake this
product is about.

Printing happens only here and in the example and validation scripts; the
library raises and returns, and never prints.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
from scipy import stats

from . import __version__
from .attribution import CONFIDENCE_LEVEL
from .identifiability import (
    PRACTICAL_LIMIT,
    RANK_RTOL,
    analyse,
    analytic_null_basis,
    subspace_angles_deg,
)
from .pipeline import METHODS, analyse_case, learned_study, montecarlo_directions
from .plant import NOMINAL, PARAMETER_NAMES, PARAMETER_UNITS, Campaign
from .profile import GRID_POINTS, profile_parameters
from .report import (
    format_table,
    number,
    render_attribution,
    render_identifiability,
    render_profiles,
    render_telemetry_sets,
)
from .scoring import CONFIDENT_THRESHOLD
from .synthetic import KINDS, generate_cases, observe
from .telemetry import TELEMETRY_SETS, TelemetrySet

SAFETY = (
    "Research grade. Not flight-qualified, not certified, not approved for "
    "operational aerospace use."
)


def _telemetry(name: str) -> TelemetrySet:
    if name not in TELEMETRY_SETS:
        raise SystemExit(
            f"unknown telemetry set {name!r}; declared sets are "
            f"{', '.join(TELEMETRY_SETS)}"
        )
    return TELEMETRY_SETS[name]


def _header(title: str) -> None:
    print("=" * 78)
    print(title)
    print("=" * 78)


def command_telemetry(_arguments: argparse.Namespace) -> int:
    """List the declared telemetry sets and the nominal parameters."""
    _header("twindiverge -- the declared telemetry sets")
    print(render_telemetry_sets(TELEMETRY_SETS))
    print()
    print(
        format_table(
            ["parameter", "nominal", "unit"],
            [
                [name, number(value), unit]
                for name, value, unit in zip(
                    PARAMETER_NAMES, NOMINAL.as_array(), PARAMETER_UNITS, strict=True
                )
            ],
        )
    )
    print()
    for name, telemetry in TELEMETRY_SETS.items():
        print(f"{name}:")
        print(f"  {telemetry.note}")
    print()
    print("Parameter values are DECLARED, not measured. See twindiverge.plant.")
    print(SAFETY)
    return 0


def command_identify(arguments: argparse.Namespace) -> int:
    """Baseline 1 on one telemetry set."""
    telemetry = _telemetry(arguments.telemetry)
    _header(f"twindiverge -- baseline 1, local identifiability of {telemetry.name}")
    report = analyse(telemetry, NOMINAL, arguments.rank_rtol, arguments.practical_limit)
    print(render_identifiability(report))
    print()
    print(f"REFUSED      : {', '.join(report.refused) or 'none'}")
    print(f"WEAK         : {', '.join(report.weak) or 'none'}")
    print(f"IDENTIFIABLE : {', '.join(report.identifiable) or 'none'}")
    print()
    print(
        "A REFUSED parameter is structurally unidentifiable from this telemetry set:"
    )
    print("no residual, and no amount of further data of the same kind, can attribute")
    print("anything to it. The identifiable combinations above are what is reported")
    print("in its place. That refusal is an answer, not an error.")
    print()
    print(SAFETY)
    return 0


def command_nullspace(arguments: argparse.Namespace) -> int:
    """The on-paper null space against the computed one."""
    telemetry = _telemetry(arguments.telemetry)
    _header(f"twindiverge -- derived against computed null space, {telemetry.name}")
    report = analyse(telemetry, NOMINAL)
    derived = analytic_null_basis(telemetry)
    angles = subspace_angles_deg(derived, report.null_basis)
    print("The derived basis is algebra from identities A and B of twindiverge.plant.")
    print("It never touches a sensitivity matrix. The computed basis is the right")
    print("singular vectors of equation (8) below the rank cutoff. They must agree.")
    print()
    print(f"derived dimension  : {derived.shape[0]}")
    print(f"computed dimension : {report.null_basis.shape[0]}")
    print()
    if derived.size:
        print("derived basis (orthonormal rows):")
        print()
        print(
            format_table(
                ["direction", *PARAMETER_NAMES],
                [
                    [f"d{index + 1}", *[f"{value:+.12f}" for value in row]]
                    for index, row in enumerate(derived)
                ],
            )
        )
        print()
    if report.null_basis.size:
        print("computed basis (orthonormal rows):")
        print()
        print(
            format_table(
                ["direction", *PARAMETER_NAMES],
                [
                    [f"c{index + 1}", *[f"{value:+.12f}" for value in row]]
                    for index, row in enumerate(report.null_basis)
                ],
            )
        )
        print()
    if angles.size:
        print("principal angles between the two subspaces, degrees:")
        for index, angle in enumerate(angles):
            print(f"  angle {index + 1} : {angle:.12e}")
        print()
        print(f"worst principal angle : {float(angles.max()):.12e} degrees")
    else:
        print("both subspaces are empty: nothing to compare, and nothing is refused.")
    print()
    print("Exact in exact arithmetic; the angles above are the measured rounding.")
    print(SAFETY)
    return 0


def command_attribute(arguments: argparse.Namespace) -> int:
    """Baseline 2 on one synthetic case, with baseline 1's verdict above it."""
    telemetry = _telemetry(arguments.telemetry)
    case = generate_cases((arguments.seed,), arguments.kind)[0]
    analysis = analyse_case(telemetry, case)
    _header(
        f"twindiverge -- baselines 1 and 2 on case seed {case.seed}, kind {case.kind}"
    )
    print(f"true perturbation (log parameters) : "
          f"{np.array2string(case.delta_log, precision=6)}")
    print(f"true label                         : {case.label_name}")
    print(f"perturbation norm                  : {case.magnitude:.6f}")
    print()
    print("-- baseline 1, the gate ------------------------------------------------")
    print(render_identifiability(analysis.identifiability))
    print()
    print("-- baseline 2, attribution ---------------------------------------------")
    print(render_attribution(analysis.attribution))
    print()
    print(f"what this product delivers : {analysis.gated_answer}")
    print(f"the truth                  : {analysis.truth_name}")
    print()
    print(SAFETY)
    return 0


def command_profile(arguments: argparse.Namespace) -> int:
    """Baseline 3 on one synthetic case."""
    telemetry = _telemetry(arguments.telemetry)
    case = generate_cases((arguments.seed,), arguments.kind)[0]
    report = analyse(telemetry, NOMINAL)
    critical = float(stats.norm.ppf(0.5 + CONFIDENCE_LEVEL / 2.0))
    observed = observe(telemetry, case)
    _header(
        f"twindiverge -- baseline 3, profile likelihood on {telemetry.name}, "
        f"seed {case.seed}"
    )
    print(f"true perturbation (log parameters) : "
          f"{np.array2string(case.delta_log, precision=6)}")
    print(f"grid points per profile            : {arguments.points}")
    print()
    started = time.perf_counter()
    results = profile_parameters(
        telemetry,
        observed,
        critical * report.pinv_diagonal_not_a_bound,
        points=arguments.points,
    )
    elapsed = time.perf_counter() - started
    print(render_profiles(results))
    print()
    flat = [result.label for result in results if result.flat]
    overrides = [result.label for result in results if result.overrides_covariance]
    print(f"flat profiles                  : {', '.join(flat) or 'none'}")
    print(f"profiles overriding covariance : {', '.join(overrides) or 'none'}")
    print(f"wall clock                     : {elapsed:.3f} s")
    print()
    print("A flat profile means the data contain no information about that direction.")
    print("Where the linearised covariance nevertheless reports a finite half-width,")
    print("the covariance is wrong and the profile is the reference.")
    print()
    print(SAFETY)
    return 0


def command_crb(arguments: argparse.Namespace) -> int:
    """The Cramer-Rao bound against the measured spread of repeated fits."""
    telemetry = _telemetry(arguments.telemetry)
    seeds = tuple(range(arguments.first_seed, arguments.first_seed + arguments.realisations))
    _header(
        f"twindiverge -- Cramer-Rao bound against measured spread, {telemetry.name}"
    )
    print("The bound comes from the Fisher information of equation (9); the spread")
    print("comes from repeated NONLINEAR fits to independent noise realisations of the")
    print("same truth. The bound must lie below the spread. Neither computation uses")
    print("the other.")
    print()
    started = time.perf_counter()
    amplitudes, bound = montecarlo_directions(telemetry, NOMINAL, seeds)
    elapsed = time.perf_counter() - started
    report = analyse(telemetry, NOMINAL)
    spread = amplitudes.std(axis=0, ddof=1)
    rows = []
    for index in range(report.rank):
        rows.append(
            [
                f"direction {index + 1}",
                number(float(bound[index])),
                number(float(spread[index])),
                f"{float(spread[index] / bound[index]):.6f}",
                "ok" if spread[index] >= bound[index] * 0.5 else "BELOW",
            ]
        )
    print(
        format_table(
            ["direction", "CRB", "measured sd", "ratio", "verdict"],
            rows,
        )
    )
    print()
    print(f"realisations : {len(seeds)}")
    print(f"wall clock   : {elapsed:.3f} s")
    print()
    print("The verdict column uses a crude half-bound margin; the proper test, with a")
    print("slack that follows the conditioning of the bound expression, is in")
    print("validation/validate_independent_bound.py.")
    print()
    print(SAFETY)
    return 0


def command_sweep(arguments: argparse.Namespace) -> int:
    """Record length against structural and practical identifiability."""
    _header("twindiverge -- structural rank is a property of the model; practical is not")
    rows = []
    for samples in (101, 201, 401, 801, 1201, 2001):
        campaign = Campaign(
            f"sweep{samples}", 0.1, samples, 1.0, 0.1 * round(0.2 * (samples - 1))
        )
        telemetry = TelemetrySet(f"sweep{samples}", campaign, ("rate",))
        report = analyse(telemetry, NOMINAL)
        ratios = report.singular_values / report.singular_values[0]
        rows.append(
            [
                f"{campaign.duration_s:.1f}",
                str(samples),
                str(report.rank),
                f"{float(ratios[-1]):.3e}",
                number(float(report.direction_crb_relative[-1])),
                ", ".join(report.refused) or "none",
            ]
        )
    print(
        format_table(
            [
                "length [s]",
                "samples",
                "rank",
                "least ratio",
                "worst direction CRB",
                "refused",
            ],
            rows,
        )
    )
    print()
    print("The rank and the refusal list do not move. The worst identifiable")
    print("direction's bound moves by orders of magnitude. That is the difference")
    print("between structural and practical identifiability, measured.")
    print()
    print(f"declared rank cutoff    : {RANK_RTOL:.1e} (relative)")
    print(f"declared practical limit: {PRACTICAL_LIMIT:.2f} (relative standard error)")
    print()
    print(SAFETY)
    _ = arguments
    return 0


def command_learned(arguments: argparse.Namespace) -> int:
    """The learned study: accuracy, confidence and behaviour inside the null space."""
    telemetry = _telemetry(arguments.telemetry)
    if arguments.train_runs < 10:
        raise SystemExit("--train-runs must be at least 10 to fit five classes")
    if arguments.test_runs < 5 or arguments.null_runs < 5:
        raise SystemExit("--test-runs and --null-runs must each be at least 5")
    train = tuple(range(1000, 1000 + arguments.train_runs))
    test = tuple(range(20000, 20000 + arguments.test_runs))
    null = tuple(range(30000, 30000 + arguments.null_runs))
    _header(f"twindiverge -- the learned classifier against the baselines, {telemetry.name}")
    study = learned_study(telemetry, train, test, null)
    print(f"training cases (single-parameter only) : {study.train_cases}")
    print(f"test cases per population              : {study.test_cases}")
    print(f"confident threshold                    : {CONFIDENT_THRESHOLD:.2f}")
    print()
    print("computation budget, measured on this run:")
    print()
    print(
        format_table(
            ["item", "seconds"],
            [
                ["analytic setup, once per telemetry set", f"{study.setup_seconds:.6f}"],
                ["learned fit, once per training set", f"{study.fit_seconds:.6f}"],
                [
                    f"analytic attribution per case (mean of {study.budget_cases})",
                    f"{study.analytic_seconds_per_case:.6f}",
                ],
                [
                    f"learned prediction per case (mean of {study.budget_cases})",
                    f"{study.learned_seconds_per_case:.6f}",
                ],
            ],
        )
    )
    for block in study.populations:
        print()
        print(f"-- population {block.population} ({block.cases} cases) "
              f"-----------------------------")
        print(f"detected as a real change : {block.detection_rate}")
        print()
        print(
            format_table(
                ["method", "accuracy", "mean confidence", "confident and wrong",
                 "refusal rate", "ambiguity declared"],
                [
                    [
                        method,
                        f"{block.methods[method].accuracy.value:.6f}",
                        number(block.methods[method].mean_confidence),
                        f"{block.methods[method].confident_error.value:.6f}",
                        f"{block.methods[method].refusal_rate.value:.6f}",
                        f"{block.methods[method].ambiguity_declared.value:.6f}",
                    ]
                    for method in METHODS
                ],
            )
        )
    print()
    print(f"wall clock : {study.total_seconds:.3f} s")
    print()
    print("The population named 'equivalent' is inside the unidentifiable subspace:")
    print("each case produces telemetry identical to a single-parameter fault, so no")
    print("single-parameter answer is correct. The analytic path refuses there and")
    print("prints the family it cannot distinguish; the learned classifier answers")
    print("with high confidence and prints nothing.")
    print()
    print("This model is not certified for operational flight use.")
    print(SAFETY)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """The argument parser. Separated so the tests can exercise it directly."""
    parser = argparse.ArgumentParser(
        prog="twindiverge",
        description=(
            "Attribute a twin-versus-reality residual to a parameter subset, after "
            "reporting which parameters the supplied telemetry cannot identify at all."
        ),
    )
    parser.add_argument("--version", action="version", version=f"twindiverge {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "telemetry", help="list the declared telemetry sets and nominal parameters"
    ).set_defaults(handler=command_telemetry)

    identify = subparsers.add_parser(
        "identify", help="baseline 1: structural and practical identifiability"
    )
    identify.add_argument("--telemetry", default="rate_full")
    identify.add_argument("--rank-rtol", type=float, default=RANK_RTOL)
    identify.add_argument("--practical-limit", type=float, default=PRACTICAL_LIMIT)
    identify.set_defaults(handler=command_identify)

    nullspace = subparsers.add_parser(
        "nullspace", help="the derived null space against the computed one"
    )
    nullspace.add_argument("--telemetry", default="rate_full")
    nullspace.set_defaults(handler=command_nullspace)

    attribute_parser = subparsers.add_parser(
        "attribute", help="baseline 2: attribution on the identifiable subspace"
    )
    attribute_parser.add_argument("--telemetry", default="rate_full")
    attribute_parser.add_argument("--seed", type=int, default=20000)
    attribute_parser.add_argument("--kind", choices=KINDS, default="single")
    attribute_parser.set_defaults(handler=command_attribute)

    profile_parser = subparsers.add_parser(
        "profile", help="baseline 3: the nonlinear profile likelihood"
    )
    profile_parser.add_argument("--telemetry", default="rate_full")
    profile_parser.add_argument("--seed", type=int, default=20000)
    profile_parser.add_argument("--kind", choices=KINDS, default="single")
    profile_parser.add_argument("--points", type=int, default=GRID_POINTS)
    profile_parser.set_defaults(handler=command_profile)

    crb = subparsers.add_parser(
        "crb", help="the Cramer-Rao bound against the measured spread of repeated fits"
    )
    crb.add_argument("--telemetry", default="rate_torque_full")
    crb.add_argument("--realisations", type=int, default=24)
    crb.add_argument("--first-seed", type=int, default=50000)
    crb.set_defaults(handler=command_crb)

    subparsers.add_parser(
        "sweep", help="record length against structural and practical identifiability"
    ).set_defaults(handler=command_sweep)

    learned = subparsers.add_parser(
        "learned", help="the learned classifier against the analytic baselines"
    )
    learned.add_argument("--telemetry", default="rate_full")
    learned.add_argument("--train-runs", type=int, default=240)
    learned.add_argument("--test-runs", type=int, default=120)
    learned.add_argument("--null-runs", type=int, default=120)
    learned.set_defaults(handler=command_learned)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit status."""
    parser = build_parser()
    arguments = parser.parse_args(argv)
    points = getattr(arguments, "points", None)
    if points is not None and (points < 3 or points % 2 == 0):
        raise SystemExit("--points must be an odd integer of at least 3")
    realisations = getattr(arguments, "realisations", None)
    if realisations is not None and realisations < 4:
        raise SystemExit("--realisations must be at least 4 for a standard deviation")
    return int(arguments.handler(arguments))


if __name__ == "__main__":
    sys.exit(main())
