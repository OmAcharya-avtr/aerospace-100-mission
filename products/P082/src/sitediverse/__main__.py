"""Command line for sitediverse: ``python -m sitediverse <subcommand>``.

Eight subcommands. Each prints a table and nothing else; no subcommand writes a
file, and none requires a network connection. Every record is regenerated from
its declared seed, so two runs of the same subcommand print the same numbers.

The subcommands are ordered as the analysis is: ``sites`` and ``record`` describe
the geometry and the record, ``climatology`` the estimated marginals,
``overstate`` the headline, ``bounds`` the distribution-free check,
``correlate`` the fitted baselines, ``learned`` the comparison, and ``network``
an arbitrary station set.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from . import __version__
from .bounds import bonferroni_bracket, frechet_hoeffding
from .climatology import estimate_climatology
from .copula import network_joint_monte_carlo, pair_joint_exact
from .correlation import matrix_from_pairs
from .empirical import bootstrap_joint_rate, pair_clear_matrix
from .field import (
    RECORD_SPECS,
    REGIME_COUNT,
    generate_record,
    true_correlation,
)
from .geometry import (
    SITES,
    along_cross_matrices,
    bearing_matrix,
    separation_matrix,
)
from .independence import availability_nines, overstatement
from .learned import NOMINAL_COVERAGE
from .report import format_table, interval, number, scientific
from .splits import HELD_OUT_SITES
from .study import MODEL_NAMES, run_study
from .uncertainty import climatology_share, naive_product_relative_sd

#: Declared station sets the CLI can report on by name.
NETWORKS: dict[str, tuple[int, ...]] = {
    "near-pair": (0, 1),
    "far-pair": (0, 12),
    "cluster": (0, 1, 2, 3),
    "two-groups": (0, 1, 2, 4, 5, 6),
    "one-per-group": (0, 4, 8, 12, 16),
}

#: Declared Monte Carlo sample count of the ``network`` subcommand.
NETWORK_SAMPLES: int = 400_000

SAFETY = (
    "Research grade. Not flight-qualified, not certified, not approved for "
    "operational aerospace use."
)

SYNTHETIC = (
    "The record is SYNTHETIC, generated from declared constants and a fixed "
    "seed by sitediverse.field. No real site climatology is used and no real "
    "ground station is a data source. See DATASET_CARD.md."
)


def _record_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--record",
        default="anisotropic",
        choices=sorted(RECORD_SPECS),
        help="which declared record to use (default: anisotropic)",
    )


def _network_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--network",
        default="cluster",
        choices=sorted(NETWORKS),
        help="which declared station set to report on (default: cluster)",
    )


def build_parser() -> argparse.ArgumentParser:
    """The argument parser. Built in a function so a test can introspect it."""
    parser = argparse.ArgumentParser(
        prog="python -m sitediverse",
        description=(
            "Optical ground-station site-diversity availability under measured "
            "spatial correlation, against the independence assumption that "
            "overstates it. " + SAFETY
        ),
        epilog=SYNTHETIC,
    )
    parser.add_argument("--version", action="version", version=f"sitediverse {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("sites", help="the declared station geometry")

    record = subparsers.add_parser("record", help="the declared synthetic record")
    _record_argument(record)

    climatology = subparsers.add_parser(
        "climatology", help="the estimated marginal climatology and its uncertainty"
    )
    _record_argument(climatology)

    overstate = subparsers.add_parser(
        "overstate", help="THE HEADLINE: independence against the record"
    )
    _record_argument(overstate)
    overstate.add_argument(
        "--replicates",
        type=int,
        default=400,
        help="bootstrap replicates (default: 400, minimum 50)",
    )

    bounds = subparsers.add_parser(
        "bounds", help="distribution-free bounds the implementation does not use"
    )
    _record_argument(bounds)
    _network_argument(bounds)

    correlate = subparsers.add_parser(
        "correlate", help="the fitted correlation-length baselines"
    )
    _record_argument(correlate)

    learned = subparsers.add_parser(
        "learned", help="the learned model against the baselines, both hold-outs"
    )
    _record_argument(learned)

    network = subparsers.add_parser(
        "network", help="joint unavailability of a declared station set"
    )
    _record_argument(network)
    _network_argument(network)
    network.add_argument(
        "--samples",
        type=int,
        default=NETWORK_SAMPLES,
        help=f"Monte Carlo samples (default: {NETWORK_SAMPLES}, minimum 1000)",
    )
    return parser


def command_sites(_arguments: argparse.Namespace) -> int:
    """Print the declared geometry."""
    separation = separation_matrix()
    bearing = bearing_matrix()
    along, cross = along_cross_matrices()
    print("declared synthetic station network -- COORDINATES ONLY")
    print(SYNTHETIC)
    print("")
    print(
        format_table(
            ["index", "name", "lat deg", "lon deg", "alt m", "group"],
            [
                [
                    str(index),
                    site.name,
                    f"{site.latitude_deg:+8.3f}",
                    f"{site.longitude_deg:+9.3f}",
                    f"{site.altitude_m:7.0f}",
                    str(site.cluster),
                ]
                for index, site in enumerate(SITES)
            ],
        )
    )
    rows, cols = np.triu_indices(len(SITES), 1)
    print("")
    print(f"pairs                      : {rows.size}")
    print(f"separation range km        : {separation[rows, cols].min():.1f} to "
          f"{separation[rows, cols].max():.1f}")
    print(f"held-out stations          : "
          f"{', '.join(SITES[i].name for i in HELD_OUT_SITES)}")
    print("")
    print("the eight closest pairs, with the along/cross decomposition of equation (3)")
    print("")
    order = np.argsort(separation[rows, cols])[:8]
    print(
        format_table(
            ["pair", "separation km", "bearing deg", "along km", "cross km"],
            [
                [
                    f"{SITES[rows[k]].name}-{SITES[cols[k]].name}",
                    f"{separation[rows[k], cols[k]]:.1f}",
                    f"{bearing[rows[k], cols[k]]:.1f}",
                    f"{along[rows[k], cols[k]]:.1f}",
                    f"{cross[rows[k], cols[k]]:.1f}",
                ]
                for k in order
            ],
        )
    )
    return 0


def command_record(arguments: argparse.Namespace) -> int:
    """Print the declared record and its ground truth."""
    spec = RECORD_SPECS[arguments.record]
    record = generate_record(spec)
    print(f"record                     : {arguments.record}")
    print(SYNTHETIC)
    print("")
    print(f"epochs                     : {record.epochs} "
          f"({spec.years} declared years at 3 h)")
    print(f"stations                   : {record.site_count}")
    print(f"seed                       : {spec.seed}")
    print(f"nugget rho_0               : {spec.nugget}")
    print(f"persistence phi            : {spec.persistence}")
    print(f"integral time scale epochs : {spec.integral_time_scale_epochs:.4f}")
    print("")
    print("GROUND TRUTH correlation lengths, equation (4). Available to validation")
    print("and used by no estimator in this package.")
    print("")
    print(
        format_table(
            ["regime", "L_along km", "L_cross km", "ratio"],
            [
                [
                    str(index),
                    f"{spec.along_km[index]:.2f}",
                    f"{spec.cross_km[index]:.2f}",
                    f"{spec.along_km[index] / spec.cross_km[index]:.4f}",
                ]
                for index in range(REGIME_COUNT)
            ],
        )
    )
    print("")
    observed = record.cloud.mean(axis=0)
    truth = record.marginal_rate_true()
    print(
        format_table(
            ["station", "observed rate", "true rate", "difference"],
            [
                [
                    record.site_names[index],
                    number(observed[index]),
                    number(truth[index]),
                    f"{observed[index] - truth[index]:+.6f}",
                ]
                for index in range(record.site_count)
            ],
        )
    )
    return 0


def command_climatology(arguments: argparse.Namespace) -> int:
    """Print the estimated marginal climatology and its sampling uncertainty."""
    record = generate_record(RECORD_SPECS[arguments.record])
    climatology = estimate_climatology(record.cloud)
    truth = record.marginal_rate_true()
    effective = climatology.effective_sample_size()
    errors = climatology.rate_standard_error()
    print(f"record                     : {arguments.record}")
    print(f"season-by-hour cells       : {climatology.cell_rate.shape[0]}")
    print(f"epochs                     : {climatology.epochs}")
    print("")
    print("The effective sample size of equation (13) uses the estimated lag-one")
    print("autocorrelation of the cloud INDICATOR, not the latent field's phi.")
    print("")
    print(
        format_table(
            ["station", "rate", "lag-1 indicator", "n_eff", "rate sd", "true rate",
             "error in sd"],
            [
                [
                    record.site_names[index],
                    number(climatology.long_term_rate[index]),
                    number(climatology.lag_one_indicator[index], 4),
                    f"{effective[index]:9.1f}",
                    scientific(errors[index]),
                    number(truth[index]),
                    f"{(climatology.long_term_rate[index] - truth[index]) / errors[index]:+.3f}",
                ]
                for index in range(record.site_count)
            ],
        )
    )
    return 0


def command_overstate(arguments: argparse.Namespace) -> int:
    """THE HEADLINE. Independence against the record, for every declared network."""
    if arguments.replicates < 50:
        print(
            f"--replicates must be at least 50 for a percentile interval, got "
            f"{arguments.replicates}",
            file=sys.stderr,
        )
        return 2
    record = generate_record(RECORD_SPECS[arguments.record])
    climatology = estimate_climatology(record.cloud)
    errors = climatology.rate_standard_error()
    print(f"record                     : {arguments.record}")
    print(SYNTHETIC)
    print("")
    print("THE HEADLINE OUTPUT. P_independent is the naive product of marginals of")
    print("equation (14), which is what site-diversity practice does. P_measured is")
    print("the directly observed joint cloud rate of equation (20). Every figure")
    print("carries the record's own sampling uncertainty.")
    print("")
    rows = []
    for name, sites in sorted(NETWORKS.items()):
        result = overstatement(climatology, record.cloud, sites)
        band = bootstrap_joint_rate(
            record.cloud, sites, replicates=arguments.replicates
        )
        relative = naive_product_relative_sd(
            climatology.long_term_rate[list(sites)], errors[list(sites)]
        )
        rows.append(
            [
                name,
                str(len(sites)),
                f"{result.naive:.6f} +/- {relative * result.naive:.6f}",
                f"{result.measured:.6f} +/- {band.standard_error:.6f}",
                number(result.ratio, 4),
                f"{result.total_difference:+.6f}",
                number(result.nines_lost, 4),
            ]
        )
    print(
        format_table(
            [
                "network",
                "n",
                "P_independent",
                "P_measured",
                "ratio",
                "difference",
                "nines lost",
            ],
            rows,
        )
    )
    print("")
    print("The decomposition of equation (16). It is exact: the two terms add to the")
    print("difference above with no cross term. The first is what assuming a")
    print("STATIONARY climatology costs and can be negative; the second is what")
    print("assuming spatial INDEPENDENCE costs and is what the correlation models")
    print("of this package predict.")
    print("")
    rows = []
    for name, sites in sorted(NETWORKS.items()):
        result = overstatement(climatology, record.cloud, sites)
        band = bootstrap_joint_rate(
            record.cloud, sites, replicates=arguments.replicates
        )
        rows.append(
            [
                name,
                f"{result.naive:.6f}",
                f"{result.conditional:.6f}",
                f"{result.measured:.6f}",
                f"{result.comodulation_term:+.6f}",
                f"{result.correlation_term:+.6f}",
                scientific(abs(result.identity_residual()), 2),
                number(
                    climatology_share(
                        result.naive - result.measured, band.standard_error
                    ),
                    4,
                ),
            ]
        )
    print(
        format_table(
            [
                "network",
                "P_naive",
                "P_conditional",
                "P_measured",
                "co-modulation",
                "correlation",
                "identity resid",
                "clim share",
            ],
            rows,
        )
    )
    print("")
    print("A ratio below 1 is not an error. Two stations whose seasonal cloud maxima")
    print("fall six months apart are jointly clear more often than their annual mean")
    print("rates suggest, so the naive product OVERSTATES their joint unavailability;")
    print("the co-modulation term carries the sign. Where the clim share is near 1 the")
    print("whole discrepancy is inside the record's own sampling noise and is not")
    print("evidence of anything.")
    return 0


def command_bounds(arguments: argparse.Namespace) -> int:
    """Distribution-free bounds, derived from mathematics nothing here uses."""
    record = generate_record(RECORD_SPECS[arguments.record])
    climatology = estimate_climatology(record.cloud)
    sites = NETWORKS[arguments.network]
    rates = climatology.long_term_rate[list(sites)]
    result = overstatement(climatology, record.cloud, sites)
    frechet = frechet_hoeffding(rates)
    clear = pair_clear_matrix(record.cloud)[np.ix_(sites, sites)]
    bonferroni = bonferroni_bracket(rates, clear)
    print(f"record                     : {arguments.record}")
    print(f"network                    : {arguments.network} "
          f"({', '.join(record.site_names[i] for i in sites)})")
    print("")
    print("Equation (17), the Frechet-Hoeffding bounds. They depend on the marginals")
    print("and on nothing else: no copula, no correlation, no latent field. Both ends")
    print("are attained by real joint distributions, so neither can be tightened.")
    print("")
    print(f"  marginals                : {', '.join(number(rate) for rate in rates)}")
    print(f"  bounds                   : {interval(frechet.lower, frechet.upper)}")
    print(f"  width                    : {number(frechet.width)}")
    print(f"  condition number kappa   : {scientific(frechet.condition_number)}")
    print(f"  slack, equation (19)     : {scientific(frechet.slack)}")
    print("")
    print(
        format_table(
            ["quantity", "value", "inside bounds", "violation"],
            [
                [
                    label,
                    number(value),
                    "yes" if frechet.contains(value) else "NO",
                    scientific(frechet.violation(value)),
                ]
                for label, value in (
                    ("P_independent, equation (14)", result.naive),
                    ("P_conditional, equation (15)", result.conditional),
                    ("P_measured, equation (20)", result.measured),
                )
            ],
        )
    )
    print("")
    print("Equation (18), the second-order Bonferroni bracket. For n = 2 the upper")
    print("bound is an exact inclusion-exclusion IDENTITY and must reproduce the")
    print("measured joint rate; for n > 2 it is a bound and on data with cloud rates")
    print("of order 0.3 it is vacuous, which is reported rather than hidden.")
    print("")
    print(f"  bracket                  : {interval(bonferroni.lower, bonferroni.upper)}")
    print(f"  sum of pair clear rates  : {number(bonferroni.pair_clear_sum)}")
    print(f"  slack                    : {scientific(bonferroni.slack)}")
    print(f"  contains P_measured      : "
          f"{'yes' if bonferroni.contains(result.measured) else 'NO'}")
    if len(sites) == 2:
        print(f"  n = 2 identity residual  : "
              f"{scientific(bonferroni.upper - result.measured)}")
    else:
        vacuous = bonferroni.lower < 0.0 and bonferroni.upper > 1.0
        print(f"  vacuous for this network : {'yes' if vacuous else 'no'}")
    return 0


def command_correlate(arguments: argparse.Namespace) -> int:
    """The fitted correlation-length baselines against the ground truth."""
    spec = RECORD_SPECS[arguments.record]
    record = generate_record(spec)
    study = run_study(record, arguments.record)
    print(f"record                     : {arguments.record}")
    print("")
    print("Baseline 2 as specified, equation (23): one two-parameter exponential")
    print("fitted over the pooled training rows.")
    print("")
    print(f"  rho_0                    : {number(study.exponential.nugget)}")
    print(f"  L                        : {number(study.exponential.length_km, 2)} km")
    print(f"  residual rms             : {number(study.exponential.residual_rms)}")
    print(f"  training rows            : {study.exponential.sample_count}")
    print(f"  parameters on a bound    : "
          f"{', '.join(study.exponential.at_bound) or 'none'}")
    print("")
    print("The strengthened baseline, equation (24): one model per declared regime.")
    print("Fitted because the build rule of this mission is that where a named")
    print("baseline measures weakly the strongest honest baseline is used and the")
    print("named one is reported beside it. Both are reported everywhere.")
    print("")
    print(
        format_table(
            ["regime", "rho_0", "L km", "residual rms", "rows", "on a bound",
             "true L_along km", "true L_cross km"],
            [
                [
                    str(model.regime),
                    number(model.nugget),
                    number(model.length_km, 2),
                    number(model.residual_rms),
                    str(model.sample_count),
                    ", ".join(model.at_bound) or "none",
                    f"{spec.along_km[index]:.2f}",
                    f"{spec.cross_km[index]:.2f}",
                ]
                for index, model in enumerate(study.per_regime.models)
            ],
        )
    )
    print("")
    print(f"  per-regime residual rms  : {number(study.per_regime.residual_rms)}")
    print(f"  strongest on TRAIN       : {study.strongest_baseline}")
    print("")
    print("Equation (25): the pair (rho_0, L) is weakly identified and the figure of")
    print("merit is the correlation FUNCTION at the separations that exist, not")
    print("either parameter. The function error is reported by")
    print("validation/validate_correlation.py.")
    return 0


def command_learned(arguments: argparse.Namespace) -> int:
    """The learned model against the baselines, both hold-outs, separately."""
    record = generate_record(RECORD_SPECS[arguments.record])
    started = time.perf_counter()
    study = run_study(record, arguments.record)
    print(f"record                     : {arguments.record}")
    print(f"training rows              : {study.learned.report.rows}")
    print(f"features                   : {study.learned.report.feature_count}")
    print(f"trees x depth              : {study.learned.report.trees} x "
          f"{study.learned.report.max_depth}")
    print(f"quantile levels            : {study.learned.report.quantiles}")
    print(f"tuned against a hold-out   : "
          f"{study.learned.report.tuned_against_holdout}")
    print(f"learned fit seconds        : {study.learned_seconds:.3f}")
    print(f"whole study seconds        : {study.seconds:.3f}")
    print("")
    print("Mean absolute error in JOINT UNAVAILABILITY against the empirical")
    print("reference of equation (20). The two hold-out cells are reported")
    print("SEPARATELY and are never merged: a spatio-temporal model can pass one and")
    print("fail the other, and on this record it does.")
    print("")
    print(
        format_table(
            ["cell", "rows", *MODEL_NAMES, "interval coverage"],
            [
                [
                    score.cell,
                    str(score.rows),
                    *(number(score.joint_mae[model]) for model in MODEL_NAMES),
                    number(score.interval_coverage, 4),
                ]
                for score in study.scores.values()
            ],
        )
    )
    print("")
    print(f"nominal interval coverage  : {NOMINAL_COVERAGE:.4f}")
    print(f"quantile crossings         : {study.learned_prediction.crossing_count} of "
          f"{study.learned_prediction.rows} rows")
    print("")
    print("Equation (41). The margin is the baseline's error minus the learned")
    print("model's, so positive favours the learned model. Two intervals are")
    print("reported and they answer different questions: the paired bootstrap")
    print("interval on the MARGIN, which is narrow because the two errors move")
    print("together under resampling, and the half-width of the bootstrap interval")
    print("on the empirical REFERENCE itself, which is the scale at which the thing")
    print("being scored against is uncertain.")
    print("")
    print(
        format_table(
            [
                "cell",
                "baseline",
                "margin",
                "margin 95% CI",
                "CI contains 0",
                "reference half-width",
                "|margin| inside it",
            ],
            [
                [
                    result.cell,
                    result.baseline,
                    f"{result.margin:+.6f}",
                    interval(result.bootstrap.lower, result.bootstrap.upper),
                    "yes" if result.inside_interval else "no",
                    number(result.reference_half_width),
                    "yes" if result.inside_reference_half_width else "no",
                ]
                for result in study.margins
            ],
        )
    )
    print("")
    print("This model is not certified for operational flight use.")
    print(f"elapsed seconds            : {time.perf_counter() - started:.3f}")
    return 0


def command_network(arguments: argparse.Namespace) -> int:
    """Joint unavailability of a declared station set, every route."""
    if arguments.samples < 1000:
        print(
            f"--samples must be at least 1000 for a usable Monte Carlo error, got "
            f"{arguments.samples}",
            file=sys.stderr,
        )
        return 2
    spec = RECORD_SPECS[arguments.record]
    record = generate_record(spec)
    climatology = estimate_climatology(record.cloud)
    sites = NETWORKS[arguments.network]
    result = overstatement(climatology, record.cloud, sites)
    band = bootstrap_joint_rate(record.cloud, sites)
    threshold = climatology.threshold_series()[:, list(sites)]
    print(f"record                     : {arguments.record}")
    print(f"network                    : {arguments.network} "
          f"({', '.join(record.site_names[i] for i in sites)})")
    print(f"stations                   : {len(sites)}")
    print("")
    rows = [
        [
            "independence, equation (14)",
            number(result.naive),
            "analytic",
            number(availability_nines(result.naive), 4),
        ],
        [
            "conditional, equation (15)",
            number(result.conditional),
            "analytic",
            number(availability_nines(result.conditional), 4),
        ],
    ]
    if len(sites) == 2:
        exact = pair_joint_exact(
            threshold[:, 0],
            threshold[:, 1],
            float(true_correlation(spec, 0)[sites[0], sites[1]]),
        )
        rows.append(
            [
                "copula, equation (8), exact",
                number(exact),
                "quadrature",
                number(availability_nines(exact), 4),
            ]
        )
    projected = matrix_from_pairs(true_correlation(spec, 0)[np.ix_(sites, sites)])
    monte = network_joint_monte_carlo(
        threshold, projected.matrix, arguments.samples, seed=515151
    )
    rows.append(
        [
            "copula, equation (10), Monte Carlo",
            f"{monte.estimate:.6f} +/- {monte.standard_error:.6f}",
            f"{monte.samples} draws",
            number(availability_nines(monte.estimate), 4),
        ]
    )
    rows.append(
        [
            "empirical, equation (20)",
            f"{result.measured:.6f} +/- {band.standard_error:.6f}",
            "block bootstrap",
            number(availability_nines(result.measured), 4),
        ]
    )
    print(
        format_table(
            ["route", "joint unavailability", "error source", "nines"], rows
        )
    )
    print("")
    print("The copula rows use the GENERATOR's correlation matrix, which exists only")
    print("because this record is synthetic. They are a consistency check on the")
    print("copula machinery and NOT a prediction: no real record offers them.")
    print("")
    print(f"  bootstrap 95% interval   : {interval(band.lower, band.upper)}")
    print(f"  block length epochs      : {band.block}")
    print(f"  matrix projection needed : {'yes' if projected.projected else 'no'}")
    print(f"  least eigenvalue         : "
          f"{scientific(projected.least_eigenvalue_before)}")
    return 0


COMMANDS = {
    "sites": command_sites,
    "record": command_record,
    "climatology": command_climatology,
    "overstate": command_overstate,
    "bounds": command_bounds,
    "correlate": command_correlate,
    "learned": command_learned,
    "network": command_network,
}


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch. Returns a process exit status."""
    arguments = build_parser().parse_args(argv)
    try:
        return COMMANDS[arguments.command](arguments)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
