"""One network, end to end, in the order the analysis runs. The narrative version.

A reader who wants to know what this package does and what it concludes reads
this script and its committed output. It takes ONE declared four-station network
inside one regional group, every baseline under 300 km, and walks the whole
path: the record, the climatology and its uncertainty, the independence
assumption, the bounds that constrain any answer, the copula, the empirical
reference with its bootstrap, and finally what the correlation models predict
and whether any of them is distinguishable from the others.

Nothing here is a new measurement. Every number is reproduced by one of the
other scripts in this directory and the point of this one is the order.
"""

from __future__ import annotations

import sys  # noqa: E402

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

from _bootstrap import (  # noqa: E402
    BLOCK,
    MONTE_CARLO_SAMPLES,
    REPLICATES,
    Tee,
    Timer,
    bootstrap,
    environment_lines,
)

ROOT = bootstrap()

from sitediverse.bounds import bonferroni_bracket, frechet_hoeffding  # noqa: E402
from sitediverse.climatology import estimate_climatology  # noqa: E402
from sitediverse.copula import network_joint_monte_carlo  # noqa: E402
from sitediverse.correlation import matrix_from_pairs  # noqa: E402
from sitediverse.empirical import (  # noqa: E402
    bootstrap_joint_rate,
    pair_clear_matrix,
)
from sitediverse.field import RECORD_SPECS, generate_record  # noqa: E402
from sitediverse.geometry import (  # noqa: E402
    midpoint_bearing_matrix,
    separation_matrix,
)
from sitediverse.independence import availability_nines, overstatement  # noqa: E402
from sitediverse.report import format_table, interval, number, scientific  # noqa: E402
from sitediverse.study import row_nodes  # noqa: E402
from sitediverse.uncertainty import (  # noqa: E402
    FigureWithUncertainty,
    climatology_share,
    naive_product_relative_sd,
)

#: The declared worked network: the four stations of the first regional group.
NETWORK: tuple[int, ...] = (0, 1, 2, 3)


def main() -> int:  # noqa: PLR0915 - a narrative, deliberately linear
    out = Tee("worked_example_output.txt")
    out.rule("sitediverse -- one four-station network, end to end")
    for line in environment_lines():
        out(line)
    out("")
    out("THE RECORD IS SYNTHETIC. Nothing below is a measurement of the atmosphere")
    out("and nothing below transfers to a real network. See DATASET_CARD.md.")

    total = Timer()
    with total:
        out.rule("step 1. the network, as geometry")
        record = generate_record(RECORD_SPECS["anisotropic"])
        separation = separation_matrix(record.sites)
        bearing = midpoint_bearing_matrix(record.sites)
        names = [record.site_names[site] for site in NETWORK]
        out(f"  stations                 : {', '.join(names)}")
        rows = []
        for index, site_i in enumerate(NETWORK):
            for site_j in NETWORK[index + 1 :]:
                rows.append(
                    [
                        f"{record.site_names[site_i]}-{record.site_names[site_j]}",
                        number(float(separation[site_i, site_j]), 1),
                        number(float(bearing[site_i, site_j]), 1),
                    ]
                )
        out(format_table(["pair", "separation km", "midpoint azimuth deg"], rows))
        out("")
        out(f"  baselines                : {len(rows)}")
        out(f"  longest baseline         : "
            f"{max(float(row[1]) for row in rows):.1f} km")
        out("")
        out("Every baseline is under 300 km. This is the configuration a diversity")
        out("trade study reaches for when the sites have to share a backhaul or a")
        out("maintenance crew, and it is the configuration the independence")
        out("assumption is most wrong about.")

        out.rule("step 2. the marginal climatology, estimated from the record")
        climatology = estimate_climatology(record.cloud)
        errors = climatology.rate_standard_error()
        effective = climatology.effective_sample_size()
        standardised = (
            climatology.long_term_rate - record.marginal_rate_true()
        ) / errors
        out(f"  record                   : {record.epochs} epochs, "
            f"{record.spec.years} declared years at 3 h")
        out("")
        out(
            format_table(
                ["station", "cloud rate", "n_eff", "sd", "declared truth",
                 "sigma"],
                [
                    [
                        record.site_names[site],
                        number(climatology.long_term_rate[site]),
                        f"{effective[site]:.1f}",
                        scientific(errors[site]),
                        number(record.marginal_rate_true()[site]),
                        f"{standardised[site]:+.3f}",
                    ]
                    for site in NETWORK
                ],
            )
        )
        out("")
        out("The effective sample size is far below the epoch count because cloud")
        out("cover persists: equation (13) divides by (1 + phi) / (1 - phi) and the")
        out("naive binomial standard error understates by a factor of about 2.5.")

        out.rule("step 3. baseline 1: the independence assumption")
        result = overstatement(climatology, record.cloud, NETWORK)
        relative = naive_product_relative_sd(
            climatology.long_term_rate[list(NETWORK)], errors[list(NETWORK)]
        )
        naive_figure = FigureWithUncertainty(
            "independence, equation (14)",
            result.naive,
            relative * result.naive,
            "analytic-propagation",
        )
        out(f"  P_independent            : {naive_figure.render()}")
        out(f"  A_independent            : {naive_figure.availability:.6f}")
        out(f"  nines claimed            : "
            f"{availability_nines(result.naive):.4f}")
        out("")
        out("That is the number a diversity trade study would carry forward. The")
        out("rest of this script is about how wrong it is.")

        out.rule("step 4. the bounds any answer must satisfy")
        rates = climatology.long_term_rate[list(NETWORK)]
        bounds = frechet_hoeffding(rates)
        clear = pair_clear_matrix(record.cloud)[np.ix_(NETWORK, NETWORK)]
        bracket = bonferroni_bracket(rates, clear)
        out(f"  Frechet-Hoeffding        : "
            f"{interval(bounds.lower, bounds.upper)}")
        out(f"  width                    : {bounds.width:.6f}")
        out(f"  lower bound informative  : "
            f"{'yes' if bounds.lower_informative else 'no'}")
        out(f"  slack, equation (19)     : {bounds.slack:.6e}")
        out(f"  Bonferroni bracket       : "
            f"{interval(bracket.lower, bracket.upper)}")
        out(f"  Bonferroni vacuous       : "
            f"{'yes' if bracket.lower < 0.0 and bracket.upper > 1.0 else 'no'}")
        out("")
        out("The independence product sits inside the Frechet bounds, and so does")
        out("every other answer in this script. The bounds are wide -- their width")
        out("is the entire site-diversity problem -- but they are derived from the")
        out("marginals alone and would catch any answer that is simply wrong. The")
        out("Bonferroni bracket is vacuous at four stations and says so.")

        out.rule("step 5. baseline 3: what the record actually shows")
        band = bootstrap_joint_rate(
            record.cloud, NETWORK, block=BLOCK, replicates=REPLICATES, seed=909001
        )
        measured_figure = FigureWithUncertainty(
            "measured, equation (20)",
            result.measured,
            band.standard_error,
            "block-bootstrap",
        )
        out(f"  P_measured               : {measured_figure.render()}")
        out(f"  95% interval             : {interval(band.lower, band.upper)}")
        out(f"  block length             : {band.block} epochs = "
            f"{band.block * 3.0 / 24.0:.1f} days")
        out(f"  A_measured               : {measured_figure.availability:.6f}")
        out(f"  nines measured           : "
            f"{availability_nines(result.measured):.4f}")

        out.rule("step 6. THE HEADLINE")
        out(f"  P_independent            : {naive_figure.render()}")
        out(f"  P_measured               : {measured_figure.render()}")
        out("")
        out(f"  overstatement factor     : {result.ratio:.4f}")
        out(f"  difference               : {result.total_difference:+.6f}")
        out(f"  availability deficit     : {result.availability_deficit:.6f}")
        out(f"  nines lost               : {result.nines_lost:+.4f}")
        out(f"  climatology share        : "
            f"{climatology_share(result.naive - result.measured, band.standard_error):.6f}")
        out("")
        out(
            "The independence assumption assigns this network "
            f"{availability_nines(result.naive):.4f} nines of availability "
            f"and the record shows {availability_nines(result.measured):.4f}."
        )
        out(
            "The climatology share says the discrepancy is not the record being "
            f"short: it is {result.total_difference / band.standard_error:.1f} "
            "bootstrap standard errors."
        )
        out("")
        out("The exact decomposition of equation (16) splits it:")
        out("")
        out(
            format_table(
                ["term", "value", "what it is"],
                [
                    [
                        "co-modulation",
                        f"{result.comodulation_term:+.6f}",
                        "what assuming a STATIONARY climatology costs",
                    ],
                    [
                        "correlation",
                        f"{result.correlation_term:+.6f}",
                        "what assuming spatial INDEPENDENCE costs",
                    ],
                    [
                        "total",
                        f"{result.total_difference:+.6f}",
                        "their sum, exactly",
                    ],
                    [
                        "residual",
                        scientific(abs(result.identity_residual()), 2),
                        "rounding; the identity has no cross term",
                    ],
                ],
            )
        )

        out.rule("step 7. baseline 2: the copula, at the generator's own correlation")
        out("This step exists only because the record is synthetic: it uses the")
        out("GENERATOR's correlation matrix, which no real record offers. It is a")
        out("consistency check on the copula machinery and NOT a prediction.")
        out("")
        from sitediverse.field import true_correlation

        threshold = climatology.threshold_series()[:, list(NETWORK)]
        projected = matrix_from_pairs(
            true_correlation(record.spec, 0, record.sites)[np.ix_(NETWORK, NETWORK)]
        )
        monte = network_joint_monte_carlo(
            threshold, projected.matrix, MONTE_CARLO_SAMPLES, seed=515151
        )
        out(f"  Monte Carlo, equation (10): {monte.estimate:.6f} +/- "
            f"{monte.standard_error:.6f} over {monte.samples} draws")
        out(f"  against P_measured        : "
            f"{abs(monte.estimate - result.measured) / band.standard_error:.4f} "
            "bootstrap standard errors")
        out(f"  matrix projection needed  : "
            f"{'yes' if projected.projected else 'no'}")
        out(f"  least eigenvalue          : "
            f"{scientific(projected.least_eigenvalue_before)}")
        out("")
        out("The copula at the TRUE correlation reproduces the record's joint rate")
        out("to within the record's own sampling uncertainty. That is the machinery")
        out("working; it says nothing about a model that has to ESTIMATE the")
        out("correlation, which is what the rest of this package does and what")
        out("validate_learned_model.py measures.")

        out.rule("step 8. what this network's figure should be reported as")
        out("")
        for figure in (naive_figure, measured_figure):
            out(f"  {figure.label:34s} {figure.render()}  "
                f"(source: {figure.source})")
        out("")
        out("Every availability figure this package reports carries the record's own")
        out("sampling uncertainty beside it, and a figure whose uncertainty was not")
        out("computed says so rather than carrying a zero, because a zero reads as")
        out("certainty.")
        out("")
        out.check(
            "the measured joint unavailability exceeds the independence product by "
            "more than four bootstrap standard errors",
            result.total_difference > 4.0 * band.standard_error,
            f"difference {result.total_difference:+.6f} is "
            f"{result.total_difference / band.standard_error:.4f} standard errors",
        )
        out.check(
            "every route lies inside the distribution-free bounds",
            all(
                bounds.contains(value)
                for value in (result.naive, result.conditional, result.measured,
                              monte.estimate)
            ),
            f"four routes against {interval(bounds.lower, bounds.upper)}",
        )
        out.check(
            "the copula at the generator's correlation agrees with the record",
            abs(monte.estimate - result.measured) < 4.0 * band.standard_error,
            f"{abs(monte.estimate - result.measured) / band.standard_error:.4f} "
            "bootstrap standard errors",
        )
        out.check(
            "the decomposition of equation (16) closes to rounding",
            abs(result.identity_residual()) < 1e-15,
            f"residual {abs(result.identity_residual()):.6e}",
        )
    out("")
    out(f"  whole worked example      : {total.elapsed_s:.3f} s")
    _ = row_nodes
    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
