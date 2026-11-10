"""Bounds derived from mathematics the implementation does not use.

Three bound checks, and the conditioning rule that makes them checks rather than
decoration.

1. **The Frechet-Hoeffding bounds of equation (17)** against every route by
   which this package computes a joint unavailability. They depend on the
   marginals and on nothing else: no copula, no correlation, no latent field.
2. **The Bonferroni bracket of equation (18).** For two stations its upper bound
   is an exact inclusion-exclusion IDENTITY computed entirely from CLEAR-rate
   statistics, so it must reproduce the measured joint rate taken by counting
   cloudy epochs. For more than two it is a bound, and on cloud rates of order
   0.3 it is vacuous -- which is reported rather than hidden.
3. **The slack of equation (19).** The absolute slack is set by the magnitude of
   the SUMMANDS and never by the result, so the relative slack diverges exactly
   where the expression cancels. Both are tabulated so a reader can see the
   slack is not a constant, and a bound that falls below its own slack is
   reported as UNINFORMATIVE rather than as a pass.
"""

from __future__ import annotations

import sys  # noqa: E402

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

from _bootstrap import (  # noqa: E402
    NETWORKS,
    Tee,
    bootstrap,
    environment_lines,
)

ROOT = bootstrap()

from sitediverse.bounds import (  # noqa: E402
    MACHINE_EPSILON,
    SLACK_FACTOR,
    bonferroni_bracket,
    bound_condition_number,
    bound_slack,
    frechet_hoeffding,
)
from sitediverse.climatology import estimate_climatology  # noqa: E402
from sitediverse.empirical import (  # noqa: E402
    joint_rate,
    pair_clear_matrix,
)
from sitediverse.field import RECORD_SPECS, generate_record  # noqa: E402
from sitediverse.geometry import SITES, upper_triangle_pairs  # noqa: E402
from sitediverse.independence import overstatement, product_of_marginals  # noqa: E402
from sitediverse.report import format_table, interval, number, scientific  # noqa: E402


def frechet(out: Tee, record, climatology) -> None:
    """Part 1: equation (17) against every route, on every declared network."""
    out.rule("1. the Frechet-Hoeffding bounds of equation (17)")
    out("max(0, sum q_i - (n - 1))  <=  P_joint  <=  min_i q_i")
    out("")
    out("Both ends are ATTAINED, by comonotone and countermonotone arrangements, so")
    out("neither can be tightened without an assumption. The independence product")
    out("lies strictly inside them for n >= 2, and the width of the interval is the")
    out("entire content of the site-diversity problem: independence is one point in")
    out("it and the record picks another.")
    out("")
    rows = []
    violations = 0
    for name, sites in NETWORKS.items():
        rates = climatology.long_term_rate[list(sites)]
        bounds = frechet_hoeffding(rates)
        result = overstatement(climatology, record.cloud, sites)
        for label, value in (
            ("P_independent", result.naive),
            ("P_conditional", result.conditional),
            ("P_measured", result.measured),
        ):
            inside = bounds.contains(value)
            if not inside:
                violations += 1
            rows.append(
                [
                    name if label == "P_independent" else "",
                    label,
                    number(value),
                    interval(bounds.lower, bounds.upper),
                    "yes" if inside else "NO",
                    scientific(bounds.violation(value)),
                ]
            )
    out(
        format_table(
            ["network", "route", "value", "bounds", "inside", "violation"], rows
        )
    )
    out("")
    out(f"  routes checked           : {len(rows)}")
    out(f"  violations               : {violations}")
    out.check(
        "every route lies inside the distribution-free bounds on every network",
        violations == 0,
        f"{violations} violations in {len(rows)} comparisons",
    )
    out("")
    out("The informativeness of each side is reported, because a check against a")
    out("bound that admits everything is not a check. The lower bound is identically")
    out("zero whenever the union bound is vacuous, which on cloud rates near 0.3 it")
    out("usually is, and that is stated rather than counted as a pass.")
    out("")
    rows = []
    for name, sites in NETWORKS.items():
        rates = climatology.long_term_rate[list(sites)]
        bounds = frechet_hoeffding(rates)
        rows.append(
            [
                name,
                str(len(sites)),
                number(bounds.lower),
                number(bounds.upper),
                number(bounds.width),
                "yes" if bounds.lower_informative else "no",
                "yes" if bounds.upper_informative else "no",
            ]
        )
    out(
        format_table(
            ["network", "n", "lower", "upper", "width", "lower informative",
             "upper informative"],
            rows,
        )
    )


def all_pairs(out: Tee, record, climatology) -> None:
    """Part 2: equation (17) over all 190 pairs, and where the lower bound bites."""
    out.rule("2. equation (17) over all 190 pairs")
    pairs = upper_triangle_pairs(len(SITES))
    violations = 0
    informative = 0
    conditions = []
    for site_i, site_j in pairs:
        rates = climatology.long_term_rate[[site_i, site_j]]
        bounds = frechet_hoeffding(rates)
        measured = joint_rate(record.cloud, (site_i, site_j))
        naive = product_of_marginals(rates)
        if not bounds.contains(measured):
            violations += 1
        if not bounds.contains(naive):
            violations += 1
        if bounds.lower_informative:
            informative += 1
        conditions.append(bounds.condition_number)
    finite = [value for value in conditions if np.isfinite(value)]
    out(f"  pairs                    : {len(pairs)}")
    out(f"  comparisons              : {2 * len(pairs)}")
    out(f"  violations               : {violations}")
    out(f"  pairs with an informative lower bound : {informative}")
    out(f"  condition number range   : {min(finite):.6e} to {max(finite):.6e}")
    out(f"  infinite condition numbers            : "
        f"{len(conditions) - len(finite)}")
    out.check(
        "no pair violates the distribution-free bounds by either route",
        violations == 0,
        f"{violations} violations in {2 * len(pairs)} comparisons",
    )
    out.check(
        "the lower bound bites on at least one pair, so the check is not vacuous "
        "everywhere",
        informative > 0,
        f"{informative} of {len(pairs)} pairs have a strictly informative lower bound",
    )


def bonferroni(out: Tee, record, climatology) -> None:
    """Part 3: equation (18), the identity for pairs and the bracket beyond."""
    out.rule("3. the Bonferroni bracket of equation (18)")
    out("For n = 2 inclusion-exclusion is exact:")
    out("")
    out("    P(both cloudy) = 1 - P(i clear) - P(j clear) + P(both clear)")
    out("")
    out("The right-hand side is computed entirely from CLEAR-rate statistics and the")
    out("left-hand side by counting cloudy epochs, so the two are different")
    out("computations of the same number. This is the sharpest independent-route")
    out("check in this package.")
    out("")
    clear = pair_clear_matrix(record.cloud)
    pairs = upper_triangle_pairs(len(SITES))
    residuals = []
    for site_i, site_j in pairs:
        rates = climatology.long_term_rate[[site_i, site_j]]
        bracket = bonferroni_bracket(
            rates, clear[np.ix_([site_i, site_j], [site_i, site_j])]
        )
        residuals.append(
            abs(bracket.upper - joint_rate(record.cloud, (site_i, site_j)))
        )
    worst = max(residuals)
    out(f"  pairs                    : {len(pairs)}")
    out(f"  worst identity residual  : {worst:.6e}")
    out(f"  median identity residual : {float(np.median(residuals)):.6e}")
    out.check(
        "the n = 2 inclusion-exclusion identity closes over all 190 pairs",
        worst < 1e-14,
        f"worst residual {worst:.6e} against a tolerance of 1e-14",
    )
    out("")
    out("For n > 2 it is a bound and not an identity, and on cloud rates of order")
    out("0.3 it is usually vacuous: the sum of the clear rates exceeds 1 and the")
    out("lower bound 1 - sum u_i falls below zero. Reported rather than hidden.")
    out("")
    rows = []
    violations = 0
    vacuous = 0
    for name, sites in NETWORKS.items():
        rates = climatology.long_term_rate[list(sites)]
        bracket = bonferroni_bracket(rates, clear[np.ix_(sites, sites)])
        measured = joint_rate(record.cloud, sites)
        inside = bracket.contains(measured)
        is_vacuous = bracket.lower < 0.0 and bracket.upper > 1.0
        if not inside:
            violations += 1
        if is_vacuous:
            vacuous += 1
        rows.append(
            [
                name,
                str(len(sites)),
                interval(bracket.lower, bracket.upper),
                number(bracket.pair_clear_sum, 4),
                number(measured),
                "yes" if inside else "NO",
                "VACUOUS" if is_vacuous else "informative",
            ]
        )
    out(
        format_table(
            ["network", "n", "bracket", "pair clear sum", "P_measured", "inside",
             "verdict"],
            rows,
        )
    )
    out("")
    out(f"  violations               : {violations}")
    out(f"  vacuous brackets         : {vacuous} of {len(NETWORKS)}")
    out.check(
        "the bracket contains the measured joint rate on every network",
        violations == 0,
        f"{violations} violations in {len(NETWORKS)} networks",
    )
    out.check(
        "the bracket is reported as vacuous where it is, rather than as a pass",
        vacuous > 0,
        f"{vacuous} of {len(NETWORKS)} networks have a bracket leaving the unit "
        "interval; the CLI prints the flag",
    )


def conditioning(out: Tee) -> None:
    """Part 4: equation (19), the slack that follows the conditioning."""
    out.rule("4. the slack of equation (19) follows the conditioning")
    out("The absolute slack is c eps (sum|q_i| + |f(q)|), set by the magnitude of the")
    out("SUMMANDS and never by the result. The relative slack is therefore kappa")
    out("times machine epsilon and diverges exactly where the expression cancels.")
    out("")
    out("This product's FIRST version of equation (19) returned")
    out("c eps kappa |f(q)|, which cancels to c eps sum|q_i|: it looked conditioned")
    out("and was independent of f(q) altogether. Measured then: a well-conditioned")
    out("bound of 0.9 got a slack of 2.700e-14 and an ill-conditioned bound of 1e-07")
    out("got 1.421e-14 -- the ill-conditioned case got the SMALLER slack. The table")
    out("below is the current form over eight decades of cancellation.")
    out("")
    out(f"  SLACK_FACTOR             : {SLACK_FACTOR}")
    out(f"  machine epsilon          : {MACHINE_EPSILON:.6e}")
    out("")
    deltas = (1e-1, 1e-2, 1e-4, 1e-6, 1e-8, 1e-12, 1e-17)
    rows = []
    ratios = []
    absolutes = []
    for delta in deltas:
        rates = np.array([0.5, 0.5 + delta])
        slack = bound_slack(rates, delta)
        condition = bound_condition_number(rates, delta)
        bounds = frechet_hoeffding(rates)
        ratios.append(slack / delta)
        absolutes.append(slack)
        rows.append(
            [
                scientific(delta, 0),
                scientific(slack),
                scientific(condition),
                scientific(slack / delta),
                "yes" if bounds.lower_informative else "NO",
            ]
        )
    out(
        format_table(
            [
                "lower bound f(q)",
                "absolute slack",
                "condition number",
                "relative slack",
                "informative",
            ],
            rows,
        )
    )
    out("")
    out(f"  absolute slack range     : {min(absolutes):.3e} to {max(absolutes):.3e}")
    out(f"  relative slack range     : {min(ratios):.3e} to {max(ratios):.3e}")
    out(f"  relative slack grows by  : a factor of {max(ratios) / min(ratios):.3e} "
        "over sixteen decades of cancellation")
    out("")
    out("Two statements, and they are the two halves of equation (19). The ABSOLUTE")
    out("slack is nearly constant across the table -- it changes only through the")
    out("|f(q)| term, which is why the largest and smallest differ by less than a")
    out("factor of two. The RELATIVE slack is proportional to 1 / f(q) and therefore")
    out("grows exactly as the bound cancels, decade for decade. Neither statement is")
    out("true of a slack floored at an absolute constant, which is the defect class")
    out("this check belongs to.")
    out("")
    scaled = [ratio * delta for ratio, delta in zip(ratios, deltas, strict=True)]
    out(f"  (relative slack) x f(q)  : {min(scaled):.6e} to {max(scaled):.6e}")
    out("")
    monotone = all(
        later > earlier
        for earlier, later in zip(ratios[:-1], ratios[1:], strict=True)
    )
    proportional = all(
        abs(
            (ratios[index + 1] / ratios[index])
            / (deltas[index] / deltas[index + 1])
            - 1.0
        )
        < 0.25
        for index in range(len(deltas) - 1)
    )
    out.check(
        "the relative slack rises monotonically as the bound expression cancels",
        monotone,
        f"from {min(ratios):.3e} to {max(ratios):.3e} over sixteen decades",
    )
    out.check(
        "the relative slack is proportional to 1 / f(q), decade for decade",
        proportional,
        "every consecutive pair's ratio matches the ratio of their f(q) values to "
        f"within 25 per cent, which is the spread of the absolute slack itself "
        f"({max(absolutes) / min(absolutes):.4f}): the |f(q)| term in equation (19) "
        "makes the absolute slack 20 per cent larger at f(q) = 0.1 than at f(q) = 0",
    )
    out.check(
        "the absolute slack is nearly constant, which is what makes it conditioned "
        "rather than relative",
        max(absolutes) / min(absolutes) < 2.0,
        f"absolute slack varies by a factor of "
        f"{max(absolutes) / min(absolutes):.6f} across the table",
    )
    out.check(
        "a bound below its own slack is reported as uninformative, not as a pass",
        not frechet_hoeffding(np.array([0.5, 0.5 + 1e-17])).lower_informative,
        "the lower bound at a cancellation of 1e-17 is flagged uninformative",
    )
    out.control(
        "the first, wrong form of equation (19) gave the ill-conditioned case the "
        "SMALLER slack",
        fired=(
            SLACK_FACTOR * MACHINE_EPSILON * bound_condition_number(
                np.array([0.5, 0.5000001]), 1e-7
            ) * 1e-7
            < SLACK_FACTOR * MACHINE_EPSILON * bound_condition_number(
                np.array([0.95, 0.95]), 0.9
            ) * 0.9
        ),
        must_fire=True,
        detail=(
            "reconstructed: the wrong form is monotone in sum|q| alone, so a "
            "conditioned bound of 1e-07 on marginals near 0.5 got less slack than "
            "a well-conditioned bound of 0.9 on marginals near 0.95"
        ),
    )


def main() -> int:
    out = Tee("validate_bounds_output.txt")
    out.rule("sitediverse -- bounds from mathematics the implementation does not use")
    for line in environment_lines():
        out(line)
    record = generate_record(RECORD_SPECS["anisotropic"])
    climatology = estimate_climatology(record.cloud)
    frechet(out, record, climatology)
    all_pairs(out, record, climatology)
    bonferroni(out, record, climatology)
    conditioning(out)
    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
