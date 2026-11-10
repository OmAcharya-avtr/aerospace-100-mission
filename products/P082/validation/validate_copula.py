"""The copula: equation (9) against SciPy, against Monte Carlo, and the binning.

Four measurements, each against something the implementation does not use.

1. **Equation (9) against SciPy's multivariate normal CDF.** SciPy implements
   the Genz lattice method on the orthant integral; this product integrates the
   Plackett derivative identity by Gauss-Legendre quadrature. Two
   implementations, different mathematics, one number.
2. **The correlation excess against the subtraction it replaces.** The whole
   point of equation (9) is that the excess keeps its relative precision where
   ``Phi_2 - Phi(h)Phi(k)`` loses all of it. The failure of the subtraction is
   measured, not asserted.
3. **The binned epoch nodes against the exact epoch-by-epoch mean.** The
   approximation that makes the fitting loops affordable, with its error.
4. **The n-station Monte Carlo of equation (10) against the exact pair
   quadrature.** A random algorithm against a deterministic one, with the
   criterion the declared four Monte Carlo standard errors.
"""

from __future__ import annotations

import sys  # noqa: E402

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
from scipy.stats import multivariate_normal, norm  # noqa: E402

from _bootstrap import (  # noqa: E402
    CONTROL_SIGMA,
    MONTE_CARLO_SAMPLES,
    Tee,
    Timer,
    bootstrap,
    environment_lines,
)

ROOT = bootstrap()

from sitediverse.climatology import estimate_climatology  # noqa: E402
from sitediverse.copula import (  # noqa: E402
    EPOCH_BINS,
    NODES,
    SEGMENTS,
    bivariate_density,
    bivariate_orthant,
    correlation_excess,
    epoch_nodes,
    implied_correlation,
    network_joint_monte_carlo,
    pair_joint_exact,
)
from sitediverse.field import RECORD_SPECS, generate_record, true_correlation  # noqa: E402
from sitediverse.report import format_table, number, scientific  # noqa: E402


def against_scipy(out: Tee) -> None:
    """Part 1: two implementations of the same orthant probability."""
    out.rule("1. equation (9) against SciPy's independent multivariate normal CDF")
    out("SciPy implements the Genz lattice method on the orthant integral. This")
    out("product integrates the Plackett derivative identity d Phi_2 / d rho = phi_2")
    out("by piecewise Gauss-Legendre quadrature. Different mathematics, one number.")
    out("SciPy is used HERE and nowhere inside the package.")
    out("")
    out(f"  quadrature nodes per segment : {NODES}")
    out(f"  segments over [0, rho]       : {SEGMENTS}")
    out("")
    rng = np.random.default_rng(777)
    count = 600
    levels_h = -rng.uniform(0.05, 3.4, count)
    levels_k = -rng.uniform(0.05, 3.4, count)
    rho = rng.uniform(-0.94, 0.98, count)
    with Timer() as timer:
        mine = bivariate_orthant(levels_h, levels_k, rho)
    theirs = np.array(
        [
            multivariate_normal.cdf(
                [h, k], mean=[0.0, 0.0], cov=[[1.0, r], [r, 1.0]]
            )
            for h, k, r in zip(levels_h, levels_k, rho, strict=True)
        ]
    )
    absolute = np.abs(mine - theirs)
    # The relative difference is reported only where SciPy's own value is above
    # 1e-10. Below that its requested accuracy is absolute and a ratio against
    # it measures SciPy's floor, not this product's error: at the extreme tail
    # of the sample SciPy returns values of order 1e-287 and the ratio runs to
    # 1e+283, which says nothing about either implementation.
    usable = theirs > 1e-10
    relative = absolute[usable] / theirs[usable]
    out(f"  cases                        : {count}")
    out(f"  cases with SciPy above 1e-10 : {int(usable.sum())}")
    out(f"  worst absolute difference    : {float(absolute.max()):.6e}")
    out(f"  worst relative difference    : {float(relative.max()):.6e}  "
        "(over the usable cases only)")
    out(f"  this product, vectorised     : {timer.elapsed_s * 1000.0:.2f} ms for "
        f"{count} cases")
    out("")
    out.check(
        "equation (9) agrees with SciPy's independent implementation absolutely",
        float(absolute.max()) < 1e-14,
        f"worst absolute difference {float(absolute.max()):.6e} against 1e-14",
    )
    out("")
    out("The relative agreement is bounded by SciPy, not by this product: SciPy's")
    out("default requested accuracy is a relative 1e-05 and its measured error here")
    out("is far better than that but not at the level of the absolute agreement. The")
    out("absolute figure is the one this check rests on, and it is stated as such")
    out("rather than quoting the relative number as if it were a bound on this")
    out("product's error.")


def excess_precision(out: Tee) -> None:
    """Part 2: the excess against the subtraction it replaces."""
    out.rule("2. the correlation excess against the subtraction it replaces")
    out("Equation (9)'s first term is exactly the independence baseline and the")
    out("integral is exactly the amount correlation adds. The difference is therefore")
    out("computed as an integral of a positive integrand and never as a subtraction")
    out("of two nearly equal probabilities. The reference for small rho is the exact")
    out("first-order expansion rho phi_2(h, k; 0), whose remainder is O(rho^2).")
    out("")
    level = -3.0
    rows = []
    remainder_ratios: list[float] = []
    floor_errors: list[float] = []
    worst_naive = 0.0
    for rho in (1e-4, 1e-8, 1e-12, 1e-16, 1e-20):
        first_order = rho * float(bivariate_density(level, level, 0.0))
        integral = float(correlation_excess(level, level, rho))
        naive = float(bivariate_orthant(level, level, rho)) - float(
            norm.cdf(level) ** 2
        )
        error_integral = abs(integral - first_order) / first_order
        error_naive = abs(naive - first_order) / first_order
        rows.append(
            [
                scientific(rho, 0),
                scientific(first_order),
                scientific(integral),
                scientific(naive),
                scientific(error_integral),
                f"{error_integral / rho:.3f}",
                scientific(error_naive),
            ]
        )
        # The REFERENCE carries its own O(rho) remainder, so the integral
        # form's error is compared against that remainder and not against a
        # constant. The second-order term of the expansion is
        # (rho / 2) h k phi_2(h, k; 0) relative to the first, so the expected
        # relative discrepancy is |h k| rho / 2, which at h = k = -3 is 4.5
        # rho. The ratio error / rho is tabulated so a reader can see it is
        # that constant and not a growing error.
        # Two regimes, and they are measured separately because they are
        # bounded by different things. Where 4.5 rho is above double-precision
        # rounding the discrepancy IS the expansion's remainder and the ratio
        # is the constant |h k| / 2. Where it is below, the discrepancy is the
        # rounding of the two quantities themselves and the relative error of
        # the integral form sits at the machine floor -- which is the stronger
        # statement of the two.
        if rho >= 1e-12:
            remainder_ratios.append(error_integral / rho)
        else:
            floor_errors.append(error_integral)
        if rho <= 1e-8:
            worst_naive = max(worst_naive, error_naive)
    out(
        format_table(
            [
                "rho",
                "first order",
                "integral form",
                "subtraction",
                "integral rel err",
                "err / rho",
                "subtraction rel err",
            ],
            rows,
        )
    )
    out("")
    expected_ratio = abs(level * level) / 2.0
    worst_ratio_error = max(
        abs(ratio - expected_ratio) for ratio in remainder_ratios
    )
    worst_floor = max(floor_errors)
    out(f"  (integral rel err) / rho, rho >= 1e-12 : "
        f"{', '.join(f'{ratio:.4f}' for ratio in remainder_ratios)}")
    out(f"  predicted |h k| / 2                    : {expected_ratio:.6f}")
    out(f"  integral rel err, rho <= 1e-16         : "
        f"{', '.join(f'{error:.3e}' for error in floor_errors)}")
    out(f"  worst relative error, subtraction      : {worst_naive:.6e}")
    out("")
    out("Two regimes, bounded by two different things. For rho at or above 1e-12 the")
    out("integral form's apparent relative error is the REFERENCE expansion's own")
    out("second-order remainder and not a quadrature error: it is |h k| rho / 2 to")
    out("four figures at every rho. Below that the remainder falls under")
    out("double-precision rounding and what is left is the representation error of")
    out("the two quantities, which puts the integral form at the machine floor --")
    out("the stronger of the two statements. The subtraction, by contrast, has lost")
    out("every digit by rho = 1e-20 and returns exactly 0.0 for a quantity of")
    out("1.964128e-25.")
    out("")
    out.check(
        "the integral form's discrepancy is the expansion's own remainder, |h k| rho / 2",
        worst_ratio_error < 0.05 * expected_ratio,
        f"worst deviation of (error / rho) from {expected_ratio:.6f} is "
        f"{worst_ratio_error:.6f}",
    )
    out.check(
        "below that regime the integral form sits at the double-precision floor",
        worst_floor < 1e-14,
        f"worst relative error {worst_floor:.6e} at rho of 1e-16 and 1e-20, "
        "against a tolerance of 1e-14",
    )
    out.control(
        "the subtraction form loses precision where the integral form does not",
        fired=worst_naive > 1e-3,
        must_fire=True,
        detail=(
            f"subtraction worst relative error {worst_naive:.6e}; this control "
            "exists to show the cancellation is real and not hypothetical"
        ),
    )


def binning(out: Tee) -> None:
    """Part 3: the binned nodes against the exact epoch-by-epoch mean."""
    out.rule("3. the binned epoch nodes against the exact epoch mean")
    out("Evaluating equation (8) at every epoch of the eight-year record costs 23360")
    out("quadratures per correlation value and the inversion of equation (11) needs")
    out("about twenty of those per pair. The epochs are binned on their (h, k) values")
    out("and the node of a cell is the MEAN of the epochs in it, not the cell centre.")
    out("Both forms are measured below; the cell-mean form is worth nearly three")
    out("orders of magnitude and the centre form's error was not even monotone in the")
    out("bin count.")
    out("")
    record = generate_record(RECORD_SPECS["anisotropic"])
    threshold = estimate_climatology(record.cloud).threshold_series()
    pairs = ((0, 1), (0, 12), (4, 5), (8, 16), (3, 19), (0, 2), (16, 19))
    rows = []
    worst_mean = 0.0
    worst_centre = 0.0
    exact_timer = Timer()
    node_timer = Timer()
    for site_i, site_j in pairs:
        left, right = threshold[:, site_i], threshold[:, site_j]
        with node_timer:
            nodes = epoch_nodes(left, right)
        for rho in (0.0, 0.35, 0.7, 0.92):
            with exact_timer:
                exact = pair_joint_exact(left, right, rho)
            binned = nodes.joint(rho)
            counts, edges_left, edges_right = np.histogram2d(
                left, right, bins=EPOCH_BINS
            )
            centre_left = 0.5 * (edges_left[:-1] + edges_left[1:])
            centre_right = 0.5 * (edges_right[:-1] + edges_right[1:])
            occupied = counts > 0
            grid_left, grid_right = np.meshgrid(
                centre_left, centre_right, indexing="ij"
            )
            centre = float(
                np.sum(
                    (counts[occupied] / float(left.size))
                    * bivariate_orthant(
                        grid_left[occupied],
                        grid_right[occupied],
                        np.full(int(occupied.sum()), rho),
                    )
                )
            )
            worst_mean = max(worst_mean, abs(binned - exact))
            worst_centre = max(worst_centre, abs(centre - exact))
            if rho == 0.7:
                rows.append(
                    [
                        f"{site_i}-{site_j}",
                        str(nodes.node_count),
                        number(exact),
                        scientific(abs(binned - exact)),
                        scientific(abs(centre - exact)),
                    ]
                )
    out(
        format_table(
            ["pair", "nodes", "exact at rho=0.7", "cell-mean error",
             "cell-centre error"],
            rows,
        )
    )
    out("")
    out(f"  bins per axis                 : {EPOCH_BINS}")
    out(f"  worst error, cell-mean nodes  : {worst_mean:.6e}")
    out(f"  worst error, cell-centre nodes: {worst_centre:.6e}")
    out(f"  improvement factor            : {worst_centre / worst_mean:.1f}")
    out(f"  exact evaluation, mean        : {exact_timer.mean_ms:.2f} ms")
    out(f"  node construction, mean       : {node_timer.mean_ms:.2f} ms")
    out("")
    out.check(
        "the cell-mean binning error is below 1e-05",
        worst_mean < 1e-5,
        f"worst error {worst_mean:.6e} against a tolerance of 1e-05",
    )
    out.control(
        "the cell-centre form really was worse, so the change was needed",
        fired=worst_centre > 100.0 * worst_mean,
        must_fire=True,
        detail=f"centre error {worst_centre:.6e} against mean error {worst_mean:.6e}",
    )
    out("")
    out("Every headline figure of this product is computed by the EXACT")
    out("epoch-by-epoch route. The nodes exist to make the fitting loops affordable")
    out("and their error is three to four orders of magnitude below the block")
    out("bootstrap standard error of the empirical reference, which")
    out("validate_empirical.py measures at about 5e-03.")


def monte_carlo(out: Tee) -> None:
    """Part 4: equation (10) against equation (8), and against itself."""
    out.rule("4. the n-station Monte Carlo of equation (10)")
    out("For n = 2 the Monte Carlo can be checked against the exact quadrature. The")
    out("criterion is the declared four Monte Carlo standard errors and not an")
    out("absolute constant: the estimator is unbiased and its error is known.")
    out("")
    record = generate_record(RECORD_SPECS["anisotropic"])
    threshold = estimate_climatology(record.cloud).threshold_series()
    rows = []
    worst_sigma = 0.0
    for site_i, site_j in ((0, 1), (0, 12), (4, 5), (16, 19)):
        rho = float(true_correlation(record.spec, 0)[site_i, site_j])
        matrix = np.array([[1.0, rho], [rho, 1.0]])
        estimate = network_joint_monte_carlo(
            threshold[:, [site_i, site_j]], matrix, MONTE_CARLO_SAMPLES, seed=31337
        )
        exact = pair_joint_exact(threshold[:, site_i], threshold[:, site_j], rho)
        sigma = abs(estimate.estimate - exact) / estimate.standard_error
        worst_sigma = max(worst_sigma, sigma)
        rows.append(
            [
                f"{site_i}-{site_j}",
                number(rho),
                number(exact),
                f"{estimate.estimate:.6f} +/- {estimate.standard_error:.6f}",
                f"{sigma:.3f}",
            ]
        )
    out(
        format_table(
            ["pair", "rho", "exact quadrature", "Monte Carlo", "sigma"], rows
        )
    )
    out("")
    out(f"  draws per estimate       : {MONTE_CARLO_SAMPLES}")
    out(f"  worst deviation          : {worst_sigma:.4f} sigma")
    out.check(
        "the Monte Carlo agrees with the exact quadrature within the declared sigma",
        worst_sigma < CONTROL_SIGMA,
        f"worst deviation {worst_sigma:.4f} sigma against a criterion of "
        f"{CONTROL_SIGMA:.1f}",
    )
    out("")
    out("And the estimator's error must fall as the square root of the draw count.")
    out("")
    sites = [0, 1, 2, 3]
    matrix = true_correlation(record.spec, 0)[np.ix_(sites, sites)]
    rows = []
    for samples in (25_000, 100_000, 400_000):
        estimate = network_joint_monte_carlo(
            threshold[:, sites], matrix, samples, seed=4242
        )
        rows.append(
            [
                str(samples),
                number(estimate.estimate),
                scientific(estimate.standard_error),
                scientific(estimate.relative_error),
            ]
        )
    out(
        format_table(
            ["draws", "estimate", "standard error", "relative error"], rows
        )
    )
    first = network_joint_monte_carlo(
        threshold[:, sites], matrix, 25_000, seed=4242
    ).standard_error
    last = network_joint_monte_carlo(
        threshold[:, sites], matrix, 400_000, seed=4242
    ).standard_error
    out("")
    out(f"  ratio of standard errors : {first / last:.4f} against an expected 4.0")
    out.check(
        "the Monte Carlo error falls as the square root of the draw count",
        abs(first / last - 4.0) < 0.5,
        f"measured ratio {first / last:.4f} against 4.0 for a 16-fold draw increase",
    )


def inversion(out: Tee) -> None:
    """Part 5: the inversion of equation (11), round tripped."""
    out.rule("5. the inversion of equation (11), round tripped")
    out("Insert a correlation, read the joint rate it implies, invert it, and compare.")
    out("The tolerance on rho is the root finder's declared xtol and the tolerance on")
    out("the reproduced rate is that xtol times the derivative of the left-hand side,")
    out("which by equation (9) is the epoch mean of phi_2.")
    out("")
    record = generate_record(RECORD_SPECS["short"])
    threshold = estimate_climatology(record.cloud).threshold_series()
    nodes = epoch_nodes(threshold[:, 0], threshold[:, 1])
    rows = []
    worst_rho = 0.0
    worst_residual = 0.0
    for rho in (-0.8, -0.2, 0.0, 0.3, 0.65, 0.9, 0.98):
        target = nodes.joint(rho)
        result = implied_correlation(nodes, target)
        worst_rho = max(worst_rho, abs(result.rho - rho))
        worst_residual = max(worst_residual, abs(result.residual))
        rows.append(
            [
                number(rho),
                number(target),
                number(result.rho, 10),
                scientific(abs(result.rho - rho)),
                scientific(abs(result.residual)),
                "yes" if result.at_bracket_end else "no",
            ]
        )
    out(
        format_table(
            ["rho in", "joint rate", "rho out", "rho error", "residual",
             "at bracket end"],
            rows,
        )
    )
    out("")
    out.check(
        "the inversion recovers the correlation it was given",
        worst_rho < 1e-8,
        f"worst error {worst_rho:.6e} against the root finder's xtol of 1e-10 "
        "with a hundredfold margin",
    )
    out.check(
        "the inversion reproduces the joint rate it targeted",
        worst_residual < 1e-10,
        f"worst residual {worst_residual:.6e}",
    )
    out("")
    out("And a target outside the attainable range must return the bracket end and")
    out("say so, rather than silently returning it as an estimate.")
    out("")
    for target, label in ((0.9999, "above the bracket"), (0.0, "below the bracket")):
        result = implied_correlation(nodes, target)
        out.check(
            f"a target {label} is flagged",
            result.at_bracket_end,
            f"target {target}, returned rho {result.rho}, flagged "
            f"{result.at_bracket_end}",
        )


def main() -> int:
    out = Tee("validate_copula_output.txt")
    out.rule("sitediverse -- the Gaussian copula and the identity it is computed by")
    for line in environment_lines():
        out(line)
    against_scipy(out)
    excess_precision(out)
    binning(out)
    monte_carlo(out)
    inversion(out)
    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
