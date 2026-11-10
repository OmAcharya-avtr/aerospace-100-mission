"""Negative controls. Every one anchored on a TRUE value, and asserted to FIRE.

The rule this script implements, from the build guide of this mission: injecting
a violation by OFFSETTING an output does not guarantee a violation, because the
output already sits inside the band. Every control below is anchored on a value
the GENERATOR declared -- a correlation, a threshold, a joint probability -- or
on a bound derived in closed form, and the offset is strictly larger than twice
the criterion so that a violation is guaranteed by construction rather than by
luck.

Each control records whether it FIRED and compares that against what it was
declared to do. A control that does not fire is a failure of the control, not a
success of the product: it means the check it guards would not have caught the
violation it exists to catch. Controls that must NOT fire are declared too,
because a control firing when it should not is equally wrong.

No control's criterion is a seed. Where a criterion is a statistical interval
the declared level is four sigma and not 95 per cent, because a 95 per cent
criterion fires once in twenty by construction.
"""

from __future__ import annotations

import sys  # noqa: E402

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
from scipy.stats import multivariate_normal, norm  # noqa: E402

from _bootstrap import (  # noqa: E402
    BLOCK,
    CONTROL_SIGMA,
    MONTE_CARLO_SAMPLES,
    NETWORKS,
    REPLICATES,
    Tee,
    bootstrap,
    environment_lines,
)

ROOT = bootstrap()

from sitediverse.bounds import (  # noqa: E402
    bonferroni_bracket,
    frechet_hoeffding,
)
from sitediverse.climatology import estimate_climatology  # noqa: E402
from sitediverse.copula import (  # noqa: E402
    bivariate_orthant,
    epoch_nodes,
    implied_correlation,
    network_joint_monte_carlo,
    pair_joint_exact,
)
from sitediverse.correlation import fit_exponential, matrix_from_pairs  # noqa: E402
from sitediverse.empirical import (  # noqa: E402
    bootstrap_joint_rate,
    joint_rate,
    pair_clear_matrix,
)
from sitediverse.field import (  # noqa: E402
    RECORD_SPECS,
    REGIME_COUNT,
    generate_record,
    true_correlation,
)
from sitediverse.independence import overstatement, product_of_marginals  # noqa: E402
from sitediverse.report import format_table, number, scientific  # noqa: E402


def bound_controls(out: Tee, record, climatology) -> None:
    """Controls 1 to 6: the distribution-free bounds."""
    out.rule("controls 1 to 6: the bounds of equations (17) and (18)")
    out("Anchored on the bounds themselves, which are closed-form functions of the")
    out("marginals. The offset is twice the slack plus a tenth of the bound's own")
    out("width, so a violation is guaranteed by construction.")
    out("")
    sites = NETWORKS["cluster"]
    rates = climatology.long_term_rate[list(sites)]
    bounds = frechet_hoeffding(rates)
    width = max(bounds.width, 1e-6)
    out(f"  network                  : cluster, marginals "
        f"{', '.join(f'{rate:.6f}' for rate in rates)}")
    out(f"  Frechet bounds           : [{bounds.lower:.6f}, {bounds.upper:.6f}]")
    out(f"  slack, equation (19)     : {bounds.slack:.6e}")
    out(f"  offset applied           : {2.0 * bounds.slack + 0.1 * width:.6e}")
    out("")
    offset = 2.0 * bounds.slack + 0.1 * width
    out.control(
        "a value above the Frechet upper bound is detected",
        fired=not bounds.contains(bounds.upper + offset),
        must_fire=True,
        detail=f"probe {bounds.upper + offset:.6f} against an upper bound of "
        f"{bounds.upper:.6f}",
    )
    out.control(
        "a value below the Frechet lower bound is detected",
        fired=not bounds.contains(bounds.lower - offset),
        must_fire=True,
        detail=f"probe {bounds.lower - offset:.6f} against a lower bound of "
        f"{bounds.lower:.6f}",
    )
    out.control(
        "a value inside the Frechet bounds is NOT flagged",
        fired=not bounds.contains(0.5 * (bounds.lower + bounds.upper)),
        must_fire=False,
        detail=f"probe {0.5 * (bounds.lower + bounds.upper):.6f}, the midpoint of "
        "the bounds",
    )
    out("")
    out("A pair whose marginals sum above one has an INFORMATIVE lower bound, so the")
    out("lower-bound control is run again on one of those: on a network whose lower")
    out("bound has clipped to zero the control below the bound is a control against")
    out("a negative probability and is weaker than it looks.")
    out("")
    informative = None
    for site_i in range(record.site_count):
        for site_j in range(site_i + 1, record.site_count):
            probe = frechet_hoeffding(climatology.long_term_rate[[site_i, site_j]])
            if probe.lower_informative:
                informative = (site_i, site_j, probe)
                break
        if informative is not None:
            break
    if informative is None:
        out.control(
            "a pair with an informative lower bound exists to run the control on",
            fired=False,
            must_fire=True,
            detail="no pair of the declared network has sum q > 1",
        )
    else:
        site_i, site_j, probe = informative
        width = max(probe.width, 1e-6)
        offset = 2.0 * probe.slack + 0.1 * width
        out(f"  pair                     : {record.site_names[site_i]}-"
            f"{record.site_names[site_j]}, marginals "
            f"{climatology.long_term_rate[site_i]:.6f}, "
            f"{climatology.long_term_rate[site_j]:.6f}")
        out(f"  lower bound              : {probe.lower:.6f} (informative)")
        out("")
        out.control(
            "a value below an INFORMATIVE Frechet lower bound is detected",
            fired=not probe.contains(probe.lower - offset),
            must_fire=True,
            detail=f"probe {probe.lower - offset:.6f} against a lower bound of "
            f"{probe.lower:.6f}, slack {probe.slack:.6e}",
        )
        out.control(
            "the measured joint rate of that pair is NOT flagged",
            fired=not probe.contains(joint_rate(record.cloud, (site_i, site_j))),
            must_fire=False,
            detail=f"measured {joint_rate(record.cloud, (site_i, site_j)):.6f}",
        )
    out("")
    out("And the n = 2 Bonferroni identity, which is exact. The clear-rate matrix is")
    out("perturbed by a hundred times the measured residual, which is anchored on")
    out("the identity's own closure and not on any output.")
    out("")
    clear = pair_clear_matrix(record.cloud)
    pair = (0, 1)
    rates = climatology.long_term_rate[list(pair)]
    bracket = bonferroni_bracket(rates, clear[np.ix_(pair, pair)])
    measured = joint_rate(record.cloud, pair)
    residual = abs(bracket.upper - measured)
    perturbation = 100.0 * max(residual, bracket.slack)
    corrupted = clear[np.ix_(pair, pair)].copy()
    corrupted[0, 1] = corrupted[1, 0] = min(
        1.0, corrupted[0, 1] + perturbation
    )
    broken = bonferroni_bracket(rates, corrupted)
    out(f"  identity residual        : {residual:.6e}")
    out(f"  perturbation applied     : {perturbation:.6e}")
    out.control(
        "a corrupted pair clear rate breaks the n = 2 inclusion-exclusion identity",
        fired=abs(broken.upper - measured) > 10.0 * max(residual, bracket.slack),
        must_fire=True,
        detail=f"corrupted residual {abs(broken.upper - measured):.6e} against the "
        f"true residual {residual:.6e}",
    )


def copula_controls(out: Tee, record, climatology) -> None:
    """Controls 7 to 11: the copula against SciPy and against Monte Carlo."""
    out.rule("controls 7 to 11: the copula of equations (8) to (11)")
    out("Anchored on SciPy's independent implementation and on the exact quadrature.")
    out("The perturbation in rho is chosen so that the resulting change in the")
    out("orthant probability exceeds the check's tolerance by a known factor,")
    out("computed from the derivative of equation (9), which is the density.")
    out("")
    level = -1.5
    rho = 0.6
    tolerance = 1e-14
    density = float(np.abs(bivariate_orthant(level, level, rho) * 0.0) + 1.0)
    derivative = float(
        np.abs(
            bivariate_orthant(level, level, rho + 1e-6)
            - bivariate_orthant(level, level, rho)
        )
        / 1e-6
    )
    delta = 10.0 * tolerance / derivative
    reference = float(
        multivariate_normal.cdf(
            [level, level], mean=[0.0, 0.0], cov=[[1.0, rho], [rho, 1.0]]
        )
    )
    out(f"  h = k                    : {level}")
    out(f"  rho                      : {rho}")
    out(f"  d Phi_2 / d rho          : {derivative:.6e}")
    out(f"  tolerance                : {tolerance:.6e}")
    out(f"  perturbation in rho      : {delta:.6e}")
    out("")
    out.control(
        "a perturbed correlation makes the copula disagree with SciPy",
        fired=abs(float(bivariate_orthant(level, level, rho + delta)) - reference)
        > tolerance,
        must_fire=True,
        detail=f"disagreement "
        f"{abs(float(bivariate_orthant(level, level, rho + delta)) - reference):.6e} "
        f"against a tolerance of {tolerance:.6e}",
    )
    out.control(
        "the unperturbed correlation does NOT make it disagree",
        fired=abs(float(bivariate_orthant(level, level, rho)) - reference)
        > tolerance,
        must_fire=False,
        detail=f"disagreement "
        f"{abs(float(bivariate_orthant(level, level, rho)) - reference):.6e}",
    )
    _ = density
    out("")
    out("The Monte Carlo of equation (10) against the exact pair quadrature. The")
    out("perturbation in rho is scaled so that the induced change in the orthant")
    out("probability is eight Monte Carlo standard errors, which is twice the")
    out("declared four-sigma criterion.")
    out("")
    threshold = climatology.threshold_series()
    pair = (0, 1)
    true_rho = float(true_correlation(record.spec, 0, record.sites)[pair])
    exact = pair_joint_exact(threshold[:, pair[0]], threshold[:, pair[1]], true_rho)
    honest = network_joint_monte_carlo(
        threshold[:, list(pair)],
        np.array([[1.0, true_rho], [true_rho, 1.0]]),
        MONTE_CARLO_SAMPLES,
        seed=31337,
    )
    slope = (
        pair_joint_exact(
            threshold[:, pair[0]], threshold[:, pair[1]], true_rho + 1e-4
        )
        - exact
    ) / 1e-4
    rho_offset = 2.0 * CONTROL_SIGMA * honest.standard_error / abs(slope)
    broken = network_joint_monte_carlo(
        threshold[:, list(pair)],
        np.array(
            [[1.0, true_rho + rho_offset], [true_rho + rho_offset, 1.0]]
        ),
        MONTE_CARLO_SAMPLES,
        seed=31337,
    )
    out(f"  true rho                 : {true_rho:.6f}")
    out(f"  exact joint              : {exact:.6f}")
    out(f"  Monte Carlo standard err : {honest.standard_error:.6e}")
    out(f"  d joint / d rho          : {slope:.6f}")
    out(f"  rho offset for 8 sigma   : {rho_offset:.6e}")
    out("")
    out.control(
        "a Monte Carlo run at a perturbed correlation misses the exact quadrature",
        fired=abs(broken.estimate - exact) / broken.standard_error > CONTROL_SIGMA,
        must_fire=True,
        detail=f"{abs(broken.estimate - exact) / broken.standard_error:.4f} sigma "
        f"against a criterion of {CONTROL_SIGMA:.1f}",
    )
    out.control(
        "the Monte Carlo at the TRUE correlation does NOT miss it",
        fired=abs(honest.estimate - exact) / honest.standard_error > CONTROL_SIGMA,
        must_fire=False,
        detail=f"{abs(honest.estimate - exact) / honest.standard_error:.4f} sigma",
    )
    out("")
    out("And the inversion of equation (11) must flag a target the copula cannot")
    out("reach. Anchored on the attainable range itself: the target is the Frechet")
    out("upper bound, which the copula approaches only as rho tends to one.")
    out("")
    nodes = epoch_nodes(threshold[:, pair[0]], threshold[:, pair[1]])
    unreachable = float(min(climatology.long_term_rate[list(pair)]))
    out.control(
        "an unattainable target joint rate is flagged at the bracket end",
        fired=implied_correlation(nodes, unreachable).at_bracket_end,
        must_fire=True,
        detail=f"target {unreachable:.6f}, the Frechet upper bound of the pair",
    )


def generator_controls(out: Tee, record, climatology) -> None:
    """Controls 12 to 16: the generator checks, anchored on declared constants."""
    out.rule("controls 12 to 16: the generator checks of validate_field.py")
    out("Anchored on the generator's declared constants. Every offset is a multiple")
    out("of the check's own standard error, computed from the record length and the")
    out("declared persistence, so a violation is guaranteed.")
    out("")
    observed = record.cloud.mean(axis=0)
    truth = record.marginal_rate_true()
    effective = record.epochs * (1.0 - record.spec.persistence) / (
        1.0 + record.spec.persistence
    )
    errors = np.sqrt(truth * (1.0 - truth) / effective)
    honest = float(np.max(np.abs(observed - truth) / errors))
    offset = 2.0 * CONTROL_SIGMA * errors
    out(f"  effective sample size    : {effective:.1f}")
    out(f"  station standard error   : {float(errors.min()):.6e} to "
        f"{float(errors.max()):.6e}")
    out(f"  honest worst deviation   : {honest:.4f} sigma")
    out(f"  offset applied           : {2.0 * CONTROL_SIGMA:.1f} sigma")
    out("")
    out.control(
        "a marginal rate offset by 8 sigma fails the declared 4-sigma check",
        fired=float(np.max(np.abs(observed + offset - truth) / errors))
        > CONTROL_SIGMA,
        must_fire=True,
        detail=f"{float(np.max(np.abs(observed + offset - truth) / errors)):.4f} "
        f"sigma against a criterion of {CONTROL_SIGMA:.1f}",
    )
    out.control(
        "the unperturbed marginal rate does NOT fail it",
        fired=honest > CONTROL_SIGMA,
        must_fire=False,
        detail=f"{honest:.4f} sigma",
    )
    out("")
    out("The spatial correlation check, anchored on the WRONG regime's declared")
    out("matrix. This is not an offset applied to an output: it is the true matrix of")
    out("a different regime, which is a correct matrix for the wrong data.")
    out("")
    mask = record.regime == 0
    measured = np.corrcoef(record.latent[mask], rowvar=False)
    upper = np.triu_indices(record.site_count, 1)
    regime_effective = mask.sum() * (1.0 - record.spec.persistence) / (
        1.0 + record.spec.persistence
    )
    error = 1.0 / float(np.sqrt(regime_effective))
    right = true_correlation(record.spec, 0, record.sites)
    wrong = true_correlation(record.spec, 2, record.sites)
    honest_sigma = float(np.max(np.abs(measured[upper] - right[upper]))) / error
    wrong_sigma = float(np.max(np.abs(measured[upper] - wrong[upper]))) / error
    out(f"  regime 0 epochs          : {int(mask.sum())}")
    out(f"  correlation standard err : {error:.6e}")
    out(f"  against regime 0's matrix: {honest_sigma:.4f} sigma")
    out(f"  against regime 2's matrix: {wrong_sigma:.4f} sigma")
    out("")
    out.control(
        "regime 0's latent field does not match regime 2's declared matrix",
        fired=wrong_sigma > CONTROL_SIGMA,
        must_fire=True,
        detail=f"{wrong_sigma:.4f} sigma against a criterion of "
        f"{CONTROL_SIGMA:.1f}; the regimes differ by a factor of "
        f"{record.spec.along_km[0] / record.spec.along_km[2]:.2f} in L_along",
    )
    out.control(
        "regime 0's latent field DOES match regime 0's declared matrix",
        fired=honest_sigma > CONTROL_SIGMA,
        must_fire=False,
        detail=f"{honest_sigma:.4f} sigma",
    )
    out("")
    out("The decomposition of equation (16), which is exact. A perturbation of one")
    out("term by a hundred times the identity's own residual must open it.")
    out("")
    result = overstatement(climatology, record.cloud, NETWORKS["cluster"])
    residual = abs(result.identity_residual())
    perturbation = max(100.0 * residual, 1e-12)
    broken = (
        result.total_difference
        - (result.comodulation_term + perturbation + result.correlation_term)
    )
    out(f"  honest residual          : {residual:.6e}")
    out(f"  perturbation applied     : {perturbation:.6e}")
    out.control(
        "a perturbed co-modulation term opens the exact decomposition",
        fired=abs(broken) > 1e-15,
        must_fire=True,
        detail=f"perturbed residual {abs(broken):.6e} against a tolerance of 1e-15",
    )


def model_controls(out: Tee, record, climatology) -> None:
    """Controls 17 to 22: the fitted baselines and the learned comparison."""
    out.rule("controls 17 to 22: the fitted models")
    out("A correctly specified fit on noiseless data must recover its own")
    out("parameters, and a fit on data generated by a DIFFERENT law must not. Both")
    out("are anchored on declared parameters.")
    out("")
    separation = np.linspace(90.0, 7000.0, 120)
    declared = (0.92, 480.0)
    exact = fit_exponential(
        separation, declared[0] * np.exp(-separation / declared[1])
    )
    out(f"  declared rho_0, L        : {declared[0]}, {declared[1]} km")
    out(f"  fitted rho_0, L          : {exact.nugget:.6f}, {exact.length_km:.4f} km")
    out(f"  residual rms             : {exact.residual_rms:.6e}")
    out.control(
        "a correctly specified noiseless fit does NOT fail its own parameters",
        fired=abs(exact.length_km - declared[1]) / declared[1] > 1e-4,
        must_fire=False,
        detail=f"relative error in L "
        f"{abs(exact.length_km - declared[1]) / declared[1]:.6e}",
    )
    gaussian = declared[0] * np.exp(-((separation / declared[1]) ** 2))
    wrong = fit_exponential(separation, gaussian)
    function_error = float(np.max(np.abs(wrong.predict(separation) - gaussian)))
    out("")
    out("  a GAUSSIAN correlation law, fitted by the exponential form:")
    out(f"    fitted rho_0, L        : {wrong.nugget:.6f}, {wrong.length_km:.4f} km")
    out(f"    worst function error   : {function_error:.6f}")
    out.control(
        "an exponential fit to a GAUSSIAN correlation law leaves a visible "
        "function error",
        fired=function_error > 0.02,
        must_fire=True,
        detail=f"worst function error {function_error:.6f} against a tolerance of "
        "0.02; the exponential form cannot represent a Gaussian decay and the "
        "mis-specification is visible in the residual",
    )
    out("")
    out("The correlation matrix projection of equation (26), anchored on a matrix")
    out("constructed to be indefinite: a three-station cycle with two strong positive")
    out("correlations and one strong negative one, which no Gaussian field has.")
    out("")
    indefinite = np.array(
        [[1.0, 0.95, -0.95], [0.95, 1.0, 0.95], [-0.95, 0.95, 1.0]]
    )
    projected = matrix_from_pairs(indefinite)
    definite = matrix_from_pairs(
        true_correlation(record.spec, 0, record.sites)[np.ix_((0, 1, 2), (0, 1, 2))]
    )
    out(f"  least eigenvalue, indefinite : "
        f"{projected.least_eigenvalue_before:.6e}")
    out(f"  Frobenius shift              : {projected.frobenius_shift:.6e}")
    out(f"  least eigenvalue, generator  : "
        f"{definite.least_eigenvalue_before:.6e}")
    out.control(
        "an indefinite pair matrix is detected and projected, and the projection "
        "is reported",
        fired=projected.projected and projected.frobenius_shift > 0.0,
        must_fire=True,
        detail=f"least eigenvalue {projected.least_eigenvalue_before:.6e}, "
        f"Frobenius shift {projected.frobenius_shift:.6e}",
    )
    out.control(
        "the generator's own matrix is NOT projected",
        fired=definite.projected,
        must_fire=False,
        detail=f"least eigenvalue {definite.least_eigenvalue_before:.6e}",
    )
    out("")
    out("The learned comparison of equation (41), anchored on the TRUE joint")
    out("unavailability. A model whose prediction is the true value plus an offset of")
    out("twice the reference half-width must produce a margin the paired bootstrap")
    out("interval excludes.")
    out("")
    sites = NETWORKS["cluster"]
    band = bootstrap_joint_rate(
        record.cloud, sites, block=BLOCK, replicates=REPLICATES, seed=55
    )
    measured = joint_rate(record.cloud, sites)
    offset = 2.0 * band.half_width
    out(f"  measured joint rate      : {measured:.6f}")
    out(f"  bootstrap half-width     : {band.half_width:.6f}")
    out(f"  offset applied           : {offset:.6f}")
    out.control(
        "a model offset by twice the bootstrap half-width is outside the interval",
        fired=not band.contains(measured + offset),
        must_fire=True,
        detail=f"probe {measured + offset:.6f} against an interval of "
        f"[{band.lower:.6f}, {band.upper:.6f}]",
    )
    out.control(
        "a model offset by a TENTH of the half-width is NOT outside it",
        fired=not band.contains(measured + 0.1 * band.half_width),
        must_fire=False,
        detail=f"probe {measured + 0.1 * band.half_width:.6f}",
    )


def independence_controls(out: Tee, record, climatology) -> None:
    """Controls 23 to 26: the headline, and the record it rests on."""
    out.rule("controls 23 to 26: the headline overstatement")
    out("Anchored on a record generated with NO spatial correlation at all. On such")
    out("a record the independence assumption is TRUE, so the overstatement must")
    out("vanish -- and a check that reports an overstatement there is reporting")
    out("something other than correlation.")
    out("")
    from sitediverse.field import FieldSpec

    independent_spec = FieldSpec(
        **{
            **RECORD_SPECS["short"].__dict__,
            "nugget": 1e-9,
            "along_km": (1.0, 1.0, 1.0, 1.0),
            "cross_km": (1.0, 1.0, 1.0, 1.0),
            "seasonal_amplitude": (0.0, 0.0, 0.0, 0.0, 0.0),
            "diurnal_amplitude": 0.0,
            "persistence": 0.0,
            "seed": 606060,
        }
    )
    uncorrelated = generate_record(independent_spec)
    flat = estimate_climatology(uncorrelated.cloud)
    sites = NETWORKS["cluster"]
    control_result = overstatement(flat, uncorrelated.cloud, sites)
    control_band = bootstrap_joint_rate(
        uncorrelated.cloud, sites, block=1, replicates=REPLICATES, seed=66
    )
    honest_result = overstatement(climatology, record.cloud, sites)
    honest_band = bootstrap_joint_rate(
        record.cloud, sites, block=BLOCK, replicates=REPLICATES, seed=66
    )
    out("  A record with no spatial correlation, no seasonal or diurnal modulation")
    out("  and no persistence. Everything the independence assumption assumes is")
    out("  true of it by construction.")
    out("")
    out(
        format_table(
            ["record", "P_naive", "P_measured", "ratio", "difference",
             "difference / bootstrap sd"],
            [
                [
                    "uncorrelated control",
                    number(control_result.naive),
                    number(control_result.measured),
                    number(control_result.ratio, 4),
                    f"{control_result.total_difference:+.6f}",
                    f"{control_result.total_difference / control_band.standard_error:+.4f}",
                ],
                [
                    "anisotropic reference",
                    number(honest_result.naive),
                    number(honest_result.measured),
                    number(honest_result.ratio, 4),
                    f"{honest_result.total_difference:+.6f}",
                    f"{honest_result.total_difference / honest_band.standard_error:+.4f}",
                ],
            ],
        )
    )
    out("")
    out.control(
        "the overstatement VANISHES on a record with no spatial correlation",
        fired=abs(control_result.total_difference)
        > CONTROL_SIGMA * control_band.standard_error,
        must_fire=False,
        detail=f"difference {control_result.total_difference:+.6f} is "
        f"{abs(control_result.total_difference) / control_band.standard_error:.4f} "
        f"bootstrap standard errors, against a criterion of {CONTROL_SIGMA:.1f}",
    )
    out.control(
        "the overstatement is PRESENT on the correlated reference record",
        fired=honest_result.total_difference
        > CONTROL_SIGMA * honest_band.standard_error,
        must_fire=True,
        detail=f"difference {honest_result.total_difference:+.6f} is "
        f"{honest_result.total_difference / honest_band.standard_error:.4f} "
        f"bootstrap standard errors",
    )
    out("")
    out("And the co-modulation term of equation (16) must vanish on a record with no")
    out("seasonal or diurnal modulation, because that is exactly what it measures.")
    out("")
    out(f"  control co-modulation    : {control_result.comodulation_term:+.6e}")
    out(f"  reference co-modulation  : {honest_result.comodulation_term:+.6e}")
    out.control(
        "the co-modulation term vanishes with no seasonal or diurnal modulation",
        fired=abs(control_result.comodulation_term) > 1e-4,
        must_fire=False,
        detail=f"{control_result.comodulation_term:+.6e} against a tolerance of "
        "1e-04; the control record's marginals are constant in time so the "
        "epoch-wise product is the long-term product",
    )
    out.control(
        "the co-modulation term is non-zero on the modulated reference record",
        fired=abs(honest_result.comodulation_term) > 1e-4,
        must_fire=True,
        detail=f"{honest_result.comodulation_term:+.6e}",
    )
    _ = norm, product_of_marginals, scientific, REGIME_COUNT


def main() -> int:
    out = Tee("validate_negative_controls_output.txt")
    out.rule("sitediverse -- negative controls, every one anchored on a true value")
    for line in environment_lines():
        out(line)
    out("")
    out(f"declared control sigma       : {CONTROL_SIGMA}")
    out("")
    out("Four, not two: a control whose criterion is a 95 per cent interval fires")
    out("once in twenty by construction, and a predecessor in this mission corrected")
    out("such a control rather than changing the seed until it passed. No control's")
    out("criterion here is a seed, and every offset is a multiple of the check's own")
    out("criterion rather than a chosen constant.")
    record = generate_record(RECORD_SPECS["anisotropic"])
    climatology = estimate_climatology(record.cloud)
    bound_controls(out, record, climatology)
    copula_controls(out, record, climatology)
    generator_controls(out, record, climatology)
    model_controls(out, record, climatology)
    independence_controls(out, record, climatology)
    out.rule("summary")
    out(f"  controls declared        : {out.controls_expected}")
    out(f"  controls that fired      : {out.controls_fired}")
    out(f"  controls that must fire  : {out.controls_fired} fired of the declared "
        "must-fire set; see each line above")
    out("")
    out("A control that does not fire is a failure of the CONTROL, not a success of")
    out("the product. Both directions are declared and both are asserted.")
    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
