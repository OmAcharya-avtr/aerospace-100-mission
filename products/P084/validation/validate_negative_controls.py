"""Negative controls, every one anchored on the TRUE value, every one asserted to fire.

Why the anchoring matters
-------------------------
A control that injects a violation by offsetting an OUTPUT does not guarantee
a violation: the output already sits inside the band, so only an offset
strictly larger than the band guarantees leaving it. One product of this
mission shipped two of eleven controls silently too weak that way. The
certificate was right and the control was wrong, which is the worse of the two
failures because it reads as evidence.

So every control below is anchored on a quantity this package's synthetic
generator makes KNOWN -- the true mean gap, the true population quantile, the
true conditional mean gap of equation (22), or the declared alpha of a
hypothesis test -- and never on a bound, a model prediction or a seed.

Both directions are asserted. A control declared must-NOT-fire that fires is
as wrong as a must-fire that does not, and several of the pairs below exist
precisely so that the firing one cannot be passing for a trivial reason.
"""

from __future__ import annotations

# Bytecode must not be written anywhere inside this tree. validation/
# validate_cli.py asserts that no __pycache__ directory and no .pyc file
# exists in the repository, so an interpreter that caches its own imports
# here makes that check report itself. It is set in each entry script,
# before _bootstrap is imported, and not inside _bootstrap, because CPython
# writes a module's cache BEFORE executing that module: _bootstrap cannot
# set this on its own behalf. The imports below therefore follow a statement
# and carry the same E402 waiver the post-bootstrap imports already carry.
import sys  # noqa: E402

sys.dont_write_bytecode = True

import math  # noqa: E402

import numpy as np  # noqa: E402

from _bootstrap import (  # noqa: E402
    CONFIDENCE,
    PROPORTION,
    SEED_MODEL,
    SEED_POOL,
    SEED_TRUTH,
    Tee,
    bootstrap,
    environment_lines,
)

ROOT = bootstrap()

from transfergap import synthetic  # noqa: E402
from transfergap.bounds import hoeffding_mean_upper, required_n_wilks  # noqa: E402
from transfergap.discrepancy import validate_gaps  # noqa: E402
from transfergap.exchangeability import permutation_test, trend_statistic  # noqa: E402
from transfergap.learned import DiscrepancyModel, plugin_mean_upper  # noqa: E402
from transfergap.report import format_table  # noqa: E402
from transfergap.resample import conformal_order_index  # noqa: E402

TRIALS = 400
ALPHA = 1.0 - CONFIDENCE
R = synthetic.BOUND_RANGE_KNM


def coverage_of_a_mean_bound(n: int, true_mean: float, slack_factor: float) -> float:
    """Fraction of campaigns whose Hoeffding bound covers the TRUE mean.

    ``slack_factor`` multiplies the slack of equation (3): 1.0 is the correct
    bound, anything below 1.0 is a deliberately corrupted one. The comparison
    is against ``true_mean``, which comes from the synthetic generator and not
    from any bound.
    """
    covered = 0
    for trial in range(TRIALS):
        sample = synthetic.generate(n, "random", seed=1_100_000 + 1019 * trial)
        bound = hoeffding_mean_upper(sample.gaps, R, CONFIDENCE)
        corrupted = bound.estimate + slack_factor * bound.slack
        covered += int(corrupted >= true_mean)
    return covered / TRIALS


def tolerance_rate_at(n: int, reference: float, proportion: float) -> float:
    """Fraction of campaigns whose largest observation reaches the TRUE quantile."""
    reached = 0
    for trial in range(TRIALS):
        sample = synthetic.generate(n, "random", seed=1_200_000 + 1021 * trial)
        reached += int(float(np.max(sample.gaps)) >= reference)
    return reached / TRIALS


def conformal_coverage_at(n: int, offset: int, pool: np.ndarray) -> float:
    """Mean coverage of the conformal limit with the order index shifted by ``offset``.

    ``offset = 0`` is equation (12). A negative offset is a deliberately
    corrupted index. The coverage is measured against a fixed held-out pool of
    real draws from the population, so the reference is the population and not
    another limit.
    """
    coverages = []
    for trial in range(TRIALS):
        sample = synthetic.generate(n, "random", seed=1_300_000 + 1031 * trial)
        k = conformal_order_index(n, ALPHA) + offset
        if not 1 <= k <= n:
            continue
        limit = float(np.sort(sample.gaps)[k - 1])
        coverages.append(float(np.mean(pool <= limit)))
    return float(np.mean(coverages)) if coverages else float("nan")


def main() -> int:
    out = Tee("validate_negative_controls_output.txt")
    out.rule("transfergap -- negative controls, anchored on the truth")
    for line in environment_lines():
        out(line)
    truth = synthetic.population_truth(draws=400_000, seed=SEED_TRUTH)
    pool = synthetic.generate(4000, "random", seed=SEED_POOL + 1)
    conditional_truth = synthetic.conditional_mean_gap(pool.features)
    reference_quantile = truth.quantiles[PROPORTION]
    out("")
    out(f"true E[d]                       : {truth.mean:.6f} +/- "
        f"{truth.standard_error:.6f} {synthetic.UNIT}")
    out(f"true {PROPORTION:g} quantile of d           : "
        f"{reference_quantile:.6f} {synthetic.UNIT}")
    out(f"declared confidence             : {CONFIDENCE}")
    out(f"trials per control              : {TRIALS}")
    out(f"binomial se at a rate of {CONFIDENCE:g}  : "
        f"{math.sqrt(CONFIDENCE * ALPHA / TRIALS):.6f}")
    out("")
    out("A control FIRES when the quantity it watches crosses the declared criterion.")
    out("Every criterion below is the DECLARED level with a three-standard-error")
    out("Monte Carlo allowance, never a level chosen until the control passed.")

    allowance = 3.0 * math.sqrt(CONFIDENCE * ALPHA / TRIALS)

    out.rule("1. the mean bound: a corrupted slack must lose coverage")
    out("The corruption is a factor on the slack of equation (3). The factor needed")
    out("to GUARANTEE a miss is not assumed: it is derived from the sampling")
    out("standard deviation of the sample mean at this n, printed below, so that the")
    out("control cannot be too weak to fire. A control that cannot fire reads as")
    out("evidence and is worse than no control.")
    out("")
    sample_sd = math.sqrt(
        float(np.var(synthetic.generate(400_000, "random", seed=SEED_TRUTH + 3).gaps, ddof=1))
        / 40.0
    )
    nominal_slack = R * math.sqrt(math.log(1.0 / ALPHA) / 80.0)
    out(f"  sampling sd of the mean of 40 gaps : {sample_sd:.6f} {synthetic.UNIT}")
    out(f"  Hoeffding slack at n=40            : {nominal_slack:.6f} {synthetic.UNIT}")
    out(f"  true E[d]                          : {truth.mean:.6f} {synthetic.UNIT}")
    out("")
    rows = []
    for factor in (1.0, 0.75, 0.25, 0.0, -0.5, -1.0):
        rate = coverage_of_a_mean_bound(40, truth.mean, factor)
        margin = (truth.mean - factor * nominal_slack - truth.mean) / sample_sd
        rows.append([f"{factor:+.2f}", f"{-margin:+.2f}", f"{rate:.4f}",
                     f"{CONFIDENCE - allowance:.4f}",
                     "FIRED" if rate < CONFIDENCE - allowance else "did not fire"])
    out(format_table(
        ["slack factor", "corrupted bound minus truth, in sd of the mean",
         "coverage of the TRUE mean", "criterion", "control"], rows))
    out("")
    out.control(
        "the correct Hoeffding bound at n=40 keeps its coverage",
        coverage_of_a_mean_bound(40, truth.mean, 1.0) < CONFIDENCE - allowance,
        must_fire=False,
        detail=f"coverage {coverage_of_a_mean_bound(40, truth.mean, 1.0):.4f} against a "
               f"criterion of {CONFIDENCE - allowance:.4f}",
    )
    for factor in (-0.5, -1.0):
        rate = coverage_of_a_mean_bound(40, truth.mean, factor)
        sigmas = factor * nominal_slack / sample_sd
        out.control(
            f"a Hoeffding slack with the WRONG SIGN, factor {factor:+g}, loses coverage "
            f"of the true mean",
            rate < CONFIDENCE - allowance,
            must_fire=True,
            detail=f"coverage {rate:.4f} against a criterion of "
                   f"{CONFIDENCE - allowance:.4f}; the corrupted bound sits "
                   f"{-sigmas:.2f} sd of the sample mean BELOW the truth, so a miss is "
                   f"guaranteed and not lucky",
        )
    out("")
    out("REPORTED AS FOUND, and it changed the design of this control. A slack")
    out("SCALED DOWN does not lose coverage at n = 40 at any factor down to 0.25, and")
    out("not even at 0.00 -- the bound reduced to the bare sample mean still covers")
    out("the true mean on essentially every draw, because the sample mean of 40 gaps")
    out("has a sampling sd of about 0.61 kN.m and is unbiased. The first version of")
    out("this control used factors of 0.5 and 0.25 and did NOT fire, which is exactly")
    out("the too-weak-control defect this mission has shipped before: the corruption")
    out("was smaller than the quantity it had to overcome. The must-fire controls")
    out("above are therefore SIGN errors, whose margin below the truth is computed")
    out("from the sampling sd and printed, so the control is guaranteed to fire by")
    out("construction rather than by luck.")
    out("")
    out("The same measurement says something about the bound itself: Hoeffding's")
    out("inequality on a population with R = 40 kN.m and a mean of 5.1 is loose")
    out("enough that its entire slack could be deleted without losing coverage at")
    out("this sample size. That is the cost of distribution-freeness and it is the")
    out("reason this package reports the slack in its own column.")

    out.rule("2. Wilks' limit: a sample below the requirement must fail the statement")
    needed = required_n_wilks(PROPORTION, CONFIDENCE)
    rows = []
    for n in (needed, 45, 30, 20):
        rate = tolerance_rate_at(n, reference_quantile, PROPORTION)
        predicted = 1.0 - PROPORTION**n
        rows.append([str(n), f"{predicted:.4f}", f"{rate:.4f}",
                     f"{CONFIDENCE - allowance:.4f}",
                     "FIRED" if rate < CONFIDENCE - allowance else "did not fire"])
    out(format_table(["n", "predicted 1 - p^n", "measured rate", "criterion",
                      "control"], rows))
    out("")
    out.control(
        f"Wilks' limit at the required n={needed} reaches the declared confidence",
        tolerance_rate_at(needed, reference_quantile, PROPORTION) < CONFIDENCE - allowance,
        must_fire=False,
        detail=f"measured rate "
               f"{tolerance_rate_at(needed, reference_quantile, PROPORTION):.4f}",
    )
    for n in (30, 20):
        rate = tolerance_rate_at(n, reference_quantile, PROPORTION)
        out.control(
            f"Wilks' limit from only {n} measurements misses the declared confidence",
            rate < CONFIDENCE - allowance,
            must_fire=True,
            detail=f"measured rate {rate:.4f}, predicted 1 - p^n = "
                   f"{1.0 - PROPORTION**n:.4f}",
        )

    out.rule("3. split conformal: an off-by-one order index must lose coverage")
    rows = []
    for offset in (0, -1, -3, -8):
        rate = conformal_coverage_at(60, offset, pool.gaps)
        rows.append([str(offset), str(conformal_order_index(60, ALPHA) + offset),
                     f"{rate:.4f}", f"{CONFIDENCE - allowance:.4f}",
                     "FIRED" if rate < CONFIDENCE - allowance else "did not fire"])
    out(format_table(["index offset", "k used", "mean coverage on the pool", "criterion",
                      "control"], rows))
    out("")
    out.control(
        "the conformal limit at the index of equation (12) keeps its coverage",
        conformal_coverage_at(60, 0, pool.gaps) < CONFIDENCE - allowance,
        must_fire=False,
        detail=f"coverage {conformal_coverage_at(60, 0, pool.gaps):.4f}",
    )
    for offset in (-3, -8):
        rate = conformal_coverage_at(60, offset, pool.gaps)
        out.control(
            f"a conformal order index short by {-offset} loses coverage",
            rate < CONFIDENCE - allowance,
            must_fire=True,
            detail=f"coverage {rate:.4f} against a criterion of "
                   f"{CONFIDENCE - allowance:.4f}",
        )

    out.rule("4. the exchangeability test: both directions")
    out("The criterion is the DECLARED alpha of the test plus three standard errors,")
    out("not a level chosen to make a control pass.")
    out("")
    size_allowance = 3.0 * math.sqrt(0.05 * 0.95 / 200)
    rows = []
    verdicts = {}
    for campaign in ("random", "swept"):
        for statistic in ("trend", "half_split"):
            rejected = 0
            for trial in range(200):
                gaps = synthetic.generate(120, campaign, seed=1_400_000 + 53 * trial).gaps
                rejected += int(
                    permutation_test(
                        gaps, statistic, permutations=299, alpha=0.05, seed=trial
                    ).rejected
                )
            rate = rejected / 200
            verdicts[(campaign, statistic)] = rate
            rows.append([campaign, statistic, f"{rate:.4f}",
                         f"{0.05 + size_allowance:.4f}",
                         "FIRED" if rate > 0.05 + size_allowance else "did not fire"])
    out(format_table(["campaign", "statistic", "rejection rate", "criterion (alpha + 3 se)",
                      "control"], rows))
    out("")
    for statistic in ("trend", "half_split"):
        out.control(
            f"the {statistic} test does not reject an exchangeable campaign above alpha",
            verdicts[("random", statistic)] > 0.05 + size_allowance,
            must_fire=False,
            detail=f"rejection rate {verdicts[('random', statistic)]:.4f} at n=120",
        )
        out.control(
            f"the {statistic} test rejects the non-exchangeable swept campaign",
            verdicts[("swept", statistic)] > 0.05 + size_allowance,
            must_fire=True,
            detail=f"rejection rate {verdicts[('swept', statistic)]:.4f} at n=120",
        )

    out.rule("5. the learned model: shuffled labels must destroy its discrimination")
    out("The criterion is anchored on the TRUE conditional mean gap of equation (22),")
    out("not on the model's own output. A model fitted to gaps that have been")
    out("permuted relative to their operating conditions has no information left,")
    out("and its rank correlation with the conditional truth must collapse to zero.")
    out("")
    rows = []
    discrimination = {}
    for label, shuffle in (("as measured", False), ("labels shuffled", True)):
        correlations = []
        for trial in range(20):
            sample = synthetic.generate(60, "random", seed=1_500_000 + 61 * trial)
            gaps = sample.gaps
            if shuffle:
                gaps = np.random.default_rng(trial).permutation(gaps)
            model = DiscrepancyModel(seed=SEED_MODEL).fit(sample.features, gaps)
            predicted = model.predict(pool.features)
            paired = np.column_stack([predicted, conditional_truth])
            correlations.append(float(trend_statistic(paired[np.argsort(paired[:, 1]), 0])))
        mean = float(np.mean(correlations))
        error = float(np.std(correlations, ddof=1) / math.sqrt(len(correlations)))
        discrimination[label] = (mean, error)
        rows.append([label, f"{mean:.4f}", f"{error:.4f}", "0.2000",
                     "FIRED" if mean < 0.2 else "did not fire"])
    out(format_table(["training labels", "rank correlation with the conditional truth",
                      "+/- se", "criterion", "control"], rows))
    out("")
    out.control(
        "the learned model fitted to its own measurements discriminates conditions",
        discrimination["as measured"][0] < 0.2,
        must_fire=False,
        detail=f"correlation {discrimination['as measured'][0]:.4f} +/- "
               f"{discrimination['as measured'][1]:.4f} at n=60",
    )
    out.control(
        "the learned model fitted to shuffled labels loses all discrimination",
        discrimination["labels shuffled"][0] < 0.2,
        must_fire=True,
        detail=f"correlation {discrimination['labels shuffled'][0]:.4f} +/- "
               f"{discrimination['labels shuffled'][1]:.4f}",
    )

    out.rule("6. the plug-in learned mean bound: it must be caught under-covering")
    out("Equation (29) is in this package because it is the construction a programme")
    out("would reach for and it has no guarantee. The control asserts that the")
    out("measurement CATCHES it, anchored on the true mean.")
    out("")
    covered = 0
    for trial in range(200):
        sample = synthetic.generate(40, "random", seed=1_600_000 + 67 * trial)
        bound = plugin_mean_upper(
            sample.features, sample.gaps, pool.features, CONFIDENCE, SEED_MODEL
        )
        covered += int(bound.upper >= truth.mean)
    rate = covered / 200
    plugin_allowance = 3.0 * math.sqrt(CONFIDENCE * ALPHA / 200)
    out(f"  coverage of the true mean at n=40, 200 trials: {rate:.4f}")
    out(f"  criterion                                    : "
        f"{CONFIDENCE - plugin_allowance:.4f}")
    out.control(
        "the plug-in learned mean bound of equation (29) is caught under-covering",
        rate < CONFIDENCE - plugin_allowance,
        must_fire=True,
        detail=f"coverage {rate:.4f} against a declared {CONFIDENCE:g}",
    )

    out.rule("7. input validation: a range below the data must be refused")
    sample = synthetic.generate(200, "random", seed=1_700_000)
    refused = False
    try:
        validate_gaps(sample.gaps, float(np.max(sample.gaps)) * 0.5)
    except ValueError:
        refused = True
    out.control(
        "a declared R below the observed maximum is refused",
        refused,
        must_fire=True,
        detail=f"R = {float(np.max(sample.gaps)) * 0.5:.4f} against an observed maximum "
               f"of {float(np.max(sample.gaps)):.4f} {synthetic.UNIT}",
    )

    out.rule("8. a corruption deliberately too small to detect must NOT fire")
    out("A control that fires on everything is not a control. This one perturbs the")
    out("gaps by one part in 1e13, far below any reported precision, and asserts")
    out("that nothing changes: the Hoeffding bound moves by less than 1e-9 kN.m and")
    out("the exchangeability verdict is unchanged.")
    out("")
    clean = synthetic.generate(60, "random", seed=1_800_000)
    perturbed = clean.gaps * (1.0 + 1e-13)
    before = hoeffding_mean_upper(clean.gaps, R, CONFIDENCE).upper
    after = hoeffding_mean_upper(perturbed, R, CONFIDENCE).upper
    out(f"  bound before {before:.12f}, after {after:.12f}, change "
        f"{abs(after - before):.3e} {synthetic.UNIT}")
    out.control(
        "a one-part-in-1e13 perturbation does not move the bound by 1e-9",
        abs(after - before) > 1e-9,
        must_fire=False,
        detail=f"change {abs(after - before):.3e} {synthetic.UNIT}",
    )
    first = permutation_test(clean.gaps, "trend", permutations=299, seed=5)
    second = permutation_test(perturbed, "trend", permutations=299, seed=5)
    out.control(
        "a one-part-in-1e13 perturbation does not change the exchangeability verdict",
        first.rejected != second.rejected,
        must_fire=False,
        detail=f"p {first.p_value:.4f} before, {second.p_value:.4f} after",
    )

    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
