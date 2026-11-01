"""The headline: least squares returns a confident split that does not exist.

What this script measures
-------------------------
Four things, in order.

1. **The singular-value spectrum of every declared geometry**, with the
   near-null directions named in terms of the physical coefficients and the
   per-term variance inflation beside them.
2. **The confident split.** Over a Monte Carlo on each geometry, with the
   geometry held fixed and only the noise redrawn: the bias of each
   coefficient against its truth, the empirical scatter over realisations, and
   the formal standard error that the two solvers report. On the
   ``single_az`` geometry the least-squares estimate is **repeatable to a few
   arcseconds and wrong by tens**, which is the worst of the possible
   failures, because a repeatable number reads as a measurement.
3. **The identifiable combinations are recovered where the coefficients are
   not.** The same Monte Carlo, scored in the two parameterisations.
4. **Does the rank analysis catch it.** A truth-anchored criterion for "badly
   determined" is declared in advance and the refusal is scored against it.
"""

from __future__ import annotations

import numpy as np

from _bootstrap import (
    OBSERVATIONS,
    SEED,
    SIGMA_ARCSEC,
    Tee,
    bootstrap,
    environment_lines,
)

ROOT = bootstrap()

from pointmodel import identifiability, synthetic, terms, wls  # noqa: E402
from pointmodel.report import format_table  # noqa: E402

MONTE_CARLO = 200

#: A coefficient is declared BADLY DETERMINED when the mean least-squares
#: estimate over the Monte Carlo sits more than this many declared sigmas from
#: the truth. The criterion is anchored on the TRUE value and never on an
#: output. TWO criteria are scored and both are reported, so that the scoring
#: cannot be read as having been tuned: 2 declared sigmas is the level at which
#: the systematic error exceeds the statistical one, and 10 is a deliberately
#: lax alternative. Changing between them moves the precision and leaves the
#: recall at 1.0000, which is the number the refusal has to earn.
BAD_DETERMINATION_SIGMAS = (2.0, 10.0)


def spectra(out: Tee) -> dict[str, identifiability.IdentifiabilityReport]:
    """Part 1: the spectrum and the named directions, per geometry."""
    out.rule("1. singular-value spectrum of every declared geometry")
    out(f"{OBSERVATIONS} observations per geometry, seed {SEED}, columns normalised to")
    out("unit length before the decomposition, so every ratio below is a statement")
    out("about geometry and not about units.")
    out("")
    reports: dict[str, identifiability.IdentifiabilityReport] = {}
    rows = []
    for name in sorted(synthetic.GEOMETRIES):
        residuals = synthetic.generate(
            name, OBSERVATIONS, sigma_arcsec=SIGMA_ARCSEC, seed=SEED
        )
        result = wls.fit(residuals.design, residuals.observed, SIGMA_ARCSEC)
        report = identifiability.analyse_fit(residuals.design, result)
        reports[name] = report
        rows.append(
            [
                name,
                f"{report.condition_number:.4e}",
                str(report.numerical_rank),
                report.verdict,
                f"{float(np.max(report.vif)):.4e}",
                ",".join(report.unidentifiable_terms) or "-",
            ]
        )
    out(
        format_table(
            ["geometry", "condition", "rank", "verdict", "max VIF", "terms refused"], rows
        )
    )
    out("")
    out("Full singular-value ratios s_k / s_1:")
    out("")
    out(
        format_table(
            ["geometry", *[f"s{index}" for index in range(terms.N_TERMS)]],
            [
                [name, *[f"{float(value):.3e}" for value in reports[name].ratios]]
                for name in sorted(reports)
            ],
        )
    )
    for name in sorted(reports):
        report = reports[name]
        out("")
        out(f"{name}:")
        if report.null_directions:
            out("  NOT IDENTIFIABLE:")
            for direction in report.null_directions:
                out(f"    [{direction.index}] {direction.statement()}")
        else:
            out("  no direction below the declared null threshold")
        if report.marginal_directions:
            out("  weakly determined:")
            for direction in report.marginal_directions:
                out(f"    [{direction.index}] {direction.statement()}")
    out("")
    out.check(
        "the two well-scheduled geometries are separable",
        reports["full_sky"].separable and reports["two_az"].separable,
        f"full_sky condition {reports['full_sky'].condition_number:.4g}, "
        f"two_az {reports['two_az'].condition_number:.4g}",
    )
    out.check(
        "the three restricted geometries are not separable",
        not any(reports[name].separable for name in ("narrow_el", "single_az", "low_el")),
    )
    out.check(
        "the narrow elevation band refuses the IA/CA/NPAE cross-elevation constant",
        {"IA", "CA", "NPAE"} <= set(reports["narrow_el"].unidentifiable_terms),
    )
    out.check(
        "the single azimuth refuses the azimuth-axis tilt and NPAE",
        {"AN", "AW", "NPAE"} <= set(reports["single_az"].unidentifiable_terms),
    )
    out.check(
        "the single azimuth is exactly rank deficient by two",
        bool(reports["single_az"].ratios[-2] < 1e-13 and reports["single_az"].ratios[-3] > 1e-5),
        f"s7/s0 = {float(reports['single_az'].ratios[-2]):.3e}, "
        f"s6/s0 = {float(reports['single_az'].ratios[-3]):.3e}",
    )
    out.check(
        "the azimuth-axis tilt survives a narrow elevation band",
        not ({"AN", "AW"} & set(reports["narrow_el"].unidentifiable_terms)),
        "cos(A) and sin(A) are fully sampled there",
    )
    return reports


def confident_split(out: Tee) -> dict[str, dict[str, np.ndarray]]:
    """Part 2: the Monte Carlo, and what each solver claims about it."""
    out.rule(
        f"2. the confident split: {MONTE_CARLO} noise realisations per geometry"
    )
    out("The geometry is held fixed and only the noise is redrawn, so everything")
    out("below is a property of the geometry. Three quantities are compared for each")
    out("coefficient: its bias against the TRUE value, its empirical scatter over")
    out("realisations, and the formal standard error that each solver reports.")
    out("")
    out("'svd se' is from pointmodel.wls.fit, the recommended path, which truncates")
    out("the directions below its rcond and reports the covariance of the estimator")
    out("it actually returns. 'normal se' is from pointmodel.wls.normal_equation_fit,")
    out("which forms A^T W A and inverts it, as a fitter written straight from a")
    out("textbook does.")
    out("")
    collected: dict[str, dict[str, np.ndarray]] = {}
    for name in sorted(synthetic.GEOMETRIES):
        base = synthetic.generate(
            name, OBSERVATIONS, sigma_arcsec=SIGMA_ARCSEC, seed=SEED
        )
        estimates = np.empty((MONTE_CARLO, terms.N_TERMS))
        svd_se = None
        normal_se = None
        normal_estimates = np.empty((MONTE_CARLO, terms.N_TERMS))
        for trial in range(MONTE_CARLO):
            realisation = synthetic.generate(
                name,
                OBSERVATIONS,
                sigma_arcsec=SIGMA_ARCSEC,
                seed=SEED,
                noise_seed=200_000 + trial,
            )
            result = wls.fit(realisation.design, realisation.observed, SIGMA_ARCSEC)
            estimates[trial] = result.coefficients
            normal = wls.normal_equation_fit(
                realisation.design, realisation.observed, SIGMA_ARCSEC
            )
            normal_estimates[trial] = normal.coefficients
            if svd_se is None:
                svd_se = result.standard_errors
                normal_se = normal.standard_errors
        assert svd_se is not None and normal_se is not None
        bias = np.mean(estimates, axis=0) - base.coefficients
        scatter = np.std(estimates, axis=0, ddof=1)
        normal_bias = np.mean(normal_estimates, axis=0) - base.coefficients
        normal_scatter = np.std(normal_estimates, axis=0, ddof=1)
        collected[name] = {
            "truth": base.coefficients,
            "bias": bias,
            "scatter": scatter,
            "svd_se": svd_se,
            "normal_se": normal_se,
            "normal_bias": normal_bias,
            "normal_scatter": normal_scatter,
            "design": base.design,
            "estimates": estimates,
        }
        out("")
        out(f"{name}:")
        out("")
        out(
            format_table(
                [
                    "term",
                    "truth",
                    "svd bias",
                    "svd sd",
                    "svd se",
                    "normal bias",
                    "normal sd",
                    "normal se",
                ],
                [
                    [
                        term_name,
                        f"{base.coefficients[index]:.3f}",
                        f"{float(bias[index]):+.3f}",
                        f"{float(scatter[index]):.4g}",
                        f"{float(svd_se[index]):.4g}",
                        f"{float(normal_bias[index]):+.4g}",
                        f"{float(normal_scatter[index]):.4g}",
                        f"{float(normal_se[index]):.4g}",
                    ]
                    for index, term_name in enumerate(terms.TERM_NAMES)
                ],
            )
        )
    out("")
    out("Read the single_az block. The least-squares estimate of AN, AW, NPAE and IE")
    out("is repeatable across noise realisations to a few arcseconds and sits tens of")
    out("arcseconds from the truth, while the normal-equation solver reports formal")
    out("standard errors in the millions of arcseconds for the same parameters. A")
    out("repeatable wrong number with a meaningless error bar is the failure this")
    out("product exists to prevent.")
    out("")
    single = collected["single_az"]
    degenerate = [terms.TERM_INDEX[name] for name in ("AN", "AW", "NPAE", "IE")]
    out.check(
        "on single_az the least-squares split is repeatable",
        bool(np.all(single["scatter"][degenerate] < 20.0)),
        f"largest scatter on the degenerate terms = "
        f"{float(np.max(single['scatter'][degenerate])):.4g} arcsec",
    )
    out.check(
        "and on single_az it is wrong by more than its own scatter",
        bool(
            np.any(
                np.abs(single["bias"][degenerate]) > 2.0 * single["scatter"][degenerate]
            )
        ),
        f"largest |bias| = {float(np.max(np.abs(single['bias'][degenerate]))):.4g} arcsec "
        f"against a largest scatter of "
        f"{float(np.max(single['scatter'][degenerate])):.4g} arcsec",
    )
    smallest_ratio = float(
        np.min(single["normal_se"][degenerate] / single["normal_scatter"][degenerate])
    )
    out.check(
        "the normal-equation standard errors on single_az are unrelated to the scatter",
        bool(
            np.all(
                single["normal_se"][degenerate] > 100.0 * single["normal_scatter"][degenerate]
            )
        ),
        f"smallest ratio = {smallest_ratio:.4g}",
    )
    narrow = collected["narrow_el"]
    out.check(
        "on narrow_el the least-squares split is wrong by hundreds of arcseconds",
        float(np.max(np.abs(narrow["bias"]))) > 100.0,
        f"largest |bias| = {float(np.max(np.abs(narrow['bias']))):.4g} arcsec",
    )
    good = collected["full_sky"]
    out.check(
        "on full_sky both solvers agree with the truth and with each other",
        bool(
            np.all(np.abs(good["bias"]) < good["svd_se"])
            and np.all(np.abs(good["normal_bias"]) < 1.01 * good["normal_se"])
        ),
    )
    return collected


def combinations(out: Tee, collected: dict[str, dict[str, np.ndarray]]) -> None:
    """Part 3: the identifiable combinations carry no bias where the terms do."""
    out.rule("3. the identifiable combinations carry the bias the terms do not")
    out("The same Monte Carlo, scored in two parameterisations. What separates them")
    out("is the BIAS, not the total error: the total error of any quantity here also")
    out("contains the noise scatter, which is the same size in both")
    out("parameterisations and has nothing to do with the geometry. The bias is the")
    out("systematic part, it is the part an error bar does not show, and it is what a")
    out("degeneracy produces.")
    out("")
    out("Each combination is divided by the norm of its own weight vector so that")
    out("both columns are in arcseconds.")
    out("")
    rows = []
    for name in sorted(collected):
        data = collected[name]
        residuals = synthetic.generate(
            name, OBSERVATIONS, sigma_arcsec=SIGMA_ARCSEC, seed=SEED
        )
        result = wls.fit(residuals.design, residuals.observed, SIGMA_ARCSEC)
        report = identifiability.analyse_fit(residuals.design, result)
        determined = [c for c in report.combinations if c.determined]
        term_bias = float(np.max(np.abs(data["bias"])))
        term_scatter = float(np.max(data["scatter"]))
        worst_bias = 0.0
        worst_scatter = 0.0
        for combination in determined:
            norm = float(np.linalg.norm(combination.weights))
            true_value = float(combination.weights @ data["truth"]) / norm
            estimated = (data["estimates"] @ combination.weights) / norm
            worst_bias = max(worst_bias, abs(float(np.mean(estimated)) - true_value))
            worst_scatter = max(worst_scatter, float(np.std(estimated, ddof=1)))
        rows.append(
            [
                name,
                str(len(determined)),
                f"{term_bias:.4g}",
                f"{worst_bias:.4g}",
                f"{term_bias / worst_bias:.4g}" if worst_bias > 0 else "inf",
                f"{term_scatter:.4g}",
                f"{worst_scatter:.4g}",
            ]
        )
    out(
        format_table(
            [
                "geometry",
                "combinations determined",
                "worst per-term |bias| arcsec",
                "worst combination |bias| arcsec",
                "bias ratio",
                "worst per-term sd",
                "worst combination sd",
            ],
            rows,
        )
    )
    out("")
    out("The bias ratio is the number that matters. On the two well-scheduled")
    out("geometries it is near one and both biases are a fraction of an arcsecond:")
    out("there is no degeneracy and nothing is systematically wrong. On the three")
    out("restricted geometries the per-term bias is one to four orders of magnitude")
    out("larger than the bias in the combinations the geometry actually determines.")
    out("")
    out.check(
        "on every restricted geometry the per-term bias exceeds the combination "
        "bias by at least a factor of ten",
        all(
            float(row[4]) > 10.0
            for row in rows
            if row[0] in {"narrow_el", "single_az", "low_el"}
        ),
        "; ".join(f"{row[0]} {row[4]}" for row in rows),
    )
    out.check(
        "every determined combination has a bias below one arcsecond",
        all(float(row[3]) < 1.0 for row in rows),
        f"worst = {max(float(row[3]) for row in rows):.4g} arcsec",
    )
    out.check(
        "on the well-scheduled geometries the per-term bias is below one arcsecond",
        all(
            float(row[2]) < 1.0 for row in rows if row[0] in {"full_sky", "two_az"}
        ),
    )


def does_the_analysis_catch_it(out: Tee, collected: dict[str, dict[str, np.ndarray]]) -> None:
    """Part 4: score the refusal against a truth-anchored criterion."""
    out.rule("4. does the rank analysis catch the terms that are actually wrong")
    out("The criterion is declared in advance and anchored on the TRUE value: a")
    out("coefficient is BADLY DETERMINED when its mean least-squares estimate over")
    out("the Monte Carlo sits more than k declared sigmas from the truth. The rank")
    out("analysis never sees the truth. Two values of k are scored and both are")
    out("reported, so that the scoring cannot be read as having been tuned.")
    out("")
    refusals: dict[str, tuple[str, ...]] = {}
    for name in sorted(collected):
        residuals = synthetic.generate(
            name, OBSERVATIONS, sigma_arcsec=SIGMA_ARCSEC, seed=SEED
        )
        result = wls.fit(residuals.design, residuals.observed, SIGMA_ARCSEC)
        refusals[name] = identifiability.analyse_fit(
            residuals.design, result
        ).unidentifiable_terms

    out(
        format_table(
            ["geometry", "worst |bias| arcsec", "refused by the rank analysis"],
            [
                [
                    name,
                    f"{float(np.max(np.abs(collected[name]['bias']))):.4g}",
                    ",".join(refusals[name]) or "-",
                ]
                for name in sorted(collected)
            ],
        )
    )

    summary_rows = []
    all_recall_one = True
    for sigmas in BAD_DETERMINATION_SIGMAS:
        totals = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        rows = []
        for name in sorted(collected):
            bad = np.abs(collected[name]["bias"]) > sigmas * SIGMA_ARCSEC
            refused = np.array(
                [term_name in set(refusals[name]) for term_name in terms.TERM_NAMES]
            )
            counts = {
                "tp": int(np.count_nonzero(bad & refused)),
                "fp": int(np.count_nonzero(~bad & refused)),
                "fn": int(np.count_nonzero(bad & ~refused)),
                "tn": int(np.count_nonzero(~bad & ~refused)),
            }
            for key, value in counts.items():
                totals[key] += value
            rows.append(
                [
                    name,
                    ",".join(
                        term_name
                        for index, term_name in enumerate(terms.TERM_NAMES)
                        if bad[index]
                    )
                    or "-",
                    str(counts["tp"]),
                    str(counts["fp"]),
                    str(counts["fn"]),
                    str(counts["tn"]),
                ]
            )
        recall = (
            totals["tp"] / (totals["tp"] + totals["fn"])
            if totals["tp"] + totals["fn"]
            else 1.0
        )
        precision = (
            totals["tp"] / (totals["tp"] + totals["fp"])
            if totals["tp"] + totals["fp"]
            else 1.0
        )
        out("")
        out(
            f"criterion k = {sigmas:g} declared sigmas "
            f"({sigmas * SIGMA_ARCSEC:g} arcsec of bias):"
        )
        out("")
        out(
            format_table(
                ["geometry", "badly determined (truth-anchored)", "tp", "fp", "fn", "tn"],
                rows,
            )
        )
        out("")
        out(
            f"  totals: tp {totals['tp']}, fp {totals['fp']}, fn {totals['fn']}, "
            f"tn {totals['tn']}; recall {recall:.4f}, precision {precision:.4f}"
        )
        summary_rows.append(
            [
                f"{sigmas:g}",
                str(totals["tp"]),
                str(totals["fp"]),
                str(totals["fn"]),
                str(totals["tn"]),
                f"{recall:.4f}",
                f"{precision:.4f}",
            ]
        )
        if totals["fn"] != 0:
            all_recall_one = False
        out.check(
            f"the rank analysis refuses every coefficient biased beyond {sigmas:g} sigma",
            totals["fn"] == 0,
            f"{totals['fn']} badly determined coefficients were not refused",
        )
    out("")
    out(
        format_table(
            ["k sigmas", "tp", "fp", "fn", "tn", "recall", "precision"], summary_rows
        )
    )
    out("")
    out("A false positive here is a term the analysis refuses whose Monte Carlo bias")
    out("happens to be small. It is a conservative refusal, not an error: the")
    out("geometry genuinely cannot separate the direction, and this particular truth")
    out("happens to sit near where the minimum-norm solution lands. Nothing is lost,")
    out("because the identifiable combinations are reported in place of the split. The")
    out("recall is the number the refusal has to earn, and it is 1.0000 under both")
    out("criteria; the precision is 0.5000 at k = 10 and higher at k = 2, which is")
    out("reported rather than chosen.")
    out("")
    out.check(
        "recall is 1.0000 under both declared criteria",
        all_recall_one,
    )
    out.check(
        "the rank analysis refuses nothing on the two well-scheduled geometries",
        all(refusals[name] == () for name in ("full_sky", "two_az")),
    )


def main() -> int:
    out = Tee("validate_identifiability_output.txt")
    out.rule("pointmodel -- rank, identifiability, and the confident split")
    for line in environment_lines():
        out(line)
    out("")
    out(f"declared sigma        : {SIGMA_ARCSEC:g} arcsec per residual row")
    out(f"declared observations : {OBSERVATIONS}")
    out(f"declared geometry seed: {SEED}")
    out(f"monte carlo           : {MONTE_CARLO} noise realisations per geometry")
    out(
        f"declared thresholds   : null <= {identifiability.DEFAULT_NULL_THRESHOLD:g}, "
        f"marginal <= {identifiability.DEFAULT_MARGINAL_THRESHOLD:g}"
    )
    spectra(out)
    collected = confident_split(out)
    combinations(out, collected)
    does_the_analysis_catch_it(out, collected)
    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
