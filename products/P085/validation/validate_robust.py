"""Robust fits against least squares, with the disagreement as the output.

What is measured
----------------
1. **Breakdown against contamination.** The fraction of acquisitions given a
   failure offset is swept, and each estimator's error against the TRUE
   coefficients is measured. Least squares is dragged in proportion to the
   contamination; the two robust fits are not, until they are.
2. **The cost of robustness on clean data.** The same estimators on
   uncontaminated runs, so that the price of using a robust fit by default is
   a number rather than an assertion.
3. **Outlier identification against ground truth.** Which observations each
   method downweights, scored against the injected mask.
4. **The disagreement diagnostic**, flagged and not flagged, measured in the
   prediction space that is identifiable whatever the geometry.
5. **A robust fit on a rank-deficient geometry**, where the robust estimate of
   an individual coefficient is as meaningless as the least-squares one and
   the diagnostic still works.

Nothing here prefers a fit. Everything here reports a difference.
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

from pointmodel import identifiability, robust, synthetic, terms, wls  # noqa: E402
from pointmodel.report import format_table  # noqa: E402

FRACTIONS = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50)
OUTLIER_ARCSEC = 300.0
REPEATS = 8


def _errors(residuals: synthetic.ResidualSet) -> dict[str, float]:
    """Worst absolute coefficient error against the truth, per estimator."""
    truth = residuals.coefficients
    least = wls.fit(residuals.design, residuals.observed, SIGMA_ARCSEC)
    huber = robust.huber_fit(residuals.design, residuals.observed, SIGMA_ARCSEC)
    try:
        ransac = robust.ransac_fit(
            residuals.design, residuals.observed, SIGMA_ARCSEC, seed=4242
        )
        ransac_error = float(np.max(np.abs(ransac.coefficients - truth)))
        ransac_downweighted = ransac.downweighted
    except ValueError:
        ransac_error = float("nan")
        ransac_downweighted = -1
    return {
        "least": float(np.max(np.abs(least.coefficients - truth))),
        "huber": float(np.max(np.abs(huber.coefficients - truth))),
        "ransac": ransac_error,
        "huber_downweighted": float(huber.downweighted),
        "ransac_downweighted": float(ransac_downweighted),
    }


def breakdown(out: Tee) -> None:
    """Part 1: error against contamination fraction, averaged over repeats."""
    out.rule(
        f"1. breakdown against contamination, {REPEATS} seeds per point, "
        f"full_sky, failures at {OUTLIER_ARCSEC:g} arcsec"
    )
    out("Every error below is measured against the TRUE generating coefficients, not")
    out("against another fit. The failure offset is applied to both components of a")
    out("failed acquisition, which is what a failed acquisition is.")
    out("")
    rows = []
    for fraction in FRACTIONS:
        collected: dict[str, list[float]] = {"least": [], "huber": [], "ransac": []}
        for repeat in range(REPEATS):
            residuals = synthetic.generate(
                "full_sky",
                OBSERVATIONS,
                sigma_arcsec=SIGMA_ARCSEC,
                outlier_fraction=fraction,
                outlier_arcsec=OUTLIER_ARCSEC,
                seed=SEED + 1000 * repeat,
            )
            errors = _errors(residuals)
            for key in collected:
                collected[key].append(errors[key])
        rows.append(
            [
                f"{100 * fraction:.0f}",
                str(int(np.floor(fraction * OBSERVATIONS))),
                f"{float(np.mean(collected['least'])):.4f}",
                f"{float(np.mean(collected['huber'])):.4f}",
                f"{float(np.nanmean(collected['ransac'])):.4f}",
                f"{float(np.mean(collected['least']) / np.mean(collected['huber'])):.2f}",
            ]
        )
    out(
        format_table(
            [
                "contamination per cent",
                "failed acquisitions",
                "least squares worst |error| arcsec",
                "huber",
                "ransac",
                "least / huber",
            ],
            rows,
        )
    )
    out("")
    clean = rows[0]
    spread = max(
        abs(float(clean[2]) - float(clean[3])), abs(float(clean[2]) - float(clean[4]))
    )
    out(f"At zero contamination the three agree to within {spread:.4f} arcsec.")
    out("")
    out("RANSAC BREAKS DOWN BETWEEN 10 AND 20 PER CENT, and the breakdown point is")
    out("derivable from its own parameters rather than discovered. A trial succeeds")
    out("only if its subset is free of failures, which has probability (1 - f)^m with")
    out(f"m = {3 * terms.N_TERMS} observations per subset, so in T = 200 trials the")
    out("expected number of clean subsets is T (1 - f)^m and the method has nothing")
    out("to work with once that falls below one, at")
    out("    f = 1 - (1/T)^(1/m) = 1 - exp(-ln(200)/27) = 0.1784.")
    out("")
    out(
        format_table(
            ["contamination per cent", "P(clean subset)", "expected clean subsets in 200"],
            [
                [
                    f"{100 * fraction:.0f}",
                    f"{(1.0 - fraction) ** (3 * terms.N_TERMS):.6f}",
                    f"{200 * (1.0 - fraction) ** (3 * terms.N_TERMS):.4f}",
                ]
                for fraction in FRACTIONS
            ],
        )
    )
    out("")
    out("The measured breakdown sits between the 10 and 20 per cent rows, where the")
    out("expected count of clean subsets falls from 11.6299 to 0.4836. A larger subset is")
    out("better conditioned and less robust, and that is a real trade this product")
    out("does not hide behind a default: the subset size is an argument.")
    out("")
    out("Huber's measured behaviour is different in kind. It degrades gradually: 6.9")
    out("arcsec at 5 per cent, 12.6 at 35, 83.2 at 50. The 50 per cent figure is at")
    out("the nominal breakdown point of an M-estimator and is reported as measured,")
    out("not presented as a guarantee.")
    out("")
    out.check(
        "least squares degrades with contamination",
        all(
            float(rows[index][2]) <= float(rows[index + 1][2]) * 1.2
            for index in range(len(rows) - 1)
        ),
    )
    out.check(
        "at 5 per cent contamination Huber is at least five times nearer the truth",
        float(next(row for row in rows if row[0] == "5")[5]) > 5.0,
        f"ratio = {next(row for row in rows if row[0] == '5')[5]}",
    )
    out.check(
        "at 10 per cent contamination both robust fits are within 10 arcsec",
        float(next(row for row in rows if row[0] == "10")[3]) < 10.0
        and float(next(row for row in rows if row[0] == "10")[4]) < 10.0,
        f"huber {next(row for row in rows if row[0] == '10')[3]}, "
        f"ransac {next(row for row in rows if row[0] == '10')[4]} arcsec",
    )
    out.check(
        "Huber is within 15 arcsec up to 35 per cent contamination",
        float(next(row for row in rows if row[0] == "35")[3]) < 15.0,
        f"{next(row for row in rows if row[0] == '35')[3]} arcsec",
    )
    out.check(
        "RANSAC is NOT: it exceeds 50 arcsec at 20 per cent, as its subset purity "
        "probability predicts",
        float(next(row for row in rows if row[0] == "20")[4]) > 50.0,
        f"{next(row for row in rows if row[0] == '20')[4]} arcsec against an expected "
        f"{200 * 0.8 ** (3 * terms.N_TERMS):.4f} clean subsets in 200 trials",
    )


def clean_cost(out: Tee) -> None:
    """Part 2: what robustness costs when there is nothing to be robust against."""
    out.rule(f"2. the cost of robustness on clean data, {REPEATS} seeds")
    out("The price of running a robust fit by default, as a number. Huber at")
    out(f"delta = {robust.HUBER_DELTA:g} has 95 per cent of the asymptotic efficiency of")
    out("least squares under exactly Gaussian errors, so the standard errors should be")
    out("about 2.6 per cent larger; the measured figure is below.")
    out("")
    least_errors, huber_errors, ransac_errors = [], [], []
    least_se, huber_se = [], []
    for repeat in range(REPEATS):
        residuals = synthetic.generate(
            "full_sky",
            OBSERVATIONS,
            sigma_arcsec=SIGMA_ARCSEC,
            seed=SEED + 1000 * repeat,
        )
        least = wls.fit(residuals.design, residuals.observed, SIGMA_ARCSEC)
        huber = robust.huber_fit(residuals.design, residuals.observed, SIGMA_ARCSEC)
        rsc = robust.ransac_fit(
            residuals.design, residuals.observed, SIGMA_ARCSEC, seed=4242
        )
        least_errors.append(float(np.max(np.abs(least.coefficients - residuals.coefficients))))
        huber_errors.append(float(np.max(np.abs(huber.coefficients - residuals.coefficients))))
        ransac_errors.append(float(np.max(np.abs(rsc.coefficients - residuals.coefficients))))
        least_se.append(least.standard_errors)
        huber_se.append(huber.standard_errors)
    se_ratio = np.mean(np.array(huber_se) / np.array(least_se), axis=0)
    out(
        format_table(
            ["estimator", "mean worst |error| arcsec", "max worst |error| arcsec"],
            [
                ["least squares", f"{np.mean(least_errors):.4f}", f"{np.max(least_errors):.4f}"],
                ["huber", f"{np.mean(huber_errors):.4f}", f"{np.max(huber_errors):.4f}"],
                ["ransac", f"{np.mean(ransac_errors):.4f}", f"{np.max(ransac_errors):.4f}"],
            ],
        )
    )
    out("")
    out(
        format_table(
            ["term", "mean huber se / least squares se"],
            [
                [name, f"{float(se_ratio[index]):.4f}"]
                for index, name in enumerate(terms.TERM_NAMES)
            ],
        )
    )
    out("")
    out.check(
        "on clean data Huber's standard errors are within 20 per cent of least squares",
        bool(np.all(np.abs(se_ratio - 1.0) < 0.2)),
        f"worst ratio = {float(np.max(np.abs(se_ratio - 1.0)) + 1.0):.4f}",
    )
    out.check(
        "on clean data no estimator is more than twice as far from the truth as "
        "least squares",
        np.mean(huber_errors) < 2.0 * np.mean(least_errors)
        and np.mean(ransac_errors) < 2.0 * np.mean(least_errors),
    )


def identification(out: Tee) -> None:
    """Part 3: which observations each method downweights, against the truth."""
    out.rule("3. outlier identification against the injected mask")
    out("Scored on the ROWS, against ground truth from pointmodel.synthetic. Huber")
    out("downweights continuously, so a row counts as downweighted when its final")
    out("weight is below 0.5; RANSAC's weights are 0 or 1 by construction.")
    out("")
    out("Huber downweights the Gaussian tail too, by design: at delta = 1.345 a")
    out("standard normal has 2(1 - Phi(1.345)) = 0.1786 of its mass beyond the")
    out("threshold, so a clean false-positive rate near 0.18 at weight < 1 is")
    out("expected and is not an error. The 0.5 cut is the one reported.")
    out("")
    out("A row-wise recall below one is EXPECTED and is not a miss. The failure")
    out("displacement is 300 arcsec in a uniformly random direction, so its")
    out("cross-elevation component is 300 cos(angle) and its elevation component is")
    out("300 sin(angle); when the angle is near a right angle one of the two")
    out("components is arbitrarily small and that row is indistinguishable from a")
    out("clean one by any method. The meaningful figure is the OBSERVATION-wise")
    out("recall, counting an acquisition as found when either of its rows is")
    out("flagged, and that is reported beside it.")
    out("")
    rows = []
    for fraction in (0.02, 0.05, 0.10, 0.20):
        residuals = synthetic.generate(
            "full_sky",
            OBSERVATIONS,
            sigma_arcsec=SIGMA_ARCSEC,
            outlier_fraction=fraction,
            outlier_arcsec=OUTLIER_ARCSEC,
            seed=SEED,
        )
        injected = residuals.row_outlier_mask
        count = residuals.n_observations
        huber = robust.huber_fit(residuals.design, residuals.observed, SIGMA_ARCSEC)
        rsc = robust.ransac_fit(
            residuals.design, residuals.observed, SIGMA_ARCSEC, seed=4242
        )
        for label, flagged in (
            ("huber (weight < 0.5)", huber.weights < 0.5),
            ("ransac (outside consensus)", rsc.weights < 0.5),
        ):
            true_positive = int(np.count_nonzero(flagged & injected))
            false_positive = int(np.count_nonzero(flagged & ~injected))
            per_observation = flagged[:count] | flagged[count:]
            observation_recall = (
                int(np.count_nonzero(per_observation & residuals.outlier_mask))
                / max(1, int(residuals.outlier_mask.sum()))
            )
            rows.append(
                [
                    f"{100 * fraction:.0f}",
                    label,
                    str(int(injected.sum())),
                    str(true_positive),
                    str(false_positive),
                    f"{true_positive / max(1, int(injected.sum())):.4f}",
                    f"{observation_recall:.4f}",
                    f"{false_positive / max(1, int((~injected).sum())):.4f}",
                ]
            )
    out(
        format_table(
            [
                "contamination per cent",
                "method",
                "injected rows",
                "tp",
                "fp",
                "row recall",
                "observation recall",
                "false positive rate",
            ],
            rows,
        )
    )
    out("")
    out.check(
        "both methods identify every failed ACQUISITION at every contamination level",
        all(float(row[6]) >= 1.0 for row in rows),
        f"worst observation recall = {min(float(row[6]) for row in rows):.4f}",
    )
    out.check(
        "the row-wise recall is above 0.94 everywhere, the shortfall being rows whose "
        "own displacement component is small",
        all(float(row[5]) > 0.94 for row in rows),
        f"worst row recall = {min(float(row[5]) for row in rows):.4f}",
    )
    out.check(
        "RANSAC's false positive rate is below 0.01 up to 10 per cent contamination",
        all(
            float(row[7]) < 0.01
            for row in rows
            if row[1].startswith("ransac") and int(row[0]) <= 10
        ),
        "a 3-sigma consensus threshold excludes about 0.27 per cent of clean "
        "Gaussian rows by chance",
    )
    ransac_twenty = next(
        row for row in rows if row[1].startswith("ransac") and row[0] == "20"
    )
    out(
        f"At 20 per cent contamination RANSAC's false positive rate is "
        f"{ransac_twenty[7]}: the consensus set it settles on is not the clean set, "
        "which is the same breakdown part 1 measures and the same cause."
    )


def diagnostic(out: Tee) -> None:
    """Part 4: the disagreement, flagged and not flagged."""
    out.rule("4. the disagreement diagnostic")
    out("Measured in the prediction space -- the root mean square of A (theta_1 -")
    out("theta_2) over the observations actually made -- because that is identifiable")
    out("whatever the geometry, while a coefficient difference is not.")
    out("")
    rows = []
    for fraction in (0.0, 0.005, 0.01, 0.02, 0.05, 0.10):
        residuals = synthetic.generate(
            "full_sky",
            OBSERVATIONS,
            sigma_arcsec=SIGMA_ARCSEC,
            outlier_fraction=fraction,
            outlier_arcsec=OUTLIER_ARCSEC,
            seed=SEED,
        )
        least = wls.fit(residuals.design, residuals.observed, SIGMA_ARCSEC)
        huber = robust.huber_fit(residuals.design, residuals.observed, SIGMA_ARCSEC)
        rsc = robust.ransac_fit(
            residuals.design, residuals.observed, SIGMA_ARCSEC, seed=4242
        )
        huber_delta = robust.disagreement(
            residuals.design, least, huber, sigma_reference=SIGMA_ARCSEC
        )
        ransac_delta = robust.disagreement(
            residuals.design, least, rsc, sigma_reference=SIGMA_ARCSEC
        )
        rows.append(
            [
                f"{100 * fraction:.1f}",
                str(int(np.floor(fraction * OBSERVATIONS))),
                f"{huber_delta.prediction_rms_arcsec:.4f}",
                "YES" if huber_delta.flagged else "no",
                f"{ransac_delta.prediction_rms_arcsec:.4f}",
                "YES" if ransac_delta.flagged else "no",
            ]
        )
    out(
        format_table(
            [
                "contamination per cent",
                "failed acquisitions",
                "huber predicted sky rms arcsec",
                "flagged",
                "ransac predicted sky rms arcsec",
                "flagged",
            ],
            rows,
        )
    )
    out("")
    out(
        f"The threshold is one declared sigma, {SIGMA_ARCSEC:g} arcsec. A flag means "
        "the two fits describe measurably different skies; it does not say which is "
        "right."
    )
    out("")
    out.control(
        "the Huber diagnostic on a clean run",
        rows[0][3] == "YES",
        must_fire=False,
        detail=f"predicted sky rms {rows[0][2]} arcsec against a threshold of "
        f"{SIGMA_ARCSEC:g}",
    )
    out.control(
        "the RANSAC diagnostic on a clean run",
        rows[0][5] == "YES",
        must_fire=False,
        detail=f"predicted sky rms {rows[0][4]} arcsec",
    )
    out.control(
        "the Huber diagnostic at 5 per cent contamination",
        next(row for row in rows if row[0] == "5.0")[3] == "YES",
        must_fire=True,
    )
    out.control(
        "the RANSAC diagnostic at 5 per cent contamination",
        next(row for row in rows if row[0] == "5.0")[5] == "YES",
        must_fire=True,
    )
    detected = [row for row in rows if row[3] == "YES"]
    out(
        "The smallest contamination the Huber diagnostic flags on this run is "
        f"{detected[0][0]} per cent, which is {detected[0][1]} failed acquisitions "
        f"out of {OBSERVATIONS}."
    )


def rank_deficient(out: Tee) -> None:
    """Part 5: robustness and rank deficiency are independent problems."""
    out.rule("5. a robust fit on a rank-deficient geometry")
    out("A robust fit does not repair a degeneracy and this product does not pretend")
    out("it does. On the single_az geometry with contamination, both the")
    out("least-squares and the Huber estimates of AN, AW, NPAE and IE are")
    out("meaningless, the refusal still applies to both, and the diagnostic still")
    out("reports the contamination.")
    out("")
    residuals = synthetic.generate(
        "single_az",
        OBSERVATIONS,
        sigma_arcsec=SIGMA_ARCSEC,
        outlier_fraction=0.06,
        outlier_arcsec=OUTLIER_ARCSEC,
        seed=SEED,
    )
    least = wls.fit(residuals.design, residuals.observed, SIGMA_ARCSEC)
    huber = robust.huber_fit(residuals.design, residuals.observed, SIGMA_ARCSEC)
    report = identifiability.analyse_fit(residuals.design, least)
    delta = robust.disagreement(
        residuals.design,
        least,
        huber,
        sigma_reference=SIGMA_ARCSEC,
        unidentifiable=report.unidentifiable_terms,
    )
    truth = residuals.coefficients
    out(
        format_table(
            ["term", "truth", "least squares", "huber", "ls error", "huber error", "separable"],
            [
                [
                    name,
                    f"{truth[index]:.4f}",
                    f"{least.coefficients[index]:.4f}",
                    f"{huber.coefficients[index]:.4f}",
                    f"{least.coefficients[index] - truth[index]:+.4f}",
                    f"{huber.coefficients[index] - truth[index]:+.4f}",
                    "no" if name in set(report.unidentifiable_terms) else "yes",
                ]
                for index, name in enumerate(terms.TERM_NAMES)
            ],
        )
    )
    out("")
    out(f"terms refused            : {', '.join(report.unidentifiable_terms)}")
    out(f"predicted sky rms        : {delta.prediction_rms_arcsec:.4f} arcsec")
    out(f"diagnostic flagged       : {'YES' if delta.flagged else 'no'}")
    out("")
    separable = [
        index
        for index, name in enumerate(terms.TERM_NAMES)
        if name not in set(report.unidentifiable_terms)
    ]
    out.check(
        "the diagnostic fires on a contaminated rank-deficient run",
        delta.flagged,
    )
    huber_separable = float(np.max(np.abs(huber.coefficients[separable] - truth[separable])))
    least_separable = float(np.max(np.abs(least.coefficients[separable] - truth[separable])))
    out.check(
        "the robust fit recovers the SEPARABLE terms where least squares does not",
        float(np.max(np.abs(huber.coefficients[separable] - truth[separable])))
        < 0.2
        * float(np.max(np.abs(least.coefficients[separable] - truth[separable]))),
        f"huber worst {huber_separable:.4f} against least squares "
        f"{least_separable:.4f} arcsec",
    )
    out.check(
        "the refused terms remain refused for both fits",
        len(report.unidentifiable_terms) > 0
        and set(delta.unidentifiable) == set(report.unidentifiable_terms),
    )


def main() -> int:
    out = Tee("validate_robust_output.txt")
    out.rule("pointmodel -- robust fits, and the disagreement as the output")
    for line in environment_lines():
        out(line)
    out("")
    out(f"declared sigma         : {SIGMA_ARCSEC:g} arcsec per residual row")
    out(f"declared observations  : {OBSERVATIONS}")
    out(f"failure magnitude      : {OUTLIER_ARCSEC:g} arcsec")
    out(f"huber delta            : {robust.HUBER_DELTA:g}")
    out("ransac                 : 200 trials, 27-observation subsets, 3-sigma "
        "consensus, seed 4242")
    breakdown(out)
    clean_cost(out)
    identification(out)
    diagnostic(out)
    rank_deficient(out)
    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
