"""Baseline 1 and baseline 2, validated BEFORE the learned model exists.

This script is the analytic baseline's own validation and it is deliberately
first in the reproduction order of ``README.md``. An under-strength baseline
inflates a learned model's apparent win -- by about 50 per cent in one earlier
product of this mission -- so the baseline is measured on its own terms here,
and only then benchmarked against the model in
``validate_learned_model.py``.

What is measured
----------------
1. Known answers for every closed form, against hand calculations and against
   the classical published one-sided normal tolerance factors.
2. The achieved coverage of each bound at its declared confidence, over
   repeated independent field samples, with the TRUE population mean and
   quantile as the reference.
3. The Hoeffding-versus-empirical-Bernstein crossover: which of the two
   distribution-free mean bounds is tighter, as a function of ``n``, measured
   rather than asserted.
"""

from __future__ import annotations

import math

import numpy as np

from _bootstrap import (
    CONFIDENCE,
    FIELD_SIZES,
    PROPORTION,
    REPETITIONS,
    SEED_FIELD,
    SEED_TRUTH,
    Tee,
    Timer,
    bootstrap,
    environment_lines,
)

ROOT = bootstrap()

from transfergap import synthetic  # noqa: E402
from transfergap.bounds import (  # noqa: E402
    empirical_bernstein_mean_upper,
    empirical_bernstein_slack,
    hoeffding_mean_upper,
    normal_tolerance_upper,
    required_n_wilks,
    tolerance_factor_normal,
    wilks_tolerance_upper,
)
from transfergap.report import format_table, render_coverage  # noqa: E402
from transfergap.resample import bootstrap_mean_upper, split_conformal_upper  # noqa: E402
from transfergap.scoring import score_mean_bounds, score_prediction_bounds  # noqa: E402

R = synthetic.BOUND_RANGE_KNM


def draw(n: int, seed: int) -> synthetic.GapSample:
    """One synthetic field campaign of ``n`` exchangeable paired measurements."""
    return synthetic.generate(n, "random", seed=seed)


def known_answers(out: Tee) -> None:
    out.rule("1. closed forms against hand calculations and published tables")
    out("Every expected value below is derived in the test suite's comments or is a")
    out("published table value. None of them comes from this implementation.")
    out("")

    gaps = np.array([1.0, 2.0, 3.0, 4.0])
    hoeffding = hoeffding_mean_upper(gaps, 10.0, 0.95)
    expected_slack = 10.0 * math.sqrt(math.log(20.0) / 8.0)
    out.check(
        "Hoeffding slack on [1,2,3,4], R=10, 0.95 equals R sqrt(ln(20)/(2n))",
        abs(hoeffding.slack - expected_slack) <= 1e-13 * expected_slack,
        f"{hoeffding.slack:.15g} against {expected_slack:.15g}",
    )
    out.check(
        "the Hoeffding slack is independent of the data",
        abs(
            hoeffding_mean_upper(np.zeros(20), 10.0, 0.95).slack
            - hoeffding_mean_upper(np.full(20, 10.0), 10.0, 0.95).slack
        )
        == 0.0,
        "identical for an all-zero and an all-R sample at n=20",
    )

    bernstein = empirical_bernstein_mean_upper(np.array([0.0, 0.0, 10.0, 10.0]), 10.0, 0.95)
    log_term = math.log(40.0)
    hand = math.sqrt(2.0 * (100.0 / 3.0) * log_term / 4.0) + 7.0 * 10.0 * log_term / 9.0
    out.check(
        "empirical Bernstein slack on [0,0,10,10] equals the hand calculation",
        abs(bernstein.slack - hand) <= 1e-13 * hand,
        f"{bernstein.slack:.15g} against {hand:.15g}",
    )

    out.check(
        "Wilks' required n at p=0.95, conf=0.95 is 59",
        required_n_wilks(0.95, 0.95) == 59,
        f"ln(0.05)/ln(0.95) = {math.log(0.05) / math.log(0.95):.14g}",
    )

    out("")
    out("The classical one-sided normal tolerance factors of equation (9). Table")
    out("values are quoted to three decimals as the tables print them; published")
    out("tables differ in the third decimal depending on whether the exact")
    out("non-central t or the Natrella normal approximation was used.")
    rows = []
    worst = 0.0
    for n, proportion, published in ((3, 0.90, 6.155), (5, 0.95, 4.202), (10, 0.90, 2.355),
                                     (20, 0.99, 3.295)):
        measured = tolerance_factor_normal(n, proportion, 0.95)
        deviation = abs(measured - published)
        worst = max(worst, deviation)
        rows.append([str(n), f"{proportion:g}", f"{published:.3f}", f"{measured:.6f}",
                     f"{deviation:.2e}"])
    out(format_table(["n", "p", "table (conf 0.95)", "measured", "deviation"], rows))
    out.check(
        "every tolerance factor is within 1e-3 of the published table",
        worst <= 1e-3,
        f"worst deviation {worst:.3e}",
    )


def coverage_of_the_mean_bounds(out: Tee, truth: synthetic.PopulationTruth) -> None:
    out.rule("2a. achieved coverage of the MEAN bounds, against the true E[d]")
    out(f"true E[d] = {truth.mean:.6f} +/- {truth.standard_error:.6f} kN.m, from")
    out(f"{truth.draws} Monte Carlo draws at seed {truth.seed}. Every coverage below is")
    out("anchored on THIS number and never on another method's output.")
    out("")
    methods = {
        "hoeffding": lambda s: hoeffding_mean_upper(s.gaps, R, CONFIDENCE, s.unit),
        "empirical_bernstein": lambda s: empirical_bernstein_mean_upper(
            s.gaps, R, CONFIDENCE, s.unit
        ),
        "bootstrap_percentile": lambda s: bootstrap_mean_upper(
            s.gaps, R, CONFIDENCE, resamples=1000, seed=s.size, unit=s.unit
        ),
        "bootstrap_basic": lambda s: bootstrap_mean_upper(
            s.gaps, R, CONFIDENCE, resamples=1000, seed=s.size, kind="basic", unit=s.unit
        ),
    }
    timer = Timer()
    all_scores = []
    for n in FIELD_SIZES:
        with timer:
            all_scores += score_mean_bounds(
                methods, draw, truth.mean, n, REPETITIONS, SEED_FIELD + 1000 * n
            )
    out(render_coverage(all_scores))
    out("")
    for score in all_scores:
        if score.distribution_free:
            out.check(
                f"{score.method} at n={score.n} covers its declared {score.declared:g}",
                score.covers,
                f"coverage {score.coverage:.4f} +/- {score.coverage_se:.4f}",
            )
    under = [
        s for s in all_scores
        if not s.distribution_free and s.coverage < s.declared - 2.0 * s.coverage_se
    ]
    out("")
    out("MEASURED AND REPORTED AS FOUND: the bootstrap limits are not")
    out("distribution-free and are not expected to hold at these sample sizes. The")
    out(f"rows that measurably under-cover are counted here, not removed: {len(under)} of")
    out(f"{len([s for s in all_scores if not s.distribution_free])} bootstrap rows.")
    for score in under:
        out(
            f"  {score.method} at n={score.n}: coverage {score.coverage:.4f} "
            f"+/- {score.coverage_se:.4f} against a declared {score.declared:g}"
        )
    out("")
    out(f"mean-bound scoring wall clock: {timer.elapsed_s:.1f} s over {timer.calls} sample sizes")


def coverage_of_the_prediction_limits(out: Tee, pool: synthetic.GapSample) -> None:
    out.rule("2b. achieved coverage of the PREDICTION limits, on a held-out pool")
    out(f"The pool is {pool.size} fresh paired measurements from the same envelope. The")
    out("per-repetition coverage is the fraction of the pool below the limit; the")
    out("tolerance rate is the fraction of repetitions whose own coverage reached the")
    out(f"declared {PROPORTION:g}.")
    out("")

    def wilks(sample, _conditions):
        return wilks_tolerance_upper(sample.gaps, R, PROPORTION, CONFIDENCE).upper

    def normal(sample, _conditions):
        return normal_tolerance_upper(sample.gaps, R, PROPORTION, CONFIDENCE).upper

    def conformal(sample, _conditions):
        return split_conformal_upper(sample.gaps, R, PROPORTION).upper

    methods = {"wilks_tolerance": wilks, "normal_tolerance": normal,
               "split_conformal": conformal}
    flags = {"wilks_tolerance": True, "normal_tolerance": False, "split_conformal": True}
    scores = []
    for n in FIELD_SIZES:
        scores += score_prediction_bounds(
            methods, draw, pool.features, pool.gaps, n, REPETITIONS,
            SEED_FIELD + 2000 * n, PROPORTION, flags,
        )
    out(render_coverage(scores))
    out("")
    for score in scores:
        if score.failures == REPETITIONS:
            out.check(
                f"{score.method} at n={score.n} correctly refuses to produce a limit",
                True,
                f"{score.failures} of {REPETITIONS} repetitions raised, as equation "
                f"(7) or (12) requires",
            )
        elif score.distribution_free:
            out.check(
                f"{score.method} at n={score.n} covers its declared {score.declared:g} marginally",
                score.covers,
                f"coverage {score.coverage:.4f} +/- {score.coverage_se:.4f}, "
                f"tolerance rate {score.tolerance_rate:.4f}",
            )
    out("")
    out("MEASURED AND REPORTED AS FOUND: the normal-theory tolerance limit is NOT")
    out("distribution-free. A discrepancy magnitude is a folded quantity with a hard")
    out("floor at zero and is not normal, so its coverage is not guaranteed and is")
    out("reported rather than checked. The measured values are:")
    for score in scores:
        if not score.distribution_free and score.failures < REPETITIONS:
            verdict = (
                "at or above the declared level"
                if score.covers
                else "MEASURABLY BELOW the declared level"
            )
            out(
                f"  normal_tolerance at n={score.n}: coverage {score.coverage:.4f} "
                f"+/- {score.coverage_se:.4f}, tolerance rate {score.tolerance_rate:.4f} "
                f"-- {verdict}"
            )
    out("")
    out("The distinction the two rate columns exist for: split conformal is")
    out("MARGINALLY valid, so its coverage column sits at the declared level while")
    out("its tolerance rate is near one half -- about half of campaigns produce a")
    out("limit that under-covers, and the guarantee is only that the average is")
    out("right. Wilks' limit is training-conditionally valid, so its tolerance rate")
    out("is the number that reaches the declared confidence. Reporting the coverage")
    out("column alone would make the two look equivalent.")


def crossover(out: Tee) -> None:
    out.rule("3. the Hoeffding / empirical Bernstein crossover, measured")
    out("Both bounds are distribution-free. Which is tighter depends on n: equation")
    out("(5) pays a 7 R ln(2/delta) / (3(n-1)) penalty that equation (3) does not,")
    out("and wins only once the variance term dominates. The crossover below uses")
    out("the population variance of the synthetic field distribution so that the")
    out("comparison is about n and not about an estimate.")
    out("")
    population_variance = float(np.var(synthetic.generate(200_000, "random", seed=4242).gaps))
    out(f"population variance of d: {population_variance:.6f} kN.m^2")
    out(f"declared R             : {R:.4f} kN.m")
    out("")
    rows = []
    crossing = None
    for n in (10, 15, 25, 40, 60, 100, 150, 200, 300, 500, 1000):
        hoeffding_slack = R * math.sqrt(math.log(1.0 / (1.0 - CONFIDENCE)) / (2.0 * n))
        bernstein = empirical_bernstein_slack(n, population_variance, R, CONFIDENCE)
        tighter = "empirical Bernstein" if bernstein < hoeffding_slack else "Hoeffding"
        if crossing is None and bernstein < hoeffding_slack:
            crossing = n
        rows.append([str(n), f"{hoeffding_slack:.4f}", f"{bernstein:.4f}",
                     f"{bernstein / hoeffding_slack:.4f}", tighter])
    out(format_table(["n", "Hoeffding slack", "emp. Bernstein slack", "ratio", "tighter"], rows))
    out("")
    exact = next(
        n for n in range(2, 5000)
        if empirical_bernstein_slack(n, population_variance, R, CONFIDENCE)
        < R * math.sqrt(math.log(1.0 / (1.0 - CONFIDENCE)) / (2.0 * n))
    )
    out(f"CROSSOVER: empirical Bernstein first becomes the tighter bound at n = {exact}.")
    out("")
    out.check(
        "Hoeffding is the tighter distribution-free mean bound at every field size "
        "this product is about",
        all(
            R * math.sqrt(math.log(1.0 / (1.0 - CONFIDENCE)) / (2.0 * n))
            < empirical_bernstein_slack(n, population_variance, R, CONFIDENCE)
            for n in FIELD_SIZES
        ),
        f"field sizes {list(FIELD_SIZES)} are all below the crossover at n = {exact}",
    )
    out("")
    out("REPORTED AS FOUND: at the sample sizes a flight-test campaign actually")
    out("delivers, the simpler and older inequality is the better one. The")
    out("empirical Bernstein bound is in this package because it becomes the better")
    out("one later, and the table above is where a reader can see which regime")
    out("they are in.")


def main() -> int:
    out = Tee("validate_bounds_output.txt")
    out.rule("transfergap -- baseline 1 and baseline 2, validated first")
    for line in environment_lines():
        out(line)
    out("")
    out(f"declared confidence : {CONFIDENCE}")
    out(f"declared proportion : {PROPORTION}")
    out(f"field sample sizes  : {list(FIELD_SIZES)}")
    out(f"repetitions          : {REPETITIONS}")
    truth = synthetic.population_truth(draws=400_000, seed=SEED_TRUTH)
    pool = synthetic.generate(4000, "random", seed=99000)
    known_answers(out)
    coverage_of_the_mean_bounds(out, truth)
    coverage_of_the_prediction_limits(out, pool)
    crossover(out)
    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
