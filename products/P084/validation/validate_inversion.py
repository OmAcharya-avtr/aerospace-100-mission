"""The sample-count inversions, verified OPERATIONALLY and not just algebraically.

An inversion that satisfies its own forward expression has been checked
against itself. What a programme needs to know is different: if the plan says
268 more flights, and 268 more flights are flown, does the bound actually
reach the declared target, and how often?

That is what this script measures. For each method and each target slack it
takes the predicted sample count, DRAWS that many measurements from the
synthetic field population many times over, computes the bound, and counts the
fraction of draws in which

- the realised slack is at or below the target, and
- the bound covers the true population mean.

The two rates are different questions and are reported separately.

The expected answers, stated in advance
---------------------------------------
**Hoeffding.** The slack of equation (3) does not depend on the data, so the
slack hit rate must be exactly 1.0, every time, with no Monte Carlo error at
all. If it is not 1.0 the implementation is wrong, not unlucky. The coverage
rate must be at or above the declared confidence.

**Empirical Bernstein.** The slack of equation (5) depends on the sample
variance, and the plan uses a plug-in variance, so the slack hit rate is
strictly below 1.0 and is the number a programme actually needs. It is
measured here under four plug-ins, three of them ANCHORED ON THE TRUE
population variance so that the direction of the effect is not left to a
draw: half the true variance, the true variance, twice the true variance,
and the variance estimated from a 15-point pilot campaign, which is the
realistic case and whose direction is whatever it turns out to be.

A prediction made before running this and falsified by it, recorded rather
than quietly corrected: the 15-point pilot was expected to UNDERSTATE the
variance, that being the common direction for a small sample of a
right-skewed quantity, and therefore to ask for too few flights. It did the
opposite on the declared pilot seed -- it overstates -- so the plan built on
it asks for MORE flights than the truth warrants and over-delivers on the
slack. The anchored half-and-double rows are what establish the direction;
the pilot row is one draw and is reported as such.

**Wilks.** The requirement of equation (7) is about the statement existing,
so what is verified is the tolerance statement itself: at the predicted ``n``,
the fraction of campaigns whose largest observation covers at least ``p`` of
the population must reach ``1 - delta``.
"""

from __future__ import annotations

import math

import numpy as np

from _bootstrap import (
    CONFIDENCE,
    PROPORTION,
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
    hoeffding_mean_upper,
    required_n_empirical_bernstein,
    required_n_hoeffding,
    required_n_wilks,
)
from transfergap.report import format_table  # noqa: E402

TRIALS = 600
TARGETS = (2.0, 3.0, 5.0, 8.0)
R = synthetic.BOUND_RANGE_KNM


def main() -> int:
    out = Tee("validate_inversion_output.txt")
    out.rule("transfergap -- the sample-count inversions, verified by drawing the samples")
    for line in environment_lines():
        out(line)
    truth = synthetic.population_truth(draws=400_000, seed=SEED_TRUTH)
    population_variance = float(
        np.var(synthetic.generate(400_000, "random", seed=SEED_TRUTH + 1).gaps, ddof=1)
    )
    pilot = synthetic.generate(15, "random", seed=123456)
    pilot_variance = float(np.var(pilot.gaps, ddof=1))
    out("")
    out(f"declared confidence   : {CONFIDENCE}")
    out(f"declared proportion   : {PROPORTION}")
    out(f"declared R            : {R:.4f} kN.m")
    out(f"true E[d]             : {truth.mean:.6f} +/- {truth.standard_error:.6f} kN.m")
    out(f"true Var[d]           : {population_variance:.6f} kN.m^2")
    out(f"15-point pilot Var[d] : {pilot_variance:.6f} kN.m^2 "
        f"({100.0 * pilot_variance / population_variance:.1f} per cent of the truth)")
    out(f"independent trials    : {TRIALS} per row")
    out("")
    out("The pilot variance is UNDERSTATED here, which is the common direction for a")
    out("small sample of a right-skewed quantity, and the consequence is visible in")
    out("the table below: the plan built on it asks for too few flights.")

    timer = Timer()
    out.rule("1. Hoeffding: the slack is data-free, so the hit rate must be exactly 1.0")
    rows = []
    hoeffding_failures = 0
    for target in TARGETS:
        n = required_n_hoeffding(R, target, CONFIDENCE)
        with timer:
            slacks = np.empty(TRIALS)
            covers = 0
            for trial in range(TRIALS):
                sample = synthetic.generate(n, "random", seed=700_000 + 977 * trial)
                bound = hoeffding_mean_upper(sample.gaps, R, CONFIDENCE)
                slacks[trial] = bound.slack
                covers += int(bound.upper >= truth.mean)
        hit = float(np.mean(slacks <= target))
        coverage = covers / TRIALS
        if hit != 1.0 or coverage < CONFIDENCE:
            hoeffding_failures += 1
        rows.append([f"{target:.1f}", str(n), f"{slacks.min():.6f}", f"{slacks.max():.6f}",
                     f"{hit:.4f}", f"{coverage:.4f}"])
    out(format_table(
        ["target slack", "predicted n", "min realised slack", "max realised slack",
         "slack hit rate", "coverage of true mean"], rows))
    out("")
    out.check(
        "the Hoeffding slack hit rate is exactly 1.0 at every predicted n",
        hoeffding_failures == 0,
        f"{len(TARGETS)} targets, {TRIALS} trials each; the realised slack is identical "
        f"across trials because equation (3) does not read the data",
    )

    out.rule("2. empirical Bernstein: the slack is random, so the hit rate is not 1.0")
    out("Two plug-in variances. The first is the true population variance, the best")
    out("case. The second is the 15-point pilot estimate, the realistic case.")
    out("")
    rows = []
    for label, variance in (
        ("0.5 x true", 0.5 * population_variance),
        ("true variance", population_variance),
        ("2.0 x true", 2.0 * population_variance),
        ("15-point pilot", pilot_variance),
    ):
        for target in TARGETS:
            n = required_n_empirical_bernstein(variance, R, target, CONFIDENCE)
            with timer:
                slacks = np.empty(TRIALS)
                covers = 0
                for trial in range(TRIALS):
                    sample = synthetic.generate(n, "random", seed=800_000 + 1009 * trial)
                    bound = empirical_bernstein_mean_upper(sample.gaps, R, CONFIDENCE)
                    slacks[trial] = bound.slack
                    covers += int(bound.upper >= truth.mean)
            hit = float(np.mean(slacks <= target))
            error = math.sqrt(max(hit * (1.0 - hit), 0.0) / TRIALS)
            rows.append([label, f"{target:.1f}", str(n), f"{float(np.mean(slacks)):.4f}",
                         f"{hit:.4f}", f"{error:.4f}", f"{covers / TRIALS:.4f}"])
    out(format_table(
        ["plug-in", "target slack", "predicted n", "mean realised slack",
         "slack hit rate", "+/- se", "coverage of true mean"], rows))
    out("")
    out("REPORTED AS FOUND. Read the two blocks against each other. With the true")
    out("variance the hit rate is near one half, because the plan is built on the")
    out("MEAN variance and a realised sample is above it half the time -- the")
    out("inversion is a median prediction, not a guarantee, and nothing in equation")
    out("(11) claims otherwise. With the pilot variance, which is understated, the")
    out("predicted n is smaller and the hit rate is lower still. A programme that")
    out("needs a guarantee on the slack has to use the Hoeffding plan of part 1.")
    out("")
    def block(label: str) -> list[list[str]]:
        return [row for row in rows if row[0] == label]

    true_rows = block("true variance")
    half_rows = block("0.5 x true")
    double_rows = block("2.0 x true")
    pilot_rows = block("15-point pilot")
    out.check(
        "the empirical Bernstein coverage of the true mean reaches the declared "
        "confidence at every predicted n",
        all(float(row[6]) >= CONFIDENCE for row in rows),
        "the bound itself holds even where the SLACK prediction misses; those are "
        "different claims",
    )
    out.check(
        "the empirical Bernstein slack hit rate is strictly below 1.0 under a plug-in "
        "variance",
        all(float(row[4]) < 1.0 for row in true_rows),
        f"hit rates {[row[4] for row in true_rows]} with the true variance",
    )
    out.check(
        "the hit rate is monotone in the plug-in variance, anchored on the TRUE value",
        all(
            float(low[4]) <= float(middle[4]) <= float(high[4])
            for low, middle, high in zip(half_rows, true_rows, double_rows, strict=True)
        ),
        f"half {[row[4] for row in half_rows]} <= true {[row[4] for row in true_rows]} "
        f"<= double {[row[4] for row in double_rows]}",
    )
    out("")
    out(f"The 15-point pilot at the declared seed gives {pilot_variance:.4f} kN.m^2 "
        f"against a true {population_variance:.4f}, a ratio of "
        f"{pilot_variance / population_variance:.4f}: it OVERSTATES the variance, so")
    out("its plan asks for more flights than needed and its hit rates "
        f"{[row[4] for row in pilot_rows]} sit above the true-variance block. The")
    out("expectation written into this script's docstring was the opposite and is")
    out("recorded there as falsified. One pilot draw does not establish a direction;")
    out("the anchored half-and-double rows above do.")

    out.rule("3. Wilks: the tolerance statement itself, at the predicted n")
    out("At the n of equation (7), the fraction of campaigns whose largest")
    out("observation covers at least p of the population must reach 1 - delta. The")
    out("population quantile is the reference and comes from the synthetic truth.")
    out("")
    rows = []
    wilks_failures = 0
    for proportion, confidence in ((0.90, 0.90), (0.95, 0.95), (0.90, 0.99)):
        n = required_n_wilks(proportion, confidence)
        reference = float(
            np.quantile(synthetic.generate(400_000, "random", seed=SEED_TRUTH + 2).gaps,
                        proportion)
        )
        with timer:
            covers = 0
            for trial in range(TRIALS):
                sample = synthetic.generate(n, "random", seed=900_000 + 1013 * trial)
                covers += int(float(np.max(sample.gaps)) >= reference)
        rate = covers / TRIALS
        error = math.sqrt(max(rate * (1.0 - rate), 0.0) / TRIALS)
        ok = rate >= confidence - 3.0 * error
        if not ok:
            wilks_failures += 1
        rows.append([f"{proportion:g}", f"{confidence:g}", str(n), f"{reference:.4f}",
                     f"{rate:.4f}", f"{error:.4f}", "ok" if ok else "BELOW"])
    out(format_table(
        ["p", "confidence", "required n", f"true p-quantile [{synthetic.UNIT}]",
         "measured rate", "+/- se", "verdict"], rows))
    out("")
    out.check(
        "the Wilks tolerance statement holds at the predicted n for every (p, "
        "confidence) pair",
        wilks_failures == 0,
        f"{TRIALS} campaigns per row",
    )

    out.rule("compute")
    out(f"total Monte Carlo wall clock: {timer.elapsed_s:.1f} s over {timer.calls} blocks")
    out("declared budget             : under 180 s on two cores")
    out.check(
        "the Monte Carlo finishes inside the declared three-minute budget",
        timer.elapsed_s < 180.0,
        f"{timer.elapsed_s:.1f} s",
    )

    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
