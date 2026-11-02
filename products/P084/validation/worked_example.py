"""The worked example quoted in README.md, run so the quoted output is real.

The scenario: a flight-test campaign has returned fifteen paired points -- a
twin prediction and a strain-gauge measurement of wing-root bending moment at
the same operating condition. The programme wants an upper bound on the
sim-to-field discrepancy at 95 per cent confidence, and the number of
additional flights that would bring it to 2 kN.m.

Everything printed here is produced by the package. The README quotes this
output verbatim and is re-run against it as the last step of the build.
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

import numpy as np  # noqa: E402

from _bootstrap import (  # noqa: E402
    CONFIDENCE,
    PROPORTION,
    SEED_TRUTH,
    Tee,
    bootstrap,
    environment_lines,
)

ROOT = bootstrap()

from transfergap import synthetic  # noqa: E402
from transfergap.bounds import (  # noqa: E402
    empirical_bernstein_mean_upper,
    hoeffding_mean_upper,
    normal_tolerance_upper,
    wilks_tolerance_upper,
)
from transfergap.exchangeability import permutation_test  # noqa: E402
from transfergap.learned import DiscrepancyModel  # noqa: E402
from transfergap.plan import plan_existence, plan_for_slack  # noqa: E402
from transfergap.report import (  # noqa: E402
    render_bounds,
    render_exchangeability,
    render_plan,
)
from transfergap.resample import bootstrap_mean_upper, split_conformal_upper  # noqa: E402

TARGET_SLACK = 2.0


def main() -> int:
    out = Tee("worked_example_output.txt")
    out.rule("transfergap -- a fifteen-point campaign, worked through")
    for line in environment_lines():
        out(line)

    sample = synthetic.generate(15, "random", seed=101)
    out("")
    out(f"paired measurements : {sample.size}")
    out(f"unit                : {sample.unit}")
    out(f"declared range R    : {sample.bound_range:.4f} {sample.unit}")
    out(f"operating conditions: {', '.join(sample.feature_names)}")
    out("")
    out("the fifteen discrepancies, in the order flown:")
    out("  " + "  ".join(f"{value:.3f}" for value in sample.gaps))
    out(f"  sample mean {float(np.mean(sample.gaps)):.4f} {sample.unit}, "
        f"sd {float(np.std(sample.gaps, ddof=1)):.4f}, "
        f"max {float(np.max(sample.gaps)):.4f}")

    out.rule("step 1 -- test the assumption before using anything that rests on it")
    tests = [
        permutation_test(sample.gaps, name, permutations=1999, alpha=0.05, seed=0)
        for name in ("trend", "half_split")
    ]
    out(render_exchangeability(tests))
    out("")
    out("At n = 15 the measured POWER of this test against the swept-card")
    out("alternative is 0.43, so a non-rejection here is close to uninformative.")
    out("That is itself the answer: a fifteen-point campaign cannot establish the")
    out("assumption its tighter bounds need, so the distribution-free bound is the")
    out("one to quote.")

    out.rule("step 2 -- the bounds that exist at this sample size")
    bounds = [
        hoeffding_mean_upper(sample.gaps, sample.bound_range, CONFIDENCE, sample.unit),
        empirical_bernstein_mean_upper(
            sample.gaps, sample.bound_range, CONFIDENCE, sample.unit
        ),
        bootstrap_mean_upper(
            sample.gaps, sample.bound_range, CONFIDENCE, resamples=4000, seed=101,
            unit=sample.unit,
        ),
        normal_tolerance_upper(
            sample.gaps, sample.bound_range, PROPORTION, CONFIDENCE, sample.unit
        ),
    ]
    out(render_bounds(bounds, sample.unit))
    out("")
    for builder, label in (
        (lambda: wilks_tolerance_upper(
            sample.gaps, sample.bound_range, PROPORTION, CONFIDENCE, sample.unit
        ), "Wilks' nonparametric tolerance limit"),
        (lambda: split_conformal_upper(
            sample.gaps, sample.bound_range, PROPORTION, sample.unit
        ), "the split-conformal limit"),
    ):
        try:
            builder()
        except ValueError as error:
            out(f"{label} does not exist at n = 15:")
            out(f"  {error}")
            out("")

    out.rule("step 3 -- how many more flights")
    plans = plan_for_slack(sample, TARGET_SLACK, CONFIDENCE)
    plans += plan_existence(sample, PROPORTION, CONFIDENCE)
    out(render_plan(plans))

    out.rule("step 4 -- what the learned model adds, and what it costs")
    model = DiscrepancyModel(
        upper_level=PROPORTION, lower_level=1.0 - PROPORTION, seed=990000
    ).fit(sample.features, sample.gaps)
    pool = synthetic.generate(2000, "random", seed=880001)
    prediction = model.predict_with_uncertainty(pool.features)
    truth = synthetic.population_truth(draws=200_000, seed=SEED_TRUTH)
    out(f"conditional band over a {pool.size}-point held-out pool:")
    out(f"  mean upper band      {float(np.mean(prediction.upper)):.4f} {sample.unit}")
    out(f"  mean interval width  {float(np.mean(prediction.width)):.4f} {sample.unit}")
    out(f"  quantile crossing    {prediction.crossing_rate:.6f} of conditions")
    out(f"  achieved coverage    {float(np.mean(pool.gaps <= prediction.upper)):.4f} "
        f"against a declared {PROPORTION:g}")
    out("")
    out(f"For reference, the true population {PROPORTION:g} quantile of the discrepancy")
    out(f"is {truth.quantiles[PROPORTION]:.4f} {sample.unit} and the true mean is "
        f"{truth.mean:.4f}. Those are")
    out("available here only because the data is synthetic.")
    out("")
    out("The learned band is NARROWER than the normal-theory tolerance limit above --")
    out("the only limit on a FUTURE measurement that exists at n = 15 at all -- and")
    out("its coverage is below what it declares. Over 80 independent campaigns at")
    out("this sample size the measured coverage is 0.8761 against a declared 0.95;")
    out("see validation/validate_learned_model_output.txt. The model's value here is")
    out("not the bound, it is the ranking: it says WHERE in the envelope the twin is")
    out("worst, which no marginal bound in this package can do.")
    out("")
    out("This model is not certified for operational flight use.")

    out.rule("the answer a programme would write down")
    hoeffding = bounds[0]
    out("  From 15 paired points, the mean sim-to-field discrepancy in wing-root")
    out(f"  bending moment is at most {hoeffding.upper:.2f} {sample.unit} with "
        f"{CONFIDENCE:g} confidence,")
    out("  distribution-free (Hoeffding 1963), assuming independent sampling and a")
    out(f"  declared channel range of {sample.bound_range:.0f} {sample.unit}.")
    out(f"  Reaching a slack of {TARGET_SLACK:.0f} {sample.unit} needs "
        f"{plans[0].required_n} points in total, "
        f"{plans[0].additional_n} more.")
    out("  No tolerance limit on a FUTURE measurement can be stated from 15 points:")
    out(f"  equation (7) needs {plans[2].required_n} and equation (12) needs "
        f"{plans[3].required_n}.")

    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
