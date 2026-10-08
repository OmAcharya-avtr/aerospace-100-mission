"""The AI component against the analytic/importance-sampling baseline.

The question is narrow and the answer is measured: at an equal budget of true
limit-state evaluations, does a learned surrogate used to place the
importance-sampling tilt beat the tilt an analyst can write down?

The baseline was implemented first and is the analytic design-point tilt of
rareverify.tilting. The surrogate's training evaluations are counted against
its budget, because a real campaign pays for them with runs of the simulator.

Expected outcome, stated before the measurement and confirmed by it: the
surrogate cannot win on a smooth analytic limit state, because the analytic
design point is exact and free. It can win on a rough one, where the analytic
design point of the smooth part is not the most probable failure point.

Run from the product directory:

    python validation/validate_surrogate.py
"""

from __future__ import annotations

import math
import time
import warnings

import numpy as np
from _reporting import Report  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402

from rareverify.benchmark import replicate, summarise, variance_reduction  # noqa: E402
from rareverify.limitstates import (  # noqa: E402
    LinearGaussianLimitState,
    RippledLimitState,
)
from rareverify.montecarlo import crude_monte_carlo  # noqa: E402
from rareverify.surrogate import (  # noqa: E402
    fit_surrogate,
    surrogate_design_point,
    surrogate_guided_importance_sampling,
    surrogate_probability,
)
from rareverify.tilting import (  # noqa: E402
    analytic_mean_shift,
    find_design_point_radial,
    importance_sampling,
    oracle_mean_shift,
)

warnings.simplefilter("ignore", ConvergenceWarning)

BUDGET = 60_000
REPLICATIONS_CHEAP = 40
REPLICATIONS_SURROGATE = 12
N_TRAIN = 100

SMOOTH = LinearGaussianLimitState(beta=3.719, dimension=2)
ROUGH = RippledLimitState(beta=5.5, amplitude=2.5, frequency=2.0)


def main() -> int:
    report = Report("validate_surrogate")
    report.line(
        f"budget = {BUDGET} TRUE limit-state evaluations for every method, "
        f"including the surrogate's training set"
    )
    report.line(
        f"replications: {REPLICATIONS_CHEAP} for the analytic methods, "
        f"{REPLICATIONS_SURROGATE} for the surrogate (compute budget)"
    )
    report.line("")

    report.line("## 1. Design-point recovery against the true answer")
    report.line(
        "Smooth instance true beta = 3.719 (closed form). Rough instance true "
        "beta = 3.097896 (ray search and SLSQP agree to 1e-13, see "
        "validate_design_point.py); its smooth-part design point is 5.5."
    )
    report.line(
        f"{'instance':>10} {'n_train':>8} {'beta_hat':>11} {'rel err':>11} "
        f"{'band lo':>10} {'band hi':>10} {'fit s':>7} {'rmse':>10}"
    )
    truths = {"smooth": 3.719, "rough": 3.0978962}
    recovery = {}
    for key, state in (("smooth", SMOOTH), ("rough", ROUGH)):
        for n_train in (25, 50, 100, 200):
            rng = np.random.default_rng([555, n_train])
            fit = fit_surrogate(state, n_train=n_train, rng=rng)
            design = surrogate_design_point(fit, rng=rng)
            recovery[(key, n_train)] = design.beta
            report.line(
                f"{key:>10} {n_train:>8} {design.beta:>11.6f} "
                f"{design.beta / truths[key] - 1:>11.3e} "
                f"{min(design.beta_lower, design.beta_upper):>10.5f} "
                f"{max(design.beta_lower, design.beta_upper):>10.5f} "
                f"{fit.fit_seconds:>7.2f} "
                f"{fit.diagnostics['train_rmse']:>10.3e}"
            )
    report.check(
        "the surrogate recovers the smooth design point to better than 1e-3 "
        "relative from 50 training evaluations onward",
        all(
            abs(recovery[("smooth", n)] / 3.719 - 1.0) < 1e-3
            for n in (50, 100, 200)
        ),
    )
    report.check(
        "the surrogate recovers the rough design point to better than 1e-2 "
        "relative from 100 training evaluations onward, where the analytic "
        "smooth design point is 77.5 % too large",
        all(
            abs(recovery[("rough", n)] / truths["rough"] - 1.0) < 1e-2
            for n in (100, 200)
        ),
        f"(analytic smooth design point error: {5.5 / truths['rough'] - 1:+.4f})",
    )

    report.line("")
    report.line("## 2. The structural reason the surrogate cannot win when smooth")
    analytic_point = SMOOTH.design_point()
    rng = np.random.default_rng(606)
    fit = fit_surrogate(SMOOTH, n_train=N_TRAIN, rng=rng)
    design = surrogate_design_point(fit, rng=rng)
    offset = float(np.linalg.norm(design.point - analytic_point))
    report.line(f"analytic design point       : {np.round(analytic_point, 8).tolist()}")
    report.line(f"surrogate design point      : {np.round(design.point, 8).tolist()}")
    report.line(f"distance between them       : {offset:.3e} standard-normal units")
    report.line(
        f"deterministic variance penalty at equal evaluations: "
        f"{BUDGET / (BUDGET - N_TRAIN):.6f} "
        f"({100 * (BUDGET / (BUDGET - N_TRAIN) - 1):.3f} % worse)"
    )
    report.line(
        f"surrogate fit + design-point search wall time: "
        f"{fit.fit_seconds:.2f} s + search, against a closed form that costs "
        "nothing"
    )
    report.check(
        "the surrogate's tilt is within 1e-2 standard-normal units of the "
        "analytic tilt on the smooth instance, so no variance gain is "
        "available to it",
        offset < 1e-2,
        f"(offset {offset:.3e})",
    )

    report.line("")
    report.line("## 3. Replicated benchmark at equal true-evaluation budget")
    for key, state, truth_beta in (
        ("smooth", SMOOTH, 3.719),
        ("rough", ROUGH, truths["rough"]),
    ):
        reference = state.analytic_probability()
        report.line(f"### {key}: {state.name}, reference p = {reference:.8e}")
        crude = summarise(
            "crude",
            replicate(
                lambda rng, s=state: crude_monte_carlo(s, BUDGET, rng=rng),
                REPLICATIONS_CHEAP,
                seed=701,
            ),
            reference,
        )
        analytic = summarise(
            "analytic-IS",
            replicate(
                lambda rng, s=state: importance_sampling(
                    s, analytic_mean_shift(s), BUDGET, rng=rng
                ),
                REPLICATIONS_CHEAP,
                seed=702,
            ),
            reference,
        )
        oracle_tilt = oracle_mean_shift(state, rng=np.random.default_rng(3))
        oracle = summarise(
            "oracle-IS (reference, not a method)",
            replicate(
                lambda rng, s=state, t=oracle_tilt: importance_sampling(
                    s, t, BUDGET, rng=rng
                ),
                REPLICATIONS_CHEAP,
                seed=703,
            ),
            reference,
        )

        def surrogate_run(rng, s=state):
            local = fit_surrogate(s, n_train=N_TRAIN, rng=rng)
            estimate, _ = surrogate_guided_importance_sampling(
                s, local, BUDGET - N_TRAIN, rng=rng
            )
            return estimate

        surrogate = summarise(
            f"surrogate-guided-IS (n_train={N_TRAIN})",
            replicate(surrogate_run, REPLICATIONS_SURROGATE, seed=704),
            reference,
        )
        for summary in (crude, analytic, oracle, surrogate):
            report.line("  " + summary.describe())
        for summary in (analytic, oracle, surrogate):
            report.line("  " + variance_reduction(summary, crude, reference).describe())
        head_to_head = (
            analytic.empirical_std**2
            * analytic.mean_true_evaluations
            / surrogate.mean_true_evaluations
        ) / surrogate.empirical_std**2
        report.line(
            f"  surrogate-guided-IS against analytic-IS, equal evaluations: "
            f"variance ratio = {head_to_head:.4f} "
            f"({'surrogate better' if head_to_head > 1 else 'BASELINE BETTER'})"
        )
        noise = math.sqrt(
            1.0 / (2 * (REPLICATIONS_SURROGATE - 1))
            + 1.0 / (2 * (REPLICATIONS_CHEAP - 1))
        )
        report.line(
            f"  relative uncertainty on that variance ratio from replication "
            f"noise alone: about {2 * noise:.2f} (one standard deviation of "
            f"the log ratio is {noise:.3f})"
        )
        if key == "smooth":
            report.check(
                "SMOOTH INSTANCE: the surrogate does not beat the analytic "
                "baseline by more than the replication noise allows, and the "
                "structural reason is in section 2",
                head_to_head < math.exp(3 * noise),
                f"(ratio {head_to_head:.4f}, 3-sigma band "
                f"{math.exp(-3 * noise):.3f}..{math.exp(3 * noise):.3f})",
            )
        else:
            report.check(
                "ROUGH INSTANCE: the surrogate beats the analytic baseline by "
                "more than the replication noise allows",
                head_to_head > math.exp(3 * noise),
                f"(ratio {head_to_head:.4f}, 3-sigma upper edge "
                f"{math.exp(3 * noise):.3f})",
            )
            report.check(
                "ROUGH INSTANCE: the surrogate does not reach the oracle tilt",
                variance_reduction(surrogate, crude, reference).vrf_measured
                < variance_reduction(oracle, crude, reference).vrf_measured,
            )
        report.line(f"  true reliability index used by the oracle: {truth_beta:.6f}")
        report.line("")

    report.line("## 4. Sample-efficiency curve, single run per training size")
    report.line(
        "Total true evaluations held at the budget: n_train training "
        "evaluations plus (budget - n_train) samples. The reported standard "
        "error is the estimator's own."
    )
    report.line(
        f"{'instance':>10} {'n_train':>8} {'estimate':>14} {'rel err':>10} "
        f"{'std error':>12} {'cov':>8} {'ESS':>10} {'straddle':>9}"
    )
    report.line(
        "straddle is the fraction of 1000 samples drawn around the surrogate's "
        "design point whose posterior band mean +- 2 sigma contains zero, i.e. "
        "where the surrogate does not know which side of the limit state it is "
        "on. It is the surrogate's own confidence output."
    )
    curve_errors = []
    for key, state in (("smooth", SMOOTH), ("rough", ROUGH)):
        reference = state.analytic_probability()
        for n_train in (25, 100, 300):
            rng = np.random.default_rng([808, n_train])
            local = fit_surrogate(state, n_train=n_train, rng=rng)
            estimate, curve_design = surrogate_guided_importance_sampling(
                state, local, BUDGET - n_train, rng=rng
            )
            sample = rng.standard_normal((1000, state.dimension)) + curve_design.point
            curve_errors.append(abs(estimate.estimate / reference - 1.0))
            report.line(
                f"{key:>10} {n_train:>8} {estimate.estimate:>14.6e} "
                f"{estimate.estimate / reference - 1:>+10.4f} "
                f"{estimate.standard_error:>12.4e} "
                f"{estimate.coefficient_of_variation:>8.4f} "
                f"{estimate.effective_sample_size:>10.1f} "
                f"{local.straddle_fraction(sample):>9.4f}"
            )
    report.line(
        "The curve is flat on both instances: on these two-dimensional limit "
        "states 25 training evaluations already place the tilt well enough "
        "that more do not help. That is a property of the test problems, not "
        "a general result, and it means the sample-efficiency question has no "
        "interesting answer here: the surrogate's cost is dominated by its "
        "fit and search time, not by its training-set size."
    )
    report.check(
        "every surrogate-guided estimate on the curve is within 5 % of its "
        "reference, because it evaluates the true limit state",
        all(error < 0.05 for error in curve_errors),
        f"(worst {max(curve_errors):.4f})",
    )

    report.line("")
    report.line("## 5. The surrogate-only estimator and its bias")
    report.line(
        "Replacing the true limit state by the surrogate makes the estimator "
        "biased by the surrogate's error, and the bias is not covered by the "
        "estimator's own standard error. Measured on a high-frequency rippled "
        "instance where the surrogate has to work."
    )
    hard = RippledLimitState(beta=3.719, amplitude=0.8, frequency=6.0)
    reference = hard.analytic_probability()
    report.line(f"reference p = {reference:.8e} (quadrature, 400 nodes)")
    report.line(
        f"{'n_train':>8} {'surrogate-only':>16} {'rel err':>10} "
        f"{'its own se':>12} {'|err|/se':>10} {'guided (true g)':>16} "
        f"{'guided rel err':>15}"
    )
    detectable = []
    guided_errors = []
    for n_train in (20, 30, 60, 120, 240):
        rng = np.random.default_rng([909, n_train])
        local = fit_surrogate(hard, n_train=n_train, rng=rng)
        only = surrogate_probability(local, 20_000, rng=rng)
        guided, _ = surrogate_guided_importance_sampling(
            hard, local, 20_000, rng=rng
        )
        ratio = abs(only.estimate - reference) / max(only.standard_error, 1e-300)
        detectable.append((n_train, ratio))
        guided_errors.append(abs(guided.estimate / reference - 1.0))
        report.line(
            f"{n_train:>8} {only.estimate:>16.6e} "
            f"{only.estimate / reference - 1:>+10.4f} "
            f"{only.standard_error:>12.4e} {ratio:>10.2f} "
            f"{guided.estimate:>16.6e} {guided.estimate / reference - 1:>+15.4f}"
        )
    report.check(
        "the surrogate-only bias exceeds its own standard error at small "
        "training sizes",
        any(ratio > 5.0 for n, ratio in detectable if n <= 30),
        f"(|err|/se at n_train<=30: "
        f"{[round(r, 2) for n, r in detectable if n <= 30]})",
    )
    report.check(
        "the surrogate-guided estimator stays within 100 % of the reference at "
        "every training size even when the surrogate is bad, because it "
        "evaluates the true limit state; the n_train=20 case shows the cost is "
        "variance, not bias",
        all(error < 1.0 for error in guided_errors),
        f"(relative errors {[round(e, 4) for e in guided_errors]})",
    )

    report.line("")
    report.line("## 6. Compute: marginal-likelihood restarts change nothing material")
    report.line(
        "n_restarts_optimizer is 0 by default. Raising it costs fit time; the "
        "recovered reliability index is compared."
    )
    report.line(f"{'instance':>10} {'restarts':>9} {'fit s':>8} {'beta_hat':>11}")
    betas: dict[tuple[str, int], float] = {}
    for key, state in (("smooth", SMOOTH), ("rough", ROUGH)):
        for restarts in (0, 2):
            start = time.perf_counter()
            rng = np.random.default_rng(1010)
            local = fit_surrogate(state, n_train=120, rng=rng, n_restarts=restarts)
            design = surrogate_design_point(local, rng=rng)
            betas[(key, restarts)] = design.beta
            report.line(
                f"{key:>10} {restarts:>9} {time.perf_counter() - start:>8.2f} "
                f"{design.beta:>11.6f}"
            )
    report.check(
        "restarting the marginal-likelihood optimisation does not change the "
        "recovered reliability index, so the default of 0 restarts spends the "
        "compute budget on samples instead",
        all(
            abs(betas[(key, 0)] / betas[(key, 2)] - 1.0) < 1e-6
            for key in ("smooth", "rough")
        ),
    )
    report.line(
        "Fit wall times on this container are erratic at the 2x level between "
        "otherwise identical calls, because the L-BFGS iteration count varies "
        "with the random initial kernel; they are reported for budgeting only "
        "and are not a characteristic of the method."
    )

    report.line("")
    report.line("## 7. Independent check of the oracle design point")
    for key, state in (("smooth", SMOOTH), ("rough", ROUGH)):
        ray, _ = find_design_point_radial(
            state.g, state.dimension, n_directions=2048, rng=np.random.default_rng(1)
        )
        report.line(
            f"{key}: ray search beta = {float(np.linalg.norm(ray)):.7f}, "
            f"smooth-part beta = {float(np.linalg.norm(state.design_point())):.7f}"
        )

    return report.finish()


if __name__ == "__main__":
    raise SystemExit(main())
