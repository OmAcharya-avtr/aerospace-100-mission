"""Where importance sampling is WORSE than plain Monte Carlo.

This is a required deliverable, not an aside. A mean-shift tilt is a choice,
and the wrong choice costs variance, accuracy, or both. The sweep below is run
at the same budget and the same replication count as the favourable cases in
validate_variance_reduction.py, so the numbers are directly comparable.

Two verdicts are reported for every tilt:

- VRF, the variance reduction factor. Below 1 the tilt is worse than no tilt.
- MSERF, the mean-squared-error reduction factor against the exactly unbiased
  crude estimator. This is the verdict to believe, because an over-tilted
  sampler can have a tiny variance around a badly wrong answer.

Run from the product directory:

    python validation/validate_is_worse.py
"""

from __future__ import annotations

from _reporting import Report  # noqa: E402

from rareverify.benchmark import replicate, summarise, variance_reduction  # noqa: E402
from rareverify.limitstates import (  # noqa: E402
    LinearGaussianLimitState,
    RippledLimitState,
)
from rareverify.montecarlo import crude_monte_carlo  # noqa: E402
from rareverify.tilting import (  # noqa: E402
    importance_sampling,
    orthogonal_mean_shift,
    scaled_mean_shift,
)

BUDGET = 100_000
REPLICATIONS = 40

SCALES = (-1.0, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
ORTHOGONAL = (0.1, 0.25, 0.5, 0.75, 1.0, 1.5)


def main() -> int:
    report = Report("validate_is_worse")
    report.line(f"budget = {BUDGET} samples, replications = {REPLICATIONS}")
    report.line("")

    worse_cases: list[str] = []
    for label, state in (
        ("linear beta=3.719 d=2", LinearGaussianLimitState(beta=3.719, dimension=2)),
        ("rippled beta=5.5 A=2.5 w=2.0", RippledLimitState(beta=5.5, amplitude=2.5, frequency=2.0)),
    ):
        reference = state.analytic_probability()
        report.line(f"## {label}   reference p = {reference:.8e}")
        crude = summarise(
            "crude",
            replicate(
                lambda rng, s=state: crude_monte_carlo(s, BUDGET, rng=rng),
                REPLICATIONS,
                seed=3001,
            ),
            reference,
        )
        report.line("  " + crude.describe())
        report.line(
            f"{'tilt':>16} {'|theta|':>9} {'mean':>13} {'std':>12} "
            f"{'rel bias':>10} {'VRF':>12} {'MSERF':>12} {'mean ESS':>10} "
            f"{'verdict':>12}"
        )
        tilts = [scaled_mean_shift(state, s) for s in SCALES] + [
            orthogonal_mean_shift(state, s) for s in ORTHOGONAL
        ]
        for tilt in tilts:
            results = replicate(
                lambda rng, t=tilt, s=state: importance_sampling(
                    s, t, BUDGET, rng=rng
                ),
                REPLICATIONS,
                seed=3002,
            )
            summary = summarise(tilt.label, results, reference)
            mean_ess = sum(r.effective_sample_size for r in results) / len(results)
            if summary.empirical_std <= 0.0:
                report.line(
                    f"{tilt.label:>16} {tilt.norm:>9.4f} "
                    f"{summary.mean_estimate:>13.5e} {0.0:>12.4e} "
                    f"{summary.relative_bias:>+10.4f} {'n/a':>12} {'n/a':>12} "
                    f"{mean_ess:>10.2f} {'DEGENERATE':>12}"
                )
                worse_cases.append(f"{label} / {tilt.label}: degenerate, every run returned 0")
                continue
            reduction = variance_reduction(summary, crude, reference)
            verdict = "WORSE" if reduction.worse_in_mse else "better"
            report.line(
                f"{tilt.label:>16} {tilt.norm:>9.4f} {summary.mean_estimate:>13.5e} "
                f"{summary.empirical_std:>12.4e} {summary.relative_bias:>+10.4f} "
                f"{reduction.vrf_measured:>12.4g} "
                f"{reduction.mse_reduction_factor:>12.4g} {mean_ess:>10.2f} "
                f"{verdict:>12}"
            )
            if reduction.worse_in_mse:
                worse_cases.append(
                    f"{label} / {tilt.label}: VRF={reduction.vrf_measured:.4g}, "
                    f"MSERF={reduction.mse_reduction_factor:.4g}, "
                    f"relative bias {summary.relative_bias:+.4f}"
                )
            elif reduction.worse_than_reference:
                worse_cases.append(
                    f"{label} / {tilt.label}: WORSE in variance "
                    f"(VRF={reduction.vrf_measured:.4g}) though not in MSE"
                )
        report.line("")

    report.line("## Tilts that are worse than plain Monte Carlo")
    for case in worse_cases:
        report.line(f"  - {case}")
    report.check(
        "at least four tilts are measurably worse than plain Monte Carlo",
        len(worse_cases) >= 4,
        f"({len(worse_cases)} found)",
    )
    report.check(
        "the wrong-direction tilt at scale -1 is degenerate: no replication "
        "ever reaches the failure region",
        any("degenerate" in case for case in worse_cases),
    )

    report.line("")
    report.line("## The trap the MSE metric exists to catch")
    report.line(
        "Over-tilting at scale 3 on the linear instance: the sampler is "
        "centred deep inside the failure region, almost every sample fails, "
        "and the few samples that would carry significant weight are never "
        "drawn. The result is an estimate many orders of magnitude below the "
        "truth with a standard error to match, so the variance reduction "
        "factor is enormous and meaningless. The first version of this sweep "
        "reported that tilt as 'better'."
    )
    state = LinearGaussianLimitState(beta=3.719, dimension=2)
    reference = state.analytic_probability()
    crude = summarise(
        "crude",
        replicate(
            lambda rng: crude_monte_carlo(state, BUDGET, rng=rng),
            REPLICATIONS,
            seed=3001,
        ),
        reference,
    )
    over = summarise(
        "scale=3",
        replicate(
            lambda rng: importance_sampling(
                state, scaled_mean_shift(state, 3.0), BUDGET, rng=rng
            ),
            REPLICATIONS,
            seed=3002,
        ),
        reference,
    )
    reduction = variance_reduction(over, crude, reference)
    report.line(f"  reference p             : {reference:.8e}")
    report.line(f"  mean estimate           : {over.mean_estimate:.8e}")
    report.line(f"  relative bias           : {over.relative_bias:+.6f}")
    report.line(f"  variance reduction      : {reduction.vrf_measured:.6g}")
    report.line(f"  MSE reduction           : {reduction.mse_reduction_factor:.6g}")
    report.check(
        "the variance verdict and the MSE verdict disagree for the over-tilted "
        "sampler, which is why both are reported",
        reduction.worse_than_reference is False and reduction.worse_in_mse is True,
    )

    return report.finish()


if __name__ == "__main__":
    raise SystemExit(main())
