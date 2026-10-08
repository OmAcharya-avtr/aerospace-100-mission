"""Known-answer checks against analytic tail probabilities.

Tolerances are stated as multiples of the counting-noise floor
``sqrt(p (1 - p) / n_eff)`` of the run that produced the estimate, or of the
estimator's own standard error when that is larger. No tolerance in this
script was adjusted after a result was seen.

Run from the product directory:

    python validation/validate_known_answer.py
"""

from __future__ import annotations

import numpy as np
from _reporting import Report  # noqa: E402

from rareverify.knownanswer import check_against_reference  # noqa: E402
from rareverify.limitstates import (  # noqa: E402
    LinearGaussianLimitState,
    LognormalRatioLimitState,
    RippledLimitState,
)
from rareverify.montecarlo import crude_monte_carlo  # noqa: E402
from rareverify.subset import subset_simulation  # noqa: E402
from rareverify.tilting import analytic_mean_shift, importance_sampling  # noqa: E402

TOLERANCE = 4.0
SAMPLES = 400_000


def main() -> int:
    report = Report("validate_known_answer")
    report.line(f"tolerance = {TOLERANCE} counting-noise floors, samples = {SAMPLES}")
    report.line(
        "A 4-floor two-sided normal tolerance has a 6.3e-5 false-alarm rate "
        "per check."
    )
    report.line("")

    instances = [
        ("linear beta=2.5", LinearGaussianLimitState(beta=2.5, dimension=2)),
        ("linear beta=3.719", LinearGaussianLimitState(beta=3.719, dimension=2)),
        ("linear beta=4.753", LinearGaussianLimitState(beta=4.753, dimension=2)),
        ("linear beta=3.719 d=6", LinearGaussianLimitState(beta=3.719, dimension=6)),
        ("lognormal default", LognormalRatioLimitState()),
        ("lognormal sigma_r=0.35", LognormalRatioLimitState(mu_r=1.4, sigma_r=0.35)),
        ("rippled A=0.8 w=1.5", RippledLimitState()),
        (
            "rippled beta=5.5 A=2.5 w=2.0",
            RippledLimitState(beta=5.5, amplitude=2.5, frequency=2.0),
        ),
    ]

    report.line("## 1. Crude Monte Carlo against the reference")
    for label, state in instances:
        reference = state.analytic_probability()
        estimate = crude_monte_carlo(state, SAMPLES, rng=np.random.default_rng(101))
        check = check_against_reference(
            f"{label} crude", estimate, reference, state.reference_kind, TOLERANCE
        )
        report.check(check.name, check.passed, check.describe())

    report.line("")
    report.line("## 2. Importance sampling with the analytic design-point tilt")
    for label, state in instances:
        reference = state.analytic_probability()
        tilt = analytic_mean_shift(state)
        estimate = importance_sampling(
            state, tilt, SAMPLES, rng=np.random.default_rng(202)
        )
        check = check_against_reference(
            f"{label} analytic-IS", estimate, reference, state.reference_kind, TOLERANCE
        )
        report.check(check.name, check.passed, check.describe())

    report.line("")
    report.line("## 3. Subset simulation, mean of 20 replications")
    report.line(
        "Subset simulation is biased at finite samples per level, so a single "
        "run is not the right object to test. The check is on the mean of 20 "
        "independent runs against the reference, with the tolerance set by "
        "the standard error of that mean."
    )
    report.line(
        f"{'instance':>30} {'reference':>14} {'mean':>14} {'se of mean':>12} "
        f"{'z':>7} {'empirical cov':>14}"
    )
    for label, state in instances:
        reference = state.analytic_probability()
        estimates = np.array(
            [
                subset_simulation(
                    state, n_per_level=2000, rng=np.random.default_rng([303, i])
                ).estimate
                for i in range(20)
            ]
        )
        mean = float(estimates.mean())
        se = float(estimates.std(ddof=1)) / np.sqrt(estimates.size)
        z = abs(mean - reference) / se
        report.line(
            f"{label:>30} {reference:>14.6e} {mean:>14.6e} {se:>12.3e} "
            f"{z:>7.3f} {estimates.std(ddof=1) / mean:>14.4f}"
        )
        report.check(f"{label} subset-simulation mean within 4 se", z <= 4.0, f"(z={z:.3f})")

    report.line("")
    report.line("## 4. Deliberate negative control")
    report.line(
        "The harness must fail when the estimate is wrong. Comparing a correct "
        "importance-sampling estimate against a deliberately wrong reference "
        "of 1e-2 must be reported as FAIL."
    )
    state = LinearGaussianLimitState(beta=3.719, dimension=2)
    estimate = importance_sampling(
        state, analytic_mean_shift(state), 100_000, rng=np.random.default_rng(404)
    )
    control = check_against_reference(
        "negative control", estimate, 1e-2, "closed-form", TOLERANCE
    )
    report.line("  " + control.describe())
    report.check(
        "the harness reports FAIL for a wrong reference", control.passed is False
    )

    return report.finish()


if __name__ == "__main__":
    raise SystemExit(main())
