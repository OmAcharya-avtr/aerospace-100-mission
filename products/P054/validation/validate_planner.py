"""Validate the sample-size planner: closed forms, feasibility, monotonicity audit.

Run from the product directory:

    python validation/validate_planner.py
"""

from __future__ import annotations

import math

from _reporting import Report  # noqa: E402

from rareverify.intervals import proportion_interval, zero_failure_upper  # noqa: E402
from rareverify.montecarlo import required_samples_for_cov  # noqa: E402
from rareverify.planner import (  # noqa: E402
    detection_probability,
    plan_campaign,
    probability_of_zero_violations,
    samples_for_relative_width,
    samples_for_zero_failure_bound,
)


def main() -> int:
    report = Report("validate_planner")

    report.line("## 1. Demonstration sample size against the closed form")
    report.line("n = ceil(ln(alpha) / ln(1 - p_target)), Clopper-Pearson one-sided")
    report.line(
        f"{'p_target':>10} {'conf':>6} {'n':>12} {'closed form':>16} "
        f"{'bound at n':>14} {'bound at n-1':>14}"
    )
    all_ok = True
    for p_target in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
        for confidence in (0.90, 0.95, 0.99):
            n = samples_for_zero_failure_bound(p_target, confidence)
            closed = math.log(1.0 - confidence) / math.log1p(-p_target)
            at_n = zero_failure_upper(n, confidence, side="upper")
            at_prev = zero_failure_upper(n - 1, confidence, side="upper")
            all_ok &= at_n <= p_target < at_prev
            report.line(
                f"{p_target:>10.1e} {confidence:>6.2f} {n:>12} {closed:>16.4f} "
                f"{at_n:>14.6e} {at_prev:>14.6e}"
            )
    report.check(
        "the returned n is the smallest one meeting the bound, in all 15 cases",
        all_ok,
    )
    report.check(
        "p_target=1e-3, 95 % gives n=2995 (hand calculation 2994.23 -> 2995)",
        samples_for_zero_failure_bound(1e-3, 0.95) == 2995,
    )
    report.check(
        "p_target=1e-4, 95 % gives n=29956",
        samples_for_zero_failure_bound(1e-4, 0.95) == 29956,
    )
    report.check(
        "p_target=1e-3, 99 % gives n=4603 (hand calculation 4602.87 -> 4603)",
        samples_for_zero_failure_bound(1e-3, 0.99) == 4603,
    )

    report.line("")
    report.line("## 2. Consistency identity: P(k=0) at p_target equals alpha")
    report.line(
        "By construction the demonstration size is the n at which a campaign "
        "whose true p equals p_target has probability alpha of seeing nothing."
    )
    report.line(f"{'p_target':>10} {'n':>12} {'P(k=0)':>12} {'alpha':>8} {'detection':>12}")
    identity_ok = True
    for p_target in (1e-2, 1e-3, 1e-4, 1e-5):
        n = samples_for_zero_failure_bound(p_target, 0.95)
        zero = probability_of_zero_violations(n, p_target)
        identity_ok &= abs(zero - 0.05) < 2e-3
        report.line(
            f"{p_target:>10.1e} {n:>12} {zero:>12.6f} {0.05:>8.2f} "
            f"{detection_probability(n, p_target):>12.6f}"
        )
    report.check("P(k=0) at p_target is 0.05 to within 2e-3", identity_ok)

    report.line("")
    report.line("## 3. The Wilson plan is SMALLER, which makes it the weaker claim")
    report.line(
        "For a one-sided demonstration the Wilson k=0 bound is "
        "z^2/(n+z^2) with z = 1.6448536, i.e. 2.70554/n asymptotically, "
        "against the exact 2.99573/n. Wilson therefore asks for about 9.7 % "
        "fewer runs for the same nominal claim, and the exact coverage study "
        "in validate_intervals.py shows its one-sided coverage dropping to "
        "0.9308 on the same grid where Clopper-Pearson stays at or above "
        "0.9520. Planning with Wilson buys a smaller campaign by weakening "
        "the guarantee. The default in this package is Clopper-Pearson."
    )
    report.line(f"{'p_target':>10} {'CP n':>12} {'Wilson n':>12} {'ratio':>8}")
    for p_target in (1e-2, 1e-3, 1e-4, 1e-5):
        cp = samples_for_zero_failure_bound(p_target, 0.95)
        wi = samples_for_zero_failure_bound(p_target, 0.95, method="wilson")
        report.line(f"{p_target:>10.1e} {cp:>12} {wi:>12} {wi / cp:>8.4f}")
    ratios = [
        samples_for_zero_failure_bound(p, 0.95, method="wilson")
        / samples_for_zero_failure_bound(p, 0.95)
        for p in (1e-2, 1e-3, 1e-4, 1e-5)
    ]
    report.check(
        "the Wilson plan is smaller than the Clopper-Pearson plan at every "
        "p_target tested, and the ratio approaches 2.70554/2.99573 = 0.90313",
        all(r < 1.0 for r in ratios) and abs(ratios[-1] - 0.90313) < 2e-3,
        f"(ratios {[round(r, 5) for r in ratios]})",
    )

    report.line("")
    report.line("## 4. Estimation sample size for a relative interval width")
    report.line(
        "Assumes p equals p_target and k = round(n p). The predicate is the "
        "two-sided interval width at 95 %."
    )
    report.line(
        f"{'p':>10} {'rel width':>10} {'n (CP)':>14} {'achieved width/p':>18} "
        f"{'n (Wilson)':>12}"
    )
    feasible_ok = True
    for p in (1e-3, 1e-4):
        for width in (1.0, 0.5, 0.25):
            n = samples_for_relative_width(p, width, 0.95)
            k = round(n * p)
            achieved = proportion_interval(k, n, 0.95).width / p
            feasible_ok &= achieved <= width + 1e-12
            n_wilson = samples_for_relative_width(p, width, 0.95, method="wilson")
            report.line(
                f"{p:>10.1e} {width:>10.2f} {n:>14} {achieved:>18.6f} "
                f"{n_wilson:>12}"
            )
    report.check("every returned estimation size meets its width predicate", feasible_ok)

    report.line("")
    report.line("## 5. Non-monotonicity audit of the width predicate")
    report.line(
        "k = round(n p) jumps, so the interval width is not monotone in n. "
        "Counting the n in a window where width(n+1) > width(n)."
    )
    p = 1e-3
    target = 0.5 * p
    base = samples_for_relative_width(p, 0.5, 0.95)
    violations = 0
    window = range(base - 400, base + 400)
    widths = {}
    for n in window:
        k = round(n * p)
        widths[n] = proportion_interval(min(k, n), n, 0.95).width
    for n in list(window)[:-1]:
        if widths[n + 1] > widths[n]:
            violations += 1
    report.line(f"window {base - 400}..{base + 399} around the returned n = {base}")
    report.line(f"non-monotone steps in that window: {violations} of 799")
    smaller_feasible = [n for n in window if n < base and widths[n] <= target]
    report.line(
        f"feasible n smaller than the returned one inside the window: "
        f"{len(smaller_feasible)}"
    )
    report.check(
        "the predicate is genuinely non-monotone, which is why the search "
        "brackets and then scans downward",
        violations > 0,
        f"({violations} non-monotone steps)",
    )
    report.check(
        "no feasible n smaller than the returned one exists inside the "
        "512-step back-scan window",
        not smaller_feasible,
        f"({len(smaller_feasible)} found)",
    )

    report.line("")
    report.line("## 6. Crude cost of the same targets, for the record")
    report.line(
        "required_samples_for_cov(p, 0.1) = (1 - p) / (p * 0.01): the run count "
        "a crude campaign needs for a 10 % coefficient of variation."
    )
    report.line(f"{'p':>10} {'n for cov=0.1':>16} {'n for cov=0.05':>16}")
    for p in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
        report.line(
            f"{p:>10.1e} {required_samples_for_cov(p, 0.1):>16} "
            f"{required_samples_for_cov(p, 0.05):>16}"
        )
    report.check(
        "p=1e-6 at cov=0.1 needs 99999900 runs, which is the arithmetic that "
        "motivates variance reduction",
        required_samples_for_cov(1e-6, 0.1) == 99_999_900,
    )

    report.line("")
    report.line("## 7. Full plan for the package's headline target")
    plan = plan_campaign(1e-4, confidence=0.95, relative_width=0.5)
    for line in plan.describe().splitlines():
        report.line("  " + line)

    return report.finish()


if __name__ == "__main__":
    raise SystemExit(main())
