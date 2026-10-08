"""Validate the binomial intervals: exact coverage, closed forms, hand calculations.

Run from the product directory:

    python validation/validate_intervals.py

Everything here is exact arithmetic, not simulation: the coverage figures are
sums of binomial probability mass over the values of ``k`` whose interval
contains the given ``p``, so they carry no Monte-Carlo error.
"""

from __future__ import annotations

import math

from _reporting import Report  # noqa: E402

from rareverify.intervals import (  # noqa: E402
    clopper_pearson,
    exact_coverage,
    rule_of_three_upper,
    wilson,
    zero_failure_upper,
)


def main() -> int:
    report = Report("validate_intervals")

    report.line("## 1. Zero-failure closed forms against hand calculation")
    report.line(
        "Clopper-Pearson one-sided: 1 - alpha**(1/n). Wilson two-sided: z^2/(n+z^2)."
    )
    report.line(
        f"{'n':>8} {'CP one-sided':>16} {'CP two-sided':>16} "
        f"{'Wilson 2-sided':>16} {'3/n':>12}"
    )
    for n in (10, 100, 1000, 2995, 29956, 100000):
        report.line(
            f"{n:>8} {zero_failure_upper(n, 0.95, side='upper'):>16.10e} "
            f"{zero_failure_upper(n, 0.95, side='two-sided'):>16.10e} "
            f"{zero_failure_upper(n, 0.95, 'wilson', 'two-sided'):>16.10e} "
            f"{rule_of_three_upper(n):>12.6e}"
        )
    hand = 1.0 - 0.05 ** (1.0 / 100)
    report.check(
        "CP one-sided n=100 equals 1 - 0.05**(1/100) = 0.0295130496070399",
        abs(zero_failure_upper(100, 0.95, side="upper") - hand) < 1e-15,
        f"(difference {abs(zero_failure_upper(100, 0.95, side='upper') - hand):.3e})",
    )
    report.check(
        "closed form equals the Beta-quantile path for every n tested",
        all(
            abs(
                zero_failure_upper(n, 0.95, method=m, side="upper")
                - clopper_pearson(0, n, 0.95, side="upper").upper
            )
            <= 1e-12
            for n in (1, 2, 10, 137, 5000, 29956)
            for m in ("clopper-pearson",)
        ),
    )

    report.line("")
    report.line("## 2. The rule of three does not converge to the exact bound")
    report.line("3/n relative error, with its limit 3/(-ln 0.05) - 1 = 1.4246021e-3")
    for n in (100, 1000, 10000, 100000, 1000000):
        exact = zero_failure_upper(n, 0.95, side="upper")
        report.line(
            f"n={n:>8} exact={exact:.10e} 3/n={3.0 / n:.10e} "
            f"relative error={3.0 / n / exact - 1:.7e}"
        )
    limit = 3.0 / 2.9957322735539909 - 1.0
    report.check(
        "relative error decreases monotonically towards 1.4246021e-3 from above",
        all(
            3.0 / a / zero_failure_upper(a, 0.95, side="upper") - 1.0
            > 3.0 / b / zero_failure_upper(b, 0.95, side="upper") - 1.0
            > limit
            for a, b in ((100, 1000), (1000, 10000), (10000, 100000))
        ),
        f"(limit {limit:.7e})",
    )

    report.line("")
    report.line("## 3. At k = 0 the Wilson/Clopper-Pearson ordering crosses over")
    report.line(
        "A counter-example to the usual claim that the exact interval is always "
        "the wider one. Two-sided 95 %. The sign of (Wilson - CP) changes "
        "between n = 45 and n = 46: Wilson is NARROWER than Clopper-Pearson "
        "below that and WIDER above it. No test in this package asserts "
        "containment of one interval in the other."
    )
    report.line(f"{'n':>8} {'CP upper':>16} {'Wilson upper':>16} {'Wilson - CP':>14}")
    for n in (10, 20, 40, 45, 46, 50, 100, 1000, 10000):
        cp = clopper_pearson(0, n).upper
        wi = wilson(0, n).upper
        report.line(f"{n:>8} {cp:>16.10e} {wi:>16.10e} {wi - cp:>14.6e}")
    crossovers = [
        n
        for n in range(2, 400)
        if (wilson(0, n).upper > clopper_pearson(0, n).upper)
        != (wilson(0, n - 1).upper > clopper_pearson(0, n - 1).upper)
    ]
    report.check(
        "exactly one sign change of (Wilson - CP) at k=0 for 2 <= n <= 400, at n=46",
        crossovers == [46],
        f"(sign changes at {crossovers})",
    )
    report.check(
        "Wilson is wider at k=0 for n >= 46 up to 10000",
        all(
            wilson(0, n).upper > clopper_pearson(0, n).upper
            for n in (46, 50, 100, 500, 1000, 10000)
        ),
    )

    report.line("")
    report.line("## 4. Exact coverage, Clopper-Pearson vs Wilson, two-sided 95 %")
    report.line(
        "Coverage computed by summing the binomial mass over covering k. "
        "Nominal 0.95."
    )
    grid = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
    for n in (20, 50, 100, 1000):
        report.line(f"n = {n}")
        report.line(f"{'p':>8} {'CP coverage':>14} {'Wilson coverage':>16}")
        for p in grid:
            report.line(
                f"{p:>8.4g} {exact_coverage(n, p, method='clopper-pearson'):>14.6f} "
                f"{exact_coverage(n, p, method='wilson'):>16.6f}"
            )
    cp_min = min(
        exact_coverage(n, p, method="clopper-pearson")
        for n in (20, 50, 100, 1000)
        for p in grid
    )
    wilson_values = [
        (n, p, exact_coverage(n, p, method="wilson"))
        for n in (20, 50, 100, 1000)
        for p in grid
    ]
    wilson_min = min(v for _, _, v in wilson_values)
    worst = min(wilson_values, key=lambda t: t[2])
    report.check(
        "Clopper-Pearson coverage is at or above 0.95 everywhere on the grid",
        cp_min >= 0.95,
        f"(minimum {cp_min:.6f})",
    )
    report.check(
        "Wilson coverage dips below 0.95 somewhere on the grid, as expected",
        wilson_min < 0.95,
        f"(minimum {wilson_min:.6f} at n={worst[0]}, p={worst[1]})",
    )

    report.line("")
    report.line("## 5. One-sided exact coverage")
    cp_one = min(
        exact_coverage(n, p, method="clopper-pearson", side="upper")
        for n in (20, 50, 100)
        for p in grid
    )
    wi_one = min(
        exact_coverage(n, p, method="wilson", side="upper")
        for n in (20, 50, 100)
        for p in grid
    )
    report.line(f"minimum one-sided CP coverage over the grid: {cp_one:.6f}")
    report.line(f"minimum one-sided Wilson coverage over the grid: {wi_one:.6f}")
    report.check("one-sided CP coverage is at or above 0.95", cp_one >= 0.95)

    report.line("")
    report.line("## 6. Interior interval widths, k/n = 0.005")
    report.line(f"{'n':>8} {'k':>5} {'CP width':>14} {'Wilson width':>14} {'ratio':>8}")
    for n in (200, 1000, 10000, 100000):
        k = max(1, round(0.005 * n))
        cp = clopper_pearson(k, n)
        wi = wilson(k, n)
        report.line(
            f"{n:>8} {k:>5} {cp.width:>14.6e} {wi.width:>14.6e} "
            f"{cp.width / wi.width:>8.4f}"
        )
    report.check(
        "Clopper-Pearson is the wider interval in the interior at k/n = 0.005",
        all(
            clopper_pearson(max(1, round(0.005 * n)), n).width
            > wilson(max(1, round(0.005 * n)), n).width
            for n in (200, 1000, 10000, 100000)
        ),
    )

    report.line("")
    report.line("## 7. Degenerate inputs")
    report.check("k=0 gives lower limit exactly 0", clopper_pearson(0, 7).lower == 0.0)
    report.check("k=n gives upper limit exactly 1", clopper_pearson(7, 7).upper == 1.0)
    report.check(
        "n=1, k=0, 95 % one-sided upper is 0.95",
        abs(zero_failure_upper(1, 0.95, side="upper") - 0.95) < 1e-15,
    )
    report.check(
        "coverage at p=0 and p=1 is 1 for Clopper-Pearson",
        math.isclose(exact_coverage(10, 0.0), 1.0)
        and math.isclose(exact_coverage(10, 1.0), 1.0),
    )

    return report.finish()


if __name__ == "__main__":
    raise SystemExit(main())
