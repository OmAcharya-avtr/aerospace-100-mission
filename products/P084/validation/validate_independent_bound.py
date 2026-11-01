"""Two bound checks whose bounds come from mathematics this code does not use.

Why this is a separate script
-----------------------------
"Two implementations agree" and "the answer lies inside a bound derived from
different mathematics" are different checks, and the second catches what the
first cannot. One product of this mission had 777 tests, property tests
included, that could not catch catastrophic cancellation in a closed form,
because every test compared the series against expressions sharing the same
cancellation. A check against an independently derived bound caught it
immediately.

Check A: Hoeffding against the EXACT binomial tail
--------------------------------------------------
Hoeffding's inequality, equation (2), asserts

    P( dbar - mu >= t ) <= exp(-2 n t^2 / R^2).

For the one distribution where the left-hand side can be written down exactly
-- a two-point distribution on ``{0, R}``, which is a scaled Bernoulli -- the
tail is a finite binomial sum,

    P( dbar - mu >= t ) = P( Binomial(n, p) >= n(p + t/R) ),              (28)

evaluated by ``scipy.stats.binom.sf``. The implementation under test computes
the exponential closed form and never forms a binomial coefficient, a
factorial or a sum, so equation (28) is mathematics it does not use. The
two-point distribution is also the extremal case for Hoeffding's inequality at
fixed range, so the comparison is where the bound is tightest and a violation
would be most visible.

The slack of the comparison follows the CONDITIONING of the bound expression.
Both sides span nineteen decades across the grid below, so the comparison is
made in LOG space and the allowance is relative to the magnitude of the
logarithm: ``64 * eps * |log(bound)|``. An absolute allowance on the
probabilities would be vacuous at 1e-19 and impossible at 1.

Check B: split conformal against the finite-sample coverage bracket
-------------------------------------------------------------------
Equation (13), ``1 - alpha <= P(s_{n+1} <= qhat) <= 1 - alpha + 1/(n+1)``, is
derived from the exchangeability and order-statistic argument of Vovk et al.
(2005) and Lei et al. (2018), Theorem 2.2. This implementation takes the
``k``-th smallest calibration score by integer index and never computes a
coverage probability, a rank distribution or a Beta law, so the bracket is a
bound from mathematics it does not use. It is also TWO-SIDED, which makes it
much harder to satisfy by accident than a one-sided bound: a limit that is
merely conservative violates the upper half.

The Monte Carlo allowance on the bracket follows the conditioning of the
estimate, four binomial standard errors at the repetition count used, and is
printed.

Check C: Wilks against the Beta(n, 1) order-statistic law
---------------------------------------------------------
Wilks (1941) shows the population proportion below the largest of ``n``
i.i.d. continuous draws is ``Beta(n, 1)``-distributed. The implementation uses
only ``ceil(ln delta / ln p)``. The coverage distribution is measured by Monte
Carlo and compared against ``scipy.stats.beta(n, 1)``, which is a different
derivation of the same fact.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

from _bootstrap import CONFIDENCE, PROPORTION, Tee, bootstrap, environment_lines

ROOT = bootstrap()

from transfergap.bounds import required_n_wilks  # noqa: E402
from transfergap.report import format_table  # noqa: E402
from transfergap.resample import conformal_order_index  # noqa: E402

EPS = float(np.finfo(np.float64).eps)
R = 40.0


def hoeffding_against_the_exact_binomial_tail(out: Tee) -> None:
    out.rule("A. Hoeffding's exponential bound against the exact binomial tail")
    out("The two-point distribution on {0, R} is the extremal case for Hoeffding's")
    out("inequality at a fixed range, so this is where the bound is tightest. The")
    out("exact tail of equation (28) must never exceed the exponential bound.")
    out("")
    out("The comparison is in log space with an allowance of 64 eps times the")
    out("magnitude of log(bound), because the two sides span nineteen decades and an")
    out("absolute allowance on the probabilities would be vacuous at one end and")
    out("impossible at the other.")
    out("")
    rows = []
    violations = 0
    vacuous = 0
    worst_ratio = 0.0
    tightest = 1.0
    for n in (10, 25, 50, 100, 250, 500):
        for p in (0.1, 0.3, 0.5):
            for t_over_r in (0.05, 0.10, 0.20, 0.35):
                t = t_over_r * R
                threshold = n * (p + t_over_r)
                if threshold > n:
                    continue
                exact = float(stats.binom.sf(math.ceil(threshold) - 1, n, p))
                bound = math.exp(-2.0 * n * t * t / (R * R))
                allowance = 64.0 * EPS * abs(math.log(bound))
                ok = exact <= 0.0 or math.log(exact) <= math.log(bound) + allowance
                if not ok:
                    violations += 1
                ratio = exact / bound if bound > 0 else float("inf")
                worst_ratio = max(worst_ratio, ratio)
                if exact > 0.0:
                    tightest = min(tightest, ratio) if ratio > 0 else tightest
                if ratio < 1e-6:
                    vacuous += 1
                rows.append(
                    [str(n), f"{p:g}", f"{t_over_r:g}", f"{exact:.4e}", f"{bound:.4e}",
                     f"{ratio:.4e}", "ok" if ok else "VIOLATED"]
                )
    out(format_table(["n", "p", "t/R", "exact tail", "Hoeffding bound", "exact/bound",
                      "verdict"], rows))
    out("")
    out(f"grid points                     : {len(rows)}")
    out(f"violations                      : {violations}")
    out(f"worst exact/bound ratio         : {worst_ratio:.6f}")
    out(f"points where the ratio is < 1e-6: {vacuous} (the bound is loose there, which")
    out("                                  is a property of the inequality and is why")
    out("                                  this product reports the bound and not a")
    out("                                  claim that it is tight)")
    out("")
    out.check(
        "the exact binomial tail never exceeds Hoeffding's exponential bound",
        violations == 0,
        f"{len(rows)} grid points, worst ratio {worst_ratio:.6f}",
    )
    out.check(
        "the bound is not vacuous everywhere: at least one grid point has a ratio "
        "above 0.1",
        worst_ratio > 0.1,
        f"worst ratio {worst_ratio:.6f}; a bound never within a decade of the truth "
        f"would be untestable",
    )


def conformal_against_the_coverage_bracket(out: Tee) -> None:
    out.rule("B. split conformal against the two-sided finite-sample bracket")
    out("Equation (13) is two-sided, so a merely conservative limit fails it. The")
    out("test uses uniform scores, where the truth is known exactly, and the")
    out("allowance is four binomial standard errors of the measured rate.")
    out("")
    repetitions = 40000
    rng = np.random.default_rng(31337)
    rows = []
    failures = 0
    for n in (19, 39, 59, 99, 199):
        scores = rng.uniform(size=(repetitions, n))
        fresh = rng.uniform(size=repetitions)
        k = conformal_order_index(n, 1.0 - CONFIDENCE)
        limits = np.sort(scores, axis=1)[:, k - 1]
        rate = float(np.mean(fresh <= limits))
        lower = CONFIDENCE
        upper = CONFIDENCE + 1.0 / (n + 1)
        allowance = 4.0 * math.sqrt(CONFIDENCE * (1.0 - CONFIDENCE) / repetitions)
        inside = lower - allowance <= rate <= upper + allowance
        if not inside:
            failures += 1
        rows.append([str(n), str(k), f"{lower:.6f}", f"{upper:.6f}", f"{rate:.6f}",
                     f"{allowance:.6f}", "inside" if inside else "OUTSIDE"])
    out(format_table(["n", "k", "bracket low", "bracket high", "measured", "+/- 4 se",
                      "verdict"], rows))
    out("")
    out.check(
        "the measured conformal coverage lies inside the bracket of equation (13) at "
        "every n",
        failures == 0,
        f"{repetitions} repetitions per row, {failures} rows outside",
    )
    out("")
    out("The upper half of the bracket is the part that matters here. A limit that")
    out("simply takes the sample maximum would have a coverage above it at every n,")
    out("so passing this check is evidence that the order index of equation (12) is")
    out("the right one and not merely a safe one.")


def wilks_against_the_beta_law(out: Tee) -> None:
    out.rule("C. Wilks' required n against the Beta(n, 1) order-statistic law")
    out("The implementation computes ceil(ln delta / ln p). The check measures the")
    out("distribution of the achieved coverage by Monte Carlo and compares it with")
    out("scipy.stats.beta(n, 1), a different derivation of the same fact.")
    out("")
    repetitions = 40000
    rng = np.random.default_rng(4242)
    rows = []
    failures = 0
    for proportion, confidence in ((0.90, 0.90), (0.95, 0.95), (0.99, 0.95)):
        n = required_n_wilks(proportion, confidence)
        draws = rng.uniform(size=(repetitions, n))
        coverage = np.max(draws, axis=1)
        measured = float(np.mean(coverage >= proportion))
        from_beta = float(stats.beta.sf(proportion, n, 1))
        closed = 1.0 - proportion**n
        allowance = 4.0 * math.sqrt(from_beta * (1.0 - from_beta) / repetitions)
        ok = abs(measured - from_beta) <= allowance and from_beta >= confidence
        if not ok:
            failures += 1
        rows.append([f"{proportion:g}", f"{confidence:g}", str(n), f"{from_beta:.6f}",
                     f"{closed:.6f}", f"{measured:.6f}", f"{allowance:.6f}",
                     "ok" if ok else "FAILED"])
    out(format_table(["p", "conf", "required n", "Beta(n,1) sf", "1 - p^n", "measured",
                      "+/- 4 se", "verdict"], rows))
    out("")
    out.check(
        "the measured coverage distribution matches the Beta(n, 1) law and reaches "
        "the declared confidence",
        failures == 0,
        f"{repetitions} repetitions per row",
    )
    out("")
    out(f"The required n at p = {PROPORTION:g}, confidence {CONFIDENCE:g} is "
        f"{required_n_wilks(PROPORTION, CONFIDENCE)}.")


def main() -> int:
    out = Tee("validate_independent_bound_output.txt")
    out.rule("transfergap -- bounds from mathematics this implementation does not use")
    for line in environment_lines():
        out(line)
    hoeffding_against_the_exact_binomial_tail(out)
    conformal_against_the_coverage_bracket(out)
    wilks_against_the_beta_law(out)
    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
