"""The closed-form re-run count, against hand arithmetic, a recursion and a simulation.

Three independent checks of the mathematical core, in increasing order of how
much they could have caught.

1. **Hand arithmetic.** Values a reader can check with a calculator, printed
   with the arithmetic beside them.
2. **A recursion that shares no expression with the closed form.** Equation (5)
   divides by ``1 - p``; ``expected_failed_attempts_by_recursion`` evaluates
   ``E[F_m] = p (1 + E[F_{m-1}])`` with no division at all, so the two cannot
   share a cancellation.
3. **A Monte Carlo simulation of the actual retry process.** This is the only
   check here that could catch a wrong MODEL rather than a wrong evaluation of
   the right model: it draws Bernoulli attempts and counts failures, with no
   reference to any closed form. Agreement is asserted against the simulation's
   own standard error, at four standard errors, which is a 1-in-16000 two-sided
   criterion per cell -- the criterion is declared rather than the seed being
   changed until a tighter one passes.

The divergence at ``p -> 1`` is checked as a raised error with an actionable
message, not as a returned ``inf``.

NO CLOCK IS READ. The Monte Carlo draws come from ``numpy.random.default_rng``
at the declared seed, so this script's numbers are the same on every machine.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import math  # noqa: E402

import numpy as np  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

from flakerank.geometric import (  # noqa: E402
    expected_attempts_by_recursion,
    expected_attempts_to_first_pass,
    expected_failed_attempts,
    expected_failed_attempts_by_recursion,
    expected_wasted_seconds,
)

#: Monte Carlo invocations per cell. 20000 at 2 cores costs well under a
#: second; the whole sweep is far inside the mission's 60 s budget.
TRIALS = 4000 if _bootstrap.FAST else 20000

#: Agreement criterion, in standard errors of the simulation mean.
SIGMA = 4.0

#: Failure probabilities swept. 1/44 is the rate the mission's own validation
#: probe actually failed at.
PROBABILITIES = (0.0, 1.0 / 44.0, 0.05, 0.25, 0.5, 0.75)

#: Retry budgets swept, with ``None`` for an unbounded policy.
BUDGETS = (1, 2, 3, 5, 10, None)


def simulate(probability: float, budget: int | None, trials: int, seed: int) -> tuple[float, float]:
    """Mean and standard error of the simulated failed-attempt count.

    Draws the retry process directly: attempt until a pass or until the budget
    is exhausted, counting failures. No closed form appears anywhere in this
    function, which is the point of it.
    """
    generator = np.random.default_rng(seed)
    cap = budget if budget is not None else 200
    draws = generator.random((trials, cap)) < probability
    # Failed attempts per invocation: the number of leading True values, capped
    # at the budget. ``argmin`` of the boolean row gives the first False, which
    # is the first pass; a row that is all True has no pass and contributes the
    # whole cap.
    all_failed = draws.all(axis=1)
    first_pass = draws.argmin(axis=1)
    counts = np.where(all_failed, cap, first_pass).astype(np.float64)
    return float(counts.mean()), float(counts.std(ddof=1) / math.sqrt(trials))


def main() -> int:
    """Check the closed forms three ways and report every number."""
    out = _bootstrap.Tee("validate_geometric_output.txt")
    out("Closed-form geometric re-run count -- P097 flakerank")
    out("")
    out("Three independent checks of the mathematical core: hand arithmetic, a")
    out("recursion sharing no expression with the closed form, and a Monte Carlo")
    out("simulation of the retry process that references no closed form at all.")
    out("")
    out(f"fast path (FLAKERANK_FAST=1): {_bootstrap.FAST}")
    out(f"declared seed: {_bootstrap.SEED}")
    out(f"Monte Carlo invocations per cell: {TRIALS}")
    out(f"agreement criterion, standard errors: {SIGMA}")
    out("")
    out("Sources. Feller, W. (1968), An Introduction to Probability Theory and Its")
    out("Applications, Volume I, 3rd edition, Wiley -- the geometric distribution")
    out("and its mean. Johnson, N. L., Kemp, A. W. & Kotz, S. (2005), Univariate")
    out("Discrete Distributions, 3rd edition, Wiley, chapter 5 -- the same result")
    out("with the shifted-support convention used here.")

    out.rule("1. HAND ARITHMETIC -- values a reader can check with a calculator")
    out("")
    out("p = 1/4, unbounded:")
    out("  1 - 0.25 = 0.75;  E[N] = 1/0.75 = 4/3")
    out(f"  E[N] computed: {expected_attempts_to_first_pass(0.25)!r}")
    out("  E[F] = 0.25/0.75 = 1/3")
    out(f"  E[F] computed: {expected_failed_attempts(0.25)!r}")
    out.check(
        "E[N] at p = 1/4 equals 4/3",
        abs(expected_attempts_to_first_pass(0.25) - 4.0 / 3.0) <= 1e-15 * (4.0 / 3.0),
        f"{expected_attempts_to_first_pass(0.25)!r}",
    )
    out.check(
        "E[F] at p = 1/4 equals 1/3",
        abs(expected_failed_attempts(0.25) - 1.0 / 3.0) <= 1e-15 * (1.0 / 3.0),
        f"{expected_failed_attempts(0.25)!r}",
    )
    out("")
    out("p = 1/4, retry budget 3, equation (5): (p - p^3)/(1 - p) + p^3")
    out("  p^3 = 0.015625;  p - p^3 = 0.234375;  0.234375/0.75 = 0.3125")
    out("  0.3125 + 0.015625 = 0.328125")
    capped = expected_failed_attempts(0.25, retry_budget=3)
    out(f"  computed: {capped!r}")
    out.check(
        "E[F_3] at p = 1/4 equals 0.328125",
        abs(capped - 0.328125) <= 1e-15,
        f"{capped!r}",
    )
    out("")
    out("THE MISSION'S OWN CASE. A validation probe of this mission failed 1 time")
    out("in 44, and no number of agreeing serial verifications saw it.")
    out("  p = 1/44;  1 - p = 43/44;  E[F] = (1/44)/(43/44) = 1/43")
    mission = expected_failed_attempts(1.0 / 44.0)
    out(f"  E[F] computed: {mission!r}")
    out(f"  1/43 exactly  : {1.0 / 43.0!r}")
    out("  in a 600 s test: 600/43 s of wasted time per invocation")
    wasted = expected_wasted_seconds(1.0 / 44.0, 600.0)
    out(f"  computed: {wasted!r} s")
    over_history = expected_wasted_seconds(1.0 / 44.0, 600.0, None, 330)
    out(f"  over the 330 recorded sessions: {over_history!r} s")
    out.check(
        "E[F] at p = 1/44 equals 1/43",
        abs(mission - 1.0 / 43.0) <= 1e-15 * (1.0 / 43.0),
        f"{mission!r}",
    )
    out("")
    out("THE DASHBOARD INVERSION, which is this product's whole argument:")
    slow = expected_wasted_seconds(1.0 / 44.0, 600.0)
    fast = expected_wasted_seconds(0.5, 0.1)
    out(f"  a 1-in-44 failure in a 600 s test  : {slow!r} s wasted per invocation")
    out(f"  a 1-in-2 failure in a 0.1 s test   : {fast!r} s wasted per invocation")
    out(f"  cost ratio, slow over fast         : {slow / fast!r}")
    out(f"  flake-rate ratio, fast over slow   : {0.5 / (1.0 / 44.0)!r}")
    out(f"  the two rankings disagree on this ONE pair by a factor of "
        f"{(slow / fast) * (0.5 / (1.0 / 44.0))!r}")
    out.check(
        "the cost ranking and the rate ranking order this pair oppositely",
        slow > fast and 0.5 > 1.0 / 44.0,
        f"cost ratio {slow / fast:.6f}, rate ratio {0.5 / (1.0 / 44.0):.6f}",
    )

    out.rule("2. THE RECURSION -- no division, so no shared cancellation")
    out("")
    out("E[F_m] = p (1 + E[F_{m-1}]) with E[F_0] = 0, against equation (5).")
    out("")
    out(f"{'p':>12} {'m':>4} {'closed form':>22} {'recursion':>22} {'rel diff':>12}")
    worst_recursion = 0.0
    for probability in PROBABILITIES:
        for budget in (1, 2, 3, 5, 10):
            closed = expected_failed_attempts(probability, retry_budget=budget)
            recursive = expected_failed_attempts_by_recursion(probability, budget)
            scale = max(abs(closed), 1e-300)
            relative = abs(closed - recursive) / scale
            worst_recursion = max(worst_recursion, relative)
            out(f"{probability:>12.8f} {budget:>4} {closed:>22.15e} {recursive:>22.15e}"
                f" {relative:>12.3e}")
    out("")
    out.check(
        "equation (5) and its division-free recursion agree to 1e-12 relative",
        worst_recursion <= 1e-12,
        f"worst relative difference {worst_recursion:.3e}",
    )
    out("")
    out("E[N] = 1 + p E[N], as a fixed-point iteration, against 1/(1 - p):")
    out(f"{'p':>12} {'closed form':>22} {'fixed point':>22} {'rel diff':>12}")
    worst_fixed = 0.0
    for probability in PROBABILITIES:
        if probability >= 1.0:
            continue
        closed = expected_attempts_to_first_pass(probability)
        iterated = expected_attempts_by_recursion(probability)
        relative = abs(closed - iterated) / abs(closed)
        worst_fixed = max(worst_fixed, relative)
        out(f"{probability:>12.8f} {closed:>22.15e} {iterated:>22.15e} {relative:>12.3e}")
    out("")
    out.check(
        "the fixed-point iteration agrees with 1/(1 - p) to 1e-12 relative",
        worst_fixed <= 1e-12,
        f"worst relative difference {worst_fixed:.3e}",
    )

    out.rule("3. MONTE CARLO -- the retry process itself, referencing no closed form")
    out("")
    out("Bernoulli attempts are drawn and failures counted. This is the only")
    out("check here that could catch a wrong MODEL rather than a wrong")
    out("evaluation of the right one. The criterion is four standard errors of")
    out("the simulation mean, declared in advance.")
    out("")
    out(f"{'p':>12} {'m':>10} {'analytic':>14} {'simulated':>14} {'s.e.':>11} {'sigmas':>8}")
    worst_sigma = 0.0
    cells = 0
    outside = 0
    for index, probability in enumerate(PROBABILITIES):
        for position, budget in enumerate(BUDGETS):
            if budget is None and probability >= 1.0:
                continue
            analytic = expected_failed_attempts(probability, budget)
            simulated, error = simulate(
                probability, budget, TRIALS, _bootstrap.SEED + 1000 * index + position
            )
            sigmas = 0.0 if error == 0.0 else abs(simulated - analytic) / error
            cells += 1
            worst_sigma = max(worst_sigma, sigmas)
            if sigmas > SIGMA:
                outside += 1
            label = "unbounded" if budget is None else str(budget)
            out(f"{probability:>12.8f} {label:>10} {analytic:>14.6f} {simulated:>14.6f}"
                f" {error:>11.3e} {sigmas:>8.2f}")
    out("")
    out(f"cells compared: {cells}")
    out(f"cells outside {SIGMA} standard errors: {outside}")
    out(f"worst deviation, standard errors: {worst_sigma:.3f}")
    out.check(
        f"every analytic value lies within {SIGMA} standard errors of the simulation",
        outside == 0,
        f"{outside} of {cells} outside, worst {worst_sigma:.3f} sigma",
    )
    out("")
    out("A note on the unbounded cells. The simulation caps attempts at 200,")
    out("which is not the analytic unbounded process. At p = 0.75 the")
    out("probability of 200 consecutive failures is 0.75**200 = "
        f"{0.75 ** 200:.3e}, so the truncation bias is below the simulation's own")
    out("standard error by many orders of magnitude and is not what the")
    out("agreement is testing. At p = 0.95 it would not be, which is why the")
    out("sweep stops at 0.75.")

    out.rule("4. THE DIVERGENCE AT p -> 1 -- raised, never returned as inf")
    out("")
    for call, label in (
        (lambda: expected_failed_attempts(1.0), "expected_failed_attempts(1.0)"),
        (lambda: expected_attempts_to_first_pass(1.0), "expected_attempts_to_first_pass(1.0)"),
        (lambda: expected_wasted_seconds(1.0, 10.0), "expected_wasted_seconds(1.0, 10.0)"),
    ):
        try:
            value = call()
        except ValueError as error:
            out(f"  {label} raised ValueError")
            out(f"    message begins: {str(error)[:72]}...")
            out.control(
                f"{label} raises rather than returning inf",
                True,
                must_fire=True,
                detail="ValueError",
            )
        else:
            out.control(
                f"{label} raises rather than returning inf",
                False,
                must_fire=True,
                detail=f"returned {value!r}",
            )
    out("")
    out("With a retry budget the same quantity is FINITE and equals the budget,")
    out("because all m attempts fail and all m are wasted. That is the analytic")
    out("limit of equation (5) at p = 1, taken by L'Hopital and returned.")
    for budget in (1, 2, 3, 7, 100):
        value = expected_failed_attempts(1.0, retry_budget=budget)
        out(f"  E[F_{budget}] at p = 1 = {value!r}")
        out.check(
            f"E[F_{budget}] at p = 1 equals {budget}",
            value == float(budget),
            f"{value!r}",
        )
    out("")
    out("CONTROL that must STAY SILENT: just below 1 the unbounded form returns a")
    out("large finite number rather than raising, so the error above is a")
    out("statement about p = 1 and not a refusal to work near it.")
    near = expected_failed_attempts(1.0 - 1e-12)
    out(f"  E[F] at p = 1 - 1e-12 = {near!r}")
    out.control(
        "the unbounded form raises just below p = 1",
        not math.isfinite(near),
        must_fire=False,
        detail=f"{near!r}",
    )
    return out.finish()


if __name__ == "__main__":
    sys.exit(main())
