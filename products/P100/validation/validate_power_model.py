"""The closed-form detection power, checked against a simulation and two bounds.

A closed form whose only evidence is the algebra that produced it is the P077
class of this mission: every test compared an expression against expressions
sharing its own defect, and a bound derived from different mathematics caught
it immediately. So this script checks the closed forms three ways:

1. against a Monte Carlo estimate, with agreement required inside the Monte
   Carlo standard error;
2. against a two-term inclusion-exclusion bracket, which is derived from the
   union of the events rather than from the complement product the
   implementation evaluates;
3. against a bracket from the exponential inequality on ``log(1-p)``, which is
   derived from a bound on the logarithm rather than from the product at all.

It also checks the two inversions against each other and reports the published
detection-power table.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import math  # noqa: E402

import numpy as np  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

from seedaudit.power import (  # noqa: E402
    MISSION_RATE,
    bonferroni_bracket_reference,
    detection_probability_pairwise,
    detection_probability_reference,
    escaped_rate_upper_bound,
    exponential_bracket_reference,
    monte_carlo_detection,
    repeats_for_detection,
    repeats_for_escaped_rate_bound,
    seed_axis_detection,
    smallest_detectable_rate,
    union_bound_reference,
)
from seedaudit.report import REPORTED_REPEATS, format_power_table  # noqa: E402

#: Monte Carlo audits per grid cell. 200000 at the mission rate gives a
#: standard error near 4.6e-4 on a probability of 0.0449, which is small
#: against the quantity being checked.
TRIALS = 20000 if _bootstrap.FAST else 200000

#: The grid the simulation is run over.
GRID_RATES = (0.001, 0.01, MISSION_RATE, 0.1, 0.5, 0.9)
GRID_REPEATS = (2, 3, 5, 10)


def main() -> int:
    """Check the closed forms and print the published table."""
    out = _bootstrap.Tee("validate_power_model_output.txt")
    out("Closed-form detection power -- P100 SeedAudit")
    out("")
    out("P_ref(p, n)  = 1 - (1-p)^n        reference-comparison design")
    out("P_pair(p, n) = 1 - p^n - (1-p)^n  pairwise-difference design")
    out("p_min(n, a)  = 1 - a^(1/n)        smallest rate caught with prob 1-a")
    out("p_max(n, a)  = 1 - a^(1/n)        upper bound on p after n agreeing runs")
    out("")
    out(f"fast path (SEEDAUDIT_FAST=1): {_bootstrap.FAST}")
    out(f"declared seed: {_bootstrap.SEED}")
    out(f"Monte Carlo audits per grid cell: {TRIALS}")
    out(f"alpha: {_bootstrap.ALPHA!r}")

    out.rule("1. THE HEADLINE CONSTANTS, by hand and by the implementation")
    hand_reference = 1.0 - (43.0 / 44.0) ** 2
    hand_pairwise = 1.0 - (1.0 / 44.0) ** 2 - (43.0 / 44.0) ** 2
    out(f"p = 1/44 = {MISSION_RATE!r}")
    out(f"(43/44)^2                  = {(43.0 / 44.0) ** 2!r}")
    out(f"1 - (43/44)^2    by hand   = {hand_reference!r}")
    out(f"P_ref(1/44, 2)   computed  = {detection_probability_reference(MISSION_RATE, 2)!r}")
    out(f"1 - (1/44)^2 - (43/44)^2   = {hand_pairwise!r}")
    out(f"P_pair(1/44, 2)  computed  = {detection_probability_pairwise(MISSION_RATE, 2)!r}")
    out.check(
        "P_ref(1/44, 2) agrees with the hand arithmetic to 1e-15 relative",
        abs(detection_probability_reference(MISSION_RATE, 2) - hand_reference)
        <= 1e-15 * hand_reference,
    )
    out.check(
        "P_pair(1/44, 2) agrees with the hand arithmetic to 1e-13 relative",
        abs(detection_probability_pairwise(MISSION_RATE, 2) - hand_pairwise)
        <= 1e-13 * hand_pairwise,
    )
    out("")
    out("THIS IS THE REFUTATION OF BASELINE 1, derived. Run-twice-and-diff has")
    out(f"detection power {detection_probability_pairwise(MISSION_RATE, 2)!r} at p = 1/44:")
    out("it finds a 1-in-44 intermittent defect about one time in twenty-two.")

    out.rule("2. MONTE CARLO against the closed forms")
    rng = np.random.default_rng(_bootstrap.SEED)
    out(
        f"{'p':>12}  {'n':>3}  {'MC P_ref':>12}  {'closed P_ref':>14}  {'z':>7}"
        f"  {'MC P_pair':>12}  {'closed P_pair':>14}  {'z':>7}"
    )
    worst_z = 0.0
    for rate in GRID_RATES:
        for repeats in GRID_REPEATS:
            mc_ref, ref_se, mc_pair, pair_se = monte_carlo_detection(rate, repeats, TRIALS, rng)
            exact_ref = detection_probability_reference(rate, repeats)
            exact_pair = detection_probability_pairwise(rate, repeats)
            z_ref = abs(mc_ref - exact_ref) / ref_se if ref_se > 0.0 else 0.0
            z_pair = abs(mc_pair - exact_pair) / pair_se if pair_se > 0.0 else 0.0
            worst_z = max(worst_z, z_ref, z_pair)
            out(
                f"{rate:>12.6f}  {repeats:>3}  {mc_ref:>12.6f}  {exact_ref:>14.6f}"
                f"  {z_ref:>7.3f}  {mc_pair:>12.6f}  {exact_pair:>14.6f}  {z_pair:>7.3f}"
            )
    out("")
    out(f"worst standardised deviation over the {len(GRID_RATES) * len(GRID_REPEATS)} cells: "
        f"{worst_z:.4f}")
    out.check(
        "every Monte Carlo estimate lies within 4 standard errors of its closed form",
        worst_z <= 4.0,
        f"worst z = {worst_z:.4f}",
    )

    out.rule("3. INDEPENDENTLY DERIVED BRACKETS")
    out("the inclusion-exclusion bracket is DEGENERATE at n = 2: with two events")
    out("S1 - S2 = 2p - p^2 = 1 - (1-p)^2 exactly, so it carries no information")
    out("there and the check below uses n >= 3.")
    out("")
    out(f"{'p':>12}  {'n':>3}  {'IE lower':>14}  {'exact':>14}  {'IE upper':>14}  {'inside':>7}")
    ie_inside = 0
    ie_total = 0
    for rate in (1e-5, 1e-3, MISSION_RATE, 0.05):
        for repeats in (3, 4, 8, 20):
            low, high = bonferroni_bracket_reference(rate, repeats)
            exact = detection_probability_reference(rate, repeats)
            slack = 4.0 * math.comb(repeats, 3) * rate**3 + 8.0 * np.finfo(float).eps * max(
                exact, 1e-300
            )
            inside = low - slack <= exact <= high + slack
            ie_inside += int(inside)
            ie_total += 1
            out(
                f"{rate:>12.6g}  {repeats:>3}  {low:>14.9f}  {exact:>14.9f}  {high:>14.9f}"
                f"  {str(inside):>7}"
            )
    out("")
    out("the bracket slack follows the CONDITIONING of the bracket expression --")
    out("the third inclusion-exclusion term C(n,3) p^3 -- and not the operation")
    out("count and not an absolute constant. An absolute floor made a bracket of")
    out("width 4.5e-23 vacuous in an earlier product of this mission.")
    out.check(
        f"the closed form lies inside the inclusion-exclusion bracket on all {ie_total} cells",
        ie_inside == ie_total,
        f"{ie_inside} of {ie_total}",
    )

    exp_inside = 0
    exp_total = 0
    out("")
    out(f"{'p':>12}  {'n':>3}  {'exp lower':>14}  {'exact':>14}  {'exp upper':>14}  {'inside':>7}")
    for rate in GRID_RATES:
        for repeats in (1, 2, 5, 30, 200):
            low, high = exponential_bracket_reference(rate, repeats)
            exact = detection_probability_reference(rate, repeats)
            slack = 16.0 * np.finfo(float).eps * max(exact, 1e-300)
            inside = low - slack <= exact <= high + slack
            exp_inside += int(inside)
            exp_total += 1
            out(
                f"{rate:>12.6g}  {repeats:>3}  {low:>14.9f}  {exact:>14.9f}  {high:>14.9f}"
                f"  {str(inside):>7}"
            )
    out("")
    out.check(
        f"the closed form lies inside the exponential bracket on all {exp_total} cells",
        exp_inside == exp_total,
        f"{exp_inside} of {exp_total}",
    )
    union_ok = all(
        union_bound_reference(rate, repeats) >= detection_probability_reference(rate, repeats)
        for rate in GRID_RATES
        for repeats in GRID_REPEATS
    )
    out.check("the first-order union bound sits above the closed form everywhere", union_ok)

    out.rule("4. THE TWO INVERSIONS, checked against each other")
    out("p_min and p_max are numerically one expression and are two claims: the")
    out("smallest rate the design catches with probability 1-a is the largest")
    out("rate consistent with having caught nothing.")
    out("")
    worst_gap = 0.0
    for repeats in REPORTED_REPEATS:
        gap = abs(smallest_detectable_rate(repeats) - escaped_rate_upper_bound(repeats))
        worst_gap = max(worst_gap, gap)
    out(f"worst absolute disagreement over {len(REPORTED_REPEATS)} budgets: {worst_gap!r}")
    out.check("p_min and p_max agree to within rounding on every reported budget", worst_gap == 0.0)
    out("")
    round_trip_worst = 0.0
    for repeats in REPORTED_REPEATS:
        recovered = detection_probability_reference(escaped_rate_upper_bound(repeats), repeats)
        round_trip_worst = max(round_trip_worst, abs(recovered - 0.95))
    out(f"worst |P_ref(p_max(n), n) - 0.95| over the same budgets: {round_trip_worst!r}")
    out.check(
        "the escaped-rate bound recovers its own detection probability to 1e-12",
        round_trip_worst <= 1e-12,
    )
    out("")
    needed = repeats_for_detection(MISSION_RATE, 0.95)
    out(f"repeats needed for 95 per cent detection at p = 1/44: {needed}")
    out(f"  P_ref(1/44, {needed})     = {detection_probability_reference(MISSION_RATE, needed)!r}")
    out(f"  P_ref(1/44, {needed - 1}) = "
        f"{detection_probability_reference(MISSION_RATE, needed - 1)!r}")
    out.check(
        "131 repeats reach 0.95 at p = 1/44 and 130 do not",
        needed == 131
        and detection_probability_reference(MISSION_RATE, 131) >= 0.95
        and detection_probability_reference(MISSION_RATE, 130) < 0.95,
    )
    out(f"repeats needed for an escaped-rate bound at or below 1/44: "
        f"{repeats_for_escaped_rate_bound(MISSION_RATE)}")

    out.rule("5. THE SEED AXIS IS A DIFFERENT AXIS")
    out("a quantity that is a deterministic function of the seed yields the same")
    out("value on every repeat at a fixed seed, so repeats carry ZERO information")
    out("about seed-dependence and the Bernoulli formulae above do not apply.")
    out("seeds drawn without replacement from a finite declared set give a")
    out("HYPERGEOMETRIC count, not a binomial one.")
    out("")
    out(f"{'population':>11}  {'sensitive':>10}  {'draws':>6}  {'hypergeometric':>15}"
        f"  {'binomial (wrong)':>17}")
    for population, sensitive, draws in ((10, 5, 2), (100, 50, 3), (1000, 999, 1), (44, 2, 5)):
        hyper = seed_axis_detection(population, sensitive, draws)
        binomial = detection_probability_reference(sensitive / population, draws)
        out(f"{population:>11}  {sensitive:>10}  {draws:>6}  {hyper:>15.9f}  {binomial:>17.9f}")
    out.check(
        "the hypergeometric and binomial answers differ, so the distinction is not cosmetic",
        abs(seed_axis_detection(10, 5, 2) - detection_probability_reference(0.5, 2)) > 1e-3,
    )

    out.rule("6. THE PUBLISHED DETECTION-POWER TABLE")
    out(format_power_table())
    out("")
    out("read the last column as the answer to: after n agreeing runs, how high")
    out("could the rate still be? At n = 2 it is 0.776. That is what a two-run")
    out("reproducibility report has actually established.")

    out.rule("7. WHAT THE BOUND ASSUMES, and the direction the assumption fails in")
    out("independence across repeats is the strong assumption. Repeats inside one")
    out("process share RNG state, import caches, a fixed PYTHONHASHSEED and a warm")
    out("filesystem cache, so they are POSITIVELY CORRELATED, the true detection")
    out("power is LOWER than the formula, and p_max is OPTIMISTIC. The mitigation")
    out("is a fresh subprocess in a fresh pristine copy with a fresh hash seed,")
    out("which seedaudit.dynamic does. The residual correlation is not zero and")
    out("this product does not estimate it.")
    return out.finish()


if __name__ == "__main__":
    sys.exit(main())
