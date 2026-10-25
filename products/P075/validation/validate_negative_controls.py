"""Negative controls: each injected defect is verified to actually fire.

The defect class this exists for
--------------------------------

A negative control that does not fire proves nothing, and the way they fail to
fire is arithmetic: **offsetting a value by the tolerance does not guarantee a
violation.** If a check passes when ``|computed - reference| <= tol``, then
perturbing the computed value by exactly ``tol`` leaves it on the boundary, and
by anything less than ``2 * tol`` can leave it inside -- because the reference
itself may already sit up to ``tol`` away on the other side. Only an offset
**strictly greater than twice the tolerance** guarantees the check fails.

Every control below is therefore:

1. anchored on the **true** value, not on a nominal one;
2. offset by a factor strictly greater than 2 times the tolerance it is meant
   to break;
3. **verified to have fired**, with the verification reported as its own check.

A control that does not fire is reported as a failure of the control, which is
a failure of the validation and not of the product -- and it is the one kind of
failure that would otherwise be invisible.
"""

from __future__ import annotations

import math

from _bootstrap import Tee, Timer, bootstrap, header

bootstrap()

from reliablock.bounds import bonferroni_series_bounds, esary_proschan_bounds  # noqa: E402
from reliablock.bridge import bridge_diagram, bridge_reliability_equal  # noqa: E402
from reliablock.diagram import Block, Diagram, KOfN, Parallel, Series  # noqa: E402
from reliablock.distributions import Exponential  # noqa: E402
from reliablock.importance import (  # noqa: E402
    boolean_system_unreliability,
    importance_table,
    system_unreliability_at,
)
from reliablock.montecarlo import simulate  # noqa: E402
from reliablock.quantify import quantify  # noqa: E402
from reliablock.structure import minimal_cut_sets  # noqa: E402

#: Multiplier on a tolerance when injecting a violation. Strictly greater than
#: 2, for the reason in the module docstring. 10 is used so that the control is
#: unambiguous rather than marginal.
OFFSET_FACTOR = 10.0

HOURS = 1000.0


def _block(name: str, rate: float) -> Block:
    return Block(name=name, distribution=Exponential(rate))


def _series_parallel() -> Diagram:
    return Diagram(
        "series parallel",
        Series(
            items=(
                _block("single", 1.0e-4),
                Parallel(items=(_block("a", 3.0e-4), _block("b", 3.0e-4)), label="pair"),
            ),
            label="chain",
        ),
    )


def control_closed_form_tolerance(tee: Tee) -> None:
    tee.rule("control 1: a closed-form agreement check, broken on purpose")
    tee("")
    tee("  The real check: the k-of-n reliability agrees with the binomial")
    tee("  survival to 1e-13 relative. The control perturbs the computed value by")
    tee(f"  {OFFSET_FACTOR:g} times that tolerance, which is strictly more than twice it, so")
    tee("  the check MUST fail.")
    tee("")
    tolerance = 1e-13
    items = tuple(_block(f"u{i}", 3.0e-4) for i in range(4))
    diagram = Diagram("knn", KOfN(k=3, items=items))
    truth = quantify(diagram, HOURS).reliability
    reference = float(
        __import__("scipy.stats", fromlist=["binom"]).binom.sf(2, 4, math.exp(-0.3))
    )
    real_error = abs(truth - reference) / abs(reference)
    injected = truth * (1.0 + OFFSET_FACTOR * tolerance)
    injected_error = abs(injected - reference) / abs(reference)
    tee(f"  true computed reliability          : {truth:.15f}")
    tee(f"  independent reference              : {reference:.15f}")
    tee(f"  real relative error                : {real_error:.6e}")
    tee(f"  declared tolerance                 : {tolerance:.1e}")
    tee(f"  injected offset                    : {OFFSET_FACTOR:g} x tolerance, relative")
    tee(f"  perturbed value                    : {injected:.15f}")
    tee(f"  perturbed relative error           : {injected_error:.6e}")
    tee("")
    tee.check("the real check passes", real_error <= tolerance, f"{real_error:.6e}")
    tee.check(
        "the injected offset DOES break the check, so the control fires",
        injected_error > tolerance,
        f"perturbed error {injected_error:.6e} exceeds tolerance {tolerance:.1e}",
    )
    tee("")
    tee("  for contrast, an offset of exactly 1 x the tolerance:")
    marginal = truth * (1.0 + tolerance)
    marginal_error = abs(marginal - reference) / abs(reference)
    tee(f"    perturbed relative error         : {marginal_error:.6e}")
    tee(f"    would the check fail             : {marginal_error > tolerance}")
    tee("")
    tee("  That is the point of the 2x rule: a 1x offset is on the boundary and")
    tee("  its verdict depends on which side the real error already sat.")


def control_bound_bracket(tee: Tee) -> None:
    tee.rule("control 2: the Bonferroni bracket, broken on purpose")
    tee("")
    tee("  The real check: the series unreliability lies inside the first- and")
    tee("  second-order Bonferroni bracket. The bracket's width here is 4.5e-23")
    tee("  and its slack 1.2e-25, so the offset is anchored on the BRACKET WIDTH")
    tee("  plus the slack, times the factor, and not on a fixed relative amount.")
    tee("")
    q, n = 1.0e-12, 10
    truth = quantify(
        Diagram(
            "chain",
            Series(items=tuple(_block(f"b{i}", -math.log1p(-q)) for i in range(n))),
        ),
        1.0,
    ).unreliability
    bracket = bonferroni_series_bounds([q] * n, truth)
    offset = OFFSET_FACTOR * (bracket.width + bracket.slack)
    injected = bracket.upper + offset
    injected_bracket = bonferroni_series_bounds([q] * n, injected)
    tee(f"  true unreliability                 : {truth:.15e}")
    tee(f"  bracket                            : [{bracket.lower:.15e}, {bracket.upper:.15e}]")
    tee(f"  bracket width                      : {bracket.width:.6e}")
    tee(f"  bracket slack                      : {bracket.slack:.6e}")
    tee(f"  injected offset                    : {offset:.6e}")
    tee(f"  perturbed value                    : {injected:.15e}")
    tee("")
    tee.check("the real check passes", bracket.contains, str(bracket))
    tee.check(
        "the injected offset DOES break the bracket, so the control fires",
        not injected_bracket.contains,
        str(injected_bracket),
    )
    tee("")
    tee("  and the already-measured naive expression, which is a real defect and")
    tee("  not an injected one:")
    naive = 1.0 - (1.0 - q) ** n
    naive_bracket = bonferroni_series_bounds([q] * n, naive)
    tee(f"    naive 1 - prod(1 - q_i)          : {naive:.15e}")
    tee(f"    inside the bracket               : {naive_bracket.contains}")
    tee.check(
        "the bracket also rejects the naive expression, with no injection at all",
        not naive_bracket.contains,
        str(naive_bracket),
    )


def control_cut_sets(tee: Tee) -> None:
    tee.rule("control 3: a wrong cut set, broken on purpose")
    tee("")
    tee("  The real check: every reported minimal cut set fails the system when")
    tee("  all its events fail, and restoring any one of them restores the")
    tee("  system. The control feeds a deliberately wrong cut set -- a proper")
    tee("  SUBSET of a real one -- to the same test and requires it to be")
    tee("  rejected. A subset is the right injection here because a superset")
    tee("  would still fail the system and would pass the first half of the")
    tee("  check.")
    tee("")
    from reliablock.structure import basic_events, structure_function

    diagram = _series_parallel()
    events = set(basic_events(diagram))
    cuts = minimal_cut_sets(diagram)
    pair = next(c for c in cuts if len(c) == 2)
    tee(f"  real minimal cut set               : {{{', '.join(sorted(pair))}}}")
    real_fails = not structure_function(diagram, events - pair)
    tee(f"  system fails when it is down       : {real_fails}")
    subset = frozenset({sorted(pair)[0]})
    subset_fails = not structure_function(diagram, events - subset)
    tee(f"  injected subset                    : {{{', '.join(sorted(subset))}}}")
    tee(f"  system fails when only it is down  : {subset_fails}")
    # Minimality means: restoring ANY single event makes the system SUCCEED.
    # The first version of this control had the predicate inverted -- it asked
    # whether the system still failed -- and the control therefore reported a
    # superset as minimal. The inverted form is easy to write and the control
    # firing is what caught it.
    def is_minimal(candidate: frozenset[str]) -> bool:
        return all(
            structure_function(diagram, (events - candidate) | {e}) for e in sorted(candidate)
        )

    superset = pair | {"single"}
    tee(f"  real cut set is minimal            : {is_minimal(pair)}")
    tee(f"  injected superset                  : {{{', '.join(sorted(superset))}}}")
    tee(f"  superset passes the minimality test: {is_minimal(superset)}")
    superset_minimal = is_minimal(superset)
    tee("")
    tee.check("the real cut set fails the system", real_fails)
    tee.check("the real cut set is minimal", is_minimal(pair))
    tee.check(
        "the injected subset DOES NOT fail the system, so the control fires",
        not subset_fails,
        "a proper subset of a minimal cut set cannot be a cut set",
    )
    tee.check(
        "the injected superset DOES fail the minimality test, so the control fires",
        not superset_minimal,
        "a superset of a minimal cut set is not minimal",
    )


def control_importance(tee: Tee) -> None:
    tee.rule("control 4: a wrong Birnbaum value, broken on purpose")
    tee("")
    tee("  The real check: Birnbaum agrees with a central finite difference to")
    tee("  1e-6 relative. The control perturbs the implementation's value by")
    tee(f"  {OFFSET_FACTOR:g} times that tolerance.")
    tee("")
    tolerance = 1e-6
    diagram = _series_parallel()
    base = {b.name: b.distribution.unreliability(HOURS) for b in diagram.blocks()}
    step = 1e-4
    rows = importance_table(diagram, HOURS)
    worst_real = 0.0
    worst_injected = 0.0
    for row in rows:
        high = dict(base)
        low = dict(base)
        high[row.event] = min(base[row.event] + step, 1.0)
        low[row.event] = max(base[row.event] - step, 0.0)
        span = high[row.event] - low[row.event]
        difference = (
            system_unreliability_at(diagram, high) - system_unreliability_at(diagram, low)
        ) / span
        worst_real = max(worst_real, abs(row.birnbaum - difference) / abs(difference))
        injected = row.birnbaum * (1.0 + OFFSET_FACTOR * tolerance)
        worst_injected = max(worst_injected, abs(injected - difference) / abs(difference))
    tee(f"  worst real relative error          : {worst_real:.6e}")
    tee(f"  declared tolerance                 : {tolerance:.1e}")
    tee(f"  worst perturbed relative error     : {worst_injected:.6e}")
    tee("")
    tee.check("the real check passes", worst_real <= tolerance, f"{worst_real:.6e}")
    tee.check(
        "the injected offset DOES break the check, so the control fires",
        worst_injected > tolerance,
        f"perturbed error {worst_injected:.6e} exceeds tolerance {tolerance:.1e}",
    )


def control_monte_carlo(tee: Tee) -> None:
    tee.rule("control 5: a Monte Carlo interval, broken on purpose")
    tee("")
    tee("  The real check is the 4-standard-error criterion this repository")
    tee("  declares everywhere else, not membership of the 95 percent interval:")
    tee("  a 95 percent interval is expected to miss about 1 run in 20 BY")
    tee("  CONSTRUCTION, so making it the pass criterion of a single seeded run")
    tee("  would be a check that fails 5 percent of the time for no reason.")
    tee("  Interval membership is reported alongside, as information.")
    tee("")
    tee("  The control offsets the closed form by the FULL interval width times")
    tee(f"  {OFFSET_FACTOR:g}, which is more than twice the half-width, so it must fall")
    tee("  outside.")
    tee("")
    diagram = _series_parallel()
    closed = quantify(diagram, HOURS)
    result = simulate(diagram, HOURS, 400_000, seed=20261025)
    offset = OFFSET_FACTOR * (result.upper - result.lower)
    injected = closed.reliability + offset
    tee(f"  closed form R                      : {closed.reliability:.12f}")
    tee(f"  simulated R                        : {result.reliability:.12f}")
    tee(f"  interval                           : [{result.lower:.12f}, {result.upper:.12f}]")
    tee(f"  interval half-width                : {result.half_width:.6e}")
    tee(f"  injected offset                    : {offset:.6e}")
    tee(f"  perturbed closed form              : {injected:.12f}")
    tee("")
    sigma = result.standard_errors_from(closed.reliability)
    tee(f"  distance in standard errors        : {sigma:.4f}")
    tee(f"  inside the 95 percent interval     : {result.brackets(closed.reliability)}")
    tee("")
    tee.check(
        "the real check passes the declared 4-standard-error criterion",
        sigma < 4.0,
        f"{sigma:.4f} standard errors",
    )
    tee.check(
        "the injected offset DOES fall outside the interval, so the control fires",
        not result.brackets(injected),
        f"{result.standard_errors_from(injected):.4f} standard errors from the estimate",
    )


def control_enumeration(tee: Tee) -> None:
    tee.rule("control 6: a wrong structural quantification, broken on purpose")
    tee("")
    tee("  The real check: the structural quantification agrees with exhaustive")
    tee("  state enumeration to 1e-12 relative. The control uses the WRONG")
    tee("  expression for a parallel group -- Q = 1 - prod(1 - Q_i), which is a")
    tee("  real mistake rather than an arbitrary offset -- and requires the")
    tee("  enumeration to reject it.")
    tee("")
    tolerance = 1e-12
    diagram = Diagram(
        "pair", Parallel(items=(_block("a", 3.0e-4), _block("b", 3.0e-4)), label="pair")
    )
    base = {b.name: b.distribution.unreliability(HOURS) for b in diagram.blocks()}
    structural = system_unreliability_at(diagram, base)
    exhaustive = boolean_system_unreliability(diagram, base)
    wrong = 1.0 - (1.0 - base["a"]) * (1.0 - base["b"])
    real_error = abs(structural - exhaustive) / exhaustive
    wrong_error = abs(wrong - exhaustive) / exhaustive
    tee(f"  structural Q                       : {structural:.15e}")
    tee(f"  exhaustive Q                       : {exhaustive:.15e}")
    tee(f"  wrong expression Q                 : {wrong:.15e}")
    tee(f"  real relative error                : {real_error:.6e}")
    tee(f"  wrong expression relative error    : {wrong_error:.6e}")
    tee("")
    tee.check("the real check passes", real_error <= tolerance, f"{real_error:.6e}")
    tee.check(
        "the wrong parallel expression IS rejected by the enumeration, so the control fires",
        wrong_error > tolerance,
        f"{wrong_error:.6e} exceeds {tolerance:.1e}",
    )


def control_bridge(tee: Tee) -> None:
    tee.rule("control 7: a wrong bridge polynomial, broken on purpose")
    tee("")
    tee("  The real check: four routes agree on R = 2p^5 - 5p^4 + 2p^3 + 2p^2.")
    tee("  The control uses the two series-parallel decompositions a reader")
    tee("  might reach for instead, which are real mistakes and not offsets, and")
    tee("  requires both to be rejected at p = 0.9.")
    tee("")
    p = 0.9
    tolerance = 1e-13
    exact = bridge_reliability_equal(p)
    package = quantify(bridge_diagram(p), 1.0).reliability
    two_paths = 1.0 - (1.0 - p * p) ** 2
    merged = (1.0 - (1.0 - p) ** 2) ** 2
    tee(f"  derived closed form at p = 0.9     : {exact:.15f}")
    tee(f"  package pivotal decomposition      : {package:.15f}")
    tee(f"  two-disjoint-paths mistake         : {two_paths:.15f}")
    tee(f"  merged-vertices mistake            : {merged:.15f}")
    tee(f"  two-paths relative error           : {abs(two_paths - exact) / exact:.6e}")
    tee(f"  merged relative error              : {abs(merged - exact) / exact:.6e}")
    tee("")
    tee.check(
        "the real check passes",
        abs(package - exact) / exact <= tolerance,
        f"{abs(package - exact) / exact:.6e}",
    )
    tee.check(
        "the two-disjoint-paths mistake IS rejected, so the control fires",
        abs(two_paths - exact) / exact > tolerance,
    )
    tee.check(
        "the merged-vertices mistake IS rejected, so the control fires",
        abs(merged - exact) / exact > tolerance,
    )


def control_esary_proschan(tee: Tee) -> None:
    tee.rule("control 8: the Esary-Proschan bracket, broken on purpose")
    tee("")
    tee("  The real check: the exact bridge reliability lies inside the")
    tee("  Esary-Proschan bracket. The bracket has non-trivial width here, so")
    tee("  the offset is the bracket width times the factor.")
    tee("")
    diagram = bridge_diagram(0.9)
    truth = quantify(diagram, 1.0).reliability
    bracket = esary_proschan_bounds(diagram, 1.0, truth)
    offset = OFFSET_FACTOR * (bracket.width + bracket.slack)
    injected = esary_proschan_bounds(diagram, 1.0, bracket.upper + offset)
    tee(f"  {bracket}")
    tee(f"  injected offset                    : {offset:.6e}")
    tee(f"  {injected}")
    tee("")
    tee.check("the real check passes", bracket.contains)
    tee.check(
        "the injected offset DOES break the bracket, so the control fires",
        not injected.contains,
    )


def main() -> int:
    tee = Tee("validate_negative_controls_output.txt")
    header(tee, "negative controls, each verified to actually fire")
    tee("")
    tee("  Offsetting a value by its tolerance does NOT guarantee a violation:")
    tee("  the reference may already sit up to one tolerance away on the other")
    tee("  side, so only an offset strictly greater than TWICE the tolerance")
    tee("  guarantees failure. Every control here uses a factor of")
    tee(f"  {OFFSET_FACTOR:g}, anchors on the true value or on the bracket's own width, and")
    tee("  reports whether it fired as its own check.")
    tee("")
    tee("  A control that does not fire is a failure of the VALIDATION, not of")
    tee("  the product, and it is the one kind of failure that would otherwise")
    tee("  be invisible.")

    timer = Timer()
    with timer:
        control_closed_form_tolerance(tee)
        control_bound_bracket(tee)
        control_cut_sets(tee)
        control_importance(tee)
        control_monte_carlo(tee)
        control_enumeration(tee)
        control_bridge(tee)
        control_esary_proschan(tee)

    tee.rule("summary")
    tee("")
    tee("  Controls are counted in two groups: the real checks, which must pass,")
    tee("  and the injected violations, which must fire. Both are ordinary checks")
    tee("  above, so a control that silently stopped firing would appear as a")
    tee("  failed check and not as a quietly weakened test.")

    tee.rule("timing, declared unstable")
    tee("")
    tee("  | quantity | value | stability |")
    tee("  |---|---|---|")
    tee(f"  | whole script wall clock | {timer.elapsed_s:.2f} s | UNSTABLE, shared hardware |")

    tee.summarise_failures()
    tee("")
    tee(f"  written: {tee.save()}")
    return 1 if tee.blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())
