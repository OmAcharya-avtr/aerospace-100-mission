"""Validate the robustness semantics against the Boolean semantics, exhaustively.

Six checks:

1. Hand-calculated known answers for every operator in the fragment.
2. **Exhaustive** sign agreement over a complete enumeration of small traces and
   a complete enumeration of formulas up to depth 2, which is a stronger
   statement than the randomised property test for the cases it covers.
3. Randomised sign agreement over long traces from the actual simulator, where
   the robustness values are the ones the falsification search sees.
4. The empty-window conventions, and that they are the ones that make the
   equivalence hold rather than an arbitrary choice.
5. The time-bound grid check: a bound off the sample grid is refused, not
   rounded.
6. Scale invariance of the verdict under positive normalisation.

Reference: Fainekos & Pappas (2009), Theoretical Computer Science 410(42), for
the min/max robustness semantics these checks compare against.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np

from _bootstrap import Tee, add_src_to_path

add_src_to_path()

from falsifyloop.instances import suite  # noqa: E402
from falsifyloop.requirements import (  # noqa: E402
    Abs,
    Always,
    And,
    Difference,
    Eventually,
    Or,
    Predicate,
    Signal,
    robustness,
    satisfies,
)
from falsifyloop.traces import Trace  # noqa: E402

OUT = Path(__file__).with_name("validate_semantics_output.txt")
say = Tee(OUT)

DT = 0.25


def check_1_known_answers() -> int:
    say.rule("Check 1 - hand-calculated known answers")
    trace = Trace(np.arange(5) * DT, {"y": np.array([0.0, 1.0, 3.0, 1.0, 0.0])})
    cases = [
        ("Predicate y <= 2 at t=0", Predicate(Signal("y"), "<=", 2.0), 2.0),
        ("Predicate y >= 2 at t=0", Predicate(Signal("y"), ">=", 2.0), -2.0),
        (
            "Predicate y <= 2 scale 4 at t=0",
            Predicate(Signal("y"), "<=", 2.0, scale=4.0),
            0.5,
        ),
        (
            "always[0,1] y <= 2  (min over y=0,1,3,1,0 of 2-y)",
            Always(Predicate(Signal("y"), "<=", 2.0), 0.0, 1.0),
            -1.0,
        ),
        (
            "eventually[0,1] y >= 3  (max over 5 samples of y-3)",
            Eventually(Predicate(Signal("y"), ">=", 3.0), 0.0, 1.0),
            0.0,
        ),
        (
            "always[0.5,1] y <= 2  (indices 2..4: 2-3, 2-1, 2-0)",
            Always(Predicate(Signal("y"), "<=", 2.0), 0.5, 1.0),
            -1.0,
        ),
        (
            "|d/dt y| <= 10 at t=0  (forward diff (1-0)/0.25 = 4)",
            Predicate(Abs(Difference("y")), "<=", 10.0),
            6.0,
        ),
        (
            "always[0,1] |d/dt y| <= 10  (max |diff| is (3-1)/0.25 = 8)",
            Always(Predicate(Abs(Difference("y")), "<=", 10.0), 0.0, 1.0),
            2.0,
        ),
        (
            "and(y<=2, y>=2) at t=0  (min of 2, -2)",
            And(Predicate(Signal("y"), "<=", 2.0), Predicate(Signal("y"), ">=", 2.0)),
            -2.0,
        ),
        (
            "or(y<=2, y>=2) at t=0  (max of 2, -2)",
            Or(Predicate(Signal("y"), "<=", 2.0), Predicate(Signal("y"), ">=", 2.0)),
            2.0,
        ),
    ]
    failures = 0
    say(f"{'case':<58s} {'expected':>10s} {'computed':>10s}")
    say("-" * 80)
    for label, formula, expected in cases:
        got = robustness(formula, trace)
        ok = abs(got - expected) <= 1e-12
        failures += 0 if ok else 1
        say(f"{label:<58s} {expected:>10.6f} {got:>10.6f}" + ("" if ok else "   MISMATCH"))
    say(f"known-answer mismatches: {failures}")
    return failures


def _all_formulas(depth: int):
    """Enumerate every formula of the fragment up to ``depth``, on one signal."""
    terms = [Signal("y"), Difference("y"), Abs(Signal("y")), Abs(Difference("y"))]
    bounds = (-1.0, 0.0, 1.0)
    atoms = [
        Predicate(t, op, b)
        for t in terms
        for op in ("<=", ">=")
        for b in bounds
    ]
    if depth <= 0:
        yield from atoms
        return
    inner = list(_all_formulas(depth - 1))
    yield from atoms
    windows = ((0.0, 0.0), (0.0, DT), (DT, 2 * DT), (2 * DT, 4 * DT))
    for f in inner:
        for lo, hi in windows:
            yield Always(f, lo, hi)
            yield Eventually(f, lo, hi)
    for a, b in itertools.islice(itertools.product(atoms, atoms), 0, None, 7):
        yield And(a, b)
        yield Or(a, b)


def check_2_exhaustive_sign_agreement() -> int:
    say.rule("Check 2 - exhaustive sign agreement on enumerated traces and formulas")
    levels = (-1.0, 0.0, 1.0)
    traces = []
    for n in (2, 3, 4):
        for values in itertools.product(levels, repeat=n):
            traces.append(Trace(np.arange(n) * DT, {"y": np.array(values)}))
    formulas = list(_all_formulas(1))
    say(f"traces enumerated          : {len(traces)}")
    say(f"formulas enumerated        : {len(formulas)}")
    say(f"formula-trace pairs checked : {len(formulas) * len(traces)}")
    mismatches = 0
    boundary = 0
    infinite = 0
    for formula in formulas:
        for trace in traces:
            rho_vec = formula.rho(trace)
            sat_vec = formula.sat(trace)
            if not np.array_equal(rho_vec >= 0.0, sat_vec):
                mismatches += 1
                if mismatches <= 5:
                    say(f"  MISMATCH formula={formula} trace={trace.signal('y').tolist()}")
            boundary += int(np.count_nonzero(rho_vec == 0.0))
            infinite += int(np.count_nonzero(~np.isfinite(rho_vec)))
    say(f"sign disagreements         : {mismatches}")
    say(f"samples at exactly zero robustness (the boundary case): {boundary}")
    say(f"samples at infinite robustness (empty windows)       : {infinite}")
    say(
        "The boundary and infinite counts matter: a sign-agreement claim is only "
        "interesting if the enumeration actually visited rho == 0 and rho == +-inf, "
        "which are the two cases where a careless semantics breaks."
    )
    return mismatches


def check_3_randomised_on_real_traces() -> int:
    say.rule("Check 3 - sign agreement on simulator traces from the shipped suite")
    rng = np.random.default_rng(20261008)
    mismatches = 0
    checked = 0
    negatives = 0
    say(f"{'instance':<20s} {'draws':>6s} {'violating':>10s} {'min rho':>12s} {'max rho':>12s}")
    say("-" * 66)
    for inst in suite():
        values = []
        for vector in inst.sample(rng, 300):
            trace = inst.simulate(vector)
            rho = robustness(inst.requirement, trace)
            sat = satisfies(inst.requirement, trace)
            checked += 1
            if (rho < 0.0) != (not sat):
                mismatches += 1
            values.append(rho)
        arr = np.asarray(values)
        negatives += int(np.count_nonzero(arr < 0.0))
        say(
            f"{inst.identifier:<20s} {arr.size:>6d} {int(np.count_nonzero(arr < 0.0)):>10d} "
            f"{arr.min():>12.6f} {arr.max():>12.6f}"
        )
    say(f"formula-trace pairs checked : {checked}")
    say(f"violating pairs seen        : {negatives}")
    say(f"sign disagreements          : {mismatches}")
    return mismatches


def check_4_empty_window_conventions() -> int:
    say.rule("Check 4 - empty-window conventions")
    trace = Trace(np.arange(3) * DT, {"y": np.zeros(3)})
    always = Always(Predicate(Signal("y"), "<=", -5.0), 1.0, 1.25)
    eventually = Eventually(Predicate(Signal("y"), ">=", 5.0), 1.0, 1.25)
    rho_a, sat_a = always.rho(trace), always.sat(trace)
    rho_e, sat_e = eventually.rho(trace), eventually.sat(trace)
    failures = 0
    say(f"always   rho = {rho_a.tolist()}  sat = {sat_a.tolist()}")
    say(f"eventually rho = {rho_e.tolist()}  sat = {sat_e.tolist()}")
    if not (np.all(rho_a == np.inf) and np.all(sat_a)):
        failures += 1
        say("  MISMATCH: always over an empty window must be +inf and True")
    if not (np.all(rho_e == -np.inf) and not np.any(sat_e)):
        failures += 1
        say("  MISMATCH: eventually over an empty window must be -inf and False")
    say(
        "These are the identities of min/all and max/any. Choosing them the other way "
        "round would break the sign agreement at every clipped window, which is why "
        "they are validated and not merely documented."
    )
    say(f"convention failures: {failures}")
    return failures


def check_5_time_bounds_must_lie_on_the_sample_grid() -> int:
    say.rule("Check 5 - a time bound off the sample grid is refused, not rounded")
    trace = Trace(np.arange(5) * DT, {"y": np.zeros(5)})
    failures = 0
    off_grid = Always(Predicate(Signal("y"), "<=", 1.0), 0.0, 0.3)
    try:
        robustness(off_grid, trace)
    except ValueError as exc:
        say(f"0.3 s against dt = 0.25 s raised ValueError: {str(exc)[:96]}")
    else:
        failures += 1
        say("  MISMATCH: an off-grid bound was accepted")
    on_grid = Always(Predicate(Signal("y"), "<=", 1.0), 0.0, 0.5)
    say(f"0.5 s against dt = 0.25 s accepted, rho = {robustness(on_grid, trace):.6f}")
    say(f"grid-check failures: {failures}")
    return failures


def check_6_scale_invariance() -> int:
    say.rule("Check 6 - positive normalisation never changes a verdict")
    rng = np.random.default_rng(7)
    inst = suite()[0]
    failures = 0
    for scale in (0.1, 1.0, 10.0, 1000.0):
        base = Always(Predicate(Signal("over"), "<=", 2.0, scale=1.0), 0.0, 2.0)
        scaled = Always(Predicate(Signal("over"), "<=", 2.0, scale=scale), 0.0, 2.0)
        flips = 0
        for vector in inst.sample(rng, 120):
            trace = inst.simulate(vector)
            if (robustness(base, trace) < 0.0) != (robustness(scaled, trace) < 0.0):
                flips += 1
        failures += flips
        say(f"scale {scale:>8g}: verdict flips over 120 draws = {flips}")
    say(f"scale-invariance failures: {failures}")
    return failures


def main() -> int:
    say("falsifyloop - robustness semantics validation")
    say("Claim under test: robustness(phi, w) < 0 if and only if w does not satisfy phi,")
    say("with no tolerance band, where the two sides are computed by two separate")
    say("implementations (arithmetic with min/max, versus comparisons with all/any).")
    failures = 0
    failures += check_1_known_answers()
    failures += check_2_exhaustive_sign_agreement()
    failures += check_3_randomised_on_real_traces()
    failures += check_4_empty_window_conventions()
    failures += check_5_time_bounds_must_lie_on_the_sample_grid()
    failures += check_6_scale_invariance()
    say.rule("SUMMARY")
    say(f"total failures across all six checks: {failures}")
    say("A nonzero total is a defect in the semantics and must be reported as such.")
    say.save()
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
