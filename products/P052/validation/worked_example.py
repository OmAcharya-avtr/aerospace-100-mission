"""The worked example reproduced verbatim in README.md, with its real output.

Kept as a validation script rather than only as README prose so that the output
in the README cannot drift from the code: this script is re-run and its output
re-committed whenever anything changes.
"""

from __future__ import annotations

from pathlib import Path

from _bootstrap import Tee, add_src_to_path

add_src_to_path()

import numpy as np  # noqa: E402

from falsifyloop import (  # noqa: E402
    Abs,
    Always,
    Predicate,
    Signal,
    bootstrap_band,
    efficiency_curve,
    instance,
    robustness,
    satisfies,
    simulate,
    surrogate_guided,
    uniform_random,
)
from falsifyloop.systems import LoopInput  # noqa: E402

OUT = Path(__file__).with_name("worked_example_output.txt")
say = Tee(OUT)


def main() -> int:
    say("falsifyloop - worked example (this is the code in README.md)")
    say("")

    # ---------------------------------------------------------------- state a
    # requirement and evaluate it two independent ways on one simulation
    trace = simulate(LoopInput(5.0, 1.8, 0.30, 2.0, 8.0, 0.9))
    requirement = Always(Predicate(Abs(Signal("error")), "<=", 3.0, scale=3.0), 1.2, 2.0)
    rho = robustness(requirement, trace)
    say(f"requirement        : {requirement}")
    say(f"robustness         : {rho:+.6f}  (dimensionless, normalised by the 3 deg band)")
    say(f"Boolean semantics  : satisfied = {satisfies(requirement, trace)}")
    say(f"sign agreement     : (rho < 0) == (not satisfied) -> "
        f"{(rho < 0.0) == (not satisfies(requirement, trace))}")
    say("")

    # ---------------------------------------------------------------- find a
    # counterexample on a shipped instance with the baseline, then with the
    # surrogate, at the same budget and seed
    inst = instance("attitude-envelope")
    say(f"instance           : {inst.identifier} [{inst.tier}]")
    say(f"requirement        : {inst.requirement}")
    say("")
    for name, run in (("uniform-random", uniform_random), ("surrogate-guided", surrogate_guided)):
        result = run(inst, budget=100, seed=7)
        where = (
            f"simulation {result.first_violation}"
            if result.found
            else "NOT FOUND within the budget"
        )
        say(f"{name:<18s} first violation at {where}")
        say(f"{'':<18s} min robustness {result.best_robustness:+.6f} "
            f"after {result.simulations} simulations")
    say("")

    # ---------------------------------------------------------------- the
    # sample-efficiency curve, with a bootstrap band
    runs = [uniform_random(inst, 100, s).first_violation for s in range(20)]
    curve = efficiency_curve(runs, 100)
    lower, upper = bootstrap_band(runs, 100, n_boot=2000, seed=0)
    say("uniform-random sample-efficiency curve over 20 seeds, budget 100:")
    say(f"{'n':>5s} {'P(found by n)':>14s} {'95% bootstrap band':>24s}")
    say("-" * 45)
    for n in (10, 25, 50, 75, 100):
        i = n - 1
        say(f"{n:>5d} {curve[i]:>14.3f} [{lower[i]:>9.3f}, {upper[i]:>9.3f}]")
    say("")
    say(f"runs that found nothing: {sum(1 for r in runs if r is None)} of {len(runs)}")
    say(
        "A run that found nothing found nothing. Falsification is one-sided: finding "
        "no violation is not evidence of correctness."
    )
    say("")

    # ---------------------------------------------------------------- replay
    # the counterexample and show what it looks like
    found = surrogate_guided(inst, 100, 7)
    if found.found:
        replay = inst.simulate(found.best_vector)
        theta = replay.signal("theta")
        say("counterexample, replayed:")
        for field, value in zip(LoopInput.FIELDS, found.best_vector, strict=True):
            say(f"  {field:<16s} {value:+.6f}")
        say(f"  peak |theta|     {np.max(np.abs(theta)):.4f} deg "
            f"against the 17 deg envelope")
        say(f"  robustness       {inst.evaluate(found.best_vector):+.6f}")
    say.save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
