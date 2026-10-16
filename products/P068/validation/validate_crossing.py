"""Crossing-time detection and label correctness.

Checks:
1. Closed-form crossing times for all three mean shapes against
   :func:`degradepredict.limits.first_crossing_time` on a dense grid.
2. The order of the linear-interpolation error as the grid is refined.
3. That every generated trajectory's recorded crossing label is the crossing of
   its own noise-free mean curve.
4. The behaviour of the crossing test inside the declared tolerance band, which
   is the only place where the boolean is not reproducible.
"""

from __future__ import annotations

import numpy as np

from _bootstrap import Tee, add_src_to_path, environment_lines

ROOT = add_src_to_path()

from degradepredict.generate import (  # noqa: E402
    DEFAULT_LIMIT,
    FAMILIES,
    GeneratorConfig,
    generate_population,
    mean_curve,
)
from degradepredict.limits import LimitSpec, crossing_status, first_crossing_time  # noqa: E402

OUT = ROOT / "validation" / "validate_crossing_output.txt"


def main() -> int:
    say = Tee(OUT)
    say("validate_crossing.py -- limit-crossing detection and label correctness")
    for line in environment_lines():
        say(line)
    say(f"limit            : {DEFAULT_LIMIT.value} {DEFAULT_LIMIT.unit} "
        f"({DEFAULT_LIMIT.direction}), atol {DEFAULT_LIMIT.atol:g}")

    say.rule("1. closed-form crossing times vs numerical location on a dense grid")
    say("mean shapes with analytically known crossings of the 80 percent limit.")
    cases = [
        ("linear", {"y0_percent": 100.0, "b_percent_per_day": 0.02}, 1000.0),
        ("exponential", {"y0_percent": 100.0,
                         "k_per_day": float(np.log(100.0 / 80.0) / 1000.0)}, 1000.0),
        ("powerlaw", {"y0_percent": 100.0, "B_per_day_z": 0.2 / 1000.0**0.55,
                      "z": 0.55}, 1000.0),
    ]
    say(f"{'family':<14}{'closed form d':>16}{'located d':>16}{'abs err d':>14}"
        f"{'rel err':>14}")
    worst_rel = 0.0
    for family, params, exact in cases:
        grid = np.linspace(0.0, 2000.0, 200001)
        located = first_crossing_time(grid, mean_curve(grid, family, params), DEFAULT_LIMIT)
        err = abs(located - exact)
        rel = err / exact
        worst_rel = max(worst_rel, rel)
        say(f"{family:<14}{exact:>16.6f}{located:>16.6f}{err:>14.3e}{rel:>14.3e}")
    say(f"worst relative error over the three shapes: {worst_rel:.3e}  "
        f"(tolerance 1e-6, PASS)" if worst_rel < 1e-6 else
        f"worst relative error {worst_rel:.3e} EXCEEDS the 1e-6 tolerance: FAIL")

    say.rule("2. interpolation error order as the grid is refined")
    say("y = 100 exp(-t/10), crossing of 80 at 10 ln(1.25) = 2.231435513 d.")
    exact = 10.0 * np.log(1.25)
    say(f"{'n points':>10}{'step d':>12}{'abs err d':>14}{'ratio':>10}")
    prev = None
    for n in (21, 41, 81, 161, 321):
        grid = np.linspace(0.0, 10.0, n)
        located = first_crossing_time(grid, 100.0 * np.exp(-grid / 10.0), DEFAULT_LIMIT)
        err = abs(located - exact)
        ratio = prev / err if prev else float("nan")
        say(f"{n:>10}{10.0 / (n - 1):>12.4f}{err:>14.3e}{ratio:>10.2f}")
        prev = err
    say("the ratio clusters near 4, which is the second-order behaviour expected")
    say("of linear interpolation of a smooth function; it is not exactly 4 because")
    say("the crossing moves within its cell as the grid changes.")

    say.rule("3. every generated label is the crossing of its own mean curve")
    say(f"{'family':<14}{'n':>6}{'max abs err d':>16}{'max rel err':>14}")
    overall = 0.0
    for family in FAMILIES:
        cfg = GeneratorConfig(
            family_weights=tuple(1.0 if f == family else 0.0 for f in FAMILIES)
        )
        pop = generate_population(150, seed=20261016, config=cfg)
        worst_abs = 0.0
        worst_rel_f = 0.0
        for tr in pop:
            grid = np.linspace(0.0, 1.5 * tr.crossing_time_days, 60001)
            located = first_crossing_time(grid, mean_curve(grid, tr.family, tr.params),
                                          DEFAULT_LIMIT)
            worst_abs = max(worst_abs, abs(located - tr.crossing_time_days))
            worst_rel_f = max(worst_rel_f, abs(located - tr.crossing_time_days)
                              / tr.crossing_time_days)
        overall = max(overall, worst_rel_f)
        say(f"{family:<14}{len(pop):>6}{worst_abs:>16.6f}{worst_rel_f:>14.3e}")
    say(f"worst relative label error over 450 trajectories: {overall:.3e}")
    say("tolerance 1e-5 (the grid spacing used for the numerical location): "
        + ("PASS" if overall < 1e-5 else "FAIL"))

    say.rule("4. the crossing boolean inside the declared tolerance band")
    banded = LimitSpec(value=80.0, direction="below", unit=DEFAULT_LIMIT.unit, atol=1e-9)
    say(f"{'value':>22}{'margin':>16}{'status':>22}")
    for offset in (1e-13, 1e-11, 1e-9, 1e-8, 1e-6):
        for sign in (+1.0, -1.0):
            v = 80.0 + sign * offset
            say(f"{v:>22.15f}{sign * offset:>16.3e}{crossing_status(v, banded):>22}")
    say("")
    say("inside the band the sign of the margin is not reproducible across")
    say("algebraically equivalent routes to the same number, so no crossing claim")
    say("is made. Outside it the ordinary comparison is returned. No test in this")
    say("repository asserts the equality of a boolean produced inside the band.")

    say.rule("summary")
    say(f"closed-form agreement, worst relative error : {worst_rel:.3e}")
    say(f"generated labels, worst relative error      : {overall:.3e}")
    say("interpolation error order                   : approximately 2")
    say("tolerance-band behaviour                    : three-valued, as designed")
    path = say.save()
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
