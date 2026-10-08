"""Measure each instance's difficulty: the uniform-violation probability.

This is the number everything else is read against. For a violating-set volume
fraction ``p``, uniform random needs ``1/p`` simulations on average and succeeds
within ``n`` with probability ``1 - (1-p)^n``, so ``p`` is simultaneously the
instance's difficulty and the baseline's performance.

Measured by direct Monte Carlo with an exact Clopper-Pearson interval, at a seed
**different from the pilot seed** the bounds were originally chosen with, so
that this is a re-measurement and not a replay. The design targets in
``falsifyloop.instances`` came from the pilot; the figures here are the ones the
README quotes.

Reference: Clopper & Pearson (1934), Biometrika 26(4), 404-413, for the exact
binomial interval.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from _bootstrap import Tee, add_src_to_path

add_src_to_path()

from falsifyloop.curves import clopper_pearson  # noqa: E402
from falsifyloop.instances import suite  # noqa: E402
from falsifyloop.search import analytic_random_curve  # noqa: E402

OUT = Path(__file__).with_name("validate_difficulty_output.txt")
say = Tee(OUT)

DRAWS = 15000
SEED = 52052


def main() -> int:
    say("falsifyloop - instance difficulty measurement")
    say(
        f"{DRAWS} uniform draws per instance from the declared box, seed {SEED}. "
        "The design targets were set from a 40000-draw pilot at seed 13; this run uses "
        "a different seed so the figures are a re-measurement, not a replay."
    )
    say("")
    rows = []
    started = time.perf_counter()
    for inst in suite():
        rng = np.random.default_rng(SEED)
        points = inst.sample(rng, DRAWS)
        values = np.fromiter((inst.evaluate(p) for p in points), dtype=float, count=DRAWS)
        violations = int(np.count_nonzero(values < 0.0))
        rows.append((inst, violations, values))
    elapsed = time.perf_counter() - started

    header = (
        f"{'instance':<20s} {'tier':<10s} {'viol':>6s} {'p measured':>12s} "
        f"{'95% CI low':>12s} {'95% CI high':>12s} {'target':>8s} {'1/p':>9s}"
    )
    say(header)
    say("-" * len(header))
    for inst, violations, _ in rows:
        p = violations / DRAWS
        lo, hi = clopper_pearson(violations, DRAWS)
        inv = f"{1.0 / p:.0f}" if p > 0.0 else ">1e4"
        say(
            f"{inst.identifier:<20s} {inst.tier:<10s} {violations:>6d} {p:>12.6f} "
            f"{lo:>12.6f} {hi:>12.6f} {inst.design_target_probability:>8.4f} {inv:>9s}"
        )
    say("")
    say(
        "p is the fraction of uniform draws that violate. 1/p is the expected number of "
        "uniform simulations to the first violation. Both are properties of the "
        "instance, not of any strategy."
    )

    say.rule("Design target against re-measured value")
    say(
        f"{'instance':<20s} {'target':>9s} {'measured':>10s} "
        f"{'target inside the 95% CI':>26s}"
    )
    say("-" * 68)
    outside = 0
    for inst, violations, _ in rows:
        lo, hi = clopper_pearson(violations, DRAWS)
        inside = lo <= inst.design_target_probability <= hi
        outside += 0 if inside else 1
        say(
            f"{inst.identifier:<20s} {inst.design_target_probability:>9.4f} "
            f"{violations / DRAWS:>10.6f} {('yes' if inside else 'NO'):>26s}"
        )
    say("")
    say(f"design targets outside their re-measured interval: {outside} of {len(rows)}")
    say(
        "A target outside its interval is not a defect: the target came from a "
        "40000-draw pilot and this is a 15000-draw re-measurement, so the two differ by "
        "sampling error plus whatever the pilot's quantile rounding cost. The measured "
        "column is the one the README quotes, and the tier labels are checked against "
        "it below."
    )

    say.rule("Tier labels against the measured probability")
    bands = {
        "easy": (0.1, 1.0),
        "moderate": (0.01, 0.1),
        "hard": (0.002, 0.01),
        "very hard": (0.0, 0.002),
    }
    mislabelled = []
    for inst, violations, _ in rows:
        p = violations / DRAWS
        lo, hi = bands[inst.tier]
        ok = lo <= p < hi or (inst.tier == "easy" and p >= lo)
        if not ok:
            mislabelled.append((inst.identifier, inst.tier, p))
        say(
            f"{inst.identifier:<20s} tier {inst.tier:<10s} band [{lo:g}, {hi:g}) "
            f"measured {p:.6f}  {'ok' if ok else 'OUTSIDE ITS BAND'}"
        )
    say("")
    say(f"instances whose measured p falls outside their declared tier band: {len(mislabelled)}")
    for identifier, tier, p in mislabelled:
        say(f"  {identifier}: labelled {tier}, measured p = {p:.6f}")

    say.rule("Robustness distribution per instance")
    say(
        f"{'instance':<20s} {'min rho':>10s} {'p01':>10s} {'median':>10s} "
        f"{'max rho':>10s} {'frac < 0':>10s}"
    )
    say("-" * 74)
    for inst, violations, values in rows:
        say(
            f"{inst.identifier:<20s} {values.min():>10.5f} "
            f"{np.quantile(values, 0.01):>10.5f} {np.median(values):>10.5f} "
            f"{values.max():>10.5f} {violations / DRAWS:>10.6f}"
        )
    say("")
    say(
        "Robustness is dimensionless on every shipped instance because each predicate "
        "is normalised by its own tolerance, so these columns are comparable across "
        "rows."
    )

    say.rule("Analytic uniform-random curve implied by the measured difficulty")
    marks = (10, 50, 100, 200, 500)
    say(f"{'instance':<20s} " + " ".join(f"{'n=' + str(m):>9s}" for m in marks))
    say("-" * (21 + 10 * len(marks)))
    for inst, violations, _ in rows:
        p = violations / DRAWS
        if p <= 0.0:
            say(f"{inst.identifier:<20s} " + " ".join(f"{'n/a':>9s}" for _ in marks))
            continue
        curve = analytic_random_curve(p, max(marks))
        say(
            f"{inst.identifier:<20s} "
            + " ".join(f"{curve[m - 1]:>9.4f}" for m in marks)
        )
    say("")
    say(
        "1 - (1-p)^n, the exact probability a uniform-random search has found a "
        "violation by simulation n. This is the baseline curve every strategy in "
        "validate_benchmark.py is measured against, and it is a closed form, not an "
        "estimate."
    )

    say.rule("COMPUTE")
    say(f"draws per instance        : {DRAWS}")
    say(f"instances                 : {len(rows)}")
    say(f"total simulations         : {DRAWS * len(rows)}")
    say(f"wall clock                : {elapsed:.1f} s")
    say(f"per simulation            : {elapsed / (DRAWS * len(rows)) * 1e6:.1f} us")
    say(
        "Measured on a container reporting 2 available CPU cores "
        f"(os.sched_getaffinity: {len(__import__('os').sched_getaffinity(0))}). "
        "Single process, no parallelism. A wall clock on a shared container is not a "
        "hardware characteristic and must not be read as one."
    )
    say("")
    say("Falsification is one-sided: finding no violation is not evidence of correctness.")
    say.save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
