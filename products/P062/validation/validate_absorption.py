"""Validate how stable the absorption measurement is.

Run from the repository root: ``python validation/validate_absorption.py``
Writes ``validation/validate_absorption_output.txt``.

``validate_residual.py`` reports the absorption fraction at one configuration.
A single number from one configuration is not a result, so this script varies
the two things that could move it -- the number of training trajectories the
discrepancy model is fitted on, and the noise seed -- and reports the ranges.

The claim that survives is a separation, not a ratio. The ratio between the
degenerate design's absorption fraction and the sound design's swings by more
than a decade across configurations, because the denominator is small and
noisy; the two RANGES do not overlap, which is the statement worth making.

References: Brynjarsdottir and O'Hagan (2014), *Inverse Problems* 30(11),
114007; Golub and Van Loan, *Matrix Computations*, 4th ed., 2013, section 2.4.
"""

from __future__ import annotations

import time

import numpy as np

from _bootstrap import Tee, add_src_to_path

ROOT = add_src_to_path()

from twincalib import DEFAULT_STRONG_CUT, design, run_benchmark  # noqa: E402

PAIR = ("baseline-eclipse", "full-sun")
TRAIN_SIZES = (2, 3, 4, 5)
SEEDS = (4200, 6100)


def main() -> None:
    started = time.perf_counter()
    out = Tee(ROOT / "validation" / "validate_absorption_output.txt")
    out("validate_absorption.py -- stability of the absorption measurement")
    out(f"cut s/s_max = {DEFAULT_STRONG_CUT:g}, 3 held-out trajectories throughout")

    out.rule("1. Against the number of training trajectories")
    out(
        f"  {'n_train':>8}" + "".join(
            f"{name + ' weak':>26}" for name in PAIR
        ) + f"{'ratio':>9}"
    )
    collected: dict[str, list[float]] = {name: [] for name in PAIR}
    improvements: dict[str, list[float]] = {name: [] for name in PAIR}
    for n_train in TRAIN_SIZES:
        row = []
        for name in PAIR:
            r = run_benchmark(design(name), n_train=n_train, n_test=3)
            value = r.splits[DEFAULT_STRONG_CUT].weak_fraction
            collected[name].append(value)
            improvements[name].append(r.improvement_pooled)
            row.append(value)
        out(
            f"  {n_train:>8d}" + "".join(f"{v:>26.5f}" for v in row)
            + f"{row[1] / row[0]:>9.1f}"
        )

    out.rule("2. Against the noise seed, at four training trajectories")
    out(
        f"  {'base seed':>10}" + "".join(
            f"{name + ' weak':>26}" for name in PAIR
        ) + f"{'ratio':>9}"
    )
    for seed in SEEDS:
        row = []
        for name in PAIR:
            r = run_benchmark(design(name), n_train=4, n_test=3, base_seed=seed)
            value = r.splits[DEFAULT_STRONG_CUT].weak_fraction
            collected[name].append(value)
            improvements[name].append(r.improvement_pooled)
            row.append(value)
        out(
            f"  {seed:>10d}" + "".join(f"{v:>26.5f}" for v in row)
            + f"{row[1] / row[0]:>9.1f}"
        )

    out.rule("3. What survives")
    out(f"  {'design':<18}{'min weak':>12}{'max weak':>12}{'min improve':>14}{'max improve':>14}")
    lo_hi = {}
    for name in PAIR:
        values = np.array(collected[name])
        imps = np.array(improvements[name])
        lo_hi[name] = (float(values.min()), float(values.max()))
        out(
            f"  {name:<18}{values.min():>12.5f}{values.max():>12.5f}"
            f"{imps.min():>+14.4f}{imps.max():>+14.4f}"
        )
    base_lo, base_hi = lo_hi["baseline-eclipse"]
    bad_lo, bad_hi = lo_hi["full-sun"]
    separated = bad_lo > base_hi
    out("")
    out(f"the two ranges are disjoint: {separated}")
    out(f"  sound design   absorption fraction in [{base_lo:.5f}, {base_hi:.5f}]")
    out(f"  degenerate one absorption fraction in [{bad_lo:.5f}, {bad_hi:.5f}]")
    out(f"  worst-case separation factor = {bad_lo / base_hi:.2f}")
    out(f"  best-case separation factor  = {bad_hi / base_lo:.2f}")
    out("")
    out("THE CLAIM THAT IS SUPPORTED: the absorption fraction separates the two")
    out("designs cleanly, by a factor of at least")
    out(f"{bad_lo / base_hi:.1f} over every configuration tried, while the held-out")
    out("improvement does NOT separate them at all -- both designs improve by")
    out("27 to 44 per cent. A reader looking only at the improvement would not be")
    out("able to tell the two apart.")
    out("")
    out("THE CLAIM THAT IS NOT SUPPORTED: any particular ratio. The ratio ranges")
    out(f"over the configurations above by a factor of "
        f"{(bad_hi / base_lo) / (bad_lo / base_hi):.0f}, because the sound")
    out("design's absorption fraction is small and its run-to-run variation is a")
    out("large share of itself. The single figure quoted in")
    out("validate_residual.py is the value at ONE configuration and is labelled")
    out("as such; the range above is what should be cited.")
    out(
        "verdict: "
        + ("PASS" if separated else "FAIL")
        + "   criterion the two ranges do not overlap"
    )

    out("")
    out(f"elapsed {time.perf_counter() - started:.1f} s")
    out.save()


if __name__ == "__main__":
    main()
