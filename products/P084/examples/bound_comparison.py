"""Every bound in the package, as a function of the number of paired measurements.

What to notice: the two distribution-free mean bounds cross at n = 82 -- the
older and simpler Hoeffding inequality is the tighter one at every sample size
a flight-test campaign actually delivers -- and the two prediction limits
simply do not exist below n = 59 and n = 19, where the panels are empty and
the refusal is marked rather than filled in with a number.
"""

from __future__ import annotations

# Bytecode must not be written anywhere inside this tree. validation/
# validate_cli.py asserts that no __pycache__ directory and no .pyc file
# exists in the repository, so an interpreter that caches its own imports
# here makes that check report itself. It is set in each entry script,
# before _bootstrap is imported, and not inside _bootstrap, because CPython
# writes a module's cache BEFORE executing that module: _bootstrap cannot
# set this on its own behalf. The imports below therefore follow a statement
# and carry the same E402 waiver the post-bootstrap imports already carry.
import sys  # noqa: E402

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

from _common import (  # noqa: E402
    METHOD_COLOURS,
    METHOD_MARKERS,
    NEUTRAL,
    REFUSED,
    TRUTH,
    bootstrap,
    save,
    style,
)

ROOT = bootstrap()

import matplotlib.pyplot as plt  # noqa: E402

from transfergap import synthetic  # noqa: E402
from transfergap.bounds import (  # noqa: E402
    empirical_bernstein_mean_upper,
    hoeffding_mean_upper,
    normal_tolerance_upper,
    wilks_tolerance_upper,
)
from transfergap.resample import bootstrap_mean_upper, split_conformal_upper  # noqa: E402

CONFIDENCE = 0.95
PROPORTION = 0.95
SIZES = (10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 150, 200, 300)


def main() -> int:
    style()
    truth = synthetic.population_truth(draws=200_000, seed=20260101)
    R = synthetic.BOUND_RANGE_KNM
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))

    mean_methods = {
        "hoeffding": lambda g: hoeffding_mean_upper(g, R, CONFIDENCE).upper,
        "empirical_bernstein": lambda g: empirical_bernstein_mean_upper(
            g, R, CONFIDENCE
        ).upper,
        "bootstrap_percentile": lambda g: bootstrap_mean_upper(
            g, R, CONFIDENCE, resamples=2000, seed=7
        ).upper,
    }
    quantile_methods = {
        "wilks_tolerance": lambda g: wilks_tolerance_upper(
            g, R, PROPORTION, CONFIDENCE
        ).upper,
        "split_conformal": lambda g: split_conformal_upper(g, R, PROPORTION).upper,
        "normal_tolerance": lambda g: normal_tolerance_upper(
            g, R, PROPORTION, CONFIDENCE
        ).upper,
    }

    for axis, methods, reference, reference_label, title in (
        (axes[0], mean_methods, truth.mean, "true E[d]",
         "bounds on the MEAN gap"),
        (axes[1], quantile_methods, truth.quantiles[PROPORTION],
         f"true {PROPORTION:g} quantile",
         f"limits covering a FUTURE gap at p = {PROPORTION:g}"),
    ):
        for name, builder in methods.items():
            xs, ys = [], []
            for n in SIZES:
                gaps = synthetic.generate(n, "random", seed=4242).gaps
                try:
                    ys.append(builder(gaps))
                    xs.append(n)
                except ValueError:
                    continue
            axis.plot(xs, ys, marker=METHOD_MARKERS[name], ms=4.5, lw=1.5,
                      color=METHOD_COLOURS[name], label=name)
            if xs and min(xs) > min(SIZES):
                axis.axvspan(min(SIZES), min(xs), color=REFUSED, alpha=0.08)
        axis.axhline(reference, color=TRUTH, ls="--", lw=1.2, label=reference_label)
        axis.set_xscale("log")
        axis.set_xlabel("paired field measurements n (log scale)")
        axis.set_ylabel(f"upper bound [{synthetic.UNIT}]")
        axis.set_title(title)
        axis.legend(loc="upper right", fontsize=8)

    axes[0].axvline(82, color=NEUTRAL, ls=":", lw=1.2)
    axes[0].text(84, axes[0].get_ylim()[1] * 0.55, "crossover\nn = 82", fontsize=8,
                 color=NEUTRAL)
    axes[1].text(10.5, axes[1].get_ylim()[0] + 1.0,
                 "shaded: no limit\nexists at this n", fontsize=8, color=REFUSED)

    figure.suptitle(
        "transfergap: one synthetic campaign, every bound, at confidence "
        f"{CONFIDENCE:g}",
        fontsize=11,
    )
    figure.subplots_adjust(top=0.86, wspace=0.26)
    path = save(figure, "bound_comparison")
    print(f"written: {path}")
    print(f"true E[d] = {truth.mean:.4f} {synthetic.UNIT}, true {PROPORTION:g} quantile "
          f"= {truth.quantiles[PROPORTION]:.4f}")
    print(f"sizes plotted: {list(SIZES)}")
    print(f"R = {R:.1f} {synthetic.UNIT}; np version {np.__version__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
