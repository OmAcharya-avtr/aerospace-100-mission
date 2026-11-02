"""Split conformal's finite-sample coverage, against the bracket of equation (13).

What to notice: the measured coverage tracks the lower edge of the bracket,
``1 - alpha``, and never the upper edge ``1 - alpha + 1/(n+1)``. The bracket
narrows as ``1/(n+1)``, so the conservatism of a conformal limit is entirely a
small-sample effect and is bounded in advance -- which is the thing a
bootstrap interval cannot say about itself at any sample size.

The bracket comes from the order-statistic argument of Vovk et al. (2005) and
Lei et al. (2018), Theorem 2.2. The implementation takes the k-th smallest
score by integer index and computes no coverage probability at all, so this is
a check against mathematics it does not use.
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

import math  # noqa: E402

import numpy as np  # noqa: E402

from _common import (  # noqa: E402
    DISTRIBUTION_FREE,
    NEUTRAL,
    REFUSED,
    RESAMPLE,
    bootstrap,
    save,
    style,
)

ROOT = bootstrap()

import matplotlib.pyplot as plt  # noqa: E402

from transfergap.resample import conformal_order_index, conformal_required_n  # noqa: E402

ALPHA = 0.05
REPETITIONS = 20000
SIZES = (19, 24, 29, 39, 49, 59, 79, 99, 149, 199)


def main() -> int:
    style()
    rng = np.random.default_rng(31337)
    measured, lower, upper, errors = [], [], [], []
    for n in SIZES:
        scores = rng.uniform(size=(REPETITIONS, n))
        fresh = rng.uniform(size=REPETITIONS)
        k = conformal_order_index(n, ALPHA)
        limits = np.sort(scores, axis=1)[:, k - 1]
        rate = float(np.mean(fresh <= limits))
        measured.append(rate)
        lower.append(1.0 - ALPHA)
        upper.append(1.0 - ALPHA + 1.0 / (n + 1))
        errors.append(math.sqrt(rate * (1.0 - rate) / REPETITIONS))

    figure, axis = plt.subplots(figsize=(7.8, 4.8))
    axis.fill_between(SIZES, lower, upper, color=DISTRIBUTION_FREE, alpha=0.16,
                      label="bracket of equation (13)")
    axis.plot(SIZES, upper, color=DISTRIBUTION_FREE, lw=1.3, ls="--",
              label=r"$1-\alpha+1/(n+1)$")
    axis.axhline(1.0 - ALPHA, color=NEUTRAL, lw=1.3,
                 label=rf"$1-\alpha$ = {1 - ALPHA:g}")
    axis.errorbar(SIZES, measured, yerr=[4.0 * e for e in errors], fmt="o", ms=4.5,
                  color=RESAMPLE, lw=1.2, capsize=3,
                  label=f"measured, {REPETITIONS} repetitions, +/- 4 se")
    axis.axvline(conformal_required_n(ALPHA), color=REFUSED, ls=":", lw=1.4,
                 label=f"no limit exists below n = {conformal_required_n(ALPHA)}")
    axis.set_xscale("log")
    axis.set_xlabel("calibration scores n (log scale)")
    axis.set_ylabel("coverage of a fresh exchangeable score")
    axis.set_title(
        "split conformal is exactly as conservative as equation (13) says and no more",
        fontsize=10,
    )
    axis.legend(loc="upper right", fontsize=8)
    path = save(figure, "conformal_bracket")
    print(f"written: {path}")
    for n, rate, high, error in zip(SIZES, measured, upper, errors, strict=True):
        print(f"  n={n:<4d} k={conformal_order_index(n, ALPHA):<4d} measured {rate:.6f} "
              f"+/- {error:.6f}  bracket [{1 - ALPHA:.6f}, {high:.6f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
