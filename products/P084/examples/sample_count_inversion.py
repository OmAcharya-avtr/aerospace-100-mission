"""How many paired measurements a declared target slack needs, by method.

What to notice: the Hoeffding curve is the only one that is a guarantee, and
it is the steepest -- the ``1/eps^2`` of equation (10) means halving the
target quadruples the flights. The two prediction limits are flat lines,
because their requirement is for the statement to exist at all and has nothing
to do with a width target.
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
    DISTRIBUTION_FREE,
    NEUTRAL,
    RESAMPLE,
    bootstrap,
    save,
    style,
)

ROOT = bootstrap()

import matplotlib.pyplot as plt  # noqa: E402

from transfergap import synthetic  # noqa: E402
from transfergap.bounds import (  # noqa: E402
    required_n_empirical_bernstein,
    required_n_hoeffding,
    required_n_wilks,
)
from transfergap.resample import conformal_required_n  # noqa: E402

CONFIDENCE = 0.95
PROPORTION = 0.95
R = synthetic.BOUND_RANGE_KNM


def main() -> int:
    style()
    variance = float(np.var(synthetic.generate(200_000, "random", seed=4242).gaps, ddof=1))
    targets = np.linspace(1.0, 12.0, 60)
    hoeffding = [required_n_hoeffding(R, float(t), CONFIDENCE) for t in targets]
    bernstein = [
        required_n_empirical_bernstein(variance, R, float(t), CONFIDENCE) for t in targets
    ]
    wilks = required_n_wilks(PROPORTION, CONFIDENCE)
    conformal = conformal_required_n(1.0 - PROPORTION)

    figure, axis = plt.subplots(figsize=(7.8, 4.8))
    axis.plot(targets, hoeffding, color=DISTRIBUTION_FREE, lw=1.8,
              label="Hoeffding, equation (10) -- exact, a guarantee")
    axis.plot(targets, bernstein, color="#2f7fb4", lw=1.8, ls="-.",
              label="empirical Bernstein, equation (11) -- plug-in, a prediction")
    axis.axhline(wilks, color=DISTRIBUTION_FREE, ls="--", lw=1.4,
                 label=f"Wilks' limit exists at n = {wilks}, equation (7)")
    axis.axhline(conformal, color=RESAMPLE, ls=":", lw=1.6,
                 label=f"split conformal exists at n = {conformal}, equation (12)")
    for n in (15, 25, 40, 60):
        axis.axhline(n, color=NEUTRAL, lw=0.5, alpha=0.5)
        axis.text(11.8, n * 1.04, f"n = {n}", fontsize=7, color=NEUTRAL, ha="right")
    axis.set_yscale("log")
    axis.set_xlabel(f"target slack on the mean gap [{synthetic.UNIT}]")
    axis.set_ylabel("paired field measurements required (log scale)")
    axis.set_title(
        f"inverting the bound: R = {R:.0f} {synthetic.UNIT}, confidence {CONFIDENCE:g}, "
        f"plug-in variance {variance:.2f} {synthetic.UNIT}^2",
        fontsize=10,
    )
    axis.legend(loc="upper right", fontsize=8)
    path = save(figure, "sample_count_inversion")
    print(f"written: {path}")
    print(f"plug-in variance {variance:.6f} {synthetic.UNIT}^2")
    for target in (2.0, 3.0, 5.0, 8.0):
        print(f"  target {target:.1f} {synthetic.UNIT}: Hoeffding "
              f"{required_n_hoeffding(R, target, CONFIDENCE)}, empirical Bernstein "
              f"{required_n_empirical_bernstein(variance, R, target, CONFIDENCE)}")
    print(f"  Wilks exists at n = {wilks}; split conformal at n = {conformal}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
