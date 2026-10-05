"""ASM correlation detection: distance histogram and false-sync probability.

Produces ../screenshots/asm_false_sync.png with two panels:
left, the Hamming-distance histogram of the 32-bit marker against random
bit windows, with the Binomial(32, 1/2) prediction over it; right, the
false-sync probability against the correlation threshold T on a log axis,
Eq. (3), with the Monte Carlo points where they are resolvable.
"""

from __future__ import annotations

import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import binom  # noqa: E402

sys.path.insert(0, "../src")

from framesync.asm import (  # noqa: E402
    ASM_32_HEX,
    expected_false_syncs,
    false_sync_probability,
    hamming_distances,
)

SEED = 20261005
N_WINDOWS = 1_000_000


def main() -> int:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    stream = rng.integers(0, 2, N_WINDOWS + 31, dtype=np.uint8)
    dist = hamming_distances(stream)
    counts = np.bincount(dist, minlength=33)[:33]
    n = int(dist.size)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.8))

    k = np.arange(33)
    ax1.bar(k, counts / n, width=0.85, color="#1f3b73", alpha=0.75,
            label=f"measured, {n} windows")
    ax1.plot(k, binom.pmf(k, 32, 0.5), "o-", color="#9b2226", ms=3.5, lw=1.2,
             label="Binomial(32, 1/2)")
    ax1.set_xlabel("Hamming distance D to the marker (bits)")
    ax1.set_ylabel("fraction of windows")
    ax1.set_title(f"marker 0x{ASM_32_HEX:08X} against random bits")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8.5)

    tol = np.arange(0, 17)
    exact = np.array([false_sync_probability(int(t)) for t in tol])
    ax2.semilogy(tol, exact, "-", color="#1f3b73", lw=1.8,
                 label="Eq. (3): $2^{-32}\\sum_{k\\leq T}\\binom{32}{k}$")
    resolvable = [t for t in tol if exact[t] * n > 30]
    meas = [float(np.count_nonzero(dist <= t)) / n for t in resolvable]
    ax2.semilogy(resolvable, meas, "s", color="#9b2226", ms=6, ls="none",
                 label="Monte Carlo, where resolvable")
    ax2.axhline(1.0 / n, color="grey", ls=":", lw=1.0,
                label=f"Monte Carlo floor, 1/{n}")
    ax2.set_xlabel("correlation threshold T (tolerated bit errors)")
    ax2.set_ylabel("false-sync probability per window")
    ax2.set_title("false sync against the detector threshold")
    ax2.set_ylim(1e-11, 2.0)
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=8.5, loc="lower right")

    fig.tight_layout()
    out = "../screenshots/asm_false_sync.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)

    print("ASM false sync (framesync example 2)")
    print("-" * 72)
    print(f"{'T':>3}  {'Eq. (3)':>13}  {'measured':>13}  "
          f"{'expected in 1 Gbit':>19}")
    for t in range(0, 13):
        m = float(np.count_nonzero(dist <= t)) / n
        tag = f"{m:13.6e}" if exact[t] * n > 30 else "below MC floor"
        print(f"{t:3d}  {exact[t]:13.6e}  {tag:>13}  "
              f"{expected_false_syncs(10**9, t):19.6e}")
    print(f"\nsaved {out} ({time.time() - t0:.1f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
