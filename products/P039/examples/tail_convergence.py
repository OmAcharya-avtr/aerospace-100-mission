#!/usr/bin/env python3
"""Tail-latency estimates converge at the rate order statistics predict.

How many passes do you need before a p99.9 estimate stops moving? The answer
is set by the asymptotic normality of sample quantiles: the error decays as
n^(-1/2) with a coefficient sqrt(p(1-p))/f(q_p). This example measures that
decay on a lognormal whose exact quantile is known, and plots it against the
predicted line on log-log axes, where the prediction is a straight line of
slope -1/2.

What to look at: the measured points lie on the predicted line, and the
p = 0.999 series peels off it at small n -- that is the validity condition
n(1-p) >= 5 making itself felt, not a defect.

Writes ../screenshots/tail_convergence.png. Runtime about 15 s.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from latencynet.tails import (
    lognormal_quantile,
    lognormal_tail_convergence,
    quantile_min_samples,
)

MEAN_S = 2.0e-4
CV = 0.45
N_GRID = (200, 500, 1500, 5000, 15_000, 50_000)
N_REPEATS = 250
SEED = 20260405
OUT = Path(__file__).resolve().parents[1] / "screenshots" / "tail_convergence.png"


def main() -> None:
    sigma_sq = math.log1p(CV**2)
    sigma = math.sqrt(sigma_sq)
    mu = math.log(MEAN_S) - 0.5 * sigma_sq
    print(f"lognormal mu={mu:.9f} sigma={sigma:.9f} (mean {MEAN_S * 1e6:.1f} us, cv {CV})")
    print(f"repeats per n: {N_REPEATS}; log-RMSE sd 1/sqrt(2R) = "
          f"{1.0 / math.sqrt(2 * N_REPEATS):.4f}")

    fig, (ax_abs, ax_ratio) = plt.subplots(1, 2, figsize=(12.5, 5.0))
    colours = {0.99: "tab:orange", 0.999: "tab:red"}
    for p in (0.99, 0.999):
        result = lognormal_tail_convergence(mu, sigma, p, N_GRID, N_REPEATS, SEED)
        truth = lognormal_quantile(mu, sigma, p)
        rmse_us = np.array(result.rmse_s) * 1e6
        pred_us = np.array(result.analytic_se_s) * 1e6
        n_min = quantile_min_samples(p)
        print(f"\np = {p}   exact quantile {truth * 1e6:.4f} us   "
              f"smallest interior n {n_min}   fitted slope {result.fitted_slope:+.5f}")
        for n, r, a, ratio in zip(
            N_GRID, rmse_us, pred_us, result.ratio_measured_over_analytic, strict=True
        ):
            flag = "" if n * (1 - p) >= 5 else "  (below n(1-p) >= 5)"
            print(f"  n={n:>7}  RMSE {r:9.4f} us  predicted {a:9.4f} us  "
                  f"ratio {ratio:.4f}{flag}")
        ax_abs.loglog(N_GRID, rmse_us, "o", color=colours[p], ms=6,
                      label=f"measured RMSE, p{p * 100:g}")
        ax_abs.loglog(N_GRID, pred_us, "-", color=colours[p], lw=1.3,
                      label=f"sqrt(p(1-p)/n)/f(q_p), p{p * 100:g}")
        ax_ratio.semilogx(N_GRID, result.ratio_measured_over_analytic, "o-",
                          color=colours[p], ms=5, lw=1.2,
                          label=f"p{p * 100:g}, fitted slope {result.fitted_slope:+.4f}")
        ax_ratio.axvline(5.0 / (1 - p), color=colours[p], ls=":", lw=1.0,
                         label=f"n(1-p) = 5 for p{p * 100:g}")

    ax_abs.set_xlabel("passes n")
    ax_abs.set_ylabel("RMSE of the quantile estimate (us)")
    ax_abs.set_title("predicted slope -1/2; points are measured")
    ax_abs.legend(fontsize=7.5)
    ax_abs.grid(alpha=0.3, which="both")

    ax_ratio.axhline(1.0, color="black", lw=1.0, ls="--", label="agreement")
    ax_ratio.set_xlabel("passes n")
    ax_ratio.set_ylabel("measured RMSE / analytic standard error")
    ax_ratio.set_title("magnitude, not just rate")
    ax_ratio.set_ylim(0.5, 1.3)
    ax_ratio.legend(fontsize=7.5, loc="lower right")
    ax_ratio.grid(alpha=0.3)

    fig.suptitle(
        "latencynet: tail-quantile convergence against order-statistic theory", fontsize=11
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)
    plt.close(fig)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
