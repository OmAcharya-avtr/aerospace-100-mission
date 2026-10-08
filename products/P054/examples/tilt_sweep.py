"""Where importance sampling is worse than no importance sampling.

Writes ../screenshots/tilt_sweep.png.

The left panel sweeps the tilt along the design-point direction; the right
panel sweeps it sideways. The shaded region is where the method is worse than
plain Monte Carlo at the same cost. Two curves are drawn because they
disagree: an over-tilted sampler has a tiny variance around a badly wrong
answer, so the variance reduction factor alone is a trap.

Run from the product directory:

    python examples/tilt_sweep.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from rareverify.benchmark import replicate, summarise, variance_reduction  # noqa: E402
from rareverify.limitstates import LinearGaussianLimitState  # noqa: E402
from rareverify.montecarlo import crude_monte_carlo  # noqa: E402
from rareverify.tilting import (  # noqa: E402
    importance_sampling,
    orthogonal_mean_shift,
    scaled_mean_shift,
)

OUTPUT = Path(__file__).resolve().parent.parent / "screenshots" / "tilt_sweep.png"
BUDGET = 100_000
REPLICATIONS = 30
SCALES = (-0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5)
ORTHOGONAL = (0.1, 0.25, 0.5, 0.75, 1.0, 1.5)


def sweep(state, tilts, crude, reference):
    vrf, mserf, bias = [], [], []
    for tilt in tilts:
        summary = summarise(
            tilt.label,
            replicate(
                lambda rng, t=tilt: importance_sampling(state, t, BUDGET, rng=rng),
                REPLICATIONS,
                seed=3002,
            ),
            reference,
        )
        if summary.empirical_std <= 0.0:
            vrf.append(np.nan)
            mserf.append(np.nan)
            bias.append(summary.relative_bias)
            continue
        reduction = variance_reduction(summary, crude, reference)
        vrf.append(reduction.vrf_measured)
        mserf.append(reduction.mse_reduction_factor)
        bias.append(summary.relative_bias)
    return np.array(vrf), np.array(mserf), np.array(bias)


def main() -> None:
    state = LinearGaussianLimitState(beta=3.719, dimension=2)
    reference = state.analytic_probability()
    crude = summarise(
        "crude",
        replicate(
            lambda rng: crude_monte_carlo(state, BUDGET, rng=rng),
            REPLICATIONS,
            seed=3001,
        ),
        reference,
    )

    scale_vrf, scale_mserf, scale_bias = sweep(
        state, [scaled_mean_shift(state, s) for s in SCALES], crude, reference
    )
    orth_vrf, orth_mserf, orth_bias = sweep(
        state, [orthogonal_mean_shift(state, s) for s in ORTHOGONAL], crude, reference
    )

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    axes[0].axhspan(1e-8, 1.0, color="tab:red", alpha=0.08)
    axes[0].semilogy(SCALES, scale_vrf, marker="o", markersize=4, label="variance (VRF)")
    axes[0].semilogy(
        SCALES, scale_mserf, marker="s", markersize=4, label="mean squared error (MSERF)"
    )
    axes[0].axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    axes[0].axvline(1.0, color="grey", linewidth=0.8, linestyle=":")
    axes[0].set_xlabel("tilt scale (1.0 = the analytic design point)")
    axes[0].set_ylabel("reduction factor vs crude, equal evaluations")
    axes[0].set_title("Tilting along the design-point direction")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, which="both")
    axes[0].text(
        -0.45, 1e-2, "worse than\nplain Monte Carlo", fontsize=8, color="tab:red"
    )
    # Point at the tilt where the two verdicts disagree most, not at the best one.
    divergence = int(np.nanargmax(scale_vrf / scale_mserf))
    axes[0].annotate(
        f"scale {SCALES[divergence]}: variance says {scale_vrf[divergence]:.3g},\n"
        f"MSE says {scale_mserf[divergence]:.3g}, "
        f"bias {scale_bias[divergence]:+.3f}",
        xy=(SCALES[divergence], scale_mserf[divergence]),
        xytext=(-0.45, 1e-6),
        arrowprops={"arrowstyle": "->", "linewidth": 0.8},
        fontsize=8,
    )

    axes[1].axhspan(1e-8, 1.0, color="tab:red", alpha=0.08)
    axes[1].semilogy(ORTHOGONAL, orth_vrf, marker="o", markersize=4, label="variance (VRF)")
    axes[1].semilogy(
        ORTHOGONAL, orth_mserf, marker="s", markersize=4, label="mean squared error (MSERF)"
    )
    axes[1].axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    axes[1].set_xlabel("orthogonal tilt magnitude, in units of |design point|")
    axes[1].set_ylabel("reduction factor vs crude, equal evaluations")
    axes[1].set_title("Tilting sideways: no gain, pure weight variance")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, which="both")

    figure.suptitle(
        "rareverify: a badly chosen tilt is worse than none. "
        f"Linear limit state, p = {reference:.4e}, {REPLICATIONS} replications.",
        fontsize=9,
    )
    figure.tight_layout()
    OUTPUT.parent.mkdir(exist_ok=True)
    figure.savefig(OUTPUT, dpi=140)
    plt.close(figure)


if __name__ == "__main__":
    main()
