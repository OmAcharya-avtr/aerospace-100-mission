"""Measured validity map: where each analytic method leaves the 1 per cent band.

What to notice: the boundary for Chan's equivalent-area series falls steeply
with covariance aspect ratio and is exact at aspect one, while the
small-hard-body closed form has a boundary that is nearly flat in aspect ratio
and set almost entirely by the hard-body size. These are interpolated from
computed values, not drawn from theory.
"""

from __future__ import annotations

import numpy as np

from _common import CHAN, REFERENCE, SMALL_BODY, bootstrap, save, style

bootstrap()

from conjprob.regimes import regime_sweep  # noqa: E402

RHO = np.logspace(-3.0, 0.8, 70)
ASPECTS = np.logspace(0.0, 1.5, 16)
MAHALANOBIS = np.array([0.5, 1.0, 2.0, 4.0])
TOLERANCE = 1e-2


def main() -> None:
    style()
    import matplotlib.pyplot as plt

    sweep = regime_sweep(RHO, ASPECTS, MAHALANOBIS)
    figure, axes = plt.subplots(1, 2, figsize=(11.4, 4.4), sharey=True)
    for axis, (method, colour) in zip(
        axes, (("chan", CHAN), ("small_body", SMALL_BODY)), strict=True
    ):
        boundary = sweep.boundary(method, TOLERANCE)
        for index, mahalanobis in enumerate(MAHALANOBIS):
            axis.loglog(
                ASPECTS,
                boundary[:, index],
                marker="o",
                markersize=3,
                linewidth=1.3,
                color=colour,
                alpha=0.34 + 0.22 * index,
                label=f"Mahalanobis {mahalanobis:g}",
            )
        axis.set_title(
            f"{method}: measured R / sigma_eff at 1 per cent relative error\n"
            f"worst error anywhere on the grid {sweep.worst_error(method):.3e}"
        )
        axis.set_xlabel("covariance aspect ratio sigma_x / sigma_y")
        axis.legend(loc="lower left")
    axes[0].set_ylabel("R / sqrt(sigma_x sigma_y) at the boundary")
    axes[0].annotate(
        "points absent: tolerance never exceeded",
        xy=(1.0, RHO[-1]),
        xytext=(1.05, RHO[-1] * 0.55),
        color=REFERENCE,
        fontsize=8,
    )
    figure.suptitle(
        "Validity boundaries measured against the independent 2-D quadrature reference",
        y=1.02,
    )
    print(f"wrote {save(figure, 'regime_map.png')}")
    for method in ("chan", "small_body"):
        boundary = sweep.boundary(method, TOLERANCE)
        finite = boundary[np.isfinite(boundary)]
        print(
            f"{method:<11s} boundary range over the grid: "
            f"{finite.min():.4g} to {finite.max():.4g}  "
            f"(nan count {int(np.count_nonzero(~np.isfinite(boundary)))} of {boundary.size})"
        )


if __name__ == "__main__":
    main()
