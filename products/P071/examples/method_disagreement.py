"""Relative error of each analytic method against the quadrature reference.

What to notice: Foster-Estes tracks the independent 2-D quadrature to the
quadrature's own accuracy across the whole sweep, Alfano sits at a fixed floor
set by its Simpson node count, and Chan and the small-hard-body approximation
both diverge as the hard body grows against the covariance. The disagreement is
the output of this product, and it is six orders of magnitude wide across this
one axis.
"""

from __future__ import annotations

import numpy as np

from _common import METHOD_COLOURS, REFERENCE, bootstrap, save, style

bootstrap()

from conjprob.regimes import METHOD_NAMES, regime_sweep  # noqa: E402

RHO = np.logspace(-2.5, 0.7, 60)
ASPECTS = (1.0, 5.0, 20.0)
MAHALANOBIS = 1.0


def main() -> None:
    style()
    import matplotlib.pyplot as plt

    sweep = regime_sweep(RHO, np.array(ASPECTS), np.array([MAHALANOBIS]))
    figure, axes = plt.subplots(1, 3, figsize=(12.6, 4.2), sharey=True)
    for index, (axis, aspect) in enumerate(zip(axes, ASPECTS, strict=True)):
        for name in METHOD_NAMES:
            errors = np.abs(sweep.errors[name][index, 0, :])
            axis.loglog(
                RHO,
                np.maximum(errors, 1e-17),
                color=METHOD_COLOURS[name],
                linewidth=1.4,
                label=name,
            )
        for tolerance, style_ in ((1e-2, "--"), (1e-3, ":")):
            axis.axhline(tolerance, color=REFERENCE, linewidth=0.8, linestyle=style_)
        axis.set_title(f"aspect ratio {aspect:g}, Mahalanobis {MAHALANOBIS:g}")
        axis.set_xlabel("R / sqrt(sigma_x sigma_y)")
    axes[0].set_ylabel("|relative error| against 2-D quadrature")
    axes[0].legend(loc="lower right")
    figure.suptitle(
        "Analytic Pc against the independent quadrature reference; dashed 1 per cent, "
        "dotted 0.1 per cent",
        y=1.02,
    )
    print(f"wrote {save(figure, 'method_disagreement.png')}")
    for index, aspect in enumerate(ASPECTS):
        for name in METHOD_NAMES:
            boundary = sweep.boundary(name, 1e-2)[index, 0]
            print(
                f"aspect {aspect:5g}  {name:<11s} 1 per cent boundary at "
                f"R/sigma_eff = {boundary:.4g}"
            )


if __name__ == "__main__":
    main()
