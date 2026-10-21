"""Pc against covariance scale: the dilution region, with its turn marked.

What to notice: the curve is not monotone. Left of the peak a larger covariance
raises Pc; right of it a larger covariance lowers Pc, so a reassuring number can
be produced by a worse state estimate. The shipped case sits on the right-hand
branch, which is why the library raises a DilutionWarning for it.
"""

from __future__ import annotations

import warnings

import numpy as np

from _common import FOSTER, REFERENCE, SHADE, SMALL_BODY, bootstrap, save, style

bootstrap()

from conjprob.cases import load_case  # noqa: E402
from conjprob.dilution import dilution_curve  # noqa: E402
from conjprob.report import assess_conjunction  # noqa: E402

SCALES = np.logspace(-1.5, 2.5, 241)


def main() -> None:
    style()
    import matplotlib.pyplot as plt

    case = load_case("examples/cases/leo_crossing.yaml")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = assess_conjunction(case.primary, case.secondary, case.radius_m)
    frame = report.geometry.frame
    exact, approximate = dilution_curve(
        frame.sigma_x_m, frame.sigma_y_m, frame.x_bar_m, frame.y_bar_m, case.radius_m, SCALES
    )
    dilution = report.dilution

    figure, axis = plt.subplots(figsize=(7.4, 4.6))
    axis.axvspan(
        dilution.scale_at_max_numerical, SCALES[-1], color=SHADE, label="dilution region"
    )
    axis.loglog(SCALES, exact, color=FOSTER, linewidth=1.8, label="Foster-Estes")
    axis.loglog(
        SCALES,
        approximate,
        color=SMALL_BODY,
        linewidth=1.1,
        linestyle="--",
        label="small-hard-body closed form",
    )
    axis.plot(
        [dilution.scale_at_max_numerical],
        [dilution.pc_max_numerical],
        marker="o",
        color=REFERENCE,
        markersize=6,
        label=(
            f"maximum {dilution.pc_max_numerical:.3e} at scale "
            f"{dilution.scale_at_max_numerical:.3f}"
        ),
    )
    axis.plot(
        [1.0],
        [dilution.pc_at_unit_scale],
        marker="s",
        color=FOSTER,
        markersize=6,
        label=(
            f"declared covariance {dilution.pc_at_unit_scale:.3e}, diluted by "
            f"{dilution.dilution_factor:.2f}x"
        ),
    )
    axis.set_xlabel("covariance scale factor applied to both principal sigmas")
    axis.set_ylabel("collision probability")
    axis.set_title(
        f"{case.name}: Pc is not monotone in the covariance\n"
        f"Mahalanobis {dilution.mahalanobis:.4f}, dilution ratio "
        f"{dilution.dilution_ratio:.4f}"
    )
    axis.legend(loc="lower left")
    print(f"wrote {save(figure, 'dilution_curve.png')}")
    print(f"warnings raised by the library: {[w.category.__name__ for w in caught]}")
    print(
        f"dPc/dln(scale) at the declared covariance: "
        f"{dilution.log_scale_sensitivity:+.6e}"
    )


if __name__ == "__main__":
    main()
