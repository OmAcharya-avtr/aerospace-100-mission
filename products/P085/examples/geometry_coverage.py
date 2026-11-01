"""The five declared geometries on the sky, with their verdicts.

What to notice: two_az has only two azimuths and is as well conditioned as the
full sky, while narrow_el samples every azimuth and is not. Conditioning is not
a function of how many pointings there are.
"""

from __future__ import annotations

import numpy as np

from _common import GEOMETRY_COLOURS, bootstrap, save, style

ROOT = bootstrap()

import matplotlib.pyplot as plt  # noqa: E402

from pointmodel import identifiability, synthetic, wls  # noqa: E402

OBSERVATIONS = 300
SIGMA = 2.0
ORDER = ("full_sky", "two_az", "low_el", "narrow_el", "single_az")


def main() -> int:
    style()
    figure, axes = plt.subplots(
        1, len(ORDER), figsize=(15.0, 3.6), subplot_kw={"projection": "polar"}
    )
    verdicts = []
    for axis, name in zip(axes, ORDER, strict=True):
        residuals = synthetic.generate(name, OBSERVATIONS, sigma_arcsec=SIGMA, seed=20260101)
        result = wls.fit(residuals.design, residuals.observed, SIGMA)
        report = identifiability.analyse_fit(residuals.design, result)
        verdicts.append((name, report))
        radius = 90.0 - np.rad2deg(residuals.elevation_rad)
        axis.scatter(
            residuals.azimuth_rad,
            radius,
            s=6,
            color=GEOMETRY_COLOURS[name],
            alpha=0.6,
            edgecolors="none",
        )
        axis.set_theta_zero_location("N")
        axis.set_theta_direction(-1)
        axis.set_rlim(0.0, 90.0)
        axis.set_rticks([30.0, 60.0, 90.0])
        axis.set_yticklabels(["60", "30", "0"], fontsize=7)
        axis.set_xticks(np.radians([0.0, 90.0, 180.0, 270.0]))
        axis.set_xticklabels(["N", "E", "S", "W"], fontsize=8)
        axis.grid(lw=0.4)
        axis.set_title(
            f"{name}\nrank {report.numerical_rank}/9  "
            f"cond {report.condition_number:.2e}\n{report.verdict}",
            fontsize=8,
            pad=14,
        )
    figure.suptitle(
        "the five declared observing geometries, radius is zenith angle\n"
        "two_az has two azimuths and is well conditioned; narrow_el has all of them "
        "and is not",
        fontsize=11,
    )
    figure.subplots_adjust(top=0.70, wspace=0.42)
    path = save(figure, "geometry_coverage")
    print(f"written: {path}")
    print("")
    for name, report in verdicts:
        print(
            f"{name:<10s} rank {report.numerical_rank}/9  "
            f"condition {report.condition_number:12.4e}  "
            f"max VIF {float(np.max(report.vif)):12.4e}  "
            f"refused {', '.join(report.unidentifiable_terms) or 'nothing'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
