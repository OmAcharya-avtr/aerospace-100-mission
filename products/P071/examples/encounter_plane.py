"""Draw the encounter plane for each shipped case: covariance, hard body, miss.

What to notice: the hard-body circle is tiny against the covariance in the two
low-probability cases and comparable to it in the large-body case, and that
ratio is what decides which analytic method may be used.
"""

from __future__ import annotations

import warnings

import numpy as np

from _common import MISS_SCREEN, REFERENCE, SMALL_BODY, bootstrap, save, style

bootstrap()

from conjprob.cases import load_case  # noqa: E402
from conjprob.dilution import DilutionWarning  # noqa: E402
from conjprob.report import assess_conjunction  # noqa: E402

CASES = (
    "examples/cases/leo_crossing.yaml",
    "examples/cases/large_body_tight_covariance.yaml",
    "examples/cases/far_miss_tight_covariance.yaml",
)


def ellipse(sigma_x: float, sigma_y: float, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Points of the ``scale``-sigma covariance ellipse in the principal frame."""
    theta = np.linspace(0.0, 2.0 * np.pi, 361)
    return scale * sigma_x * np.cos(theta), scale * sigma_y * np.sin(theta)


def main() -> None:
    style()
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(12.0, 4.1))
    for axis, path in zip(axes, CASES, strict=True):
        case = load_case(path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DilutionWarning)
            report = assess_conjunction(case.primary, case.secondary, case.radius_m)
        frame = report.geometry.frame
        for scale, width in ((1.0, 1.4), (2.0, 1.0), (3.0, 0.7)):
            x, y = ellipse(frame.sigma_x_m, frame.sigma_y_m, scale)
            axis.plot(x, y, color=REFERENCE, linewidth=width, alpha=0.8)
        theta = np.linspace(0.0, 2.0 * np.pi, 361)
        axis.fill(
            frame.x_bar_m + case.radius_m * np.cos(theta),
            frame.y_bar_m + case.radius_m * np.sin(theta),
            color=SMALL_BODY,
            alpha=0.85,
        )
        axis.plot(
            [0.0, frame.x_bar_m],
            [0.0, frame.y_bar_m],
            color=MISS_SCREEN,
            linewidth=1.2,
            linestyle="--",
        )
        axis.plot([0.0], [0.0], marker="+", color=REFERENCE, markersize=9)
        axis.set_aspect("equal")
        axis.set_title(
            f"{case.name}\n"
            f"R/sigma_eff = {report.regime.radius_over_sigma:.3f}, "
            f"M = {report.geometry.mahalanobis:.3f}\n"
            f"Pc (Foster) = {report.analytic['foster']:.3e}, "
            f"spread = {report.disagreement.max_relative_spread:.2e}"
        )
        axis.set_xlabel("major principal axis [m]")
        axis.set_ylabel("minor principal axis [m]")
    figure.suptitle(
        "Encounter plane: 1, 2 and 3 sigma covariance contours (black), hard body (red), "
        "miss vector (dashed)",
        y=1.02,
    )
    print(f"wrote {save(figure, 'encounter_plane.png')}")


if __name__ == "__main__":
    main()
