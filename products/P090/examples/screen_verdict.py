"""One quality verdict, rendered the way the library returns it.

What to notice: the determinability block is above the verdict block, not
beside it. The element set in the lower panel carries a valid checksum and sits
inside every physical bound, and its measured consequence is 121.68 km over
seven days.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import BENIGN, SILENT, THRESHOLD, UNDETERMINABLE, bootstrap, save, style

bootstrap()

import matplotlib.pyplot as plt  # noqa: E402

from tlecheck import dataset, screen  # noqa: E402
from tlecheck.consequence import measure  # noqa: E402
from tlecheck.elements import format_tle, parse_tle  # noqa: E402
from tlecheck.epoch import epoch_julian_date  # noqa: E402

FIELDS = {
    "satnum": 25544,
    "epoch_year": 2019,
    "epoch_day": 343.69339541,
    "inclination_deg": 51.6439,
    "raan_deg": 211.2001,
    "eccentricity": 0.0007417,
    "arg_perigee_deg": 17.6667,
    "mean_anomaly_deg": 85.6398,
    "mean_motion_rev_per_day": 15.50103472,
    "bstar": 3.8792e-05,
    "mean_motion_dot_rev_per_day2": 1.764e-05,
    "designator": "98067A",
    "element_number": 999,
    "revolution_number": 20248,
}


def main() -> None:
    style()
    truth = parse_tle(*format_tle(**FIELDS), source="catalogue")
    rng = np.random.default_rng(7)
    history = [
        element_set
        for index in range(12)
        if (
            element_set := dataset.rewound(
                truth, -1.5 * (12 - index), source="catalogue", rng=rng,
                element_number=100 + index,
            )
        )
        is not None
    ]
    flipped = parse_tle(
        *format_tle(**{**FIELDS, "bstar": -FIELDS["bstar"]}), source="catalogue"
    )
    use_jd = epoch_julian_date(truth) + 1.0
    verdict = screen(flipped, history=history, reference_jd=use_jd)
    consequence = measure(flipped, truth, use_jd, horizon_days=7.0, samples=49)

    figure = plt.figure(figsize=(11.0, 6.4))
    grid = figure.add_gridspec(2, 1, height_ratios=[1.45, 1.0], hspace=0.30)

    axis = figure.add_subplot(grid[0])
    axis.axis("off")
    axis.set_title(
        "returned first: what this screen cannot determine", loc="left", fontsize=10
    )
    for index, limit in enumerate(verdict.determinability):
        colour = BENIGN if limit.determinable else UNDETERMINABLE
        mark = "determinable" if limit.determinable else "NOT determinable"
        value = "-" if limit.value is None else f"{limit.value:.6g}"
        axis.text(
            0.0,
            1.0 - index * 0.092,
            f"{mark:17s} {limit.basis:15s} {limit.name:44s} {value}",
            family="monospace",
            fontsize=7.3,
            color=colour,
            va="top",
        )

    axis = figure.add_subplot(grid[1])
    axis.axis("off")
    axis.set_title("returned second: the quality verdict", loc="left", fontsize=10)
    lines = [
        flipped.line1,
        flipped.line2,
        "",
        f"integrity intact            : {verdict.integrity_ok}",
        f"predicted consequence class : {verdict.predicted_class}",
        f"equation (15) risk score    : {verdict.silent_risk_score:.6f}",
        f"problems found              : {len(verdict.problems)}",
    ]
    for problem in verdict.problems:
        lines.append(f"  [{problem.screen}] {problem.name} -> {problem.predicted_class}")
    lines += [
        "",
        f"measured consequence, 7 days : {consequence.max_error_km:.4f} km "
        f"({consequence.consequence_class})",
    ]
    for index, line in enumerate(lines):
        colour = "#333333"
        if "silently_wrong" in line:
            colour = SILENT
        if line.startswith("measured consequence"):
            colour = THRESHOLD
        axis.text(
            0.0,
            1.0 - index * 0.088,
            line,
            family="monospace",
            fontsize=7.6,
            color=colour,
            va="top",
        )
    print(save(figure, "screen_verdict"))


if __name__ == "__main__":
    main()
