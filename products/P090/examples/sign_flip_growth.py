"""One flipped sign, in kilometres, as the prediction horizon grows.

What to notice: the error is 1.6e-12 km at the epoch and 1704 km a month later.
The element set is inside every physical bound at every horizon, carries a valid
checksum, and the propagator reports success at every sample. Nothing about the
element set changes along this curve; only how far it is asked to predict.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import FAMILY_COLOURS, THRESHOLD, bootstrap, save, style

bootstrap()

import matplotlib.pyplot as plt  # noqa: E402

from tlecheck import dataset  # noqa: E402
from tlecheck.consequence import position_errors_km  # noqa: E402
from tlecheck.elements import format_tle, parse_tle  # noqa: E402
from tlecheck.epoch import epoch_julian_date  # noqa: E402
from tlecheck.propagate import propagate, time_grid  # noqa: E402


def main() -> None:
    style()
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))

    base = {
        "satnum": 26000,
        "epoch_year": 2026,
        "epoch_day": 200.5,
        "raan_deg": 120.0,
        "arg_perigee_deg": 40.0,
        "mean_anomaly_deg": 200.0,
    }
    days = np.linspace(0.0, 30.0, 241)
    for family in dataset.FAMILIES:
        fields = {
            **base,
            "inclination_deg": family.inclination_deg,
            "eccentricity": family.eccentricity,
            "mean_motion_rev_per_day": family.mean_motion_rev_per_day,
            "bstar": family.bstar,
            "mean_motion_dot_rev_per_day2": max(0.0, family.bstar) * 0.45,
        }
        reference = parse_tle(*format_tle(**fields))
        flipped = parse_tle(*format_tle(**{**fields, "bstar": -family.bstar}))
        grid = time_grid(epoch_julian_date(reference), 30.0, 241)
        first = propagate(reference, grid)
        second = propagate(flipped, grid)
        if first.loud or second.loud:
            continue
        errors = position_errors_km(second, first)
        axes[0].plot(
            days,
            np.maximum(errors, 1.0e-12),
            color=FAMILY_COLOURS[family.name],
            linewidth=1.4,
            label=family.name,
        )
    axes[0].axhline(1.0, color=THRESHOLD, linewidth=1.3, linestyle="--")
    axes[0].text(0.4, 1.35, "declared silent threshold, 1 km", color=THRESHOLD, fontsize=7.5)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("days after the element set epoch")
    axes[0].set_ylabel("position error against the uncorrupted set (km)")
    axes[0].set_title("A flipped drag-term sign, by orbit family", loc="left")
    axes[0].legend(fontsize=7.0, loc="lower right")

    fields = {
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
    reference = parse_tle(*format_tle(**fields))
    labels = []
    values = []
    for name, overrides in (
        ("drag sign flipped", {"bstar": -fields["bstar"]}),
        ("drag magnitude x10", {"bstar": 10.0 * fields["bstar"]}),
        ("mean motion +1e-6 rev/d", {"mean_motion_rev_per_day": 15.50103572}),
        ("mean motion +1e-5 rev/d", {"mean_motion_rev_per_day": 15.50104472}),
        ("inclination +0.001 deg", {"inclination_deg": 51.6449}),
        ("node +0.01 deg", {"raan_deg": 211.2101}),
        ("eccentricity +1e-5", {"eccentricity": 0.0007517}),
        ("anomaly +0.001 deg", {"mean_anomaly_deg": 85.6408}),
    ):
        candidate = parse_tle(*format_tle(**{**fields, **overrides}))
        grid = time_grid(epoch_julian_date(reference), 7.0, 49)
        errors = position_errors_km(
            propagate(candidate, grid), propagate(reference, grid)
        )
        labels.append(name)
        values.append(float(np.max(errors)))
    order = np.argsort(values)
    axes[1].barh(
        np.arange(len(values)),
        [values[index] for index in order],
        color="#1f4e79",
    )
    axes[1].axvline(1.0, color=THRESHOLD, linewidth=1.3, linestyle="--")
    axes[1].set_yticks(np.arange(len(values)))
    axes[1].set_yticklabels([labels[index] for index in order], fontsize=7.5)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("maximum position error over seven days (km, log scale)")
    axes[1].set_title("What one element change is worth, on one object", loc="left")

    figure.subplots_adjust(wspace=0.35)
    print(save(figure, "sign_flip_growth"))


if __name__ == "__main__":
    main()
