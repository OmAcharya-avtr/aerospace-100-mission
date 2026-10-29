"""Measured position error by injected corruption family, against the threshold.

What to notice: the checksum_digit family sits at exactly zero. That is the
corruption the loudest screen in the package detects with certainty, and its
propagated consequence is nothing. The three families that straddle the
threshold line are the ones the silent class is made of.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import BENIGN, HISTORY_LENGTH, LOUD, OBJECTS, SILENT, THRESHOLD, bootstrap, save, style

bootstrap()

import matplotlib.pyplot as plt  # noqa: E402

from tlecheck import dataset  # noqa: E402
from tlecheck.consequence import (  # noqa: E402
    SILENTLY_WRONG,
    WILL_FAIL_LOUDLY,
    measure,
)

FLOOR_KM = 1.0e-4


def main() -> None:
    style()
    records = dataset.split("test", objects=OBJECTS, history_length=HISTORY_LENGTH)
    errors: dict[str, list[float]] = {}
    loud: dict[str, int] = {}
    silent: dict[str, int] = {}
    for record in records:
        consequence = measure(
            record.candidate, record.reference, record.use_jd, 1.0, 25, 1.0
        )
        errors.setdefault(record.corruption, [])
        loud.setdefault(record.corruption, 0)
        silent.setdefault(record.corruption, 0)
        if consequence.consequence_class == WILL_FAIL_LOUDLY:
            loud[record.corruption] += 1
        else:
            errors[record.corruption].append(max(consequence.max_error_km, FLOOR_KM))
            if consequence.consequence_class == SILENTLY_WRONG:
                silent[record.corruption] += 1

    families = [name for name in dataset.CORRUPTION_FAMILIES if name in errors]
    figure, axes = plt.subplots(
        1, 2, figsize=(11.0, 4.2), gridspec_kw={"width_ratios": [2.1, 1.0]}
    )

    axis = axes[0]
    for index, family in enumerate(families):
        values = np.array(errors[family]) if errors[family] else np.array([FLOOR_KM])
        jitter = (np.random.default_rng(index).random(values.size) - 0.5) * 0.45
        colours = [SILENT if value > 1.0 else BENIGN for value in values]
        axis.scatter(values, index + jitter, s=11, c=colours, alpha=0.75, linewidths=0)
        axis.plot(
            [float(np.median(values))],
            [index],
            marker="|",
            markersize=18,
            color="#222222",
        )
    axis.axvline(1.0, color=THRESHOLD, linewidth=1.4, linestyle="--")
    axis.text(
        1.15,
        len(families) - 0.4,
        "declared silent\nthreshold, 1 km",
        color=THRESHOLD,
        fontsize=7.5,
        va="top",
    )
    axis.set_xscale("log")
    axis.set_yticks(range(len(families)))
    axis.set_yticklabels(families, fontsize=8)
    axis.set_xlabel("measured maximum position error over one day (km, log scale)")
    axis.set_title(
        "Equation (11), propagated with sgp4 against the uncorrupted element set",
        loc="left",
    )
    axis.set_ylim(-0.7, len(families) - 0.3)
    axis.invert_yaxis()

    axis = axes[1]
    positions = np.arange(len(families))
    benign_counts = [
        len(errors[family]) - silent[family] for family in families
    ]
    silent_counts = [silent[family] for family in families]
    loud_counts = [loud[family] for family in families]
    axis.barh(positions, benign_counts, color=BENIGN, label="benign")
    axis.barh(positions, silent_counts, left=benign_counts, color=SILENT, label="silently wrong")
    axis.barh(
        positions,
        loud_counts,
        left=np.array(benign_counts) + np.array(silent_counts),
        color=LOUD,
        label="will fail loudly",
    )
    axis.set_yticks(positions)
    axis.set_yticklabels([])
    axis.set_xlabel("element sets")
    axis.set_title("measured class", loc="left")
    axis.set_ylim(-0.7, len(families) - 0.3)
    axis.invert_yaxis()
    axis.legend(fontsize=7.5, loc="lower right")

    figure.subplots_adjust(wspace=0.08)
    print(save(figure, "consequence_by_corruption"))


if __name__ == "__main__":
    main()
