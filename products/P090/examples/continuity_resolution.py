"""Equation (27): what the continuity screen can and cannot see, by orbit family.

What to notice: four of the six orbit families have a continuity resolution
above the declared one-kilometre threshold, so for those objects comparing
element sets cannot see a one-kilometre consequence at all and the screen
withholds its absolute test. The decaying low-Earth family is two orders of
magnitude worse than the sun-synchronous one, and the reason is physical rather
than numerical.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import FAMILY_COLOURS, HISTORY_LENGTH, OBJECTS, THRESHOLD, bootstrap, save, style

bootstrap()

import matplotlib.pyplot as plt  # noqa: E402

from tlecheck import benchmark, dataset  # noqa: E402


def main() -> None:
    style()
    records = dataset.split("test", objects=OBJECTS, history_length=HISTORY_LENGTH)
    prepared = benchmark.prepare(records, silent_threshold_km=1.0)
    offsets: dict[str, list[float]] = defaultdict(list)
    resolutions: dict[str, list[float]] = defaultdict(list)
    for record, evidence in zip(records, prepared.evidences, strict=True):
        if record.corruption != "none":
            continue
        continuity = evidence.continuity
        if continuity is None or not continuity.usable:
            continue
        offsets[record.family].append(continuity.offset_km)
        resolutions[record.family].append(continuity.resolution_km)

    families = sorted(resolutions, key=lambda name: float(np.median(resolutions[name])))
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))

    axis = axes[0]
    for index, family in enumerate(families):
        values = np.array(offsets[family])
        jitter = (np.random.default_rng(index).random(values.size) - 0.5) * 0.4
        axis.scatter(
            values,
            index + jitter,
            s=12,
            color=FAMILY_COLOURS[family],
            alpha=0.7,
            linewidths=0,
        )
    axis.axvline(1.0, color=THRESHOLD, linewidth=1.3, linestyle="--")
    axis.set_xscale("log")
    axis.set_yticks(range(len(families)))
    axis.set_yticklabels(families, fontsize=7.6)
    axis.set_xlabel("equation (24) offset against the previous element set (km)")
    axis.set_title("Uncorrupted element sets only", loc="left")
    axis.invert_yaxis()

    axis = axes[1]
    medians = [float(np.median(resolutions[family])) for family in families]
    colours = [FAMILY_COLOURS[family] for family in families]
    axis.barh(range(len(families)), medians, color=colours)
    axis.axvline(1.0, color=THRESHOLD, linewidth=1.3, linestyle="--")
    axis.text(
        1.1,
        len(families) - 0.5,
        "declared silent\nthreshold, 1 km",
        color=THRESHOLD,
        fontsize=7.5,
        va="top",
    )
    for index, value in enumerate(medians):
        axis.text(value * 1.1, index, f"{value:.3f}", va="center", fontsize=7.0)
    axis.set_xscale("log")
    axis.set_yticks(range(len(families)))
    axis.set_yticklabels([])
    axis.set_xlabel("equation (27) median resolution (km)")
    axis.set_title("What the screen can resolve", loc="left")
    axis.invert_yaxis()

    figure.subplots_adjust(wspace=0.08)
    print(save(figure, "continuity_resolution"))


if __name__ == "__main__":
    main()
