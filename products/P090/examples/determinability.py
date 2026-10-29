"""The eleven limits, as the library returns them before any screen runs.

What to notice: three limits are structural and no amount of data moves them.
Four are computed exactly from the element set and its history. Four are
data-dependent and change when the caller supplies more. The figure is the
output of one call to ``tlecheck.determinability.report``, not a hand-drawn
summary.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import BENIGN, UNDETERMINABLE, bootstrap, save, style

bootstrap()

import matplotlib.pyplot as plt  # noqa: E402

from tlecheck import dataset  # noqa: E402
from tlecheck.determinability import report  # noqa: E402

BASIS_COLOURS = {
    "structural": UNDETERMINABLE,
    "computed": "#1f4e79",
    "data-dependent": "#d9850b",
}


def main() -> None:
    style()
    records = dataset.split("test", objects=48, history_length=12)
    record = next(r for r in records if r.corruption == "none")

    rich = report(
        record.candidate,
        list(record.history),
        sources=2,
        epoch_requirement_days=7.0,
        silent_threshold_km=1.0,
    )
    poor = report(record.candidate, [], sources=1, epoch_requirement_days=None)

    figure, axes = plt.subplots(
        1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1.0, 1.0]}, sharey=True
    )
    names = [limit.name for limit in rich]
    positions = np.arange(len(names))

    for axis, limits, title in (
        (axes[0], poor, "one element set, no history, one source"),
        (axes[1], rich, f"the same set with {len(record.history)} prior sets and two sources"),
    ):
        for index, limit in enumerate(limits):
            colour = BASIS_COLOURS[limit.basis]
            axis.barh(
                index,
                1.0 if limit.determinable else 0.0,
                color=colour,
                alpha=0.85,
                height=0.55,
            )
            if not limit.determinable:
                axis.barh(index, 1.0, color=colour, alpha=0.18, height=0.55)
            value = (
                "-"
                if limit.value is None
                else ("nan" if math.isnan(limit.value) else f"{limit.value:.6g}")
            )
            axis.text(
                1.03,
                index,
                ("determinable  " if limit.determinable else "NOT determinable  ")
                + f"value={value}",
                va="center",
                fontsize=6.8,
                color="#333333",
            )
        axis.set_yticks(positions)
        axis.set_xlim(0.0, 2.4)
        axis.set_xticks([])
        axis.set_title(title, loc="left", fontsize=9)
        axis.grid(False)
        axis.invert_yaxis()

    axes[0].set_yticklabels(names, fontsize=7.2)
    handles = [
        plt.Line2D([0], [0], color=colour, linewidth=6, label=basis)
        for basis, colour in BASIS_COLOURS.items()
    ]
    handles.append(
        plt.Line2D([0], [0], color=BENIGN, linewidth=6, alpha=0.0, label="")
    )
    axes[1].legend(handles=handles[:3], fontsize=7.2, loc="lower right")
    figure.suptitle(
        "What the screen cannot determine, computed and returned before any screen runs",
        fontsize=9.5,
        y=1.0,
    )
    figure.subplots_adjust(wspace=0.55)
    print(save(figure, "determinability"))


if __name__ == "__main__":
    main()
