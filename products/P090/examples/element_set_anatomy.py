"""The two-line element format by column, with the checksum arithmetic shown.

What to notice: the checksum is a digit sum, so it sees a single changed digit
every time and a transposition of two digits never. The three columns marked in
red are the ones whose corruption this package found to be silent and
consequential: the drag term's sign, the eccentricity and the mean motion.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import SILENT, THRESHOLD, UNDETERMINABLE, bootstrap, save, style

bootstrap()

import matplotlib.pyplot as plt  # noqa: E402

from tlecheck.elements import checksum  # noqa: E402

LINE1 = "1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991"
LINE2 = "2 25544  51.6439 211.2001 0007417  17.6667  85.6398 15.50103472202482"

FIELDS1 = [
    (1, 1, "line"),
    (3, 7, "object"),
    (8, 8, "class"),
    (10, 17, "designator"),
    (19, 32, "epoch"),
    (34, 43, "n-dot/2"),
    (45, 52, "n-ddot/6"),
    (54, 61, "B*"),
    (63, 63, "eph"),
    (65, 68, "set"),
    (69, 69, "cks"),
]
FIELDS2 = [
    (1, 1, "line"),
    (3, 7, "object"),
    (9, 16, "inclination"),
    (18, 25, "node"),
    (27, 33, "eccentricity"),
    (35, 42, "perigee"),
    (44, 51, "anomaly"),
    (53, 63, "mean motion"),
    (64, 68, "rev"),
    (69, 69, "cks"),
]
SILENT_FIELDS = {"B*", "eccentricity", "mean motion"}


def draw(axis, line: str, fields, title: str) -> None:
    axis.set_xlim(0.5, 69.5)
    axis.set_ylim(-2.6, 1.4)
    axis.axis("off")
    axis.set_title(title, loc="left")
    for column, character in enumerate(line, start=1):
        axis.text(
            column,
            0.0,
            character if character != " " else "·",
            ha="center",
            va="center",
            family="monospace",
            fontsize=6.0,
            color="#333333" if character != " " else "#bbbbbb",
        )
    for low, high, name in fields:
        colour = SILENT if name in SILENT_FIELDS else UNDETERMINABLE
        axis.plot([low - 0.4, high + 0.4], [-0.55, -0.55], color=colour, linewidth=1.6)
        axis.text(
            (low + high) / 2.0,
            -1.0,
            name,
            ha="center",
            va="top",
            fontsize=6.2,
            color=colour,
            rotation=0 if high - low > 3 else 90,
        )
    stated = line[68]
    axis.text(
        1.0,
        1.0,
        f"columns 1-68 digit sum mod 10 = {checksum(line)}, column 69 states {stated}",
        ha="left",
        va="center",
        fontsize=7.0,
        color=THRESHOLD,
    )


def main() -> None:
    style()
    figure, axes = plt.subplots(2, 1, figsize=(10.5, 3.6))
    draw(axes[0], LINE1, FIELDS1, "line 1, by column span")
    draw(axes[1], LINE2, FIELDS2, "line 2, by column span")
    figure.subplots_adjust(hspace=0.55)
    figure.suptitle(
        "Two-line element set anatomy. Red fields are the ones whose corruption "
        "measured silent and consequential.",
        fontsize=8.5,
        y=1.02,
    )
    print(save(figure, "element_set_anatomy"))


if __name__ == "__main__":
    main()
