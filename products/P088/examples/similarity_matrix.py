"""The rename score matrix between the removed and added pools of one pair.

What to notice: the true renames sit on a permutation of bright cells well
above the threshold, and the brightest off-permutation cells are the
structurally identical unrelated items. That band is what the threshold has to
clear, and the colour bar shows how much room there is.
"""

from __future__ import annotations

from _common import THRESHOLD, bootstrap, save, style

ROOT = bootstrap()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tmtcdiff.changes import ChangeClass  # noqa: E402
from tmtcdiff.diff import DEFAULT_RENAME_THRESHOLD  # noqa: E402
from tmtcdiff.rename import score_matrix  # noqa: E402
from tmtcdiff.synth import rename_pair  # noqa: E402

SIZE = 80
SEED = 4100


def main() -> None:
    style()
    pair = rename_pair(SIZE, SEED, renames=8, removals=4, additions=4)
    removed = [
        pair.old.parameter_map[name]
        for name in sorted(set(pair.old.parameter_map) - set(pair.new.parameter_map))
    ]
    added = [
        pair.new.parameter_map[name]
        for name in sorted(set(pair.new.parameter_map) - set(pair.old.parameter_map))
    ]
    truth = {c.name: c.new_name for c in pair.truth_for(ChangeClass.RENAMED)}
    matrix = np.array([[cell[0] for cell in row] for row in score_matrix(removed, added)])

    figure, axes = plt.subplots(figsize=(8.6, 6.4))
    image = axes.imshow(matrix, cmap="magma", vmin=0.0, vmax=1.0, aspect="auto")
    axes.set_xticks(range(len(added)))
    axes.set_xticklabels([item.name for item in added], rotation=90, fontsize=7)
    axes.set_yticks(range(len(removed)))
    axes.set_yticklabels([item.name for item in removed], fontsize=7)
    axes.set_xlabel("names present only in the new dictionary")
    axes.set_ylabel("names present only in the old dictionary")
    axes.grid(False)
    for row, item in enumerate(removed):
        target = truth.get(item.name)
        if target in [other.name for other in added]:
            column = [other.name for other in added].index(target)
            axes.add_patch(
                plt.Rectangle(
                    (column - 0.5, row - 0.5),
                    1.0,
                    1.0,
                    fill=False,
                    edgecolor="#3fe08f",
                    linewidth=1.8,
                )
            )
    above = int((matrix >= DEFAULT_RENAME_THRESHOLD).sum())
    axes.set_title(
        f"rename similarity score, {len(removed)} removed by {len(added)} added\n"
        f"green outlines are the {len(truth)} injected renames; "
        f"{above} of {matrix.size} cells are at or above the {DEFAULT_RENAME_THRESHOLD:.2f} "
        "threshold",
        loc="left",
    )
    bar = figure.colorbar(image, ax=axes)
    bar.set_label("combined similarity score")
    bar.ax.axhline(DEFAULT_RENAME_THRESHOLD, color=THRESHOLD, linewidth=1.6)
    print(f"written: {save(figure, 'similarity_matrix')}")


if __name__ == "__main__":
    main()
