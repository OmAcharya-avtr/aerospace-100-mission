"""The negative control: a reordered document, and what each kind of diff says.

What to notice: the text diff reports every line of the document as changed
and the semantic diff reports nothing, on a pair that carries no semantic
change. The right-hand panel is the control that stops this being a vacuous
result: one real silent change is injected into the same reordered document
and is still reported.
"""

from __future__ import annotations

import difflib

from _common import GREEDY, SILENT, bootstrap, save, style

ROOT = bootstrap()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tmtcdiff.diff import DiffConfig, diff_dictionaries  # noqa: E402
from tmtcdiff.parse import loads  # noqa: E402
from tmtcdiff.synth import (  # noqa: E402
    INJECTABLE_SILENT,
    baseline_dictionary,
    inject_changes,
    reordering_control,
)

SIZES = (20, 60, 150, 300)
SEED = 7400


def main() -> None:
    style()
    config = DiffConfig()
    text_lines = []
    semantic = []
    for size in SIZES:
        dictionary = baseline_dictionary(size, SEED)
        canonical, reordered = reordering_control(dictionary, SEED + 1)
        differing = sum(
            1
            for line in difflib.unified_diff(
                canonical.splitlines(), reordered.splitlines(), n=0, lineterm=""
            )
            if line[:1] in "+-" and not line.startswith(("---", "+++"))
        )
        result = diff_dictionaries(dictionary, loads(reordered), config)
        text_lines.append(differing)
        semantic.append(len(result.changes))

    detected = []
    for change_class in INJECTABLE_SILENT:
        old = baseline_dictionary(60, SEED)
        pair = inject_changes(old, (change_class,), SEED + 1)
        _, reordered = reordering_control(pair.new, SEED + 2)
        result = diff_dictionaries(pair.old, loads(reordered), config)
        keys = {(c.change_class, c.name) for c in result.changes}
        detected.append(1 if pair.truth_keys <= keys else 0)

    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.3))
    positions = np.arange(len(SIZES))
    axes[0].bar(positions - 0.18, text_lines, width=0.34, color=GREEDY, label="difflib text lines")
    axes[0].bar(
        positions + 0.18,
        np.maximum(semantic, 0.0),
        width=0.34,
        color=SILENT,
        label="tmtcdiff semantic changes",
    )
    axes[0].set_yscale("symlog", linthresh=1.0)
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels([str(size) for size in SIZES])
    axes[0].set_xlabel("parameters in the dictionary")
    axes[0].set_ylabel("count, symmetric log scale")
    axes[0].set_title("reordered and respelled, nothing changed in meaning", loc="left")
    axes[0].legend(fontsize=8)
    for position, value in zip(positions, semantic, strict=True):
        axes[0].text(position + 0.18, 0.6, str(value), ha="center", fontsize=8, color=SILENT)

    class_positions = np.arange(len(INJECTABLE_SILENT))
    axes[1].barh(class_positions, detected, height=0.6, color=SILENT)
    axes[1].set_yticks(class_positions)
    axes[1].set_yticklabels([c.value for c in INJECTABLE_SILENT], fontsize=8)
    axes[1].set_xlim(0.0, 1.15)
    axes[1].set_xticks([0.0, 1.0])
    axes[1].set_xticklabels(["missed", "detected"])
    axes[1].invert_yaxis()
    axes[1].set_title(
        f"reordering does not mask a real change: {sum(detected)}/{len(detected)} detected",
        loc="left",
    )
    figure.suptitle(
        "tmtcdiff negative control, and the control on the control",
        x=0.01,
        ha="left",
    )
    figure.subplots_adjust(top=0.86, wspace=0.3)
    print(f"written: {save(figure, 'negative_control')}")


if __name__ == "__main__":
    main()
