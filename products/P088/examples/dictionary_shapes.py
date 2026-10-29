"""The shape of the generated dictionaries every number in this repository uses.

What to notice: the generator is small and declared. Ten quantity templates,
six subsystems, five enumerations, byte-aligned offsets that wrap inside each
packet. The offsets wrapping is deliberate -- it is what lets two parameters
share a structural signature, and without that the rename false-match rate
would measure zero at every threshold and the sweep would mean nothing.
"""

from __future__ import annotations

from _common import ADVISORY, GREEDY, SILENT, bootstrap, save, style

ROOT = bootstrap()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tmtcdiff.rename import signature_collision_bound  # noqa: E402
from tmtcdiff.synth import baseline_dictionary  # noqa: E402

SIZES = (20, 50, 100, 200, 400)
SEED = 4100


def main() -> None:
    style()
    dictionary = baseline_dictionary(200, SEED)
    widths = [parameter.bit_width for parameter in dictionary.parameters]
    types: dict[str, int] = {}
    for parameter in dictionary.parameters:
        types[parameter.data_type] = types.get(parameter.data_type, 0) + 1
    packets: dict[str, int] = {}
    for parameter in dictionary.parameters:
        packets[parameter.packet] = packets.get(parameter.packet, 0) + 1

    collisions = []
    distinct = []
    for size in SIZES:
        colliding, unique, _ = signature_collision_bound(
            list(baseline_dictionary(size, SEED).parameters)
        )
        collisions.append(colliding / size)
        distinct.append(unique / size)

    figure, axes = plt.subplots(1, 3, figsize=(12.5, 3.9))
    axes[0].hist(widths, bins=np.arange(0, 70, 4), color=GREEDY)
    axes[0].set_xlabel("bit width")
    axes[0].set_ylabel("parameters")
    axes[0].set_title("field widths, 200-parameter dictionary", loc="left")

    names = sorted(types)
    axes[1].bar(names, [types[name] for name in names], color=ADVISORY)
    axes[1].set_ylabel("parameters")
    axes[1].set_title("data types", loc="left")
    axes[1].tick_params(axis="x", rotation=45)

    axes[2].plot(SIZES, collisions, marker="o", color=SILENT,
                 label="items sharing an exact signature")
    axes[2].plot(SIZES, distinct, marker="s", color=GREEDY, label="distinct signatures")
    axes[2].set_xscale("log")
    axes[2].set_xlabel("parameters in the dictionary")
    axes[2].set_ylabel("fraction of parameters")
    axes[2].set_ylim(-0.02, 1.05)
    axes[2].set_title("structural signature collisions", loc="left")
    axes[2].legend(fontsize=8, loc="center left")
    figure.suptitle(
        f"tmtcdiff synthetic dictionary generator, seed {SEED}; "
        f"{len(packets)} packets, no flight heritage",
        x=0.01,
        ha="left",
    )
    figure.subplots_adjust(top=0.84, wspace=0.3, bottom=0.2)
    print(f"written: {save(figure, 'dictionary_shapes')}")


if __name__ == "__main__":
    main()
