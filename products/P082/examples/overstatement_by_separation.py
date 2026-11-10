"""The headline, as a figure: what the independence assumption costs by separation.

Every pair of the declared network, with the ratio of the measured joint
unavailability to the independence product against great-circle separation.
Writes ``screenshots/overstatement_by_separation.png``.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

from _common import (  # noqa: E402
    EMPIRICAL,
    INDEPENDENT,
    NEUTRAL,
    WARN,
    bootstrap,
    save,
    style,
)

ROOT = bootstrap()

import matplotlib.pyplot as plt  # noqa: E402

from sitediverse.climatology import estimate_climatology  # noqa: E402
from sitediverse.empirical import bootstrap_joint_rate, joint_rate  # noqa: E402
from sitediverse.field import RECORD_SPECS, generate_record  # noqa: E402
from sitediverse.geometry import separation_matrix, upper_triangle_pairs  # noqa: E402
from sitediverse.independence import product_of_marginals  # noqa: E402


def main() -> int:
    style()
    record = generate_record(RECORD_SPECS["anisotropic"])
    climatology = estimate_climatology(record.cloud)
    separation = separation_matrix(record.sites)
    pairs = upper_triangle_pairs(record.site_count)
    distances = np.array([separation[i, j] for i, j in pairs])
    naive = np.array(
        [product_of_marginals(climatology.long_term_rate[[i, j]]) for i, j in pairs]
    )
    measured = np.array([joint_rate(record.cloud, (i, j)) for i, j in pairs])
    ratio = measured / naive

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    axes[0].scatter(
        distances, naive, s=18, color=INDEPENDENT, marker="^",
        label="independence, equation (14)",
    )
    axes[0].scatter(
        distances, measured, s=18, color=EMPIRICAL, marker="o",
        label="measured, equation (20)",
    )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("great-circle separation, km")
    axes[0].set_ylabel("pair joint unavailability")
    axes[0].set_title("190 pairs: assumed against measured")
    axes[0].legend(loc="lower right")

    axes[1].axhline(1.0, color=NEUTRAL, linestyle="--", linewidth=1.0)
    axes[1].scatter(distances, ratio, s=18, color=EMPIRICAL, marker="o")
    near = distances < 300.0
    axes[1].scatter(
        distances[near], ratio[near], s=40, facecolors="none",
        edgecolors=WARN, linewidths=1.2,
        label=f"{int(near.sum())} pairs under 300 km",
    )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("great-circle separation, km")
    axes[1].set_ylabel("measured / independence")
    axes[1].set_title("the overstatement factor")
    axes[1].legend(loc="upper right")

    path = save(figure, "overstatement_by_separation.png")
    print(f"written: {path}")
    print(f"  pairs                          : {len(pairs)}")
    print(f"  mean ratio under 300 km        : {float(np.mean(ratio[near])):.4f}")
    print(f"  mean ratio over 3200 km        : "
          f"{float(np.mean(ratio[distances > 3200.0])):.4f}")
    print(f"  largest pair ratio             : {float(ratio.max()):.4f} at "
          f"{float(distances[np.argmax(ratio)]):.1f} km")
    print(f"  pairs with a ratio below 1     : "
          f"{int(np.count_nonzero(ratio < 1.0))}")
    band = bootstrap_joint_rate(record.cloud, (0, 1), replicates=200)
    print(f"  near-pair bootstrap 95% interval: [{band.lower:.6f}, {band.upper:.6f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
