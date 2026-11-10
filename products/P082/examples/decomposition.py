"""The exact decomposition of equation (16), per declared network.

Two assumptions fail in opposite directions. The co-modulation term is what a
STATIONARY climatology costs and is signed; the correlation term is what
INDEPENDENCE costs. Writes ``screenshots/decomposition.png``.
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
from sitediverse.empirical import bootstrap_joint_rate  # noqa: E402
from sitediverse.field import RECORD_SPECS, generate_record  # noqa: E402
from sitediverse.independence import overstatement  # noqa: E402

NETWORKS = {
    "near pair": (0, 1),
    "far pair": (0, 12),
    "mixed trio": (0, 1, 8),
    "one group": (0, 1, 2, 3),
    "two groups": (0, 1, 2, 4, 5, 6),
    "one per group": (0, 4, 8, 12, 16),
}


def main() -> int:
    style()
    record = generate_record(RECORD_SPECS["anisotropic"])
    climatology = estimate_climatology(record.cloud)
    names = list(NETWORKS)
    results = {
        name: overstatement(climatology, record.cloud, sites)
        for name, sites in NETWORKS.items()
    }
    bands = {
        name: bootstrap_joint_rate(record.cloud, sites, replicates=200)
        for name, sites in NETWORKS.items()
    }

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    positions = np.arange(len(names))
    comodulation = np.array([results[name].comodulation_term for name in names])
    correlation = np.array([results[name].correlation_term for name in names])
    errors = np.array([bands[name].standard_error for name in names])

    axes[0].axhline(0.0, color=NEUTRAL, linewidth=1.0)
    axes[0].bar(
        positions - 0.2, comodulation, 0.38, color=WARN,
        label="co-modulation: stationarity",
    )
    axes[0].bar(
        positions + 0.2, correlation, 0.38, color=EMPIRICAL,
        label="correlation: independence",
    )
    axes[0].errorbar(
        positions + 0.2, correlation, yerr=errors, color=NEUTRAL,
        linestyle="none", capsize=3,
    )
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(names, rotation=20, ha="right")
    axes[0].set_ylabel("contribution to the overstatement")
    axes[0].set_title("equation (16): two assumptions, two signs")
    axes[0].legend(loc="upper left")

    ratio = np.array([results[name].ratio for name in names])
    axes[1].axhline(1.0, color=NEUTRAL, linestyle="--", linewidth=1.0)
    axes[1].bar(
        positions,
        ratio,
        0.6,
        color=[INDEPENDENT if value > 1.0 else NEUTRAL for value in ratio],
    )
    axes[1].set_yscale("log")
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(names, rotation=20, ha="right")
    axes[1].set_ylabel("measured / independence")
    axes[1].set_title("the overstatement factor, per network")
    for index, value in enumerate(ratio):
        axes[1].text(
            index, value * 1.08, f"{value:.2f}", ha="center", fontsize=8
        )

    path = save(figure, "decomposition.png")
    print(f"written: {path}")
    for name in names:
        result = results[name]
        print(f"  {name:16s} naive {result.naive:.6f}  conditional "
              f"{result.conditional:.6f}  measured {result.measured:.6f}  "
              f"+/- {bands[name].standard_error:.6f}")
        print(f"  {'':16s} co-modulation {result.comodulation_term:+.6f}  "
              f"correlation {result.correlation_term:+.6f}  ratio "
              f"{result.ratio:.4f}  identity residual "
              f"{abs(result.identity_residual()):.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
