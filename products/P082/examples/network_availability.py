"""Network availability as stations are added, assumed against measured.

The figure a diversity trade study wants: how the joint unavailability falls as
stations are added, under the independence assumption and as the record shows.
Writes ``screenshots/network_availability.png``.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

from _common import (  # noqa: E402
    EMPIRICAL,
    FITTED,
    INDEPENDENT,
    bootstrap,
    save,
    style,
)

ROOT = bootstrap()

import matplotlib.pyplot as plt  # noqa: E402

from sitediverse.climatology import estimate_climatology  # noqa: E402
from sitediverse.empirical import bootstrap_joint_rate  # noqa: E402
from sitediverse.field import RECORD_SPECS, generate_record  # noqa: E402
from sitediverse.independence import availability_nines, overstatement  # noqa: E402

#: Two declared growth sequences: inside one regional group, and one station per
#: group. The first is what short-baseline diversity delivers and the second is
#: what long-baseline diversity delivers.
SEQUENCES = {
    "within one group (<300 km)": (0, 1, 2, 3),
    "one per group (>1500 km)": (0, 4, 8, 12, 16),
}


def main() -> int:
    style()
    record = generate_record(RECORD_SPECS["anisotropic"])
    climatology = estimate_climatology(record.cloud)
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    printed = []
    for index, (label, sites) in enumerate(SEQUENCES.items()):
        counts = []
        naive = []
        measured = []
        errors = []
        for size in range(1, len(sites) + 1):
            subset = sites[:size]
            if size == 1:
                counts.append(1)
                naive.append(float(climatology.long_term_rate[subset[0]]))
                measured.append(float(record.cloud[:, subset[0]].mean()))
                errors.append(0.0)
                continue
            result = overstatement(climatology, record.cloud, subset)
            band = bootstrap_joint_rate(record.cloud, subset, replicates=200)
            counts.append(size)
            naive.append(result.naive)
            measured.append(result.measured)
            errors.append(band.standard_error)
        axis = axes[index]
        axis.semilogy(
            counts, naive, color=INDEPENDENT, marker="^", linestyle="--",
            label="independence, equation (14)",
        )
        axis.errorbar(
            counts, measured, yerr=errors, color=EMPIRICAL, marker="o",
            linestyle="-", capsize=3,
            label="measured, equation (20), block bootstrap",
        )
        axis.set_xticks(counts)
        axis.set_xlabel("stations in the network")
        axis.set_ylabel("joint unavailability")
        axis.set_title(label)
        axis.legend(loc="lower left")
        printed.append((label, counts[-1], naive[-1], measured[-1], errors[-1]))

    path = save(figure, "network_availability.png")
    print(f"written: {path}")
    for label, size, assumed, shown, error in printed:
        print(f"  {label}")
        print(f"    at n = {size}: independence {assumed:.6f}, measured "
              f"{shown:.6f} +/- {error:.6f}")
        print(f"    nines claimed {availability_nines(assumed):.4f}, measured "
              f"{availability_nines(shown):.4f}, lost "
              f"{availability_nines(assumed) - availability_nines(shown):+.4f}")
    _ = np, FITTED
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
