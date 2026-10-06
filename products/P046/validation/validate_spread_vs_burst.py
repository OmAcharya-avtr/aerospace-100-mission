"""Validation: minimum spread is a poor proxy for burst dispersion.

Claim under test
----------------
Minimum spread, the metric the turbo-code literature uses and the one a reader
would compute if all they had was a permutation function, does **not** rank
interleavers the same way burst dispersion does. On a fixed array the two can move
in opposite directions, so a designer who optimises minimum spread can choose a
materially worse interleaver for a fading link.

The experiment: fix a 16 x 16 helical array (N = 256, identical memory and latency
for every candidate) and sweep the read step. Report minimum spread and the
largest fully dispersed burst side by side, and report the rank correlation
between them.

Reference
---------
Both metrics are defined in ``src/interleavekit/metrics.py``. Minimum spread is
checked against brute-force pair enumeration in
``validate_block_spread.py``; burst dispersion is checked against its
definitional window scan in ``validate_burst_metric_equivalence.py``. This script
adds no new definition -- it only puts the two measured numbers next to each other.

Runtime: about 10 s on one core.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np  # noqa: E402

from interleavekit import BlockInterleaver, HelicalInterleaver, SRandomInterleaver  # noqa: E402
from interleavekit.metrics import (  # noqa: E402
    dispersion,
    max_burst_fully_dispersed,
    minimum_spread,
)

ROWS = 16
COLUMNS = 16


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation, computed from ranks and Pearson's formula."""
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def main() -> int:
    print(f"Helical interleaver on a fixed {ROWS} x {COLUMNS} array (N = {ROWS * COLUMNS})")
    print("Memory and latency are identical for every row: only the read step changes.")
    print()
    print(f"{'step':>5} {'min spread':>11} {'dispersion':>11} {'burst':>6}")
    steps = list(range(0, 17))
    spreads = []
    bursts = []
    for step in steps:
        il = HelicalInterleaver(ROWS, COLUMNS, step)
        pi = il.permutation()
        spread = minimum_spread(pi)
        burst = max_burst_fully_dispersed(il.position_of_input())
        spreads.append(spread)
        bursts.append(burst)
        print(f"{step:>5} {spread:>11} {dispersion(pi):>11.6f} {burst:>6}")

    spreads_a = np.asarray(spreads, dtype=float)
    bursts_a = np.asarray(bursts, dtype=float)
    rho = spearman(spreads_a, bursts_a)

    best_spread_step = int(np.argmax(spreads_a))
    best_burst_step = int(np.argmax(bursts_a))

    print()
    print(f"Spearman rank correlation between minimum spread and burst: {rho:.4f}")
    print(
        f"step chosen by maximising minimum spread: {best_spread_step} "
        f"(spread {spreads[best_spread_step]}, burst {bursts[best_spread_step]})"
    )
    print(
        f"step chosen by maximising burst dispersion: {best_burst_step} "
        f"(spread {spreads[best_burst_step]}, burst {bursts[best_burst_step]})"
    )
    penalty = bursts[best_burst_step] / max(bursts[best_spread_step], 1)
    print(
        f"cost of choosing on minimum spread instead of burst dispersion: "
        f"{penalty:.2f}x less fully dispersed burst, at identical memory"
    )

    print()
    print("Same comparison across constructions at N = 256, equal pair memory 512 symbols")
    print(f"{'construction':>44} {'min spread':>11} {'burst':>6}")
    candidates = [
        BlockInterleaver(16, 16),
        BlockInterleaver(32, 8),
        BlockInterleaver(8, 32),
        HelicalInterleaver(16, 16, 1),
        HelicalInterleaver(16, 16, 8),
        SRandomInterleaver(256, 8, seed=0),
    ]
    rows = []
    for il in candidates:
        pi = il.permutation()
        spread = minimum_spread(pi)
        burst = max_burst_fully_dispersed(il.position_of_input())
        rows.append((repr(il), spread, burst))
        print(f"{repr(il):>44} {spread:>11} {burst:>6}")

    sp = np.array([r[1] for r in rows], dtype=float)
    bu = np.array([r[2] for r in rows], dtype=float)
    print()
    print(f"Spearman rank correlation across these six: {spearman(sp, bu):.4f}")
    best_by_spread = rows[int(np.argmax(sp))]
    best_by_burst = rows[int(np.argmax(bu))]
    print(f"best by minimum spread: {best_by_spread[0]} (burst {best_by_spread[2]})")
    print(f"best by burst:          {best_by_burst[0]} (spread {best_by_burst[1]})")

    print()
    print("RESULT: reported (this script measures a relationship, it does not test a bound)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
