"""Where reliability space loses the answer, plotted.

The relative error of the complementary route against the per-block
unreliability, for a parallel pair, a parallel triple and a ten-block series
chain, with the cliff marked. Below about 1e-8 per block the complementary
route for a parallel pair returns exactly zero, and the figure stops there
because there is nothing left to plot.
"""

from __future__ import annotations

import math

import numpy as np

from _common import SERIES, bootstrap, save, style_axes

bootstrap()

import matplotlib.pyplot as plt  # noqa: E402

from reliablock.numerics import parallel_unreliability, series_unreliability  # noqa: E402

EPS = 2.220446049250313e-16


def main() -> int:
    qs = np.geomspace(1e-12, 1e-1, 80)

    print(f"machine epsilon          : {EPS:.6e}")
    print()
    print(f"{'q each':>12s}{'parallel pair':>16s}{'parallel triple':>18s}{'series x10':>14s}")
    print(f"{'':>12s}{'rel error':>16s}{'rel error':>18s}{'rel error':>14s}")
    print("-" * 60)

    series_rows = []
    pair_rows = []
    triple_rows = []
    for q in qs:
        exact_pair = parallel_unreliability([float(q)] * 2)
        naive_pair = 1.0 - (1.0 - float(q) ** 2)
        pair_rows.append(abs(naive_pair - exact_pair) / exact_pair)

        exact_triple = parallel_unreliability([float(q)] * 3)
        naive_triple = 1.0 - (1.0 - float(q) ** 3)
        triple_rows.append(abs(naive_triple - exact_triple) / exact_triple)

        exact_series = series_unreliability([float(q)] * 10)
        naive_series = 1.0 - (1.0 - float(q)) ** 10
        series_rows.append(abs(naive_series - exact_series) / exact_series)

    for index in range(0, len(qs), 10):
        print(
            f"{qs[index]:>12.2e}{pair_rows[index]:>16.6e}"
            f"{triple_rows[index]:>18.6e}{series_rows[index]:>14.6e}"
        )

    # The LARGEST q at which the complementary route still returns zero: the
    # sweep is ascending, so this is the last index with a relative error of 1,
    # not the first.
    pair_lost = [q for q, e in zip(qs, pair_rows, strict=True) if e >= 1.0]
    triple_lost = [q for q, e in zip(qs, triple_rows, strict=True) if e >= 1.0]
    cliff_pair = max(pair_lost) if pair_lost else math.nan
    cliff_triple = max(triple_lost) if triple_lost else math.nan
    print()
    print(f"parallel pair returns zero up to q   = {cliff_pair:.3e}")
    print(f"parallel triple returns zero up to q = {cliff_triple:.3e}")
    print(f"series x10 worst relative error      = {max(series_rows):.6e}")

    figure, axes = plt.subplots(figsize=(8.4, 4.6))
    axes.plot(qs, pair_rows, color=SERIES[0], linewidth=1.8, label="parallel pair, Q = q^2")
    axes.plot(qs, triple_rows, color=SERIES[1], linewidth=1.8, label="parallel triple, Q = q^3")
    axes.plot(
        qs,
        series_rows,
        color=SERIES[2],
        linewidth=1.8,
        label="ten in series, Q = 1 - (1-q)^10",
    )
    axes.axhline(1.0, color="#1b1b1b", linewidth=1.0, linestyle="--")
    axes.annotate(
        "relative error 1.0 means the route\nreturned zero for a positive answer",
        (2e-11, 1.0),
        textcoords="offset points",
        xytext=(6, -28),
        fontsize=8,
    )
    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.legend(fontsize=8, frameon=False, loc="lower left")
    style_axes(
        axes,
        "relative error of 1 - prod(1 - q_i) against the exact route",
        "per-block unreliability q, dimensionless",
        "relative error of the complementary route",
    )

    written = save(figure, "precision_space.png")
    print()
    print(f"written: {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
