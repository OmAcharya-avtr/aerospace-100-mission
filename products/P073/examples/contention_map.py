"""The antenna-time contention map: apportioned demand over union supply.

What to notice: a bin can be busy without being contended. The ratio is high
where several requests have few other options at that time, which is not the
same as where the antenna is most occupied, and the validation measurement shows
this per-bin ratio is a weak predictor of who actually gets starved.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _common import bootstrap, save, style

bootstrap()

from contactsched.contention import contention_map  # noqa: E402
from contactsched.dataset import generate_instance  # noqa: E402
from contactsched.exact import solve_exact  # noqa: E402
from contactsched.fairness import analyse  # noqa: E402

SEED = 7
HORIZON_S = 43200.0
BIN_S = 1800.0


def main() -> None:
    style()
    instance = generate_instance(SEED, horizon_s=HORIZON_S)
    problem = instance.problem
    cells = contention_map(problem, HORIZON_S, BIN_S)

    stations = sorted({c.station for c in cells})
    edges = sorted({c.bin_start_s for c in cells})
    ratio = np.full((len(stations), len(edges)), np.nan)
    supply = np.zeros((len(stations), len(edges)))
    for cell in cells:
        row = stations.index(cell.station)
        column = edges.index(cell.bin_start_s)
        supply[row, column] = cell.supply_s
        ratio[row, column] = cell.ratio if np.isfinite(cell.ratio) else np.nan

    figure, (heat, occupancy) = plt.subplots(2, 1, figsize=(11.0, 5.6), sharex=True)
    finite = ratio[np.isfinite(ratio)]
    image = heat.imshow(
        ratio,
        aspect="auto",
        cmap="YlOrRd",
        vmin=0.0,
        vmax=float(np.nanpercentile(finite, 97)) if finite.size else 1.0,
        extent=(0.0, HORIZON_S / 3600.0, len(stations) - 0.5, -0.5),
        interpolation="nearest",
    )
    heat.set_yticks(range(len(stations)))
    heat.set_yticklabels(stations, fontsize=8)
    heat.grid(False)
    heat.set_title(
        f"Apportioned demand over union supply, {BIN_S:.0f} s bins, instance seed {SEED}"
    )
    figure.colorbar(image, ax=heat, label="demand / supply", pad=0.01)

    image2 = occupancy.imshow(
        supply / BIN_S,
        aspect="auto",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        extent=(0.0, HORIZON_S / 3600.0, len(stations) - 0.5, -0.5),
        interpolation="nearest",
    )
    occupancy.set_yticks(range(len(stations)))
    occupancy.set_yticklabels(stations, fontsize=8)
    occupancy.grid(False)
    occupancy.set_xlabel("hours after the scenario epoch")
    occupancy.set_title("Union visibility as a fraction of the bin: busy is not contended")
    figure.colorbar(image2, ax=occupancy, label="visible fraction", pad=0.01)

    figure.tight_layout()
    path = save(figure, "contention_map")

    report = analyse(problem, solve_exact(problem))
    hottest = sorted(
        (c for c in cells if np.isfinite(c.ratio)), key=lambda c: -c.ratio
    )[:5]
    print(f"cells                      : {len(cells)}")
    print(f"cells with demand > supply : {sum(1 for c in cells if c.ratio > 1.0)}")
    print(f"cells with no supply       : {sum(1 for c in cells if c.supply_s == 0.0)}")
    print("hottest finite-ratio cells :")
    for cell in hottest:
        print(
            f"  {cell.station:<14} bin {cell.bin_start_s / 3600.0:5.2f} h  "
            f"supply {cell.supply_s:7.1f} s  demand {cell.demand_s:7.1f} s  "
            f"ratio {cell.ratio:6.3f}  requests {','.join(cell.requests)}"
        )
    print(f"requests actually starved  : {[o.request_id for o in report.starved]}")
    print(f"figure written             : {path}")


if __name__ == "__main__":
    main()
