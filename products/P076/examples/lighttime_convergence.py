"""Light-time iteration: the residual against iteration count, three geometries.

Writes ``screenshots/lighttime_convergence.png``. The residual falls by the
radial speed over c per iteration, so three iterations reach the float64 floor
at orbital speeds. The right panel shows what truncating the iteration costs
the point-ahead offset, which is the budget line.
"""

from __future__ import annotations

from _common import SERIES, bootstrap, plt, save, style_axes

bootstrap()

from pointahead.angle import (  # noqa: E402
    light_time_pair,
    offset_difference,
    reference_point_ahead,
)
from pointahead.budget import direction_noise_floor_rad  # noqa: E402
from pointahead.geometry import C_LIGHT  # noqa: E402
from pointahead.lighttime import solve_light_time  # noqa: E402
from pointahead.scenario import NAMED_SCENARIOS  # noqa: E402

CAPS = tuple(range(6))


def main() -> int:
    figure, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    print(f"{'scenario':<18s} {'range / km':>12s} {'cap':>4s} "
          f"{'|residual| / s':>16s} {'range error / m':>17s} {'offset error / rad':>20s}")
    print("-" * 94)
    for index, (name, builder) in enumerate(NAMED_SCENARIOS.items()):
        link = builder()
        remote = link.trajectories()[1]
        converged = reference_point_ahead(link)
        residuals = []
        offsets = []
        for cap in CAPS:
            solution = solve_light_time(
                remote, link.local.position_m, link.epoch_s, "transmit", max_iter=cap
            )
            offset = offset_difference(reference_point_ahead(link, max_iter=cap), converged)
            residuals.append(abs(solution.residual_s))
            offsets.append(offset)
            print(
                f"{name:<18s} {link.range_m / 1.0e3:>12.3f} {cap:>4d} "
                f"{abs(solution.residual_s):>16.6e} "
                f"{C_LIGHT * abs(solution.residual_s):>17.6e} {offset:>20.6e}"
            )
        floor = 1.0e-20
        left.semilogy(
            CAPS,
            [max(value, floor) for value in residuals],
            marker="o",
            color=SERIES[index],
            linewidth=1.5,
            label=name,
        )
        right.semilogy(
            CAPS,
            [max(value, 1.0e-18) for value in offsets],
            marker="s",
            color=SERIES[index],
            linewidth=1.5,
            label=name,
        )
        right.axhline(
            direction_noise_floor_rad(link),
            color=SERIES[index],
            linestyle=":",
            linewidth=1.0,
        )
        _, transmit = light_time_pair(link)
        print(
            f"{name:<18s} converged tau_tx = {transmit.tau_s:.15f} s after "
            f"{transmit.iterations} iterations"
        )
        print()

    style_axes(
        left,
        "Light-time residual against iteration cap",
        "iteration cap",
        "|residual| / s (log scale, zero clipped to 1e-20)",
    )
    left.legend(fontsize=8, loc="upper right", framealpha=0.95)
    style_axes(
        right,
        "Point-ahead offset error from truncating, dotted lines are the float64 floor",
        "iteration cap",
        "offset error / rad (log scale, zero clipped to 1e-18)",
    )
    right.legend(fontsize=8, loc="upper right", framealpha=0.95)
    print(f"written: {save(figure, 'lighttime_convergence.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
