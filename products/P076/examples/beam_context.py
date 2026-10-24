"""The same point-ahead error judged against four different beams.

Writes ``screenshots/beam_context.png``. The curve is the Gaussian pointing
loss against error angle for each beam, with the budgeted total marked. This is
the figure that answers "does 3 microradians matter", and the answer is that it
depends entirely on which beam.
"""

from __future__ import annotations

import numpy as np

from _common import SERIES, bootstrap, plt, save, style_axes

bootstrap()

from pointahead.budget import BudgetInputs, build_budget  # noqa: E402
from pointahead.divergence import (  # noqa: E402
    error_for_loss_db,
    gaussian_divergence_half_angle,
    pointing_loss_db,
)
from pointahead.scenario import DEFAULT_BEAM, leo_crossing_pair  # noqa: E402
from pointahead.terminal import BeamContext  # noqa: E402

BEAMS = (
    (1550.0e-9, 0.010, "1550 nm, 10 mm waist"),
    (1550.0e-9, 0.025, "1550 nm, 25 mm waist"),
    (1064.0e-9, 0.075, "1064 nm, 75 mm waist"),
    (1064.0e-9, 0.150, "1064 nm, 150 mm waist"),
)
AGE_S = 300.0


def main() -> int:
    link = leo_crossing_pair()
    reference_budget = build_budget(link, BudgetInputs(0.05, AGE_S, 2, DEFAULT_BEAM))
    total = reference_budget.total_rss_rad

    print(f"geometry                : {link.label}, range {link.range_m / 1.0e3:.3f} km")
    print(f"ephemeris age           : {AGE_S:.0f} s")
    print(f"budgeted total error    : {total * 1.0e6:.6f} urad")
    print(f"dominant term           : {reference_budget.dominant_term}")
    print()
    print(f"{'beam':<24s} {'divergence / urad':>19s} {'fraction':>12s} "
          f"{'loss / dB':>12s} {'1 dB limit / urad':>19s} {'headroom':>11s}")
    print("-" * 102)
    rows = []
    for wavelength, waist, label in BEAMS:
        budget = build_budget(
            link, BudgetInputs(0.05, AGE_S, 2, BeamContext(wavelength, waist, label))
        )
        divergence = budget.beam.divergence_half_angle_rad
        rows.append((label, divergence, budget))
        print(
            f"{label:<24s} {divergence * 1.0e6:>19.6f} "
            f"{budget.beam.error_fraction_of_beam:>12.6f} {budget.beam.loss_db:>12.6f} "
            f"{budget.beam.affordable_error_rad * 1.0e6:>19.6f} "
            f"{budget.beam.headroom:>11.4f}"
        )
    print()
    print("The total is identical in every row. The loss spans "
          f"{rows[-1][2].beam.loss_db / rows[0][2].beam.loss_db:.1f} times.")

    figure, axes = plt.subplots(figsize=(9.0, 5.0))
    errors = np.linspace(0.0, 1.2e-5, 400)
    for index, (label, divergence, budget) in enumerate(rows):
        losses = [pointing_loss_db(float(error), divergence) for error in errors]
        axes.plot(errors * 1.0e6, losses, color=SERIES[index], linewidth=1.6, label=label)
        axes.plot(
            [total * 1.0e6],
            [budget.beam.loss_db],
            marker="o",
            color=SERIES[index],
            markersize=6,
        )
    axes.axvline(
        total * 1.0e6,
        color="#555555",
        linestyle="--",
        linewidth=1.0,
        label=f"budgeted total, {total * 1.0e6:.3f} urad",
    )
    axes.axhline(1.0, color="#8c3b3b", linestyle=":", linewidth=1.2, label="1 dB")
    axes.set_ylim(0.0, 6.0)
    style_axes(
        axes,
        "Gaussian pointing loss against point-ahead error, four transmit beams",
        "point-ahead error / urad",
        "pointing loss / dB",
    )
    axes.legend(fontsize=8, loc="upper left", framealpha=0.95)

    # Sanity line printed so the figure and the text come from one run.
    check = error_for_loss_db(1.0, gaussian_divergence_half_angle(1550.0e-9, 0.025))
    print()
    print(f"1 dB limit on the 1550 nm, 25 mm beam: {check * 1.0e6:.6f} urad")
    print(f"written: {save(figure, 'beam_context.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
