"""Measure what is frame-dependent in the attribution and what is not.

This is the claim the attribution rests on, and it cuts both ways, so both
halves are measured:

* The **total** point-ahead angle depends only on the relative transverse
  velocity, so it is invariant under a change of inertial frame to first order
  in beta.
* The **split** between the light-time lead and the two aberration terms is
  not. Boost into a frame where the remote terminal is at rest and the whole
  angle moves into the aberration terms; boost into the local terminal's rest
  frame and it all moves into the light-time lead.

A reader who takes a light-time-lead number out of this package without the
frame it was computed in has taken a number that means nothing, and the only
way to make that concrete is to show it changing.

The boost forces the linear trajectory model, because adding a constant
velocity to a Keplerian state is not a Keplerian state: see
``pointahead.terminal.Crosslink.boosted``.
"""

from __future__ import annotations

import numpy as np

from _bootstrap import Tee, bootstrap, header

bootstrap()

from pointahead.angle import decompose_point_ahead, reference_point_ahead  # noqa: E402
from pointahead.budget import direction_noise_floor_rad  # noqa: E402
from pointahead.geometry import norm  # noqa: E402
from pointahead.scenario import NAMED_SCENARIOS  # noqa: E402
from pointahead.terminal import Crosslink, TerminalState  # noqa: E402


def linearised(link: Crosslink) -> Crosslink:
    """The same states under the linear trajectory model, for a fair comparison."""
    return Crosslink(
        local=link.local,
        remote=link.remote,
        trajectory_model="linear",
        mu=link.mu,
        label=link.label,
        frame=link.frame,
    )


def main() -> int:
    tee = Tee("validate_frames_output.txt")
    header(tee, "frame dependence: the total is invariant, the attribution is not")

    for name, builder in NAMED_SCENARIOS.items():
        base = linearised(builder())
        reference = reference_point_ahead(base)
        floor = direction_noise_floor_rad(base)
        tee.rule(f"{name}: range {base.range_m / 1.0e3:.3f} km")
        tee(f"  point-ahead angle in the given frame : "
            f"{reference.total_rad * 1.0e6:.9f} urad")
        tee(f"  float64 direction resolution         : {floor:.6e} rad")
        tee("")
        tee("  Each row below is the SAME physical configuration seen from a")
        tee("  different inertial frame, reached by adding a constant velocity to")
        tee("  both terminals.")
        tee("")
        tee(f"  {'boost':<34s} {'total / urad':>14s} {'lead / urad':>13s} "
            f"{'stellar / urad':>15s} {'velocity / urad':>16s}")
        tee(f"  {'-' * 34} {'-' * 14} {'-' * 13} {'-' * 15} {'-' * 16}")

        boosts = [
            ("none (the frame as supplied)", np.zeros(3)),
            ("minus the local velocity", -base.local.velocity_mps),
            ("minus the remote velocity", -base.remote.velocity_mps),
            ("minus the mean of the two", -0.5 * (
                base.local.velocity_mps + base.remote.velocity_mps)),
            ("plus 1 km/s along +x", np.array([1000.0, 0.0, 0.0])),
            ("plus 10 km/s along +z", np.array([0.0, 0.0, 10000.0])),
        ]
        totals = []
        spreads = []
        for label, delta in boosts:
            boosted = base if not np.any(delta) else base.boosted(delta)
            decomposition = decompose_point_ahead(boosted)
            total = reference_point_ahead(boosted).total_rad
            totals.append(total)
            lead = decomposition.light_time_lead_magnitude_rad
            stellar = decomposition.stellar_aberration_magnitude_rad
            velocity = decomposition.velocity_aberration_magnitude_rad
            spreads.append(lead)
            tee(
                f"  {label:<34s} {total * 1.0e6:>14.9f} {lead * 1.0e6:>13.6f} "
                f"{stellar * 1.0e6:>15.6f} {velocity * 1.0e6:>16.6f}"
            )
        spread_total = max(totals) - min(totals)
        spread_lead = max(spreads) - min(spreads)
        tee("")
        tee(f"  spread of the total across frames    : {spread_total:.6e} rad")
        tee(f"  spread of the light-time lead        : {spread_lead:.6e} rad")
        tee(f"  ratio of the two spreads             : "
            f"{spread_lead / spread_total if spread_total > 0 else float('inf'):.3e}")
        tee(f"  spread of the total / the angle      : "
            f"{spread_total / reference.total_rad:.6e}")
        tee("")
        tee.check(
            f"{name}: the total is frame-invariant to 1e-7 of itself",
            spread_total / reference.total_rad < 1.0e-7,
            f"spread {spread_total:.3e} rad over {len(boosts)} frames",
        )
        tee.check(
            f"{name}: the light-time lead is NOT frame-invariant",
            spread_lead > 100.0 * spread_total,
            f"the lead moves {spread_lead / spread_total:.3e} times further than the total",
        )
        tee("")
        tee("  In the local terminal's rest frame -- the second row -- both")
        tee("  aberration lines are zero and the light-time lead carries the whole")
        tee("  angle. In the remote terminal's rest frame -- the third row -- the")
        tee("  lead is zero and the two aberration lines carry it. Same angle, same")
        tee("  physics, completely different attribution.")

    tee.rule("the second-order residual of the frame invariance")
    tee("  The invariance is first order in beta, so the spread should scale as the")
    tee("  square of the boost. Measured by boosting the leo_crossing geometry by a")
    tee("  sequence of speeds along a fixed axis.")
    tee("")
    base = linearised(NAMED_SCENARIOS["leo_crossing"]())
    unboosted = reference_point_ahead(base).total_rad
    tee(f"  {'boost / (m/s)':>14s} {'beta':>13s} {'total / urad':>16s} "
        f"{'change / rad':>14s} {'change / beta^2':>17s}")
    tee(f"  {'-' * 14} {'-' * 13} {'-' * 16} {'-' * 14} {'-' * 17}")
    for speed in (1.0e2, 1.0e3, 1.0e4, 1.0e5, 1.0e6, 1.0e7):
        boosted = base.boosted(np.array([0.0, 0.0, speed]))
        total = reference_point_ahead(boosted).total_rad
        beta = speed / 299792458.0
        change = abs(total - unboosted)
        tee(
            f"  {speed:>14.1e} {beta:>13.6e} {total * 1.0e6:>16.9f} "
            f"{change:>14.6e} {change / beta**2:>17.6e}"
        )
    tee("")
    tee("  The last column is roughly constant, which is the second-order scaling.")
    tee("  At a 1 km/s frame error -- larger than any realistic frame ambiguity --")
    tee("  the total moves by well under a picoradian.")

    tee.rule("boosting a Keplerian link is refused by construction")
    kepler_link = NAMED_SCENARIOS["leo_crossing"]()
    boosted = kepler_link.boosted(np.array([100.0, 0.0, 0.0]))
    tee(f"  original trajectory model          : {kepler_link.trajectory_model}")
    tee(f"  boosted trajectory model           : {boosted.trajectory_model}")
    tee(f"  boosted frame label                : {boosted.frame}")
    tee.check(
        "boosting forces the linear model rather than returning a wrong Kepler link",
        boosted.trajectory_model == "linear",
        f"got {boosted.trajectory_model!r}",
    )

    tee.rule("the local rest frame: the aberration lines vanish exactly")
    rest = Crosslink(
        local=TerminalState(
            base.local.position_m, np.zeros(3), base.epoch_s, base.local.name
        ),
        remote=TerminalState(
            base.remote.position_m,
            base.remote.velocity_mps - base.local.velocity_mps,
            base.epoch_s,
            base.remote.name,
        ),
        trajectory_model="linear",
        mu=base.mu,
        label="local rest frame",
        frame="local terminal instantaneous rest frame",
    )
    decomposition = decompose_point_ahead(rest)
    tee(f"  local speed in this frame          : {norm(rest.local.velocity_mps):.6e} m/s")
    for term, value in decomposition.as_lines():
        tee(f"  {term:<33s}: {value * 1.0e6:>12.6f} urad")
    tee(f"  {'total':<33s}: {decomposition.total_rad * 1.0e6:>12.6f} urad")
    tee(f"  {'reference in this frame':<33s}: "
        f"{reference_point_ahead(rest).total_rad * 1.0e6:>12.6f} urad")
    tee.check(
        "both aberration lines are exactly zero in the local rest frame",
        decomposition.stellar_aberration_magnitude_rad == 0.0
        and decomposition.velocity_aberration_magnitude_rad == 0.0,
    )
    tee.check(
        "the total in the local rest frame matches the total in the given frame "
        "to 1e-7 of itself",
        abs(decomposition.total_rad - unboosted) / unboosted < 1.0e-7,
        f"relative difference {abs(decomposition.total_rad - unboosted) / unboosted:.3e}",
    )

    tee.summarise_failures()
    tee("")
    tee(f"  written: {tee.save()}")
    return 1 if tee.blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())
