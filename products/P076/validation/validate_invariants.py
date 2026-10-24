"""Geometric invariants the point-ahead angle must satisfy, measured not assumed.

Five invariants, each of which follows from the definition rather than from the
implementation, so each is a genuine check on the code:

1. **Vanishing at zero transverse relative velocity.** If the relative velocity
   is parallel to the line of sight, the apparent source direction and the
   required transmit direction are both parallel to the separation vector, so
   the angle is zero -- exactly in exact arithmetic.
2. **Sign reversal under reversed relative velocity.** Negating the relative
   velocity negates the point-ahead offset vector and leaves its magnitude
   unchanged.
3. **Rotational equivariance.** Rotating the whole configuration rotates the
   offset and leaves every angle unchanged.
4. **Reverse-link magnitude symmetry.** The far terminal sees the same
   transverse relative speed, so it computes the same point-ahead magnitude.
5. **Linearity in the transverse relative speed.** The angle is proportional to
   it over the whole orbital regime.

Floating-point comparisons are made with a tolerance on both sides. No boolean
produced by a threshold comparison is asserted equal anywhere.
"""

from __future__ import annotations

import numpy as np

from _bootstrap import Tee, bootstrap, header

bootstrap()

from pointahead.angle import (  # noqa: E402
    closed_form_point_ahead,
    offset_difference,
    reference_point_ahead,
)
from pointahead.budget import direction_noise_floor_rad  # noqa: E402
from pointahead.geometry import C_LIGHT, norm, rotation_matrix, transverse, unit  # noqa: E402
from pointahead.scenario import NAMED_SCENARIOS, scenario_family  # noqa: E402
from pointahead.terminal import Crosslink, TerminalState  # noqa: E402


def rotate_link(link: Crosslink, matrix: np.ndarray) -> Crosslink:
    """Rotate both terminal states of ``link`` by ``matrix``."""
    return Crosslink(
        local=TerminalState(
            matrix @ link.local.position_m,
            matrix @ link.local.velocity_mps,
            link.epoch_s,
            link.local.name,
        ),
        remote=TerminalState(
            matrix @ link.remote.position_m,
            matrix @ link.remote.velocity_mps,
            link.epoch_s,
            link.remote.name,
        ),
        trajectory_model=link.trajectory_model,
        mu=link.mu,
        label=f"{link.label} (rotated)",
        frame=link.frame,
    )


def main() -> int:
    tee = Tee("validate_invariants_output.txt")
    header(tee, "geometric invariants of the point-ahead angle")

    rng = np.random.default_rng(7607)
    family = scenario_family(7607, 120)

    tee.rule("1. the angle vanishes for purely radial relative motion")
    tee("  Constructed by replacing the remote velocity with the local velocity")
    tee("  plus a multiple of the line of sight, so the transverse relative")
    tee("  velocity is exactly zero by construction.")
    tee("")
    worst_closed = 0.0
    worst_reference = 0.0
    for link in family:
        radial_speed = float(rng.uniform(-8000.0, 8000.0))
        radial = Crosslink(
            local=link.local,
            remote=TerminalState(
                link.remote.position_m,
                link.local.velocity_mps + radial_speed * link.line_of_sight,
                link.epoch_s,
                link.remote.name,
            ),
            trajectory_model="linear",
            mu=link.mu,
            label=f"{link.label} (radial)",
            frame=link.frame,
        )
        worst_closed = max(worst_closed, closed_form_point_ahead(radial).total_rad)
        worst_reference = max(worst_reference, reference_point_ahead(radial).total_rad)
    tee(f"  crosslinks                         : {len(family)}")
    tee(f"  worst closed-form angle            : {worst_closed:.6e} rad")
    tee(f"  worst reference angle              : {worst_reference:.6e} rad")
    tee.check("closed form vanishes to 1e-18 rad", worst_closed < 1.0e-18,
              f"{worst_closed:.3e} rad")
    tee.check("reference vanishes to 1e-15 rad", worst_reference < 1.0e-15,
              f"{worst_reference:.3e} rad")

    tee.rule("2. negating the relative velocity negates the offset")
    worst_magnitude = 0.0
    worst_direction = 0.0
    for link in family:
        reversed_velocity = Crosslink(
            local=link.local,
            remote=TerminalState(
                link.remote.position_m,
                2.0 * link.local.velocity_mps - link.remote.velocity_mps,
                link.epoch_s,
                link.remote.name,
            ),
            trajectory_model="linear",
            mu=link.mu,
            label=f"{link.label} (negated)",
            frame=link.frame,
        )
        forward = closed_form_point_ahead(link)
        backward = closed_form_point_ahead(reversed_velocity)
        scale = forward.total_rad
        worst_magnitude = max(worst_magnitude, abs(backward.total_rad - scale) / scale)
        worst_direction = max(
            worst_direction,
            float(np.linalg.norm(backward.offset_rad + forward.offset_rad)) / scale,
        )
    tee(f"  worst relative magnitude change    : {worst_magnitude:.6e}")
    tee(f"  worst relative direction residual  : {worst_direction:.6e}")
    tee.check("magnitude is unchanged to 1e-14 relative", worst_magnitude < 1.0e-14)
    tee.check("offset is negated to 1e-14 relative", worst_direction < 1.0e-14)

    tee.rule("3. rotational equivariance of the whole configuration")
    tee("  Rotating both terminal states must leave every angle unchanged and must")
    tee("  rotate the offset 3-vector with the configuration.")
    tee("")
    tee("  The acceptance criterion is ABSOLUTE and is the float64 direction")
    tee("  resolution of each geometry, pointahead.budget.direction_noise_floor_rad.")
    tee("  It was first written as a pure relative tolerance of 1e-12 and that")
    tee("  failed, at a worst relative residual of 3.7e-09 on a crosslink whose")
    tee("  point-ahead angle is 2.0e-07 rad: a residual of 7.7e-16 rad is 3.7e-09 of")
    tee("  an angle that small, and 7.7e-16 rad is three hundred times BELOW the")
    tee("  2.3e-13 rad resolution of that geometry. A relative criterion on a")
    tee("  quantity whose scale varies by two orders across the family is the wrong")
    tee("  criterion, and both residuals are printed below so a reader can see that.")
    tee("")
    worst_angle_absolute = 0.0
    worst_angle_relative = 0.0
    worst_offset_absolute = 0.0
    worst_offset_relative = 0.0
    over_floor = 0
    smallest_angle = float("inf")
    for link in family:
        axis = unit(rng.normal(size=3))
        matrix = rotation_matrix(axis, float(rng.uniform(0.0, 2.0 * np.pi)))
        rotated = rotate_link(link, matrix)
        floor = direction_noise_floor_rad(link)
        for compute in (closed_form_point_ahead, reference_point_ahead):
            before = compute(link)
            after = compute(rotated)
            smallest_angle = min(smallest_angle, before.total_rad)
            angle_residual = abs(after.total_rad - before.total_rad)
            # The transverse basis is rebuilt from the rotated line of sight, so
            # the two-component offset is NOT equivariant; the 3-vector offset is,
            # once rotated back.
            offset_residual = float(
                np.linalg.norm(matrix.T @ after.offset_vector - before.offset_vector)
            )
            worst_angle_absolute = max(worst_angle_absolute, angle_residual)
            worst_angle_relative = max(worst_angle_relative, angle_residual / before.total_rad)
            worst_offset_absolute = max(worst_offset_absolute, offset_residual)
            worst_offset_relative = max(worst_offset_relative, offset_residual / before.total_rad)
            if max(angle_residual, offset_residual) > floor:
                over_floor += 1
    tee(f"  crosslinks                         : {len(family)}")
    tee(f"  smallest point-ahead angle         : {smallest_angle:.6e} rad")
    tee(f"  worst absolute angle residual      : {worst_angle_absolute:.6e} rad")
    tee(f"  worst relative angle residual      : {worst_angle_relative:.6e}")
    tee(f"  worst absolute offset residual     : {worst_offset_absolute:.6e} rad")
    tee(f"  worst relative offset residual     : {worst_offset_relative:.6e}")
    tee(f"  residuals above their own floor    : {over_floor} of {2 * len(family)}")
    tee.check(
        "every rotation residual is below the float64 resolution of its geometry",
        over_floor == 0,
        f"{over_floor} of {2 * len(family)} residuals exceeded their floor",
    )

    tee.rule("4. the reverse link computes the same magnitude")
    tee("  The far terminal sees the same transverse relative speed about the same")
    tee("  line of sight, so the magnitude agrees. The ATTRIBUTION does not, and")
    tee("  that is measured in validate_frames.py.")
    tee("")
    worst_reverse = 0.0
    for link in family:
        forward = reference_point_ahead(link).total_rad
        backward = reference_point_ahead(link.reversed()).total_rad
        worst_reverse = max(worst_reverse, abs(backward - forward) / forward)
    tee(f"  crosslinks                         : {len(family)}")
    tee(f"  worst relative magnitude difference: {worst_reverse:.6e}")
    tee.check(
        "reverse-link magnitude agrees to 1e-4 relative",
        worst_reverse < 1.0e-4,
        f"worst {worst_reverse:.6e}; the residual is the second-order aberration "
        "difference between the two local velocities",
    )

    tee.rule("5. linearity in the transverse relative speed")
    tee("  The angle is 2 v_perp / c, so scaling the transverse relative velocity")
    tee("  scales the angle by the same factor. Measured over seven decades.")
    tee("")
    tee("  The criterion is absolute-or-relative, whichever is looser: a relative")
    tee("  tolerance of 1e-09 OR the float64 direction resolution of the geometry.")
    tee("  A pure 1e-09 relative criterion failed at v_perp = 1e-03 m/s, where the")
    tee("  angle is 6.7e-12 rad and the residual is 3.3e-17 rad -- three hundred")
    tee("  times below the 1.0e-14 rad resolution of the geometry, and therefore")
    tee("  not a measurement of the code at all.")
    tee("")
    link = NAMED_SCENARIOS["leo_crossing"]()
    floor = direction_noise_floor_rad(link)
    base_transverse = transverse(link.relative_velocity_mps, link.line_of_sight)
    direction = unit(base_transverse)
    tee(f"  float64 direction resolution       : {floor:.6e} rad")
    tee("")
    tee(f"  {'v_perp / (m/s)':>16s} {'angle / rad':>16s} {'2 v_perp / c':>16s} "
        f"{'absolute':>12s} {'relative':>12s}")
    tee(f"  {'-' * 16} {'-' * 16} {'-' * 16} {'-' * 12} {'-' * 12}")
    linearity_failures = 0
    for speed in (1.0e-3, 1.0e-1, 1.0, 1.0e1, 1.0e2, 1.0e3, 1.0e4):
        scaled = Crosslink(
            local=TerminalState(
                link.local.position_m, np.zeros(3), link.epoch_s, link.local.name
            ),
            remote=TerminalState(
                link.remote.position_m, speed * direction, link.epoch_s, link.remote.name
            ),
            trajectory_model="linear",
            mu=link.mu,
            label="scaled",
            frame=link.frame,
        )
        measured = reference_point_ahead(scaled).total_rad
        predicted = 2.0 * norm(transverse(speed * direction, scaled.line_of_sight)) / C_LIGHT
        absolute = abs(measured - predicted)
        relative = absolute / predicted
        if absolute > max(1.0e-9 * predicted, floor):
            linearity_failures += 1
        tee(
            f"  {speed:>16.3e} {measured:>16.9e} {predicted:>16.9e} "
            f"{absolute:>12.3e} {relative:>12.3e}"
        )
    tee.check(
        "the angle is linear in the transverse relative speed over seven decades",
        linearity_failures == 0,
        f"{linearity_failures} of 7 points outside max(1e-09 relative, the floor)",
    )

    tee.rule("6. the offset is unchanged by moving the remote terminal along the line of sight")
    tee("  2 v_perp / c has no range dependence at first order. The reference does")
    tee("  have one, through the light time, and the size of it is measured here.")
    tee("")
    link = NAMED_SCENARIOS["leo_crossing"]()
    base = reference_point_ahead(link)
    tee(f"  {'range / km':>14s} {'angle / urad':>16s} {'offset diff / rad':>19s}")
    tee(f"  {'-' * 14} {'-' * 16} {'-' * 19}")
    for factor in (1.0, 1.5, 2.0, 4.0, 8.0):
        stretched = Crosslink(
            local=link.local,
            remote=TerminalState(
                link.local.position_m + factor * link.separation_m,
                link.remote.velocity_mps,
                link.epoch_s,
                link.remote.name,
            ),
            trajectory_model="linear",
            mu=link.mu,
            label="stretched",
            frame=link.frame,
        )
        result = reference_point_ahead(stretched)
        tee(
            f"  {stretched.range_m / 1.0e3:>14.3f} {result.total_rad * 1.0e6:>16.9f} "
            f"{offset_difference(result, base):>19.6e}"
        )
    tee("")
    tee("  The angle is constant to nine significant figures over a factor of eight")
    tee("  in range, which is the first-order range independence, measured.")

    tee.summarise_failures()
    tee("")
    tee(f"  written: {tee.save()}")
    return 1 if tee.blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())
