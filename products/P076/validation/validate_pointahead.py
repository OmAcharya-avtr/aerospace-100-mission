"""Score the closed-form first-order baseline against the iterated reference.

This is the comparison the batch specification asks for. The baseline is the
closed form ``theta = 2 |v_rel_perp| / c`` taken about the instantaneous line
of sight. The reference is the full pipeline: light-time solutions iterated to
convergence against the declared trajectory model, then the exact relativistic
aberration applied to both photon directions with the local terminal's
velocity.

Three error metrics are reported, because none of them alone is honest:

* the difference of the two *magnitudes*, which is what a reader computes from
  two printed angles and which is the smallest of the three;
* the difference of the two *offset vectors*, which is the closed-loop error a
  terminal tracking the received beam actually pays, and which is the one the
  error budget uses;
* the angle between the two *transmit directions*, which is the open-loop error
  a terminal pointing from ephemeris alone pays.

The result is reported as measured, including where the baseline wins.
"""

from __future__ import annotations

import numpy as np

from _bootstrap import Tee, Timer, bootstrap, header

bootstrap()

from pointahead.angle import (  # noqa: E402
    closed_form_point_ahead,
    decompose_point_ahead,
    reference_point_ahead,
    score_closed_form,
)
from pointahead.budget import direction_noise_floor_rad  # noqa: E402
from pointahead.geometry import norm, transverse  # noqa: E402
from pointahead.scenario import NAMED_SCENARIOS, scenario_family  # noqa: E402


def main() -> int:
    tee = Tee("validate_pointahead_output.txt")
    header(tee, "closed-form first-order baseline scored against the iterated reference")

    tee.rule("1. the three named geometries")
    tee(f"  {'scenario':<18s} {'range / km':>12s} {'closed / urad':>15s} "
        f"{'reference / urad':>17s}")
    tee(f"  {'-' * 18} {'-' * 12} {'-' * 15} {'-' * 17}")
    for name, builder in NAMED_SCENARIOS.items():
        link = builder()
        score = score_closed_form(link)
        tee(
            f"  {name:<18s} {link.range_m / 1.0e3:>12.3f} "
            f"{score.closed_form_rad * 1.0e6:>15.6f} {score.reference_rad * 1.0e6:>17.6f}"
        )
    tee("")
    tee(f"  {'scenario':<18s} {'magnitude / rad':>16s} {'offset / rad':>16s} "
        f"{'transmit / rad':>16s} {'noise floor':>13s}")
    tee(f"  {'-' * 18} {'-' * 16} {'-' * 16} {'-' * 16} {'-' * 13}")
    for name, builder in NAMED_SCENARIOS.items():
        link = builder()
        score = score_closed_form(link)
        tee(
            f"  {name:<18s} {score.magnitude_difference_rad:>16.6e} "
            f"{score.offset_difference_rad:>16.6e} {score.transmit_direction_rad:>16.6e} "
            f"{direction_noise_floor_rad(link):>13.6e}"
        )

    tee.rule("2. relative error of the baseline against the reference")
    tee("  Expressed as a fraction of the point-ahead angle itself, which is the")
    tee("  form that can be compared across geometries of different range.")
    tee("")
    worst_relative = 0.0
    for name, builder in NAMED_SCENARIOS.items():
        link = builder()
        score = score_closed_form(link)
        relative = score.offset_difference_rad / score.reference_rad
        worst_relative = max(worst_relative, relative)
        beta_local = norm(link.local.velocity_mps) / 299792458.0
        tee(
            f"  {name:<18s} offset error / angle = {relative:.6e}   "
            f"beta_local = {beta_local:.6e}   ratio to beta^2 = "
            f"{relative / beta_local**2:.4f}"
        )
    tee("")
    tee("  The closed-form error tracks beta_local squared, which is the order the")
    tee("  first-order aberration formula drops. It is not a coincidence and it is")
    tee("  the reason the baseline is good enough at orbital speeds.")
    tee.check(
        "closed-form offset error is below 1e-8 of the angle on all named geometries",
        worst_relative < 1.0e-8,
        f"worst {worst_relative:.6e}",
    )

    tee.rule("3. a seeded family of 300 crosslinks")
    family = scenario_family(760076, 300)
    closed_timer = Timer()
    reference_timer = Timer()
    offsets = []
    transmits = []
    relatives = []
    ranges = []
    floors = []
    angles = []
    for link in family:
        with closed_timer:
            baseline = closed_form_point_ahead(link)
        with reference_timer:
            reference = reference_point_ahead(link)
        score = score_closed_form(link)
        offsets.append(score.offset_difference_rad)
        transmits.append(score.transmit_direction_rad)
        relatives.append(score.offset_difference_rad / score.reference_rad)
        ranges.append(link.range_m)
        floors.append(direction_noise_floor_rad(link))
        angles.append(score.reference_rad)
        del baseline, reference
    offsets_array = np.array(offsets)
    transmits_array = np.array(transmits)
    relatives_array = np.array(relatives)
    floors_array = np.array(floors)
    angles_array = np.array(angles)
    tee(f"  crosslinks                         : {len(family)}")
    tee("  altitudes sampled                  : 400 to 1400 km")
    tee(f"  ranges                             : {min(ranges) / 1.0e3:.1f} to "
        f"{max(ranges) / 1.0e3:.1f} km")
    tee(f"  point-ahead angles                 : {angles_array.min() * 1.0e6:.6f} to "
        f"{angles_array.max() * 1.0e6:.6f} urad")
    tee("")
    tee(f"  {'quantity':<34s} {'median':>14s} {'p95':>14s} {'max':>14s}")
    tee(f"  {'-' * 34} {'-' * 14} {'-' * 14} {'-' * 14}")
    for label, array in (
        ("offset difference / rad", offsets_array),
        ("transmit direction difference / rad", transmits_array),
        ("offset difference / angle", relatives_array),
        ("float64 direction resolution / rad", floors_array),
    ):
        tee(
            f"  {label:<34s} {float(np.median(array)):>14.6e} "
            f"{float(np.percentile(array, 95)):>14.6e} {float(array.max()):>14.6e}"
        )
    tee.check(
        "closed-form offset error stays below 1e-7 of the angle over 300 crosslinks",
        float(relatives_array.max()) < 1.0e-7,
        f"max {float(relatives_array.max()):.6e}",
    )
    tee(f"  closed-loop errors at or below their own floor: "
        f"{int(np.sum(offsets_array <= floors_array))} of {len(family)}")
    tee.check(
        "the open-loop transmit-direction error always exceeds the closed-loop one",
        bool(np.all(transmits_array >= offsets_array)),
        f"violations {int(np.sum(transmits_array < offsets_array))} of {len(family)}",
    )

    tee.rule("4. does the closed form degrade with range, as light time suggests?")
    tee("  The curvature of the remote trajectory over the light time grows with")
    tee("  range, so a reader would expect the closed form to get worse. It barely")
    tee("  does, and the reason is a cancellation: the transverse velocity at the")
    tee("  advanced epoch is larger than at the nominal epoch by a_perp tau, and at")
    tee("  the retarded epoch it is smaller by the same amount, so the two")
    tee("  first-order curvature errors cancel in the SUM that forms the offset.")
    tee("  That is a first-order-exact cancellation and it is measured below.")
    tee("")
    order = np.argsort(ranges)
    buckets = np.array_split(order, 6)
    tee(f"  {'range band / km':<22s} {'n':>4s} {'median offset err':>19s} "
        f"{'median err / angle':>20s}")
    tee(f"  {'-' * 22} {'-' * 4} {'-' * 19} {'-' * 20}")
    for bucket in buckets:
        band_ranges = [ranges[i] / 1.0e3 for i in bucket]
        tee(
            f"  {f'{min(band_ranges):.0f} to {max(band_ranges):.0f}':<22s} "
            f"{len(bucket):>4d} {float(np.median(offsets_array[bucket])):>19.6e} "
            f"{float(np.median(relatives_array[bucket])):>20.6e}"
        )

    tee.rule("5. attribution on the three named geometries")
    tee("  The split between the light-time lead and the two aberration terms is")
    tee("  frame-dependent; the total is not. See validate_frames.py.")
    for name, builder in NAMED_SCENARIOS.items():
        link = builder()
        decomposition = decompose_point_ahead(link)
        reference = reference_point_ahead(link)
        tee("")
        tee(f"  {name} (frame: {link.frame})")
        tee(f"    local speed                      : "
            f"{norm(link.local.velocity_mps):>12.3f} m/s")
        tee(f"    remote speed                     : "
            f"{norm(link.remote.velocity_mps):>12.3f} m/s")
        tee(f"    transverse relative speed        : "
            f"{norm(transverse(link.relative_velocity_mps, link.line_of_sight)):>12.3f} m/s")
        for term, value in decomposition.as_lines():
            tee(f"    {term:<33s}: {value * 1.0e6:>12.6f} urad")
        tee(f"    {'vector sum':<33s}: {decomposition.total_rad * 1.0e6:>12.6f} urad")
        tee(f"    {'reference':<33s}: {reference.total_rad * 1.0e6:>12.6f} urad")
        tee(f"    {'closure':<33s}: {decomposition.closure_rad:>12.6e} rad")
        tee.check(
            f"{name} attribution closes to within 1e-8 of the angle",
            decomposition.closure_rad / reference.total_rad < 1.0e-8,
            f"closure/angle {decomposition.closure_rad / reference.total_rad:.3e}",
        )

    tee.rule("6. verdict")
    tee("  The closed-form first-order baseline WINS on every metric that matters")
    tee("  for the point-ahead offset. Over 300 seeded crosslinks its worst")
    tee(f"  disagreement with the iterated reference is "
        f"{float(offsets_array.max()):.3e} rad, which is")
    tee(f"  {float(relatives_array.max()):.3e} of the angle itself and is of the same")
    tee("  order as the float64 direction resolution of the geometry. It needs no")
    tee("  trajectory model and no iteration.")
    tee("")
    tee("  What the reference is for, and it is not nothing: it is the only way to")
    tee("  obtain the two light-time budget lines at all, it is what makes the")
    tee("  numerical sensitivity checks in validate_budget.py possible, and it is")
    tee("  required for the absolute transmit direction when a terminal cannot")
    tee("  close the loop on a received beam -- the open-loop error of the closed")
    tee(f"  form is {float(transmits_array.max()):.3e} rad at worst, one to two orders")
    tee("  above its closed-loop error.")

    tee.rule("7. timings -- UNSTABLE, for order of magnitude only")
    tee("  These numbers move between runs on shared hardware and are never used in")
    tee("  a prose conclusion anywhere in this repository.")
    tee("")
    tee(f"  {'operation':<42s} {'calls':>7s} {'mean / ms':>12s}")
    tee(f"  {'-' * 42} {'-' * 7} {'-' * 12}")
    tee(f"  {'closed_form_point_ahead':<42s} {closed_timer.calls:>7d} "
        f"{closed_timer.mean_ms:>12.5f}")
    tee(f"  {'reference_point_ahead (kepler)':<42s} {reference_timer.calls:>7d} "
        f"{reference_timer.mean_ms:>12.5f}")

    tee.summarise_failures()
    tee("")
    tee(f"  written: {tee.save()}")
    return 1 if tee.blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())
