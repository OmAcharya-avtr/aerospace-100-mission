"""Detect the five planted gaps in the committed campaign, and nothing else.

A gap detector that flags everything is useless, so this script checks both
directions: every planted gap is reported, and nothing that was not planted is
reported. The planted gaps and the generating seed are committed in
``fixtures/generation_parameters.json``.

    G1  lighting = eclipse AND closing_speed in [0.20, 0.50]   2-way tuple
    G2  thruster_health = one_failed AND sensor_mode = camera_only  2-way tuple
    G3  closing_speed in [0.20, 0.50] AND range in [200, 400]  2-way tuple and
                                                               9 empty cells
    G4  nav_position_error >= 1.5 m never executed             equivalence class
    G5  nothing within 2.5 m of range = 200 m                  boundary
"""

from __future__ import annotations

import sys

from _common import FIXTURES, rule

from scenariocov import build_report, grid_occupancy, load_campaign, load_class_document, load_space

EXPECTED_TUPLES = {
    "closing_speed=[0.2,0.5] & range_at_handover=[200,400]",
    "closing_speed=[0.2,0.5] & lighting=eclipse",
    "sensor_mode=camera_only & thruster_health=one_failed",
}


def main() -> int:
    """Check detection and the absence of false positives."""
    out = sys.stdout
    failures = 0
    space = load_space(FIXTURES / "space.json")
    campaign = load_campaign(FIXTURES / "campaign_executed.json")
    classes, boundaries = load_class_document(FIXTURES / "classes.json")
    report = build_report(space, campaign, 2, classes, boundaries)

    out.write("planted-gap detection on the committed campaign\n")
    out.write("=" * 78 + "\n")
    out.write(f"scenarios: {len(campaign)}   parameters: {len(space)}   ")
    out.write(f"discretised space: {space.n_combinations} combinations\n")

    out.write(rule("2-way tuples: G1, G2, G3") + "\n")
    found = {u.as_text() for u in report.tway.uncovered}
    out.write(f"2-way coverage {report.tway.covered}/{report.tway.total} = ")
    out.write(f"{report.tway.fraction:.6f}\n")
    for text in sorted(EXPECTED_TUPLES):
        ok = text in found
        failures += 0 if ok else 1
        out.write(f"  {'PASS' if ok else 'FAIL'}  planted and reported: {text}\n")
    extra = sorted(found - EXPECTED_TUPLES)
    failures += len(extra)
    out.write(f"  {'PASS' if not extra else 'FAIL'}  no unplanted tuple reported ")
    out.write(f"(extra: {len(extra)})\n")
    for text in extra:
        out.write(f"      unexpected: {text}\n")

    out.write(rule("equivalence classes: G4") + "\n")
    uncovered_classes = [(c.parameter, c.name) for c in report.classes.uncovered]
    ok = uncovered_classes == [("nav_position_error", "excessive")]
    failures += 0 if ok else 1
    out.write(f"class coverage {report.classes.covered}/{report.classes.total} = ")
    out.write(f"{report.classes.fraction:.6f}\n")
    out.write(f"  {'PASS' if ok else 'FAIL'}  uncovered classes = {uncovered_classes}\n")
    out.write(
        "  note: G4 is invisible to the 2-way report, because nav_position_error\n"
        "  values below 1.5 m already occupy the [0.5, 2.0] bin the class sits in.\n"
    )

    out.write(rule("boundaries: G5, and the near tie the margin exists for") + "\n")
    head = f"{'boundary':<38}{'hit':>5}{'n':>5}{'nearest':>12}{'tol':>8}{'margin':>11}"
    out.write(head + "\n")
    for hit in report.boundaries:
        out.write(
            f"{hit.spec.parameter + '/' + hit.spec.name:<38}"
            f"{str(hit.covered):>5}{hit.count:>5}{hit.nearest_distance:>12.6g}"
            f"{hit.spec.tolerance:>8g}{hit.margin:>+11.6g}\n"
        )
    by_name = {h.spec.name: h for h in report.boundaries}
    g5 = by_name["handover_200m"]
    ok = (not g5.covered) and g5.nearest_distance > g5.spec.tolerance
    failures += 0 if ok else 1
    out.write(f"  {'PASS' if ok else 'FAIL'}  G5 detected: nothing within the declared 2.0 m\n")
    tumble = by_name["tumble_limit_2p0"]
    out.write(
        f"  NOTE  tumble_limit_2p0 is a near tie: nearest {tumble.nearest_distance:.6g} deg/s\n"
        f"        against a tolerance of {tumble.spec.tolerance:g} deg/s, a margin of\n"
        f"        {tumble.margin:+.6g} deg/s. This boundary was NOT planted. The hit flag\n"
        "        is a strict floating-point comparison one part in twenty from\n"
        "        flipping, which is exactly why the margin is reported next to it\n"
        "        and why no test in this repository asserts that boolean.\n"
    )

    out.write(rule("grid occupancy: the G3 region as empty cells") + "\n")
    occ = grid_occupancy(space, space.encode_many(campaign))
    out.write(f"grid {occ.parameters} shape {occ.shape}\n")
    out.write(f"occupied {occ.occupied}/{occ.n_cells} = {occ.fraction:.6f}\n")
    out.write(
        f"uniform-random reference, equiprobable cells, Feller eq. (3): "
        f"{occ.expected_fraction_uniform:.6f}\n"
    )
    planted_block = occ.counts[3, 3, :, :]
    ok = int(planted_block.sum()) == 0 and planted_block.size == 9
    failures += 0 if ok else 1
    out.write(
        f"  {'PASS' if ok else 'FAIL'}  the 9 cells of closing_speed bin 3 x range bin 3 "
        "are all empty\n"
    )
    largest = occ.empty_blocks()[0]
    out.write(f"  largest empty connected block: {largest.size} cells\n")
    contains_planted = sum(1 for c in largest.cells if c[0] == 3 and c[1] == 3)
    ok = contains_planted == 9
    failures += 0 if ok else 1
    out.write(
        f"  {'PASS' if ok else 'FAIL'}  all 9 planted cells lie inside that block "
        f"({contains_planted} of 9)\n"
    )
    out.write(
        f"  the block is {largest.size} cells, not 9, so occupancy ALONE does not\n"
        "  isolate the planted region at this campaign size: it reports a larger\n"
        "  contiguous untested region of which the planted one is a part. The\n"
        "  2-way enumeration is what names the region exactly.\n"
    )

    out.write(f"\nfailures: {failures}\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
