"""Constructed controls, each anchored on a truth and each verified in one direction.

Sixteen controls. Eleven are constructed so that a screen must fire and five so
that no screen may fire, and both directions are asserted, because a control
that cannot fire is not evidence and a control that fires on a clean element set
is worse than no control at all.

Every control is anchored on a **constructed element set whose corruption is
known exactly**, never on an output of this package. The two the product
specification names by hand -- a valid-but-stale set and an invalid-checksum set
-- are controls 1 and 3 below; the invalid-checksum one is also measured in
``validate_bounds.py`` and the valid-but-stale one in ``validate_consequence.py``
with its position error in kilometres.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import (
    CADENCE_DAYS,
    EPOCH_REQUIREMENT_DAYS,
    HISTORY_LENGTH,
    SILENT_THRESHOLD_KM,
    Tee,
    bootstrap,
    declared_settings,
    environment_lines,
)

bootstrap()

from tlecheck import dataset  # noqa: E402
from tlecheck.elements import apply_checksum, format_tle, parse_tle  # noqa: E402
from tlecheck.epoch import epoch_julian_date  # noqa: E402
from tlecheck.verdict import screen  # noqa: E402

BASE = {
    "satnum": 27000,
    "epoch_year": 2026,
    "epoch_day": 200.5,
    "inclination_deg": 51.6000,
    "raan_deg": 120.0000,
    "eccentricity": 0.0010000,
    "arg_perigee_deg": 40.0000,
    "mean_anomaly_deg": 200.0000,
    "mean_motion_rev_per_day": 15.50000000,
    "bstar": 4.0e-5,
    "mean_motion_dot_rev_per_day2": 1.8e-5,
    "designator": "26000A",
    "element_number": 112,
    "revolution_number": 10360,
}

GEO_BASE = {
    **BASE,
    "satnum": 27001,
    "inclination_deg": 0.0600,
    "eccentricity": 0.0002000,
    "mean_motion_rev_per_day": 1.00272000,
    "bstar": -2.0e-5,
    "mean_motion_dot_rev_per_day2": 0.0,
}


def _truth(fields: dict) -> object:
    return parse_tle(*format_tle(**fields), source="control-truth")


def _history(fields: dict, length: int = HISTORY_LENGTH) -> list:
    base = _truth(fields)
    out = []
    for index in range(length):
        offset = -CADENCE_DAYS * (length - index)
        element_set = dataset.rewound(
            base,
            offset,
            source="control-history",
            element_number=100 + index,
            revolution_number=10000 + 30 * index,
        )
        if element_set is not None:
            out.append(element_set)
    return out


def _names(verdict) -> set[str]:
    return {problem.name for problem in verdict.problems}


def main() -> int:
    out = Tee("validate_controls_output.txt")
    out.rule("ENVIRONMENT")
    for line in environment_lines():
        out(line)
    out("")
    for line in declared_settings():
        out(line)

    truth = _truth(BASE)
    history = _history(BASE)
    geo_truth = _truth(GEO_BASE)
    geo_history = _history(GEO_BASE)
    use_jd = epoch_julian_date(truth) + 1.0
    geo_use_jd = epoch_julian_date(geo_truth) + 1.0

    out.rule("CONSTRUCTED TRUTH")
    out("Low-Earth control object, the anchor for controls 1 to 11 and 12 to 14:")
    out(truth.line1)
    out(truth.line2)
    out(f"history element sets : {len(history)}")
    out("")
    out("Geostationary control object, the anchor for control 15:")
    out(geo_truth.line1)
    out(geo_truth.line2)
    out(f"history element sets : {len(geo_history)}")

    baseline = screen(
        truth,
        history=history,
        reference_jd=use_jd,
        epoch_requirement_days=EPOCH_REQUIREMENT_DAYS,
        silent_threshold_km=SILENT_THRESHOLD_KM,
    )
    out("")
    out(
        f"the truth itself screens as {baseline.predicted_class!r} with "
        f"{len(baseline.problems)} problems and equation (15) score "
        f"{baseline.silent_risk_score:.6f}"
    )

    out.rule("CONTROLS THAT MUST FIRE (11)")

    must_fire: list[tuple[str, object, str, str]] = []

    broken = parse_tle(truth.line1[:68] + str((int(truth.line1[68]) + 4) % 10), truth.line2)
    must_fire.append(
        (
            "1_invalid_checksum",
            broken,
            "checksum_mismatch",
            "column 69 of line 1 altered, data untouched. The declared "
            "invalid-checksum control.",
        )
    )

    must_fire.append(
        (
            "2_non_digit_checksum_column",
            parse_tle(truth.line1[:68] + "X", truth.line2),
            "checksum_column_not_a_digit",
            "column 69 of line 1 is not a digit, so no checksum can be compared.",
        )
    )

    stale = dataset.rewound(truth, -30.0, source="control-stale")
    must_fire.append(
        (
            "3_valid_but_stale",
            stale,
            "epoch_stale",
            "a genuine uncorrupted element set 30 days old against a declared "
            "7 day requirement. The declared valid-but-stale control.",
        )
    )

    sub_surface = parse_tle(
        truth.line1, apply_checksum(truth.line2[:26] + "6000000" + truth.line2[33:68])
    )
    must_fire.append(
        (
            "4_perigee_below_surface",
            sub_surface,
            "perigee_below_surface",
            "eccentricity 0.6 on a 15.5 rev/day orbit puts the perigee underground.",
        )
    )

    too_fast = parse_tle(
        truth.line1,
        apply_checksum(truth.line2[:52] + "17.50000000" + truth.line2[63:68]),
    )
    must_fire.append(
        (
            "5_mean_motion_above_the_ceiling",
            too_fast,
            "mean_motion_above_physical_ceiling",
            "17.5 rev/day exceeds the equation (3) ceiling of 17.0436 rev/day.",
        )
    )

    flipped = parse_tle(
        apply_checksum(truth.line1[:53] + "-" + truth.line1[54:68]), truth.line2
    )
    must_fire.append(
        (
            "6_drag_sign_flip_against_a_positive_history",
            flipped,
            "history_outlier_bstar",
            "the drag term's sign is flipped; the checksum is recomputed and valid "
            "and every bound holds, so only the history screen can see it.",
        )
    )

    dropped_line1 = parse_tle(
        apply_checksum(truth.line1[:19] + truth.line1[20:68] + " "), truth.line2
    )
    must_fire.append(
        (
            "7a_one_character_dropped_from_line_1",
            dropped_line1,
            "parse_fault",
            "one character deleted from line 1 and the rest shifted left, with the "
            "checksum recomputed, so the line is internally self-consistent. The "
            "ephemeris type column ends up blank, which this package cannot read.",
        )
    )

    dropped_line2 = parse_tle(
        truth.line1, apply_checksum(truth.line2[:16] + truth.line2[17:68] + " ")
    )
    must_fire.append(
        (
            "7b_one_character_dropped_from_line_2",
            dropped_line2,
            "history_outlier_raan_deg",
            "the same fault on line 2, where every shifted field still parses as a "
            "number and every element stays inside every physical and policy bound. "
            "Measured: no field is unreadable and no bound fires, so the only "
            "screens that can see it are the ones that compare the object against "
            "itself. A shift is not one fault class; it is whatever the shifted "
            "characters happen to say, and on line 2 of this element set they say "
            "something plausible.",
        )
    )

    far_source = dataset.rewound(truth, 0.0, source="control-source-b")
    far_source = parse_tle(
        far_source.line1,
        apply_checksum(far_source.line2[:52] + "15.49900000" + far_source.line2[63:68]),
        source="control-source-b",
    )
    must_fire.append(
        (
            "8_cross_source_separation",
            truth,
            "cross_source_divergence",
            "a second source whose mean motion differs in the fourth decimal.",
        )
    )

    nudged = parse_tle(
        truth.line1,
        apply_checksum(truth.line2[:52] + "15.50002000" + truth.line2[63:68]),
    )
    must_fire.append(
        (
            "9_continuity_break",
            nudged,
            "ephemeris_continuity_exceeded",
            "the mean motion moved by 2.0e-5 rev/day, which is inside every bound "
            "and far outside the object's own element-to-element continuity.",
        )
    )

    future = dataset.rewound(truth, 6.0, source="control-future")
    must_fire.append(
        (
            "10_future_epoch",
            future,
            "epoch_in_future",
            "an element set whose epoch is 5 days ahead of the time it is wanted at.",
        )
    )

    wrong_type = parse_tle(
        apply_checksum(truth.line1[:62] + "3" + truth.line1[63:68]), truth.line2
    )
    must_fire.append(
        (
            "11_unexpected_ephemeris_type",
            wrong_type,
            "ephemeris_type_unexpected",
            "ephemeris type 3; distributed element sets carry 0.",
        )
    )

    out(f"{'control':46s} {'expected finding':36s} {'fired':>6s} {'class':>18s}")
    for name, element_set, expected, description in must_fire:
        others = [far_source] if name.startswith("8_") else None
        verdict = screen(
            element_set,
            history=history,
            other_sources=others,
            reference_jd=use_jd,
            epoch_requirement_days=EPOCH_REQUIREMENT_DAYS,
            silent_threshold_km=SILENT_THRESHOLD_KM,
        )
        fired = expected in _names(verdict)
        out(
            f"{name:46s} {expected:36s} {str(fired):>6s} "
            f"{verdict.predicted_class:>18s}"
        )
        out(f"    {description}")
        for problem in verdict.problems:
            out(f"    found: [{problem.screen}] {problem.name}")
        out.check(f"control_{name}_fires", fired, f"expected {expected}")

    out.rule("CONTROLS THAT MUST NOT FIRE (5)")

    must_not_fire: list[tuple[str, object, list | None, float, str]] = [
        (
            "12_the_truth_itself",
            truth,
            None,
            use_jd,
            "the constructed truth with its own history. No screen may fire.",
        ),
        (
            "13_valid_and_fresh",
            dataset.rewound(truth, -3.0, source="control-fresh"),
            None,
            use_jd,
            "a genuine element set 4 days old against a 7 day requirement.",
        ),
        (
            "14_last_digit_of_the_mean_anomaly",
            parse_tle(
                truth.line1,
                apply_checksum(
                    truth.line2[:43]
                    + f"{200.0001:8.4f}"
                    + truth.line2[51:68]
                ),
            ),
            None,
            use_jd,
            "the mean anomaly moved by one count of its text field, 1.0e-4 degrees, "
            "which is below the continuity screen's resolution.",
        ),
        (
            "15_negative_drag_against_a_negative_history",
            geo_truth,
            None,
            geo_use_jd,
            "a geostationary object whose drag term is legitimately negative, with "
            "a history that is also negative. The bounds screen must not treat a "
            "negative drag term as a violation and the history screen must not "
            "treat it as an outlier.",
        ),
        (
            "16_an_agreeing_second_source",
            truth,
            [
                parse_tle(
                    truth.line1,
                    apply_checksum(
                        truth.line2[:52] + "15.50000001" + truth.line2[63:68]
                    ),
                    source="control-source-b",
                )
            ],
            use_jd,
            "a second source whose mean motion differs in the eighth decimal, one "
            "count of the field.",
        ),
    ]

    out(f"{'control':46s} {'problems':>9s} {'class':>18s} {'integrity':>10s}")
    for name, element_set, others, reference_jd, description in must_not_fire:
        use_history = geo_history if name.startswith("15_") else history
        verdict = screen(
            element_set,
            history=use_history,
            other_sources=others,
            reference_jd=reference_jd,
            epoch_requirement_days=EPOCH_REQUIREMENT_DAYS,
            silent_threshold_km=SILENT_THRESHOLD_KM,
        )
        out(
            f"{name:46s} {len(verdict.problems):>9d} "
            f"{verdict.predicted_class:>18s} {str(verdict.integrity_ok):>10s}"
        )
        out(f"    {description}")
        for problem in verdict.problems:
            out(f"    found: [{problem.screen}] {problem.name} -- {problem.detail[:120]}")
        out.check(
            f"control_{name}_does_not_fire",
            not verdict.problems,
            f"{len(verdict.problems)} problems found, expected 0",
        )

    out.rule("SUMMARY")
    out(f"controls that must fire     : {len(must_fire)}")
    out(f"controls that must not fire : {len(must_not_fire)}")
    out(f"total controls              : {len(must_fire) + len(must_not_fire)}")
    out(
        "Every control is anchored on a constructed element set whose corruption "
        "is known exactly. None is anchored on an output of this package, and both "
        "directions are asserted."
    )
    out("")
    out(
        "Measured consequence of the two specification-named controls, for the "
        "record: the invalid-checksum control has a propagated position error of "
        "exactly 0.000000 km, because only column 69 changed; the valid-but-stale "
        "control at 30 days has 3.2964 km. Both figures come from "
        "validate_consequence.py. The first is the one worth noticing: the "
        "loudest screen in the package fires on the corruption with no "
        "consequence at all, and the package reports the integrity failure and "
        "the benign consequence as two separate outputs rather than one."
    )
    out("")
    out(
        "Not a control, stated because a reader will ask. Equation (15) scores "
        f"{baseline.silent_risk_score:.6f} on the truth itself, against a flag "
        "threshold of 1.0, so the margin on a clean element set is a factor of "
        f"{1.0 / baseline.silent_risk_score:.2f} if the score is non-zero."
    )

    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
