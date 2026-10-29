"""Bounds and epoch arithmetic, cross-checked against mathematics they do not use.

Three of the four checks here compare this package's analytic expressions
against the ``sgp4`` propagator's own internals, which perform a different
calculation: equation (1) inverts Kepler's law on the Kozai mean motion while
``sgp4`` inverts it on the Brouwer mean motion after its own transformation, and
equations (16) and (17) are first-order J2 secular rates while ``sgp4``'s
``nodedot`` and ``argpdot`` carry higher-order terms. In each case the
disagreement is bounded and reported rather than asserted away, and in the first
case the test also requires the two NOT to agree exactly, because a vanishing
difference would mean the check is not testing what it claims.

The fourth is the Julian date of equation (5) against a day count from the
proleptic Gregorian ordinal, which is a different algorithm from a different
origin, over the whole 1957 to 2056 window the two-digit epoch year can express.
"""

from __future__ import annotations

import math
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import OBJECTS, Tee, bootstrap, declared_settings, environment_lines

bootstrap()

from sgp4.api import Satrec  # noqa: E402

from tlecheck import dataset  # noqa: E402
from tlecheck.bounds import (  # noqa: E402
    BoundsPolicy,
    check,
    derive,
    mean_motion_ceiling_rev_per_day,
    semi_major_axis_km,
)
from tlecheck.dataset import (  # noqa: E402
    apsidal_precession_deg_per_day,
    nodal_regression_deg_per_day,
)
from tlecheck.elements import apply_checksum, format_tle, parse_tle  # noqa: E402
from tlecheck.epoch import epoch_julian_date, julian_date  # noqa: E402


def main() -> int:
    out = Tee("validate_bounds_output.txt")
    out.rule("ENVIRONMENT")
    for line in environment_lines():
        out(line)
    out("")
    for line in declared_settings():
        out(line)

    out.rule("1. EQUATION (3): THE MEAN MOTION CEILING")
    ceiling = mean_motion_ceiling_rev_per_day()
    out(f"fastest circular Earth orbit : {ceiling:.10f} rev/day")
    out(f"its period                   : {1440.0 / ceiling:.6f} minutes")
    out.check(
        "mean_motion_ceiling_is_physical",
        17.0 < ceiling < 17.1,
        f"{ceiling:.6f} rev/day at the equatorial radius",
    )

    out.rule("2. EQUATION (1) AGAINST THE PROPAGATOR'S OWN SEMI-MAJOR AXIS")
    out(
        "Equation (1) inverts Kepler's law on the Kozai mean motion written in "
        "the element set. sgp4 converts to the Brouwer mean motion first. The "
        "difference is the Kozai bias; it is reported, not removed."
    )
    out("")
    out(
        f"{'family':28s} {'n rev/day':>11s} {'eq (1) km':>13s} {'sgp4 km':>13s} "
        f"{'bias km':>10s}"
    )
    biases = []
    for family in dataset.FAMILIES:
        fields = {
            "satnum": 26000,
            "epoch_year": 2026,
            "epoch_day": 200.5,
            "inclination_deg": family.inclination_deg,
            "raan_deg": 120.0,
            "eccentricity": family.eccentricity,
            "arg_perigee_deg": 40.0,
            "mean_anomaly_deg": 200.0,
            "mean_motion_rev_per_day": family.mean_motion_rev_per_day,
            "bstar": family.bstar,
        }
        element_set = parse_tle(*format_tle(**fields))
        satrec = Satrec.twoline2rv(element_set.line1, element_set.line2)
        theirs = satrec.a * satrec.radiusearthkm
        mine = semi_major_axis_km(element_set.mean_motion_rev_per_day)
        bias = mine - theirs
        biases.append(abs(bias))
        out(
            f"{family.name:28s} {element_set.mean_motion_rev_per_day:11.5f} "
            f"{mine:13.4f} {theirs:13.4f} {bias:10.4f}"
        )
    out("")
    out(f"largest absolute Kozai bias over the six families : {max(biases):.4f} km")
    out.check(
        "kozai_bias_is_bounded",
        max(biases) < 20.0,
        f"largest {max(biases):.4f} km, against a screen whose loosest band is "
        "thousands of kilometres",
    )
    out.check(
        "kozai_bias_is_not_zero",
        min(biases) > 0.001,
        f"smallest {min(biases):.6f} km; a vanishing bias would mean the "
        "cross-check is not exercising the transformation",
    )

    out.rule("3. EQUATIONS (16) AND (17) AGAINST THE PROPAGATOR'S SECULAR RATES")
    out(
        "Two implementations of the same first-order secular theory. sgp4's "
        "nodedot and argpdot carry higher-order terms, so the agreement is "
        "reported as a relative difference and bounded, not asserted."
    )
    out("")
    out(
        f"{'family':28s} {'eq(16) deg/d':>13s} {'sgp4 deg/d':>12s} {'rel':>8s} "
        f"{'eq(17) deg/d':>13s} {'sgp4 deg/d':>12s} {'rel':>8s}"
    )
    node_worst = 0.0
    apsis_worst = 0.0
    for family in dataset.FAMILIES:
        fields = {
            "satnum": 26000,
            "epoch_year": 2026,
            "epoch_day": 200.5,
            "inclination_deg": family.inclination_deg,
            "raan_deg": 120.0,
            "eccentricity": family.eccentricity,
            "arg_perigee_deg": 40.0,
            "mean_anomaly_deg": 200.0,
            "mean_motion_rev_per_day": family.mean_motion_rev_per_day,
            "bstar": family.bstar,
        }
        element_set = parse_tle(*format_tle(**fields))
        satrec = Satrec.twoline2rv(element_set.line1, element_set.line2)
        node_theirs = math.degrees(satrec.nodedot) * 1440.0
        apsis_theirs = math.degrees(satrec.argpdot) * 1440.0
        node_mine = nodal_regression_deg_per_day(
            family.mean_motion_rev_per_day, family.eccentricity, family.inclination_deg
        )
        apsis_mine = apsidal_precession_deg_per_day(
            family.mean_motion_rev_per_day, family.eccentricity, family.inclination_deg
        )
        node_rel = (
            abs(node_mine - node_theirs) / abs(node_theirs) if node_theirs else math.nan
        )
        apsis_rel = (
            abs(apsis_mine - apsis_theirs) / abs(apsis_theirs)
            if abs(apsis_theirs) > 1.0e-9
            else math.nan
        )
        if math.isfinite(node_rel):
            node_worst = max(node_worst, node_rel)
        if math.isfinite(apsis_rel):
            apsis_worst = max(apsis_worst, apsis_rel)
        out(
            f"{family.name:28s} {node_mine:13.6f} {node_theirs:12.6f} {node_rel:8.4f} "
            f"{apsis_mine:13.6f} {apsis_theirs:12.6f} {apsis_rel:8.4f}"
        )
    out("")
    out(f"worst relative difference, equation (16) : {node_worst:.6f}")
    out(f"worst relative difference, equation (17) : {apsis_worst:.6f}")
    out.check(
        "nodal_regression_agrees_with_the_propagator",
        node_worst < 0.10,
        f"worst {node_worst:.6f} relative",
    )
    out.check(
        "apsidal_precession_agrees_with_the_propagator",
        apsis_worst < 0.50,
        f"worst {apsis_worst:.6f} relative; the critically inclined family sits "
        "where equation (17) vanishes and a relative comparison there is "
        "meaningless, which is why the bound is loose and is reported",
    )
    critical = apsidal_precession_deg_per_day(2.006, 0.72, 63.4349488)
    out(f"equation (17) at the critical inclination : {critical:.3e} deg/day")
    out.check(
        "apsidal_precession_vanishes_at_the_critical_inclination",
        abs(critical) < 1.0e-6,
        f"{critical:.3e} deg/day at 63.4349488 degrees",
    )

    out.rule("4. EQUATION (5) AGAINST AN INDEPENDENT DAY COUNT")
    out(
        "The proleptic Gregorian ordinal plus 1721424.5 is a different "
        "derivation of the Julian date: a day count from a different origin by a "
        "different algorithm. Compared on every month start of the 1957 to 2056 "
        "epoch-year window."
    )
    disagreements = 0
    compared = 0
    for year in range(1957, 2057):
        for month in range(1, 13):
            compared += 1
            independent = date(year, month, 1).toordinal() + 1721424.5
            if julian_date(year, month, 1.0) != independent:
                disagreements += 1
    out("")
    out(f"month starts compared        : {compared}")
    out(f"bit-exact disagreements      : {disagreements}")
    out.check(
        "julian_date_matches_an_independent_derivation",
        disagreements == 0,
        f"{compared} month starts, {disagreements} disagreements",
    )
    out("")
    out("Outside the declared range the two differ by exactly one day:")
    for year, month in ((1900, 1), (1900, 2), (2100, 3), (2100, 12)):
        independent = date(year, month, 1).toordinal() + 1721424.5
        out(
            f"  {year}-{month:02d}: equation (5) {julian_date(year, month, 1.0):.1f}, "
            f"independent {independent:.1f}, difference "
            f"{julian_date(year, month, 1.0) - independent:+.1f} day"
        )
    out(
        "The two-digit epoch year window starts at 1957, so no element set this "
        "package can decode reaches those dates."
    )

    out.rule("5. EPOCH DECODING AGAINST THE PROPAGATOR, OVER THE CORPUS")
    records = dataset.split("test", objects=OBJECTS, history_length=12)
    sets = [
        element_set
        for record in records
        for element_set in (*record.history, record.reference)
    ]
    worst = 0.0
    for element_set in sets:
        satrec = Satrec.twoline2rv(element_set.line1, element_set.line2)
        theirs = satrec.jdsatepoch + satrec.jdsatepochF
        worst = max(worst, abs(epoch_julian_date(element_set) - theirs))
    out(f"element sets compared        : {len(sets)}")
    out(f"worst epoch difference       : {worst:.3e} days ({worst * 86400.0:.3e} s)")
    out.check(
        "epoch_decoding_matches_the_propagator",
        worst < 1.0e-6,
        f"worst {worst * 86400.0:.3e} seconds over {len(sets)} element sets",
    )

    out.rule("6. BOUND FINDINGS ON CONSTRUCTED VIOLATIONS")
    base_fields = {
        "satnum": 26000,
        "epoch_year": 2026,
        "epoch_day": 200.5,
        "inclination_deg": 51.6,
        "raan_deg": 120.0,
        "eccentricity": 0.001,
        "arg_perigee_deg": 40.0,
        "mean_anomaly_deg": 200.0,
        "mean_motion_rev_per_day": 15.5,
        "bstar": 4.0e-5,
    }
    cases = [
        ("clean", {}, set(), 0),
        ("eccentricity_0.96", {"eccentricity": 0.96}, {"eccentricity_above_policy"}, 1),
        ("eccentricity_0.60", {"eccentricity": 0.60}, {"perigee_below_surface"}, 1),
        (
            "mean_motion_17.5",
            {"mean_motion_rev_per_day": 17.5},
            {"mean_motion_above_physical_ceiling"},
            1,
        ),
        ("mean_motion_0.40", {"mean_motion_rev_per_day": 0.40}, {"mean_motion_below_policy"}, 1),
        ("bstar_0.50", {"bstar": 0.50}, {"bstar_magnitude_above_policy"}, 1),
        ("bstar_negative", {"bstar": -4.0e-5}, set(), 0),
    ]
    out(f"{'case':24s} {'findings':>9s} {'kinds':>22s} expected")
    policy = BoundsPolicy()
    for name, overrides, expected_names, expected_count in cases:
        element_set = parse_tle(*format_tle(**{**base_fields, **overrides}))
        findings = check(element_set, policy)
        kinds = ",".join(sorted({finding.kind for finding in findings})) or "-"
        out(f"{name:24s} {len(findings):>9d} {kinds:>22s} {expected_count}")
        for finding in findings:
            out(f"    {finding.name} [{finding.kind}] {finding.detail}")
        out.check(
            f"bounds_{name}",
            expected_names <= {finding.name for finding in findings}
            and (len(findings) > 0) == (expected_count > 0),
            f"{len(findings)} findings, expected the set {sorted(expected_names) or 'empty'}",
        )
    out("")
    out(
        "A negative drag term produces no bound finding deliberately: it is legal "
        "in the text format and appears in real catalogues, so it is the history "
        "screen's business and not the bounds screen's."
    )

    out.rule("7. DERIVED GEOMETRY ON A PUBLISHED ELEMENT SET")
    published = parse_tle(
        "1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991",
        "2 25544  51.6439 211.2001 0007417  17.6667  85.6398 15.50103472202482",
    )
    orbit = derive(published)
    out(f"semi-major axis    : {orbit.semi_major_axis_km:.4f} km")
    out(f"perigee altitude   : {orbit.perigee_altitude_km:.4f} km")
    out(f"apogee altitude    : {orbit.apogee_altitude_km:.4f} km")
    out(f"period             : {orbit.period_minutes:.6f} minutes")
    satrec = Satrec.twoline2rv(published.line1, published.line2)
    out(
        f"sgp4 perigee alt   : {satrec.altp * satrec.radiusearthkm:.4f} km "
        "(Brouwer, so it differs by the Kozai bias)"
    )
    out.check(
        "derived_perigee_is_in_the_right_place",
        400.0 < orbit.perigee_altitude_km < 430.0,
        f"{orbit.perigee_altitude_km:.4f} km",
    )
    out.check(
        "period_is_the_inverse_mean_motion",
        abs(orbit.period_minutes * published.mean_motion_rev_per_day - 1440.0) < 1.0e-9,
        f"{orbit.period_minutes * published.mean_motion_rev_per_day:.9f} minutes "
        "times rev/day",
    )

    out.rule("8. A CONSTRUCTED INVALID-CHECKSUM ELEMENT SET, THE DECLARED CONTROL")
    line2 = apply_checksum(published.line2[:68])
    assert line2 == published.line2
    broken = parse_tle(published.line1[:68] + "7", published.line2)
    out(f"line 1 stated checksum  : {broken.checksum_line1_stated}")
    out(f"line 1 computed checksum: {broken.checksum_line1_computed}")
    out.check(
        "invalid_checksum_control_fires",
        broken.checksum_line1_stated != broken.checksum_line1_computed,
        "the constructed control is detected by arithmetic, not by a statistic",
    )
    out.check(
        "invalid_checksum_control_leaves_the_elements_untouched",
        np.isclose(
            broken.mean_motion_rev_per_day,
            published.mean_motion_rev_per_day,
            rtol=0.0,
            atol=0.0,
        ),
        "only column 69 differs",
    )

    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
