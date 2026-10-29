"""Element plausibility bounds, with an independent cross-check against sgp4.

The cross-check matters and is a different class of test from the rest. Every
other test here compares this package's arithmetic against this package's
arithmetic. :func:`test_semi_major_axis_against_sgp4` compares the Kepler
inversion of equation (1) against the semi-major axis the ``sgp4`` propagator
computes from the same element set after its own Kozai-to-Brouwer conversion,
which is mathematics this package does not perform. The two differ by about
half a kilometre, which is the Kozai bias, and the test bounds that difference
rather than asserting the two agree.
"""

from __future__ import annotations

import math

import pytest
from sgp4.api import Satrec

from tlecheck.bounds import (
    EARTH_RADIUS_KM,
    MU_KM3_S2,
    BoundsPolicy,
    check,
    derive,
    mean_motion_ceiling_rev_per_day,
    semi_major_axis_km,
)
from tlecheck.elements import format_tle, parse_tle

from conftest import REFERENCE_FIELDS, REFERENCE_LINE1, REFERENCE_LINE2


def _modified(**overrides) -> object:
    fields = dict(REFERENCE_FIELDS)
    fields.update(overrides)
    return parse_tle(*format_tle(**fields))


def test_mean_motion_ceiling_known_answer():
    # Equation (3): n_max = 86400 sqrt(mu / Re^3) / (2 pi).
    # With mu = 398600.8 and Re = 6378.135 this is 17.0436417 rev/day;
    # computed by hand from the same two constants below.
    expected = 86400.0 * math.sqrt(MU_KM3_S2 / EARTH_RADIUS_KM**3) / (2.0 * math.pi)
    assert mean_motion_ceiling_rev_per_day() == pytest.approx(
        expected, rel=0.0, abs=1.0e-12
    )
    assert mean_motion_ceiling_rev_per_day() == pytest.approx(
        17.043641719, rel=1.0e-9, abs=0.0
    )


def test_semi_major_axis_geostationary_known_answer():
    # One sidereal revolution per day, 1.00273790935 rev/day, is the
    # geostationary mean motion; Kepler's third law puts it at 42164 km to
    # four figures. Hand check: omega = 7.2921159e-5 rad/s,
    # a = (398600.8 / omega^2)^(1/3) = 42164.2 km.
    assert semi_major_axis_km(1.00273790935) == pytest.approx(42164.2, rel=2.0e-5, abs=0.0)


def test_semi_major_axis_inverts_kepler():
    for mean_motion in (1.0, 2.0, 14.5, 16.9):
        a = semi_major_axis_km(mean_motion)
        omega = math.sqrt(MU_KM3_S2 / a**3)
        assert omega * 86400.0 / (2.0 * math.pi) == pytest.approx(
            mean_motion, rel=1.0e-12, abs=0.0
        )


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_semi_major_axis_rejects_non_positive(bad):
    assert math.isnan(semi_major_axis_km(bad))


def test_semi_major_axis_against_sgp4(reference_lines):
    """Independent cross-check: equation (1) against sgp4's own semi-major axis.

    ``sgp4`` converts the Kozai mean motion to the Brouwer mean motion before
    inverting Kepler's law; equation (1) does not. The difference is the Kozai
    bias and it is bounded here rather than asserted away. Measured in this
    environment: 0.5031 km for the published reference element set.
    """
    satrec = Satrec.twoline2rv(*reference_lines)
    theirs = satrec.a * satrec.radiusearthkm
    mine = semi_major_axis_km(15.50103472)
    difference = abs(theirs - mine)
    assert difference < 2.0, f"Kozai bias {difference:.4f} km is larger than expected"
    assert difference > 0.1, (
        "the two should NOT agree exactly; equation (1) does not perform the "
        "Kozai conversion and a vanishing difference would mean the cross-check "
        "is not testing what it claims"
    )


def test_derive_reference_geometry(reference_set):
    orbit = derive(reference_set)
    # Perigee altitude of a 15.501 rev/day orbit with e = 0.0007417:
    # a = 6794.56 km, r_p = a (1 - e) = 6789.52 km, altitude 411.39 km.
    assert orbit.perigee_altitude_km == pytest.approx(411.39, rel=0.0, abs=0.02)
    assert orbit.apogee_altitude_km == pytest.approx(421.47, rel=0.0, abs=0.02)
    assert orbit.period_minutes == pytest.approx(92.897, rel=0.0, abs=0.01)


def test_derive_period_is_inverse_mean_motion(reference_set):
    orbit = derive(reference_set)
    assert orbit.period_minutes * reference_set.mean_motion_rev_per_day == pytest.approx(
        1440.0, rel=1.0e-12, abs=0.0
    )


def test_reference_has_no_bound_findings(reference_set):
    assert check(reference_set) == []


def test_eccentricity_above_policy_is_policy_kind():
    findings = check(_modified(eccentricity=0.96))
    names = {finding.name: finding.kind for finding in findings}
    assert names["eccentricity_above_policy"] == "policy"


def test_perigee_below_surface_is_physical():
    findings = check(_modified(eccentricity=0.6))
    names = {finding.name: finding.kind for finding in findings}
    assert names["perigee_below_surface"] == "physical"


def test_mean_motion_above_ceiling_is_physical():
    findings = check(_modified(mean_motion_rev_per_day=17.5))
    names = {finding.name: finding.kind for finding in findings}
    assert names["mean_motion_above_physical_ceiling"] == "physical"


def test_mean_motion_below_policy():
    findings = check(_modified(mean_motion_rev_per_day=0.4))
    names = {finding.name for finding in findings}
    assert "mean_motion_below_policy" in names


def test_bstar_magnitude_above_policy():
    findings = check(_modified(bstar=0.5))
    names = {finding.name for finding in findings}
    assert "bstar_magnitude_above_policy" in names


def test_negative_bstar_is_not_a_bound_violation():
    # A negative drag term is legal in the format and appears in real
    # catalogues; it is the history screen's business, not the bounds screen's.
    assert check(_modified(bstar=-3.8792e-05)) == []


def test_ephemeris_type_unexpected_is_policy():
    element_set = _modified(ephemeris_type=3)
    findings = check(element_set)
    names = {finding.name: finding.kind for finding in findings}
    assert names["ephemeris_type_unexpected"] == "policy"


def test_non_numeric_field_yields_format_finding_and_stops():
    broken = parse_tle(REFERENCE_LINE1, REFERENCE_LINE2[:26] + "ABCDEFG" + REFERENCE_LINE2[33:])
    findings = check(broken)
    assert all(finding.kind == "format" for finding in findings)
    assert any("eccentricity" in finding.name for finding in findings)


def test_inclination_out_of_range_is_physical():
    element_set = parse_tle(
        REFERENCE_LINE1, REFERENCE_LINE2[:8] + "190.0000" + REFERENCE_LINE2[16:]
    )
    findings = check(element_set)
    names = {finding.name: finding.kind for finding in findings}
    assert names["inclination_out_of_range"] == "physical"


def test_angle_out_of_range_is_physical():
    element_set = parse_tle(
        REFERENCE_LINE1, REFERENCE_LINE2[:17] + "400.0000" + REFERENCE_LINE2[25:]
    )
    findings = check(element_set)
    names = {finding.name: finding.kind for finding in findings}
    assert names["raan_deg_out_of_range"] == "physical"


def test_policy_is_configurable():
    tight = BoundsPolicy(max_eccentricity=0.0001)
    findings = check(_modified(), tight)
    assert any(finding.name == "eccentricity_above_policy" for finding in findings)


def test_policy_defaults_are_the_documented_ones():
    policy = BoundsPolicy()
    assert policy.min_perigee_altitude_km == 60.0
    assert policy.max_eccentricity == 0.95
    assert policy.max_abs_bstar == 0.1
    assert policy.min_mean_motion_rev_per_day == 0.5
    assert policy.expected_ephemeris_type == 0


def test_findings_carry_their_value():
    findings = check(_modified(eccentricity=0.96))
    finding = next(f for f in findings if f.name == "eccentricity_above_policy")
    assert finding.value == pytest.approx(0.96, rel=0.0, abs=1.0e-7)
    assert "0.96" in finding.detail
