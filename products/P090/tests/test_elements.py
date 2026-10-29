"""Element set parsing, formatting and checksum.

The known-answer tests here are the strongest in the suite because they are
against a published element set and not against anything this repository
computed. The checksum digits of that element set are 1 and 2 as distributed;
this implementation computes 1 and 2; and formatting the parsed fields back out
reproduces both lines character for character, all 69 columns of each.
"""

from __future__ import annotations

import math

import pytest

from tlecheck.elements import (
    LINE_LENGTH,
    ElementSet,
    apply_checksum,
    checksum,
    format_implied_decimal,
    format_tle,
    parse_tle,
)

from conftest import REFERENCE_FIELDS, REFERENCE_LINE1, REFERENCE_LINE2


def test_reference_lines_are_sixty_nine_columns(reference_lines):
    for line in reference_lines:
        assert len(line) == LINE_LENGTH


def test_checksum_known_answer_line1(reference_lines):
    # Hand check of the modulo-10 digit sum rule on the published line 1:
    # the stated checksum digit in column 69 is '1'.
    assert reference_lines[0][68] == "1"
    assert checksum(reference_lines[0]) == 1


def test_checksum_known_answer_line2(reference_lines):
    # The stated checksum digit in column 69 of the published line 2 is '2'.
    assert reference_lines[1][68] == "2"
    assert checksum(reference_lines[1]) == 2


def test_checksum_hand_computed_short_string():
    # "1 2-3" has digits 1, 2, 3 summing to 6, plus 1 for the minus sign, so 7.
    assert checksum("1 2-3") == 7


def test_checksum_is_modulo_ten():
    # Nine nines sum to 81, which is 1 modulo 10.
    assert checksum("9" * 9) == 1


def test_checksum_ignores_column_sixty_nine():
    body = REFERENCE_LINE1[:68]
    assert checksum(body + "0") == checksum(body + "7") == checksum(body)


def test_checksum_ignores_letters_and_dots():
    assert checksum("abc.+ ") == 0


def test_apply_checksum_round_trip():
    body = REFERENCE_LINE1[:68]
    line = apply_checksum(body)
    assert len(line) == LINE_LENGTH
    assert int(line[68]) == checksum(body)


def test_apply_checksum_pads_short_body():
    line = apply_checksum("1 25544U")
    assert len(line) == LINE_LENGTH


def test_parse_reference_has_no_faults(reference_set):
    assert reference_set.faults == ()
    assert reference_set.parsable is True


@pytest.mark.parametrize(
    ("attribute", "expected", "tolerance"),
    [
        ("satnum", 25544, 0),
        ("epoch_year", 19, 0),
        ("epoch_day", 343.69339541, 0.0),
        ("inclination_deg", 51.6439, 0.0),
        ("raan_deg", 211.2001, 0.0),
        ("eccentricity", 0.0007417, 0.0),
        ("arg_perigee_deg", 17.6667, 0.0),
        ("mean_anomaly_deg", 85.6398, 0.0),
        ("mean_motion_rev_per_day", 15.50103472, 0.0),
        ("bstar", 3.8792e-05, 0.0),
        ("mean_motion_dot_rev_per_day2", 1.764e-05, 0.0),
        ("revolution_number", 20248, 0),
        ("ephemeris_type", 0, 0),
        ("element_number", 999, 0),
    ],
)
def test_parse_reference_fields(reference_set, attribute, expected, tolerance):
    value = getattr(reference_set, attribute)
    if isinstance(expected, int) and tolerance == 0:
        assert value == expected
    else:
        assert value == pytest.approx(expected, rel=0.0, abs=1.0e-12)


def test_parse_reference_designator(reference_set):
    assert reference_set.designator == "98067A"
    assert reference_set.classification == "U"


def test_epoch_full_year_window(reference_set):
    assert reference_set.epoch_full_year == 2019


def test_epoch_full_year_pivot():
    for written, expected in ((57, 1957), (99, 1999), (0, 2000), (56, 2056)):
        line1, line2 = format_tle(
            satnum=1,
            epoch_year=written,
            epoch_day=1.0,
            inclination_deg=10.0,
            raan_deg=20.0,
            eccentricity=0.001,
            arg_perigee_deg=30.0,
            mean_anomaly_deg=40.0,
            mean_motion_rev_per_day=14.0,
        )
        assert parse_tle(line1, line2).epoch_full_year == expected


def test_format_reproduces_published_line1_exactly():
    line1, _line2 = format_tle(**REFERENCE_FIELDS)
    assert line1 == REFERENCE_LINE1


def test_format_reproduces_published_line2_exactly():
    _line1, line2 = format_tle(**REFERENCE_FIELDS)
    assert line2 == REFERENCE_LINE2


def test_format_parse_round_trip_fields():
    element_set = parse_tle(*format_tle(**REFERENCE_FIELDS))
    for name in (
        "inclination_deg",
        "raan_deg",
        "eccentricity",
        "arg_perigee_deg",
        "mean_anomaly_deg",
        "mean_motion_rev_per_day",
        "bstar",
    ):
        assert getattr(element_set, name) == pytest.approx(
            REFERENCE_FIELDS[name], rel=0.0, abs=1.0e-12
        )


def test_implied_decimal_known_answer(reference_set):
    # ' 38792-4' is +0.38792e-4 = 3.8792e-05.
    assert reference_set.bstar == pytest.approx(3.8792e-05, rel=0.0, abs=1.0e-18)


def test_implied_decimal_zero_form():
    assert format_implied_decimal(0.0) == " 00000-0"


def test_implied_decimal_negative():
    assert format_implied_decimal(-3.8792e-05) == "-38792-4"


def test_implied_decimal_positive():
    assert format_implied_decimal(3.8792e-05) == " 38792-4"


@pytest.mark.parametrize("value", [1.0e-3, 5.5e-5, -1.2e-6, 9.9999e-2, 4.0e-9])
def test_implied_decimal_round_trip(value):
    text = format_implied_decimal(value)
    assert len(text) == 8
    line1, line2 = format_tle(
        satnum=7,
        epoch_year=2026,
        epoch_day=100.5,
        inclination_deg=10.0,
        raan_deg=20.0,
        eccentricity=0.001,
        arg_perigee_deg=30.0,
        mean_anomaly_deg=40.0,
        mean_motion_rev_per_day=14.0,
        bstar=value,
    )
    parsed = parse_tle(line1, line2)
    assert parsed.bstar == pytest.approx(value, rel=1.0e-4, abs=0.0)


def test_parse_never_raises_on_garbage():
    element_set = parse_tle("nonsense", "also nonsense")
    assert isinstance(element_set, ElementSet)
    assert element_set.faults


def test_parse_reports_short_line_length():
    element_set = parse_tle(REFERENCE_LINE1[:40], REFERENCE_LINE2)
    assert any("40 columns" in fault for fault in element_set.faults)


def test_parse_reports_long_line_length():
    element_set = parse_tle(REFERENCE_LINE1 + "XX", REFERENCE_LINE2)
    assert any("71 columns" in fault for fault in element_set.faults)


def test_parse_reports_wrong_line_number():
    element_set = parse_tle("3" + REFERENCE_LINE1[1:], REFERENCE_LINE2)
    assert any("expected '1'" in fault for fault in element_set.faults)
    element_set = parse_tle(REFERENCE_LINE1, "9" + REFERENCE_LINE2[1:])
    assert any("expected '2'" in fault for fault in element_set.faults)


def test_parse_reports_satellite_number_mismatch():
    other = "2 25545" + REFERENCE_LINE2[7:]
    element_set = parse_tle(REFERENCE_LINE1, other)
    assert any("differs between lines" in fault for fault in element_set.faults)


def test_parse_reports_non_numeric_eccentricity():
    broken = REFERENCE_LINE2[:26] + "ABCDEFG" + REFERENCE_LINE2[33:]
    element_set = parse_tle(REFERENCE_LINE1, broken)
    assert math.isnan(element_set.eccentricity)
    assert any("line 2: eccentricity" in fault for fault in element_set.faults)


def test_parse_reports_non_numeric_inclination():
    broken = REFERENCE_LINE2[:8] + "QQQQQQQQ" + REFERENCE_LINE2[16:]
    element_set = parse_tle(REFERENCE_LINE1, broken)
    assert math.isnan(element_set.inclination_deg)


def test_parse_blank_lines_are_faulty_not_fatal():
    element_set = parse_tle(" " * 69, " " * 69)
    assert element_set.faults
    assert element_set.parsable is False


def test_stated_checksum_none_when_not_a_digit():
    element_set = parse_tle(REFERENCE_LINE1[:68] + "X", REFERENCE_LINE2)
    assert element_set.checksum_line1_stated is None
    assert element_set.checksum_line2_stated == 2


def test_computed_checksums_match_stated_on_reference(reference_set):
    assert reference_set.checksum_line1_stated == reference_set.checksum_line1_computed
    assert reference_set.checksum_line2_stated == reference_set.checksum_line2_computed


def test_source_label_is_carried(reference_set):
    assert reference_set.source == "published-reference"


def test_trailing_newline_is_tolerated():
    element_set = parse_tle(REFERENCE_LINE1 + "\n", REFERENCE_LINE2 + "\r\n")
    assert element_set.faults == ()


def test_eccentricity_leading_zeros_are_implied():
    line2 = REFERENCE_LINE2[:26] + "0000001" + REFERENCE_LINE2[33:]
    element_set = parse_tle(REFERENCE_LINE1, line2)
    assert element_set.eccentricity == pytest.approx(1.0e-7, rel=0.0, abs=0.0)


def test_format_rejects_nothing_and_is_total():
    # A formatter that raises on an extreme value would make the generator
    # silently skip cases; it must produce 69 columns for anything finite.
    line1, line2 = format_tle(
        satnum=99999,
        epoch_year=2056,
        epoch_day=366.99999999,
        inclination_deg=179.9999,
        raan_deg=359.9999,
        eccentricity=0.9999999,
        arg_perigee_deg=359.9999,
        mean_anomaly_deg=359.9999,
        mean_motion_rev_per_day=17.0,
        bstar=-0.099999,
        mean_motion_dot_rev_per_day2=-0.0001,
    )
    assert len(line1) == len(line2) == LINE_LENGTH
