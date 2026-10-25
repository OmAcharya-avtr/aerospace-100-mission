"""Two-line element construction, checksums and the round trip through ``sgp4``.

Every element set here is synthetic, written by
:meth:`contactsched.tle.OrbitElements.to_lines`. No catalogue element set is
quoted anywhere in this repository; see the module docstring of
:mod:`contactsched.tle` for why.
"""

from __future__ import annotations

import math

import pytest
from sgp4.api import Satrec

from contactsched.tle import TLE_LINE_LENGTH, OrbitElements, parse_lines, tle_checksum


def elements(**overrides) -> OrbitElements:
    """A valid baseline element set with fields overridable per test."""
    base = dict(
        satnum=90001,
        epoch_year=26,
        epoch_day=120.5,
        inclination_deg=97.8,
        raan_deg=120.0,
        eccentricity=0.0012345,
        argp_deg=45.0,
        mean_anomaly_deg=10.0,
        mean_motion_rev_day=15.21,
        bstar=0.00011606,
        name="SYNTH-1",
    )
    base.update(overrides)
    return OrbitElements(**base)


class TestChecksum:
    def test_minus_sign_counts_as_one(self):
        # '-' adds 1; digits add 1+2+3+4 = 10; total 11; 11 mod 10 = 1.
        assert tle_checksum("-1234") == 1

    def test_digits_only(self):
        # 1 + 0 + 0 + 0 + 0 + 5 = 6, and the letters and space add nothing.
        assert tle_checksum("1 00005U") == 6

    def test_letters_and_dots_add_nothing(self):
        assert tle_checksum("ABC.def") == 0

    def test_only_the_first_sixty_eight_characters_count(self):
        padded = "9" * 68 + "7"
        assert tle_checksum(padded) == (68 * 9) % 10

    def test_result_is_a_single_digit(self):
        for value in ("1", "99999", "-" * 40, "0"):
            assert 0 <= tle_checksum(value) <= 9


class TestLineFormat:
    def test_both_lines_are_sixty_nine_characters(self):
        line1, line2 = elements().to_lines()
        assert len(line1) == TLE_LINE_LENGTH
        assert len(line2) == TLE_LINE_LENGTH

    def test_line_numbers_are_first(self):
        line1, line2 = elements().to_lines()
        assert line1[0] == "1"
        assert line2[0] == "2"

    def test_checksums_are_self_consistent(self):
        line1, line2 = elements().to_lines()
        assert int(line1[68]) == tle_checksum(line1)
        assert int(line2[68]) == tle_checksum(line2)

    def test_zero_bstar_uses_the_canonical_field(self):
        line1, _line2 = elements(bstar=0.0).to_lines()
        assert line1[53:61] == " 00000-0"

    def test_negative_bstar_carries_a_minus(self):
        line1, _line2 = elements(bstar=-1.1606e-4).to_lines()
        assert line1[53] == "-"

    def test_satellite_number_is_zero_padded(self):
        line1, line2 = elements(satnum=5).to_lines()
        assert line1[2:7] == "00005"
        assert line2[2:7] == "00005"


class TestRoundTrip:
    def test_independent_reader_recovers_every_field(self):
        source = elements()
        parsed = parse_lines(*source.to_lines())
        assert parsed["satnum"] == pytest.approx(90001.0, rel=0.0, abs=0.0)
        assert parsed["epoch_year"] == pytest.approx(26.0, rel=0.0, abs=0.0)
        assert parsed["epoch_day"] == pytest.approx(120.5, rel=0.0, abs=1e-8)
        assert parsed["inclination_deg"] == pytest.approx(97.8, rel=0.0, abs=1e-4)
        assert parsed["raan_deg"] == pytest.approx(120.0, rel=0.0, abs=1e-4)
        assert parsed["eccentricity"] == pytest.approx(0.0012345, rel=0.0, abs=1e-7)
        assert parsed["argp_deg"] == pytest.approx(45.0, rel=0.0, abs=1e-4)
        assert parsed["mean_anomaly_deg"] == pytest.approx(10.0, rel=0.0, abs=1e-4)
        assert parsed["mean_motion_rev_day"] == pytest.approx(15.21, rel=0.0, abs=1e-8)
        assert parsed["bstar"] == pytest.approx(0.00011606, rel=1e-9, abs=0.0)

    @pytest.mark.parametrize("eccentricity", [0.0, 1e-7, 0.0005, 0.0123456, 0.3])
    def test_eccentricity_quantises_to_seven_digits(self, eccentricity):
        parsed = parse_lines(*elements(eccentricity=eccentricity).to_lines())
        assert parsed["eccentricity"] == pytest.approx(eccentricity, rel=0.0, abs=5e-8)

    @pytest.mark.parametrize("bstar", [0.0, 1e-5, -1e-5, 2.7e-4, -3.3e-3])
    def test_bstar_round_trips_within_field_precision(self, bstar):
        parsed = parse_lines(*elements(bstar=bstar).to_lines())
        assert parsed["bstar"] == pytest.approx(bstar, rel=1e-4, abs=1e-12)

    def test_sgp4_reads_back_the_same_elements(self):
        source = elements()
        satrec = Satrec.twoline2rv(*source.to_lines())
        assert satrec.ecco == pytest.approx(source.eccentricity, rel=0.0, abs=5e-8)
        assert math.degrees(satrec.inclo) == pytest.approx(
            source.inclination_deg, rel=0.0, abs=1e-4
        )
        assert math.degrees(satrec.nodeo) == pytest.approx(source.raan_deg, rel=0.0, abs=1e-4)
        assert math.degrees(satrec.argpo) == pytest.approx(source.argp_deg, rel=0.0, abs=1e-4)
        assert satrec.bstar == pytest.approx(source.bstar, rel=1e-6, abs=0.0)

    def test_sgp4_mean_motion_matches_the_declared_revolutions_per_day(self):
        source = elements(mean_motion_rev_day=14.5)
        satrec = Satrec.twoline2rv(*source.to_lines())
        # sgp4 stores the Kozai mean motion in rad/min.
        assert satrec.no_kozai * 1440.0 / (2.0 * math.pi) == pytest.approx(
            14.5, rel=0.0, abs=1e-7
        )


class TestParseErrors:
    def test_wrong_length_raises(self):
        line1, line2 = elements().to_lines()
        with pytest.raises(ValueError, match="must be 69 characters"):
            parse_lines(line1[:-1], line2)

    def test_wrong_line_number_raises(self):
        line1, line2 = elements().to_lines()
        with pytest.raises(ValueError, match="must begin with"):
            parse_lines(line2, line2)

    def test_bad_checksum_raises(self):
        line1, line2 = elements().to_lines()
        corrupt = line1[:68] + str((int(line1[68]) + 1) % 10)
        with pytest.raises(ValueError, match="checksum"):
            parse_lines(corrupt, line2)


class TestElementValidation:
    @pytest.mark.parametrize(
        "overrides, match",
        [
            (dict(satnum=0), "satnum"),
            (dict(satnum=100000), "satnum"),
            (dict(epoch_year=100), "epoch_year"),
            (dict(epoch_year=-1), "epoch_year"),
            (dict(epoch_day=0.5), "epoch_day"),
            (dict(epoch_day=400.0), "epoch_day"),
            (dict(inclination_deg=181.0), "inclination"),
            (dict(inclination_deg=-1.0), "inclination"),
            (dict(eccentricity=1.0), "eccentricity"),
            (dict(eccentricity=-0.1), "eccentricity"),
            (dict(mean_motion_rev_day=0.0), "mean motion must be positive"),
            (dict(mean_motion_rev_day=150.0), "below 100"),
        ],
    )
    def test_invalid_elements_raise(self, overrides, match):
        with pytest.raises(ValueError, match=match):
            elements(**overrides)

    def test_semi_major_axis_from_mean_motion(self):
        # n = 15.21 rev/day = 1.1060e-3 rad/s; a = (mu / n**2)**(1/3) with the
        # WGS-72 mu of 398600.5 km**3/s**2, recomputed here from the same law at
        # tight tolerance so the assembly is checked, not the algebra.
        source = elements(mean_motion_rev_day=15.21)
        n_rad_s = 15.21 * 2.0 * math.pi / 86400.0
        assert source.semi_major_axis_km == pytest.approx(
            (398600.5 / n_rad_s**2) ** (1.0 / 3.0), rel=1e-14, abs=0.0
        )

    def test_semi_major_axis_decreases_with_mean_motion(self):
        fast = elements(mean_motion_rev_day=15.5).semi_major_axis_km
        slow = elements(mean_motion_rev_day=13.5).semi_major_axis_km
        assert fast < slow

    def test_bstar_too_large_for_the_field_raises(self):
        with pytest.raises(ValueError, match="exponent field"):
            elements(bstar=1e12).to_lines()

    def test_angles_are_reduced_modulo_360_in_the_lines(self):
        parsed = parse_lines(*elements(raan_deg=400.0, argp_deg=-30.0).to_lines())
        assert parsed["raan_deg"] == pytest.approx(40.0, rel=0.0, abs=1e-4)
        assert parsed["argp_deg"] == pytest.approx(330.0, rel=0.0, abs=1e-4)
