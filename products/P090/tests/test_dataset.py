"""The generator: determinism, the Kozai inverse, and the corruption families."""

from __future__ import annotations

import math

import numpy as np
import pytest
from sgp4.api import Satrec

from tlecheck.dataset import (
    AGREEING_SOURCE_SCALE,
    CORRUPTION_FAMILIES,
    DIVERGING_SOURCE_SCALE,
    ELEMENT_SPANS,
    FAMILIES,
    J2,
    JITTER,
    XKE,
    apsidal_precession_deg_per_day,
    corrupt_checksum_digit,
    corrupt_data_digit,
    corrupt_eccentricity,
    corrupt_field_shift,
    corrupt_sign_flip,
    generate,
    kozai_mean_motion,
    nodal_regression_deg_per_day,
    reference_epoch_jd,
    split,
    unkozai_mean_motion,
)
from tlecheck.elements import checksum, parse_tle

from conftest import REFERENCE_LINE1, REFERENCE_LINE2


def test_constants_match_the_propagator(reference_lines):
    """The generator must use the propagator's own constants, not a copy."""
    satrec = Satrec.twoline2rv(*reference_lines)
    assert XKE == satrec.xke
    assert J2 == satrec.j2


def test_unkozai_matches_the_published_algorithm(reference_lines):
    satrec = Satrec.twoline2rv(*reference_lines)
    mine = unkozai_mean_motion(satrec.no_kozai, satrec.ecco, satrec.inclo)
    # The propagator's own mean-motion derivative mdot is the un-Kozai'd mean
    # motion plus first-order J2 secular terms, so it is close but not equal;
    # the agreement is bounded rather than asserted.
    assert mine < satrec.no_kozai
    assert abs(mine - satrec.mdot) / satrec.mdot < 2.0e-3


def test_kozai_inverse_round_trip_is_exact(reference_lines):
    satrec = Satrec.twoline2rv(*reference_lines)
    brouwer = unkozai_mean_motion(satrec.no_kozai, satrec.ecco, satrec.inclo)
    back = kozai_mean_motion(brouwer, satrec.ecco, satrec.inclo)
    assert back == pytest.approx(satrec.no_kozai, rel=0.0, abs=0.0)


@pytest.mark.parametrize(
    ("mean_motion_rev_day", "eccentricity", "inclination_deg"),
    [
        (15.5, 0.0008, 51.6),
        (14.6, 0.0012, 98.2),
        (2.0056, 0.0019, 55.0),
        (1.00272, 0.0002, 0.06),
        (2.006, 0.72, 63.4),
    ],
)
def test_kozai_round_trip_across_the_families(
    mean_motion_rev_day, eccentricity, inclination_deg
):
    kozai = 2.0 * math.pi * mean_motion_rev_day / 1440.0
    inclination = math.radians(inclination_deg)
    brouwer = unkozai_mean_motion(kozai, eccentricity, inclination)
    assert kozai_mean_motion(brouwer, eccentricity, inclination) == pytest.approx(
        kozai, rel=1.0e-14, abs=0.0
    )


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_unkozai_rejects_bad_mean_motion(bad):
    with pytest.raises(ValueError, match="kozai_rad_per_min"):
        unkozai_mean_motion(bad, 0.001, 0.9)


@pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5])
def test_unkozai_rejects_bad_eccentricity(bad):
    with pytest.raises(ValueError, match="eccentricity"):
        unkozai_mean_motion(0.0676, bad, 0.9)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
def test_kozai_rejects_bad_input(bad):
    with pytest.raises(ValueError, match="brouwer_rad_per_min"):
        kozai_mean_motion(bad, 0.001, 0.9)


def test_kozai_rejects_zero_iterations():
    with pytest.raises(ValueError, match="iterations"):
        kozai_mean_motion(0.0676, 0.001, 0.9, iterations=0)


def test_nodal_regression_against_the_propagator(reference_lines):
    """Independent cross-check: equation (16) against sgp4's own ``nodedot``.

    Two implementations of the same first-order secular theory. ``nodedot`` is
    in radians per minute and carries higher-order terms equation (16) does
    not, so the agreement is bounded at a few per cent rather than asserted.
    """
    satrec = Satrec.twoline2rv(*reference_lines)
    theirs = math.degrees(satrec.nodedot) * 1440.0
    mine = nodal_regression_deg_per_day(15.50103472, 0.0007417, 51.6439)
    assert mine < 0.0
    assert theirs < 0.0
    assert abs(mine - theirs) / abs(theirs) < 0.05


def test_apsidal_precession_against_the_propagator(reference_lines):
    """The same independent cross-check for equation (17) against ``argpdot``."""
    satrec = Satrec.twoline2rv(*reference_lines)
    theirs = math.degrees(satrec.argpdot) * 1440.0
    mine = apsidal_precession_deg_per_day(15.50103472, 0.0007417, 51.6439)
    assert mine > 0.0
    assert theirs > 0.0
    assert abs(mine - theirs) / abs(theirs) < 0.05


def test_apsidal_precession_vanishes_at_the_critical_inclination():
    # 4 - 5 sin^2 i is zero at i = arcsin(sqrt(4/5)) = 63.4349 degrees.
    rate = apsidal_precession_deg_per_day(2.006, 0.72, 63.4349488)
    assert abs(rate) < 1.0e-6


def test_nodal_regression_vanishes_at_polar_inclination():
    assert abs(nodal_regression_deg_per_day(14.6, 0.001, 90.0)) < 1.0e-12


def test_generation_is_deterministic():
    first = split("test", objects=24, history_length=6)
    second = split("test", objects=24, history_length=6)
    assert [r.candidate.line1 for r in first] == [r.candidate.line1 for r in second]
    assert [r.candidate.line2 for r in first] == [r.candidate.line2 for r in second]
    assert [r.corruption for r in first] == [r.corruption for r in second]


def test_splits_use_disjoint_object_blocks():
    train = {r.satnum for r in split("train", objects=24, history_length=6)}
    test = {r.satnum for r in split("test", objects=24, history_length=6)}
    calibration = {r.satnum for r in split("calibration", objects=24, history_length=6)}
    assert train.isdisjoint(test)
    assert train.isdisjoint(calibration)
    assert test.isdisjoint(calibration)


def test_splits_differ_in_content():
    train = split("train", objects=24, history_length=6)
    test = split("test", objects=24, history_length=6)
    assert [r.candidate.line2 for r in train] != [r.candidate.line2 for r in test]


def test_unknown_split_is_rejected():
    with pytest.raises(ValueError, match="unknown split"):
        split("holdout", objects=8)


def test_every_generated_line_has_a_valid_checksum(small_records):
    for record in small_records:
        if record.corruption in {"checksum_digit", "data_digit"}:
            continue
        for element_set in (*record.history, record.candidate, record.reference):
            for line in (element_set.line1, element_set.line2):
                assert int(line[68]) == checksum(line)


def test_every_uncorrupted_candidate_parses(small_records):
    for record in small_records:
        if record.corruption != "none":
            continue
        assert record.candidate.faults == ()


def test_corruption_families_are_the_eight_declared():
    assert len(CORRUPTION_FAMILIES) == 8
    assert CORRUPTION_FAMILIES[0] == "none"


def test_all_families_are_generated():
    records = split("test", objects=240, history_length=6)
    seen = {record.corruption for record in records}
    assert seen == set(CORRUPTION_FAMILIES)


def test_six_orbit_families():
    assert len(FAMILIES) == 6
    names = {family.name for family in FAMILIES}
    assert "heo_critical_inclination" in names
    assert "geo_stationary" in names


def test_critical_inclination_family_sits_at_the_critical_inclination():
    family = next(f for f in FAMILIES if f.name == "heo_critical_inclination")
    assert abs(family.inclination_deg - 63.435) < 0.1


def test_element_spans_cover_only_numeric_element_fields():
    assert ELEMENT_SPANS[1] == ((19, 32), (34, 43), (45, 52), (54, 61))
    assert ELEMENT_SPANS[2] == ((9, 16), (18, 25), (27, 33), (35, 42), (44, 51), (53, 63))


def test_jitter_entries_are_the_documented_ones():
    assert set(JITTER) == {
        "inclination_deg",
        "raan_deg",
        "arg_perigee_deg",
        "mean_anomaly_deg",
        "eccentricity",
        "mean_motion_relative",
        "bstar_relative",
    }


def test_source_scales_are_ordered():
    assert AGREEING_SOURCE_SCALE[1] < DIVERGING_SOURCE_SCALE[0]


def test_checksum_digit_corruption_changes_only_column_sixty_nine():
    base = parse_tle(REFERENCE_LINE1, REFERENCE_LINE2)
    rng = np.random.default_rng(1)
    corrupted = corrupt_checksum_digit(base, rng)
    before = base.line1 + base.line2
    after = corrupted.line1 + corrupted.line2
    changed = [
        index
        for index, (a, b) in enumerate(zip(before, after, strict=True))
        if a != b
    ]
    assert all(index % 69 == 68 for index in changed)
    assert len(changed) == 1


def test_data_digit_corruption_breaks_the_checksum():
    base = parse_tle(REFERENCE_LINE1, REFERENCE_LINE2)
    rng = np.random.default_rng(3)
    corrupted = corrupt_data_digit(base, rng)
    mismatch = (
        corrupted.checksum_line1_stated != corrupted.checksum_line1_computed
        or corrupted.checksum_line2_stated != corrupted.checksum_line2_computed
    )
    assert mismatch


def test_field_shift_corruption_keeps_the_checksum_valid():
    base = parse_tle(REFERENCE_LINE1, REFERENCE_LINE2)
    for seed in range(12):
        corrupted = corrupt_field_shift(base, np.random.default_rng(seed))
        assert int(corrupted.line1[68]) == checksum(corrupted.line1)
        assert int(corrupted.line2[68]) == checksum(corrupted.line2)


def test_sign_flip_corruption_flips_bstar_and_rechecksums():
    base = parse_tle(REFERENCE_LINE1, REFERENCE_LINE2)
    corrupted = corrupt_sign_flip(base, np.random.default_rng(0))
    assert corrupted.bstar == pytest.approx(-base.bstar, rel=0.0, abs=0.0)
    assert int(corrupted.line1[68]) == checksum(corrupted.line1)


def test_sign_flip_is_an_involution_on_the_value():
    base = parse_tle(REFERENCE_LINE1, REFERENCE_LINE2)
    rng = np.random.default_rng(0)
    once = corrupt_sign_flip(base, rng)
    twice = corrupt_sign_flip(once, rng)
    assert twice.bstar == pytest.approx(base.bstar, rel=0.0, abs=0.0)


def test_eccentricity_corruption_moves_it_far():
    base = parse_tle(REFERENCE_LINE1, REFERENCE_LINE2)
    for seed in range(8):
        corrupted = corrupt_eccentricity(base, np.random.default_rng(seed))
        assert corrupted.eccentricity > 0.29
        assert int(corrupted.line2[68]) == checksum(corrupted.line2)


def test_stale_epoch_candidate_is_a_genuine_earlier_element_set():
    records = split("test", objects=240, history_length=12)
    record = next(r for r in records if r.corruption == "stale_epoch")
    assert record.candidate.faults == ()
    assert record.candidate.line1 in {h.line1 for h in record.history}


def test_cross_source_record_carries_another_source():
    records = split("test", objects=240, history_length=12)
    record = next(r for r in records if r.corruption == "cross_source_divergence")
    assert len(record.other_sources) == 1
    assert record.other_sources[0].satnum == record.candidate.satnum


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"objects": 0}, "objects"),
        ({"corruption_rate": 1.5}, "corruption_rate"),
        ({"corruption_rate": -0.1}, "corruption_rate"),
        ({"history_length": -1}, "history_length"),
        ({"cadence_days": 0.0}, "cadence_days"),
        ({"cadence_days": float("nan")}, "cadence_days"),
    ],
)
def test_generate_validates_its_arguments(kwargs, message):
    with pytest.raises(ValueError, match=message):
        generate(**kwargs)


def test_corruption_rate_zero_gives_only_clean_records():
    records = generate(objects=24, corruption_rate=0.0, history_length=6)
    assert {record.corruption for record in records} == {"none"}


def test_corruption_rate_one_gives_no_clean_records():
    records = generate(objects=24, corruption_rate=1.0, history_length=6)
    assert "none" not in {record.corruption for record in records}


def test_reference_epoch_is_a_julian_date():
    assert 2.4e6 < reference_epoch_jd() < 2.5e6


def test_history_is_ordered_oldest_first(clean_record):
    from tlecheck.epoch import epoch_julian_date

    times = [epoch_julian_date(entry) for entry in clean_record.history]
    assert times == sorted(times)


def test_use_epoch_is_after_the_candidate_epoch(clean_record):
    from tlecheck.epoch import epoch_julian_date

    assert clean_record.use_jd > epoch_julian_date(clean_record.candidate)
