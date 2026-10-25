"""Reproduction of the published SGP4 verification vectors."""

from __future__ import annotations

import numpy as np

from orbitdrift import vectors


def test_the_published_files_are_present_in_the_installed_distribution() -> None:
    """Both files must be readable, or the whole L3 claim has no evidence."""
    assert vectors.sgp4_distribution_version() != "unknown"
    tle = vectors._read(vectors.TLE_FILE)
    expected = vectors._read(vectors.EXPECTED_FILE)
    assert "Verification test cases" in tle
    assert len(expected.splitlines()) == 700


def test_thirty_three_cases_are_parsed_with_the_published_schedules() -> None:
    cases = vectors.published_cases()
    assert len(cases) == 33
    first = cases[0]
    assert first.satnum == 5
    assert first.schedule_min == (0.0, 4320.0, 360.0)
    assert first.times_min[0] == 0.0
    assert first.times_min[-1] == 4320.0


def test_the_catalogue_number_appears_twice_and_blocks_are_matched_positionally() -> None:
    """Catalogue 20413 is listed twice with different schedules.

    This is the defect this parser was written to avoid. Keying the expected
    blocks by catalogue number merges the two and compares each element set
    against the union of both blocks, which inflated the row count from 667 to
    763 and silently compared one satellite against another test case's rows.
    """
    cases = vectors.published_cases()
    numbers = [case.satnum for case in cases]
    assert numbers.count(20413) == 2
    duplicates = [case for case in cases if case.satnum == 20413]
    assert duplicates[0].schedule_min != duplicates[1].schedule_min
    assert duplicates[0].times_min.size != duplicates[1].times_min.size


def test_every_published_row_is_reproduced_to_the_files_own_precision() -> None:
    """The published file prints 8 decimals of a km and 9 of a km/s.

    So the comparison cannot be tighter than about 5e-9 km and 5e-10 km/s from
    the file's own rounding, and the measured worst disagreement is a few
    times that. The tolerances here, 1e-6 km and 1e-8 km/s, are an order above
    the measured values and the measured values themselves are published in
    ``validation/validate_sgp4_vectors.py``.
    """
    total_compared = 0
    total_skipped = 0
    worst_position = 0.0
    worst_velocity = 0.0
    for case in vectors.published_cases():
        result = vectors.compare_to_published(case)
        total_compared += result.rows_compared
        total_skipped += result.rows_skipped_error_code
        worst_position = max(worst_position, result.max_position_error_km)
        worst_velocity = max(worst_velocity, result.max_velocity_error_km_s)
    assert total_compared == 666
    assert total_skipped == 1
    assert worst_position < 1e-6
    assert worst_velocity < 1e-8


def test_the_skipped_row_is_skipped_for_a_named_reason() -> None:
    """One row is skipped, and it is skipped because SGP4 returns an error code.

    ``tcppver.out`` handles a satellite whose propagation fails at zero
    minutes by repeating the previous satellite's last data line, a quirk of
    the reference program. Comparing that row would report a spurious
    disagreement of thousands of kilometres. The parser must skip exactly one
    row and must name the error code, not drop it silently.
    """
    skipped = [
        result
        for result in (vectors.compare_to_published(case) for case in vectors.published_cases())
        if result.rows_skipped_error_code
    ]
    assert len(skipped) == 1
    assert skipped[0].rows_skipped_error_code == 1
    assert skipped[0].error_codes == (3,)


def test_a_deliberately_wrong_element_set_does_not_reproduce_the_vectors() -> None:
    """A negative control: the comparison must be able to fail.

    The inclination of catalogue 5 is changed from 34.2682 to 44.2682 degrees,
    a one-character edit in column 9 of line 2 that keeps the line length and
    the field layout. The checksum is not recomputed, which ``twoline2rv``
    tolerates. The resulting disagreement must be enormous, which is what
    makes the passing result above evidence rather than a coincidence.
    """
    case = vectors.published_cases()[0]
    assert case.line2[8:16] == " 34.2682"
    broken_line2 = case.line2[:8] + " 44.2682" + case.line2[16:]
    broken = vectors.PublishedCase(
        satnum=case.satnum,
        line1=case.line1,
        line2=broken_line2,
        schedule_min=case.schedule_min,
        times_min=case.times_min,
        positions_km=case.positions_km,
        velocities_km_s=case.velocities_km_s,
    )
    result = vectors.compare_to_published(broken)
    assert result.rows_compared == case.times_min.size
    assert result.max_position_error_km > 100.0


def test_published_positions_are_physically_plausible() -> None:
    """Every published row must be outside the Earth and below lunar distance.

    An independently derived bound on the parsed data itself, which catches a
    column-offset error in the parser that a self-consistent comparison would
    not: a parser reading the wrong seven columns would still agree with a
    propagation driven from the same wrong columns if the error were in the
    time column alone.
    """
    for case in vectors.published_cases():
        if case.positions_km.size == 0:
            continue
        radius = np.linalg.norm(case.positions_km, axis=-1)
        assert float(radius.min()) > 6000.0
        assert float(radius.max()) < 400000.0
        speed = np.linalg.norm(case.velocities_km_s, axis=-1)
        assert float(speed.min()) > 0.5
        assert float(speed.max()) < 15.0


def test_published_states_satisfy_the_conic_inequality() -> None:
    """``a > r/2`` must hold for every bound published state.

    An independently derived inequality, not a comparison with a propagation.
    For an ellipse ``r = a(1 - e cos E) <= a(1 + e) < 2a``, so the osculating
    semi-major axis formed from the published position and velocity by
    ``a = -mu / (2 E)`` with ``E = v^2/2 - mu/r`` must exceed half the radius.
    It is sharp against the errors a parser can make: a position and velocity
    column swap, or a metre-for-kilometre scaling, moves the specific energy by
    orders of magnitude and breaks it.

    Measured: all 667 published rows are bound and all 667 satisfy the
    inequality. The unbound count is asserted to be zero rather than
    assumed, because the verification set deliberately contains pathological
    cases -- deep-space resonances, decaying orbits, a state the Lyddane
    correction drives to a negative eccentricity -- and a release that pushed
    one of them over the escape energy would fail here rather than silently
    change a published number.
    """
    from orbitdrift.constants import WGS72

    bound = 0
    unbound = 0
    for case in vectors.published_cases():
        if case.positions_km.size == 0:
            continue
        radius = np.linalg.norm(case.positions_km, axis=-1)
        speed = np.linalg.norm(case.velocities_km_s, axis=-1)
        energy = 0.5 * speed**2 - WGS72.mu_km3_s2 / radius
        for index in range(radius.size):
            if energy[index] >= 0.0:
                unbound += 1
                continue
            axis = -WGS72.mu_km3_s2 / (2.0 * energy[index])
            assert axis > 0.5 * radius[index]
            bound += 1
    assert bound + unbound == 667
    assert bound == 667
    assert unbound == 0
