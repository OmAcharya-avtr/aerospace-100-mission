"""Pass geometry: provenance, known answers, edge cases and invariants."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sgp4.propagation import gstime

from passyield.geometry import (
    TLE_LINE1,
    TLE_LINE2,
    WGS84_A_KM,
    GroundStation,
    PassWindow,
    campaign_epoch,
    find_passes,
    gmst_rad,
    satellite,
    slant_range_spherical_km,
    teme_to_ecef,
    tle_source_path,
    topocentric,
    topocentric_profile,
)


def test_committed_tle_lines_match_the_installed_verification_file():
    """The two committed lines appear verbatim in the sgp4 package's own file.

    This is the provenance check: the element set is claimed to be catalogue
    object 06251 from the official SGP4 verification set distributed with the
    `sgp4` package, and that claim is checked against the installed file
    character for character rather than asserted in a docstring.
    """
    source = tle_source_path()
    assert source.is_file(), f"the installed sgp4 package has no {Path(source).name}"
    text = source.read_text()
    assert TLE_LINE1 in text
    assert TLE_LINE2 in text


def test_gmst_agrees_with_the_sgp4_library_implementation():
    """Our IAU-1982 GMST polynomial against the one inside `sgp4`.

    Two independent implementations of the same published polynomial. They are
    compared to a tolerance, never bit-exactly: the two accumulate the Horner
    terms in a different order.
    """
    for julian in (2451545.0, 2453911.5, 2460000.25, 2470000.75):
        ours = float(gmst_rad(julian))
        theirs = float(gstime(julian))
        assert ours == pytest.approx(theirs, rel=1.0e-9, abs=1.0e-9)


def test_sgp4_reproduces_the_official_verification_output():
    """Known answer: the first line of tcppver.out for catalogue 06251.

    The official verification output shipped with the `sgp4` package records,
    for object 06251 at tsince = 0 minutes, the TEME position

        x = 3988.31022699 km   y = 5498.96657235 km   z = 0.90055879 km

    Those three numbers are read from the installed ``tcppver.out`` rather
    than written into this test, so the test cannot drift from the file it
    claims to check.
    """
    source = tle_source_path().parent / "tcppver.out"
    assert source.is_file(), "the installed sgp4 package has no tcppver.out"
    lines = source.read_text().splitlines()
    start = next(index for index, line in enumerate(lines) if line.startswith("6251 xx"))
    fields = lines[start + 1].split()
    expected = [float(value) for value in fields[1:4]]
    error, position, _velocity = satellite().sgp4_tsince(float(fields[0]))
    assert error == 0
    for computed, reference in zip(position, expected, strict=True):
        assert computed == pytest.approx(reference, rel=0.0, abs=1.0e-6)


def test_station_ecef_hand_calculation_at_the_equator():
    """Known answer, hand-calculated: a station at 0 N, 0 E, 0 m.

    At the equator the prime-vertical radius of curvature is
    ``N = a / sqrt(1 - e2 * sin(0)**2) = a``, so the site vector is
    ``(a, 0, 0)`` with ``a = 6378.137`` km exactly. The z component is
    ``(N*(1-e2) + h) * sin(0) = 0``.
    """
    station = GroundStation("EQ", 0.0, 0.0, 0.0, 10.0, 20.0, 0.4)
    vector = station.ecef_km()
    assert vector[0] == pytest.approx(WGS84_A_KM, rel=0.0, abs=1.0e-12)
    assert vector[1] == pytest.approx(0.0, rel=0.0, abs=1.0e-12)
    assert vector[2] == pytest.approx(0.0, rel=0.0, abs=1.0e-12)


def test_station_ecef_hand_calculation_at_the_pole():
    """Known answer, hand-calculated: a station at 90 N, 0 E, 0 m.

    At the pole ``sin(phi) = 1`` so ``N = a / sqrt(1 - e2)`` and the z
    component is ``N * (1 - e2) = a * sqrt(1 - e2)``, which is the polar
    semi-axis ``b``. With ``a = 6378.137`` km and
    ``e2 = 6.69437999014e-3``, ``b = 6378.137 * sqrt(0.99330562000986)``
    which is 6356.752314245 km -- the WGS-84 polar radius.
    """
    station = GroundStation("PO", 90.0, 0.0, 0.0, 10.0, 20.0, 0.4)
    vector = station.ecef_km()
    polar = WGS84_A_KM * math.sqrt(1.0 - 6.69437999014e-3)
    assert polar == pytest.approx(6356.752314245, rel=0.0, abs=1.0e-6)
    assert vector[2] == pytest.approx(polar, rel=0.0, abs=1.0e-9)
    assert float(np.hypot(vector[0], vector[1])) == pytest.approx(0.0, rel=0.0, abs=1.0e-9)


def test_topocentric_zenith_known_answer():
    """A target directly above a station is at 90 degrees elevation.

    Station at 0 N, 0 E, 0 m; target on the x axis at ``a + 500`` km. The
    station-to-target vector is ``(500, 0, 0)`` which is exactly the local up
    direction, so elevation is 90 degrees and the slant range is 500 km.
    """
    station = GroundStation("EQ", 0.0, 0.0, 0.0, 10.0, 20.0, 0.4)
    target = np.asarray([[WGS84_A_KM + 500.0, 0.0, 0.0]])
    elevation, _azimuth, slant = topocentric(target, station)
    assert float(elevation[0]) == pytest.approx(90.0, rel=0.0, abs=1.0e-9)
    assert float(slant[0]) == pytest.approx(500.0, rel=0.0, abs=1.0e-9)


def test_slant_range_spherical_at_the_zenith_is_the_altitude():
    """At 90 degrees the closed form reduces to the altitude exactly.

    ``d(90) = R[sqrt(ratio**2 - 0) - 1] = R*ratio - R = h``.
    """
    assert float(slant_range_spherical_km(90.0, 500.0)) == pytest.approx(
        500.0, rel=1.0e-12, abs=1.0e-9
    )


def test_slant_range_spherical_at_the_horizon_known_answer():
    """At 0 degrees the closed form is the tangent length to the orbit sphere.

    ``d(0) = R[sqrt(ratio**2 - 1)] = sqrt((R+h)**2 - R**2)``. With
    ``R = 6378.137`` km and ``h = 500`` km that is
    ``sqrt(6878.137**2 - 6378.137**2) = sqrt(47303568.2 - 40680631.1)``
    which the closed form must reproduce.
    """
    expected = math.sqrt((WGS84_A_KM + 500.0) ** 2 - WGS84_A_KM**2)
    assert float(slant_range_spherical_km(0.0, 500.0)) == pytest.approx(
        expected, rel=1.0e-12, abs=0.0
    )


@given(
    elevation=st.floats(min_value=0.0, max_value=90.0),
    altitude=st.floats(min_value=200.0, max_value=2000.0),
)
@settings(max_examples=120, deadline=None)
def test_slant_range_is_monotone_decreasing_in_elevation(elevation, altitude):
    """Property: the closed-form slant range falls as elevation rises."""
    lower = float(slant_range_spherical_km(elevation, altitude))
    upper = float(slant_range_spherical_km(min(elevation + 1.0, 90.0), altitude))
    assert upper <= lower + 1.0e-9


def test_windows_are_chronological_and_above_the_mask(windows, station):
    """Every window clears the mask and the windows do not overlap."""
    for window in windows:
        assert window.max_elevation_deg >= station.mask_deg
        assert window.duration_s > 0.0
        assert 0.0 <= window.rise_fr < 1.0
    absolute = [window.rise_jd + window.rise_fr for window in windows]
    assert absolute == sorted(absolute)


def test_window_endpoints_sit_on_the_mask(satrec, station, long_window):
    """The bisection roots are mask crossings to within the clamp tolerance."""
    elevation, _azimuth, _slant = topocentric_profile(
        satrec, station, long_window, np.asarray([0.0, long_window.duration_s])
    )
    for value in elevation:
        assert abs(float(value) - station.mask_deg) < 1.0e-3


def test_no_window_for_a_station_the_orbit_never_reaches(satrec):
    """Edge case: a pass that never rises above the mask.

    The committed element set is inclined 58.06 degrees, so a station at
    85 degrees north never sees it above a 10 degree mask. ``find_passes``
    returns an empty list -- a documented return, not an exception and not a
    fabricated window.
    """
    polar = GroundStation("POLAR", 85.0, 0.0, 0.0, 10.0, 21.0, 0.4)
    epoch_jd, epoch_fr = campaign_epoch(satrec)
    assert find_passes(satrec, polar, epoch_jd, epoch_fr, 2.0, coarse_step_s=30.0) == []


@pytest.mark.parametrize(
    ("latitude", "longitude", "mask", "gt", "attenuation"),
    [
        (91.0, 0.0, 10.0, 20.0, 0.4),
        (-91.0, 0.0, 10.0, 20.0, 0.4),
        (0.0, 400.0, 10.0, 20.0, 0.4),
        (0.0, 0.0, 9.9, 20.0, 0.4),
        (0.0, 0.0, 90.0, 20.0, 0.4),
        (0.0, 0.0, 10.0, 20.0, -0.1),
    ],
)
def test_station_rejects_out_of_range_inputs(latitude, longitude, mask, gt, attenuation):
    """Input validation: each bad value raises ValueError with a reason."""
    with pytest.raises(ValueError):
        GroundStation("BAD", latitude, longitude, 0.0, mask, gt, attenuation)


def test_station_rejects_a_non_finite_figure_of_merit():
    """Input validation: a non-finite G/T is refused."""
    with pytest.raises(ValueError, match="finite"):
        GroundStation("BAD", 0.0, 0.0, 0.0, 10.0, float("nan"), 0.4)


@pytest.mark.parametrize(
    ("span", "step", "iterations"),
    [(0.0, 10.0, 40), (-1.0, 10.0, 40), (1.0, 0.0, 40), (1.0, 10.0, 0)],
)
def test_find_passes_rejects_out_of_range_settings(satrec, station, span, step, iterations):
    """Input validation on the scan settings."""
    epoch_jd, epoch_fr = campaign_epoch(satrec)
    with pytest.raises(ValueError):
        find_passes(satrec, station, epoch_jd, epoch_fr, span, step, iterations)


def test_slant_range_spherical_rejects_a_non_positive_altitude():
    """Input validation: the closed form needs a positive altitude."""
    with pytest.raises(ValueError, match="altitude_km"):
        slant_range_spherical_km(30.0, 0.0)


def test_teme_to_ecef_preserves_the_vector_norm(satrec):
    """A rotation about the pole changes no length.

    The strongest available statement about the frame change that does not
    require an independent frame implementation.
    """
    epoch_jd, epoch_fr = campaign_epoch(satrec)
    error, position, _velocity = satrec.sgp4(epoch_jd, epoch_fr)
    assert error == 0
    teme = np.asarray([position], dtype=float)
    ecef = teme_to_ecef(teme, np.asarray([epoch_jd]), np.asarray([epoch_fr]))
    assert float(np.linalg.norm(ecef)) == pytest.approx(
        float(np.linalg.norm(teme)), rel=1.0e-12, abs=0.0
    )


def test_pass_window_is_frozen(long_window):
    """A window is immutable: it is evidence about a contact, not a scratchpad."""
    with pytest.raises(Exception):  # noqa: B017 - dataclasses raise FrozenInstanceError
        long_window.duration_s = 1.0  # type: ignore[misc]


def test_pass_window_fields_round_trip(long_window):
    """A window reconstructed field by field equals the original."""
    rebuilt = PassWindow(
        station=long_window.station,
        rise_jd=long_window.rise_jd,
        rise_fr=long_window.rise_fr,
        duration_s=long_window.duration_s,
        max_elevation_deg=long_window.max_elevation_deg,
        min_slant_range_km=long_window.min_slant_range_km,
    )
    assert rebuilt == long_window
