"""Pass-finder tests.

The primary known-answer test uses a TLE-free analytic ephemeris: a circular
orbit of radius r lying in the Earth-fixed equatorial plane, passing directly
over a station at (0 deg N, 0 deg E, 0 km).  This removes SGP4 and Earth
rotation from the comparison so the rise/set times have a closed form.

For a spherical Earth of radius Re, a satellite at central (Earth-centred)
angle psi from the station sub-point has elevation el given by
(Wertz & Larson (eds.), "Space Mission Analysis and Design", 3rd ed., Ch. 5
"Space Mission Geometry"):

    tan(el) = (cos psi - Re/r) / sin psi

so the mask elevation el0 is crossed at the central angle

    psi0 = arccos((Re/r) * cos el0) - el0

and, with mean motion n = sqrt(mu/r^3), the pass runs from t_c - psi0/n to
t_c + psi0/n around the zenith crossing t_c.  Because the station sits on the
equator at zero altitude, |r_site| = a_WGS84 exactly, so Re = a is exact.
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from passplanner import Station, TLE, find_passes, find_passes_from_position_fn
from passplanner.fixtures import ISS_2008, NOAA14_1997
from passplanner.frames import MU_EARTH_KM3_S2, WGS84_A_KM

R_ORBIT_KM = 7000.0
N_RAD_S = np.sqrt(MU_EARTH_KM3_S2 / R_ORBIT_KM**3)
T0 = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
T_CULM_S = 1800.0  # zenith crossing 30 min into the window


def _analytic_ephemeris(t: datetime) -> np.ndarray:
    """Circular equatorial Earth-fixed orbit passing over (0 N, 0 E)."""
    dt = (t - T0).total_seconds() - T_CULM_S
    ang = N_RAD_S * dt
    return R_ORBIT_KM * np.array([np.cos(ang), np.sin(ang), 0.0])


def _analytic_half_width_s(mask_deg: float) -> float:
    el0 = np.deg2rad(mask_deg)
    psi0 = np.arccos((WGS84_A_KM / R_ORBIT_KM) * np.cos(el0)) - el0
    return psi0 / N_RAD_S


@pytest.mark.parametrize("mask_deg", [0.0, 5.0, 10.0, 20.0, 40.0])
def test_known_answer_analytic_circular_pass(mask_deg):
    station = Station(name="Equator", lat_deg=0.0, lon_deg=0.0, alt_km=0.0,
                      min_elevation_deg=mask_deg)
    passes = find_passes_from_position_fn(
        _analytic_ephemeris, station, T0, T0 + timedelta(seconds=3600),
        coarse_step_s=10.0, refine_tol_s=0.01, satellite_name="ANALYTIC")
    assert len(passes) == 1
    p = passes[0]
    half = _analytic_half_width_s(mask_deg)
    rise_s = (p.t_rise - T0).total_seconds()
    set_s = (p.t_set - T0).total_seconds()
    assert rise_s == pytest.approx(T_CULM_S - half, abs=0.05)
    assert set_s == pytest.approx(T_CULM_S + half, abs=0.05)
    assert p.duration_s == pytest.approx(2.0 * half, abs=0.1)
    assert p.max_elevation_deg == pytest.approx(90.0, abs=0.01)
    assert (p.t_culminate - T0).total_seconds() == pytest.approx(T_CULM_S, abs=1.0)


def test_analytic_half_width_matches_hand_value_zero_mask():
    # Hand check: psi0 = arccos(6378.137 / 7000) = arccos(0.91116243) = 0.4246999 rad
    # n = sqrt(398600.4418 / 7000^3) = 1.0780076e-3 rad/s
    # half width = 0.4246999 / 1.0780076e-3 = 393.967 s
    half = _analytic_half_width_s(0.0)
    assert np.arccos(WGS84_A_KM / R_ORBIT_KM) == pytest.approx(0.4246999, abs=1e-6)
    assert N_RAD_S == pytest.approx(1.0780076e-3, rel=1e-6)
    assert half == pytest.approx(393.967, abs=0.01)


def test_pass_clipped_by_window_start_is_reported():
    station = Station(name="Equator", lat_deg=0.0, lon_deg=0.0, min_elevation_deg=0.0)
    # Window starts after the analytic rise time -> pass starts at window start.
    t_start = T0 + timedelta(seconds=T_CULM_S - 100.0)
    passes = find_passes_from_position_fn(
        _analytic_ephemeris, station, t_start, T0 + timedelta(seconds=3600),
        coarse_step_s=10.0, satellite_name="ANALYTIC")
    assert len(passes) == 1
    assert passes[0].t_rise == t_start


def test_no_pass_when_satellite_never_rises():
    station = Station(name="Pole", lat_deg=89.0, lon_deg=0.0, min_elevation_deg=20.0)
    passes = find_passes_from_position_fn(
        _analytic_ephemeris, station, T0, T0 + timedelta(seconds=3600),
        coarse_step_s=30.0, satellite_name="ANALYTIC")
    assert passes == []


def test_tle_fixture_checksums_and_parse():
    for tle in (ISS_2008, NOAA14_1997):
        sat = tle.to_satrec()
        err, r, _v = sat.sgp4(sat.jdsatepoch, sat.jdsatepochF)
        assert err == 0
        assert 6500.0 < float(np.linalg.norm(r)) < 8000.0


def test_iss_fixture_passes_are_physically_plausible():
    station = Station(name="Alpine", lat_deg=47.1, lon_deg=10.9, alt_km=2.0,
                      min_elevation_deg=10.0)
    t0 = datetime(2008, 9, 20, 12, 0, tzinfo=timezone.utc)
    passes = find_passes(ISS_2008, station, t0, t0 + timedelta(days=1))
    assert 3 <= len(passes) <= 8  # a mid-latitude site sees a few ISS passes a day
    for p in passes:
        # LEO passes above a 10 deg mask last roughly 2-10 minutes.
        assert 60.0 < p.duration_s < 700.0
        assert p.max_elevation_deg >= 10.0
        assert p.t_rise < p.t_culminate < p.t_set
        assert p.satellite == "ISS (ZARYA)"


def test_find_passes_accepts_raw_line_pair():
    station = Station(name="Alpine", lat_deg=47.1, lon_deg=10.9, min_elevation_deg=10.0)
    t0 = datetime(2008, 9, 20, 12, 0, tzinfo=timezone.utc)
    a = find_passes(ISS_2008, station, t0, t0 + timedelta(hours=6))
    b = find_passes((ISS_2008.line1, ISS_2008.line2), station, t0, t0 + timedelta(hours=6),
                    satellite_name="ISS (ZARYA)")
    assert [p.t_rise for p in a] == [p.t_rise for p in b]


def test_pass_overlap_logic():
    station = Station(name="S", lat_deg=0.0, lon_deg=0.0)
    from passplanner.passes import Pass
    t = T0
    p1 = Pass("A", station, t, t + timedelta(seconds=100), t + timedelta(seconds=50), 45.0)
    p2 = Pass("A", station, t + timedelta(seconds=150), t + timedelta(seconds=250),
              t + timedelta(seconds=200), 30.0)
    assert not p1.overlaps(p2)
    assert p1.overlaps(p2, setup_time_s=60.0)
    assert p1.overlaps(p1)


# --- input validation -------------------------------------------------------

def test_tle_rejects_wrong_length():
    with pytest.raises(ValueError, match="69 characters"):
        TLE("X", "1 too short", ISS_2008.line2)


def test_tle_rejects_bad_checksum():
    bad = ISS_2008.line1[:68] + ("0" if ISS_2008.line1[68] != "0" else "1")
    with pytest.raises(ValueError, match="checksum"):
        TLE("X", bad, ISS_2008.line2)


def test_tle_rejects_swapped_lines():
    with pytest.raises(ValueError, match="must start with"):
        TLE("X", ISS_2008.line2, ISS_2008.line1)


def test_find_passes_rejects_inverted_window():
    station = Station(name="S", lat_deg=0.0, lon_deg=0.0)
    with pytest.raises(ValueError, match="must be after"):
        find_passes(ISS_2008, station, T0, T0 - timedelta(hours=1))


@pytest.mark.parametrize("kwargs", [
    {"coarse_step_s": 0.0}, {"coarse_step_s": -5.0}, {"refine_tol_s": 0.0},
    {"min_elev_deg": -1.0}, {"min_elev_deg": 90.0},
])
def test_find_passes_rejects_bad_parameters(kwargs):
    station = Station(name="S", lat_deg=0.0, lon_deg=0.0)
    with pytest.raises(ValueError):
        find_passes(ISS_2008, station, T0, T0 + timedelta(hours=1), **kwargs)


def test_find_passes_rejects_bad_tle_sequence():
    station = Station(name="S", lat_deg=0.0, lon_deg=0.0)
    with pytest.raises(ValueError, match="exactly"):
        find_passes([ISS_2008.line1], station, T0, T0 + timedelta(hours=1))
