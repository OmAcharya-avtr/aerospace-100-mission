"""TLE handling, Walker generation, propagation and epoch guarding."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from constellink.constellation import (
    TLE,
    CircularOrbit,
    Constellation,
    GroundStation,
    Satellite,
    TleEpochError,
    time_grid,
    walker_delta,
)
from constellink.frames import MU_EARTH_KM3_S2, WGS84_A_KM

EPOCH = datetime(2026, 4, 1, tzinfo=UTC)

# ISS element set from the SGP4 verification file shipped with the sgp4
# package is not used here; this is the TEME example, satellite 00005, whose
# lines are reproduced in validation/validate_sgp4_vector.py from the
# installed package. A short, checksummed pair is enough for the TLE tests.
LINE1 = "1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753"
LINE2 = "2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667"


def test_tle_accepts_a_valid_pair():
    tle = TLE("TEST", LINE1, LINE2)
    assert tle.to_satrec().satnum == 5


def test_tle_rejects_wrong_length():
    with pytest.raises(ValueError, match="69 characters"):
        TLE("TEST", LINE1[:-1], LINE2)


def test_tle_rejects_wrong_leading_digit():
    with pytest.raises(ValueError, match="must start with"):
        TLE("TEST", LINE2, LINE1)


def test_tle_rejects_bad_checksum():
    bad = LINE1[:68] + ("0" if LINE1[68] != "0" else "1")
    with pytest.raises(ValueError, match="checksum"):
        TLE("TEST", bad, LINE2)


def test_satellite_from_tle_and_epoch_reconstruction():
    sat = Satellite.from_tle(TLE("TEST", LINE1, LINE2))
    # TLE epoch field 00179.78495062 is day 179.78495062 of year 2000, i.e.
    # 2000-06-27 (day 179) at 0.78495062 of a day = 18:50:19.73 UTC.
    assert sat.epoch.year == 2000
    assert sat.epoch.month == 6
    assert sat.epoch.day == 27
    assert sat.epoch.hour == 18


def test_satellite_rejects_bad_parameters():
    satrec = TLE("TEST", LINE1, LINE2).to_satrec()
    with pytest.raises(ValueError):
        Satellite(name="", satrec=satrec)
    with pytest.raises(ValueError):
        Satellite(name="X", satrec=satrec, max_epoch_age_days=0.0)


def test_propagate_raises_far_from_epoch():
    sat = Satellite.from_tle(TLE("TEST", LINE1, LINE2), max_epoch_age_days=7.0)
    with pytest.raises(TleEpochError, match="element epoch"):
        sat.propagate([EPOCH])


def test_propagate_escape_hatch_works():
    sat = Satellite.from_tle(TLE("TEST", LINE1, LINE2), max_epoch_age_days=7.0)
    r, v = sat.propagate([sat.epoch + timedelta(hours=1)], check_epoch=False)
    assert r.shape == (1, 3)
    assert np.all(np.isfinite(r))
    assert np.all(np.isfinite(v))


def test_propagate_empty_times_returns_empty_arrays():
    sat = Satellite.from_tle(TLE("TEST", LINE1, LINE2))
    r, v = sat.propagate([])
    assert r.shape == (0, 3)
    assert v.shape == (0, 3)


def test_walker_pattern_names_and_count():
    const = walker_delta(12, 3, 1, 53.0, 550.0, EPOCH, max_epoch_age_days=2.0)
    assert const.n_sat == 12
    assert const.satellites[0].name == "W00-00"
    assert const.satellites[-1].name == "W02-03"
    assert const.name == "Walker-53:12/3/1"


@pytest.mark.parametrize("args", [
    (25, 4, 1, 53.0, 550.0),    # not a multiple of n_planes
    (24, 0, 0, 53.0, 550.0),    # zero planes
    (24, 4, 4, 53.0, 550.0),    # phasing out of range
    (24, 4, 1, 200.0, 550.0),   # inclination out of range
    (24, 4, 1, 53.0, 0.0),      # zero altitude
])
def test_walker_rejects_bad_parameters(args):
    with pytest.raises(ValueError):
        walker_delta(*args, EPOCH)


def test_walker_mean_motion_matches_two_body_relation():
    const = walker_delta(2, 1, 0, 53.0, 550.0, EPOCH, max_epoch_age_days=2.0)
    a_km = WGS84_A_KM + 550.0
    expected_rad_min = np.sqrt(MU_EARTH_KM3_S2 / a_km ** 3) * 60.0
    assert const.satellites[0].satrec.no_kozai == pytest.approx(
        expected_rad_min, rel=1e-12)


def test_constellation_rejects_duplicate_names():
    satrec = TLE("TEST", LINE1, LINE2).to_satrec()
    with pytest.raises(ValueError, match="unique"):
        Constellation([Satellite("A", satrec), Satellite("A", satrec)])


def test_constellation_rejects_name_collision_with_station():
    satrec = TLE("TEST", LINE1, LINE2).to_satrec()
    with pytest.raises(ValueError, match="collide"):
        Constellation([Satellite("GS", satrec)],
                      stations=[GroundStation("GS", 0.0, 0.0)])


def test_constellation_rejects_empty():
    with pytest.raises(ValueError):
        Constellation([])


def test_constellation_satellite_lookup():
    const = walker_delta(4, 2, 1, 53.0, 550.0, EPOCH, max_epoch_age_days=2.0)
    assert const.satellite("W01-01").name == "W01-01"
    with pytest.raises(KeyError):
        const.satellite("nope")


def test_ephemeris_shapes_and_step():
    const = walker_delta(4, 2, 1, 53.0, 550.0, EPOCH, max_epoch_age_days=2.0)
    eph = const.ephemeris(EPOCH, EPOCH + timedelta(minutes=10), 60.0)
    assert eph.r_teme_km.shape == (4, 11, 3)
    assert eph.step_s == pytest.approx(60.0)
    assert eph.index("W01-01") == 3
    with pytest.raises(KeyError):
        eph.index("nope")


def test_ephemeris_drop():
    const = walker_delta(4, 2, 1, 53.0, 550.0, EPOCH, max_epoch_age_days=2.0)
    eph = const.ephemeris(EPOCH, EPOCH + timedelta(minutes=10), 60.0)
    smaller = eph.drop(["W00-00"])
    assert smaller.sat_names == ["W00-01", "W01-00", "W01-01"]
    assert smaller.r_teme_km.shape == (3, 11, 3)
    with pytest.raises(ValueError):
        eph.drop(list(eph.sat_names))


def test_time_grid_endpoints():
    g = time_grid(EPOCH, EPOCH + timedelta(seconds=150), 60.0)
    assert g[0] == EPOCH
    assert g[-1] == EPOCH + timedelta(seconds=150)
    assert len(g) == 4  # 0, 60, 120, then the clipped endpoint at 150


@pytest.mark.parametrize("t1,step", [(EPOCH, 60.0),
                                     (EPOCH + timedelta(seconds=60), 0.0)])
def test_time_grid_rejects_bad_input(t1, step):
    with pytest.raises(ValueError):
        time_grid(EPOCH, t1, step)


def test_circular_orbit_period_matches_keplers_third_law():
    r = WGS84_A_KM + 550.0
    orbit = CircularOrbit(radius_km=r, inclination_deg=0.0, raan_deg=0.0,
                          arg_lat0_deg=0.0, epoch=EPOCH)
    expected = 2.0 * np.pi * np.sqrt(r ** 3 / MU_EARTH_KM3_S2)
    assert orbit.period_s == pytest.approx(expected, rel=1e-14)


def test_circular_orbit_radius_is_constant_and_speed_matches():
    r = WGS84_A_KM + 550.0
    orbit = CircularOrbit(radius_km=r, inclination_deg=53.0, raan_deg=30.0,
                          arg_lat0_deg=10.0, epoch=EPOCH)
    times = [EPOCH + timedelta(seconds=k * 300.0) for k in range(20)]
    pos, vel = orbit.propagate(times)
    assert np.allclose(np.linalg.norm(pos, axis=1), r, rtol=1e-12)
    expected_speed = np.sqrt(MU_EARTH_KM3_S2 / r)
    assert np.allclose(np.linalg.norm(vel, axis=1), expected_speed, rtol=1e-12)


def test_circular_orbit_velocity_is_perpendicular_to_position():
    orbit = CircularOrbit(radius_km=7000.0, inclination_deg=45.0, raan_deg=0.0,
                          arg_lat0_deg=0.0, epoch=EPOCH)
    pos, vel = orbit.propagate([EPOCH + timedelta(seconds=1234.0)])
    assert float(np.dot(pos[0], vel[0])) == pytest.approx(0.0, abs=1e-8)


def test_circular_orbit_inclination_sets_the_z_amplitude():
    r, inc = 7000.0, 53.0
    orbit = CircularOrbit(radius_km=r, inclination_deg=inc, raan_deg=0.0,
                          arg_lat0_deg=0.0, epoch=EPOCH)
    times = [EPOCH + timedelta(seconds=k * 50.0) for k in range(200)]
    pos, _ = orbit.propagate(times)
    assert float(np.abs(pos[:, 2]).max()) == pytest.approx(
        r * np.sin(np.deg2rad(inc)), rel=1e-3)


def test_circular_orbit_rejects_bad_parameters():
    with pytest.raises(ValueError):
        CircularOrbit(radius_km=-1.0, inclination_deg=0.0, raan_deg=0.0,
                      arg_lat0_deg=0.0, epoch=EPOCH)
    with pytest.raises(ValueError):
        CircularOrbit(radius_km=7000.0, inclination_deg=200.0, raan_deg=0.0,
                      arg_lat0_deg=0.0, epoch=EPOCH)


@pytest.mark.parametrize("kwargs", [
    {"lat_deg": 100.0}, {"lon_deg": 1000.0}, {"alt_km": -5.0},
    {"min_elevation_deg": 90.0}, {"name": ""},
])
def test_ground_station_validation(kwargs):
    base = {"name": "GS", "lat_deg": 0.0, "lon_deg": 0.0, "alt_km": 0.0,
            "min_elevation_deg": 10.0}
    base.update(kwargs)
    with pytest.raises(ValueError):
        GroundStation(**base)
