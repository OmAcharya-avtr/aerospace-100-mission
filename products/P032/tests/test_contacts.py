"""Contact-window finding."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from constellink.constellation import (
    CircularOrbit,
    Constellation,
    GroundStation,
    walker_delta,
)
from constellink.contacts import (
    ContactWindow,
    all_contact_windows,
    contact_windows_ground,
    contact_windows_isl,
)
from constellink.frames import SPEED_OF_LIGHT_KM_S, WGS84_A_KM
from constellink.geometry import ground_max_central_angle, max_isl_central_angle

EPOCH = datetime(2026, 4, 1, tzinfo=UTC)
R_ORBIT = WGS84_A_KM + 550.0


def make_window(**kwargs):
    base = {"node_a": "A", "node_b": "B", "kind": "isl",
            "t_open": EPOCH, "t_close": EPOCH + timedelta(seconds=120),
            "min_range_km": 1000.0, "max_range_km": 1500.0,
            "max_elevation_deg": None, "grid_step_s": 60.0}
    base.update(kwargs)
    return ContactWindow(**base)


def test_window_derived_quantities():
    w = make_window()
    assert w.duration_s == pytest.approx(120.0)
    assert w.mid_time == EPOCH + timedelta(seconds=60)
    assert w.min_one_way_delay_s == pytest.approx(1000.0 / SPEED_OF_LIGHT_KM_S)


def test_window_rejects_bad_kind_and_order():
    with pytest.raises(ValueError, match="kind"):
        make_window(kind="laser")
    with pytest.raises(ValueError, match="t_close"):
        make_window(t_close=EPOCH)


def test_two_body_ground_window_matches_closed_form_duration():
    # Equatorial circular orbit, equatorial station: the contact lasts
    # 2 lambda_max / (n - omega_GMST). Closed form from Wertz & Larson 1999.
    orbit = CircularOrbit(radius_km=R_ORBIT, inclination_deg=0.0, raan_deg=0.0,
                          arg_lat0_deg=0.0, epoch=EPOCH, name="TB")
    station = GroundStation("EQ", 0.0, 0.0, 0.0, 10.0)
    eph = orbit.ephemeris(EPOCH, EPOCH + timedelta(hours=4), 30.0)
    windows = contact_windows_ground(eph, orbit, station, max_range_km=1e5,
                                     refine_tol_s=1e-4)
    inner = [w for w in windows if not (w.clipped_start or w.clipped_end)]
    assert inner, "expected at least one window fully inside the horizon"
    lam = ground_max_central_angle(R_ORBIT, 10.0)
    n_rad_s = orbit.mean_motion_rad_s
    omega_gmst = np.deg2rad(360.98564736629) / 86400.0
    expected = 2.0 * lam / (n_rad_s - omega_gmst)
    for w in inner:
        assert w.duration_s == pytest.approx(expected, abs=0.5)


def test_two_body_ground_window_elevation_respects_the_mask():
    orbit = CircularOrbit(radius_km=R_ORBIT, inclination_deg=0.0, raan_deg=0.0,
                          arg_lat0_deg=0.0, epoch=EPOCH, name="TB")
    station = GroundStation("EQ", 0.0, 0.0, 0.0, 25.0)
    eph = orbit.ephemeris(EPOCH, EPOCH + timedelta(hours=4), 30.0)
    windows = contact_windows_ground(eph, orbit, station, max_range_km=1e5)
    assert windows
    for w in windows:
        assert w.max_elevation_deg >= 25.0


def test_tighter_mask_gives_shorter_windows():
    orbit = CircularOrbit(radius_km=R_ORBIT, inclination_deg=0.0, raan_deg=0.0,
                          arg_lat0_deg=0.0, epoch=EPOCH, name="TB")
    station = GroundStation("EQ", 0.0, 0.0, 0.0, 10.0)
    eph = orbit.ephemeris(EPOCH, EPOCH + timedelta(hours=4), 30.0)
    low = contact_windows_ground(eph, orbit, station, max_range_km=1e5,
                                 min_elevation_deg=5.0)
    high = contact_windows_ground(eph, orbit, station, max_range_km=1e5,
                                  min_elevation_deg=30.0)
    assert sum(w.duration_s for w in low) > sum(w.duration_s for w in high)


def test_range_limit_can_close_a_geometric_window():
    orbit = CircularOrbit(radius_km=R_ORBIT, inclination_deg=0.0, raan_deg=0.0,
                          arg_lat0_deg=0.0, epoch=EPOCH, name="TB")
    station = GroundStation("EQ", 0.0, 0.0, 0.0, 10.0)
    eph = orbit.ephemeris(EPOCH, EPOCH + timedelta(hours=4), 30.0)
    wide = contact_windows_ground(eph, orbit, station, max_range_km=1e5)
    tight = contact_windows_ground(eph, orbit, station, max_range_km=700.0)
    assert sum(w.duration_s for w in tight) < sum(w.duration_s for w in wide)
    for w in tight:
        assert w.min_range_km <= 700.0


@pytest.mark.parametrize("kwargs", [{"max_range_km": 0.0},
                                    {"min_elevation_deg": 95.0}])
def test_contact_windows_ground_rejects_bad_input(kwargs):
    orbit = CircularOrbit(radius_km=R_ORBIT, inclination_deg=0.0, raan_deg=0.0,
                          arg_lat0_deg=0.0, epoch=EPOCH, name="TB")
    station = GroundStation("EQ", 0.0, 0.0, 0.0, 10.0)
    eph = orbit.ephemeris(EPOCH, EPOCH + timedelta(minutes=30), 60.0)
    with pytest.raises(ValueError):
        contact_windows_ground(eph, orbit, station, **kwargs)


def test_isl_window_requires_distinct_satellites(small_constellation, epoch):
    eph = small_constellation.ephemeris(epoch, epoch + timedelta(minutes=30), 60.0)
    sat = small_constellation.satellites[0]
    with pytest.raises(ValueError, match="distinct"):
        contact_windows_isl(eph, sat, sat)


def test_isl_window_rejects_bad_parameters(small_constellation, epoch):
    eph = small_constellation.ephemeris(epoch, epoch + timedelta(minutes=30), 60.0)
    a, b = small_constellation.satellites[:2]
    with pytest.raises(ValueError):
        contact_windows_isl(eph, a, b, max_range_km=-1.0)
    with pytest.raises(ValueError):
        contact_windows_isl(eph, a, b, refine_tol_s=0.0)


def test_intra_plane_isl_follows_the_closed_form_clearance(epoch):
    # Satellites in the same plane hold a constant central-angle separation of
    # 360/S degrees, so whether the intra-plane link exists at all is decided
    # by the closed form gamma_max = 2 arccos(r_block / r). At 550 km,
    # gamma_max is 41.53 deg, so:
    #   S = 4  -> 90 deg separation -> blocked, no window ever;
    #   S = 10 -> 36 deg separation -> clear, one window spanning the horizon.
    gamma_max_deg = np.rad2deg(max_isl_central_angle(R_ORBIT))
    assert 36.0 < gamma_max_deg < 90.0

    blocked = walker_delta(8, 2, 1, 53.0, 550.0, epoch, max_epoch_age_days=2.0)
    eph_b = blocked.ephemeris(epoch, epoch + timedelta(hours=1), 60.0)
    assert contact_windows_isl(eph_b, blocked.satellite("W00-00"),
                               blocked.satellite("W00-01"),
                               max_range_km=1e5) == []

    clear = walker_delta(20, 2, 1, 53.0, 550.0, epoch, max_epoch_age_days=2.0)
    eph_c = clear.ephemeris(epoch, epoch + timedelta(hours=1), 60.0)
    windows = contact_windows_isl(eph_c, clear.satellite("W00-00"),
                                  clear.satellite("W00-01"), max_range_km=1e5)
    assert len(windows) == 1
    assert windows[0].clipped_start and windows[0].clipped_end
    assert windows[0].duration_s == pytest.approx(3600.0, abs=1.0)


def test_isl_window_records_the_grid_step(epoch):
    const = walker_delta(20, 2, 1, 53.0, 550.0, epoch, max_epoch_age_days=2.0)
    eph = const.ephemeris(epoch, epoch + timedelta(hours=1), 90.0)
    windows = contact_windows_isl(eph, const.satellite("W00-00"),
                                  const.satellite("W00-01"), max_range_km=1e5)
    assert windows
    assert all(w.grid_step_s == pytest.approx(90.0) for w in windows)


def test_all_contact_windows_is_sorted_and_typed(small_graph):
    windows = small_graph.windows
    assert windows
    assert all(w.kind in ("isl", "ground") for w in windows)
    opens = [w.t_open for w in windows]
    assert opens == sorted(opens)
    assert any(w.kind == "ground" for w in windows)
    assert any(w.kind == "isl" for w in windows)


def test_all_contact_windows_rejects_unknown_satellite(small_constellation,
                                                        epoch):
    eph = small_constellation.ephemeris(epoch, epoch + timedelta(minutes=30), 60.0)
    with pytest.raises(KeyError):
        all_contact_windows(eph, small_constellation.satellites[:2])


def test_single_satellite_ephemeris_has_no_isl_windows(epoch):
    orbit = CircularOrbit(radius_km=R_ORBIT, inclination_deg=0.0, raan_deg=0.0,
                          arg_lat0_deg=0.0, epoch=epoch, name="TB")
    eph = orbit.ephemeris(epoch, epoch + timedelta(minutes=30), 60.0)
    windows = all_contact_windows(eph, [orbit])
    assert windows == []


def test_ground_windows_match_between_single_and_batch(small_constellation,
                                                        station, epoch):
    eph = small_constellation.ephemeris(epoch, epoch + timedelta(hours=2), 60.0)
    batch = [w for w in all_contact_windows(eph, small_constellation.satellites,
                                            [station]) if w.kind == "ground"]
    single = []
    for sat in small_constellation.satellites:
        single.extend(contact_windows_ground(eph, sat, station))
    assert len(batch) == len(single)
    assert sorted(round(w.duration_s, 6) for w in batch) == sorted(
        round(w.duration_s, 6) for w in single)


def test_constellation_ephemeris_round_trip_with_drop(small_constellation,
                                                       epoch):
    eph = small_constellation.ephemeris(epoch, epoch + timedelta(minutes=30), 60.0)
    kept = eph.drop(["W00-00"])
    sats = [s for s in small_constellation.satellites if s.name != "W00-00"]
    windows = all_contact_windows(kept, sats)
    assert all("W00-00" not in (w.node_a, w.node_b) for w in windows)
    assert Constellation(sats).n_sat == small_constellation.n_sat - 1
