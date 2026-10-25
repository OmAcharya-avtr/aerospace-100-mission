"""Ground stations and horizon masks."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from contactsched.geometry import (
    MAX_GEODETIC_GEOCENTRIC_DEVIATION_DEG,
    WGS84_A_KM,
    WGS84_B_KM,
)
from contactsched.station import GroundStation, HorizonMask, default_network


class TestHorizonMask:
    def test_flat_mask_returns_the_floor_everywhere(self):
        mask = HorizonMask(floor_deg=7.5)
        azimuths = np.linspace(-720.0, 720.0, 101)
        assert np.allclose(
            mask.min_elevation_deg(azimuths), 7.5, rtol=0.0, atol=0.0
        )

    def test_one_breakpoint_is_ignored_and_the_floor_applies(self):
        # Interpolation needs two points; with one the profile is undefined and
        # the floor is the honest answer.
        mask = HorizonMask(floor_deg=5.0, breakpoints=((90.0, 30.0),))
        assert mask.min_elevation_deg(np.array([90.0]))[0] == pytest.approx(
            5.0, rel=0.0, abs=0.0
        )

    def test_interpolates_linearly_between_breakpoints(self):
        # Breakpoints at 0 deg -> 20 and 100 deg -> 10; at azimuth 50 the linear
        # interpolant is 15 exactly.
        mask = HorizonMask(floor_deg=0.0, breakpoints=((0.0, 20.0), (100.0, 10.0)))
        assert mask.min_elevation_deg(np.array([50.0]))[0] == pytest.approx(
            15.0, rel=0.0, abs=1e-12
        )

    def test_reproduces_the_breakpoint_values_exactly(self):
        points = ((0.0, 14.0), (90.0, 6.0), (180.0, 9.0), (270.0, 11.0))
        mask = HorizonMask(floor_deg=0.0, breakpoints=points)
        for azimuth, elevation in points:
            assert mask.min_elevation_deg(np.array([azimuth]))[0] == pytest.approx(
                elevation, rel=0.0, abs=1e-12
            )

    def test_is_periodic_in_azimuth(self):
        mask = HorizonMask(floor_deg=0.0, breakpoints=((10.0, 20.0), (200.0, 4.0)))
        azimuths = np.linspace(0.0, 360.0, 73)
        assert np.allclose(
            mask.min_elevation_deg(azimuths),
            mask.min_elevation_deg(azimuths + 360.0),
            rtol=0.0,
            atol=1e-13,
        )

    def test_wraps_the_short_way_across_zero(self):
        # With breakpoints only at 90 and 270, azimuth 0 sits midway on the
        # wrapped side, so the value is the mean of the two.
        mask = HorizonMask(floor_deg=0.0, breakpoints=((90.0, 10.0), (270.0, 20.0)))
        assert mask.min_elevation_deg(np.array([0.0]))[0] == pytest.approx(
            15.0, rel=0.0, abs=1e-12
        )

    def test_floor_dominates_a_lower_profile(self):
        mask = HorizonMask(floor_deg=12.0, breakpoints=((0.0, 2.0), (180.0, 3.0)))
        assert np.all(mask.min_elevation_deg(np.linspace(0.0, 360.0, 37)) >= 12.0)

    def test_profile_dominates_a_lower_floor(self):
        mask = HorizonMask(floor_deg=5.0, breakpoints=((0.0, 25.0), (180.0, 25.0)))
        assert mask.min_elevation_deg(np.array([90.0]))[0] == pytest.approx(
            25.0, rel=0.0, abs=1e-12
        )

    def test_duplicate_azimuths_collapse_to_their_maximum(self):
        # Two elevations at one azimuth do not define a function. The maximum is
        # taken, which is the conservative choice for an obstruction.
        mask = HorizonMask(
            floor_deg=0.0, breakpoints=((10.0, 4.0), (10.0, 12.0), (200.0, 4.0))
        )
        assert mask.min_elevation_deg(np.array([10.0]))[0] == pytest.approx(
            12.0, rel=0.0, abs=1e-12
        )

    def test_duplicate_azimuths_do_not_break_periodicity(self):
        mask = HorizonMask(floor_deg=0.0, breakpoints=((1.0, 0.0), (1.0, 1.0), (180.0, 3.0)))
        assert mask.min_elevation_deg(np.array([1.0]))[0] == pytest.approx(
            mask.min_elevation_deg(np.array([361.0]))[0], rel=0.0, abs=1e-12
        )

    def test_breakpoints_at_one_azimuth_fall_back_to_the_floor(self):
        # After collapsing there is one distinct azimuth, so no profile exists
        # and the floor is the honest answer.
        mask = HorizonMask(floor_deg=6.0, breakpoints=((30.0, 20.0), (30.0, 25.0)))
        assert mask.min_elevation_deg(np.array([30.0, 120.0]))[0] == pytest.approx(
            6.0, rel=0.0, abs=0.0
        )

    def test_unsorted_breakpoints_give_the_same_profile(self):
        forward = HorizonMask(floor_deg=0.0, breakpoints=((0.0, 10.0), (120.0, 20.0)))
        reverse = HorizonMask(floor_deg=0.0, breakpoints=((120.0, 20.0), (0.0, 10.0)))
        azimuths = np.linspace(0.0, 360.0, 37)
        assert np.allclose(
            forward.min_elevation_deg(azimuths),
            reverse.min_elevation_deg(azimuths),
            rtol=0.0,
            atol=0.0,
        )

    @pytest.mark.parametrize("floor", [-90.0, 90.0, 120.0])
    def test_invalid_floor_raises(self, floor):
        with pytest.raises(ValueError, match="mask floor"):
            HorizonMask(floor_deg=floor)

    def test_invalid_breakpoint_elevation_raises(self):
        with pytest.raises(ValueError, match="mask elevation"):
            HorizonMask(floor_deg=5.0, breakpoints=((0.0, 95.0), (90.0, 5.0)))

    def test_non_finite_breakpoint_azimuth_raises(self):
        with pytest.raises(ValueError, match="mask azimuth"):
            HorizonMask(floor_deg=5.0, breakpoints=((float("inf"), 5.0), (90.0, 5.0)))


class TestGroundStation:
    def test_equatorial_station_radius_is_the_semi_major_axis(self):
        station = GroundStation("EQ", 0.0, 0.0, 0.0)
        assert station.geocentric_radius_km == pytest.approx(WGS84_A_KM, rel=1e-14, abs=0.0)

    def test_polar_station_radius_is_the_semi_minor_axis(self):
        station = GroundStation("POLE", 90.0, 0.0, 0.0)
        assert station.geocentric_radius_km == pytest.approx(WGS84_B_KM, rel=1e-12, abs=0.0)

    def test_altitude_adds_exactly_at_the_pole(self):
        # At the pole the ellipsoid normal is radial, so the geocentric radius
        # grows by exactly the altitude.
        low = GroundStation("L", 90.0, 10.0, 0.0).geocentric_radius_km
        high = GroundStation("H", 90.0, 10.0, 2.0).geocentric_radius_km
        assert high - low == pytest.approx(2.0, rel=0.0, abs=1e-9)

    def test_altitude_adds_slightly_less_than_itself_off_the_pole(self):
        # Off the pole the normal is tilted from the radial direction by at most
        # MAX_GEODETIC_GEOCENTRIC_DEVIATION_DEG, so the radial growth is
        # alt * cos(tilt) < alt, short by at most alt * (1 - cos(tilt)). This
        # bound is derived from the ellipsoid, not from the function under test,
        # and the measured deficit at 45 deg is 1.1275e-05 km for a 2 km rise.
        low = GroundStation("L", 45.0, 10.0, 0.0).geocentric_radius_km
        high = GroundStation("H", 45.0, 10.0, 2.0).geocentric_radius_km
        growth = high - low
        tilt = math.radians(MAX_GEODETIC_GEOCENTRIC_DEVIATION_DEG)
        assert 0.0 < 2.0 - growth <= 2.0 * (1.0 - math.cos(tilt)) + 1e-12

    def test_sez_is_orthonormal(self):
        matrix = GroundStation("S", -45.0, 170.0, 0.1).sez
        assert np.allclose(matrix @ matrix.T, np.eye(3), rtol=0.0, atol=1e-14)

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            GroundStation("", 0.0, 0.0)

    def test_negative_turnaround_raises(self):
        with pytest.raises(ValueError, match="turnaround_s"):
            GroundStation("S", 0.0, 0.0, turnaround_s=-1.0)

    def test_bad_latitude_raises_through_the_position_check(self):
        with pytest.raises(ValueError, match="geodetic latitude"):
            GroundStation("S", 91.0, 0.0)

    def test_station_is_frozen(self):
        station = GroundStation("S", 0.0, 0.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            station.name = "T"


class TestDefaultNetwork:
    def test_has_four_antennas(self):
        assert len(default_network()) == 4

    def test_names_are_unique(self):
        names = [s.name for s in default_network()]
        assert len(set(names)) == len(names)

    def test_two_antennas_share_a_site(self):
        network = {s.name: s for s in default_network()}
        first, second = network["SVALBARD-1"], network["SVALBARD-2"]
        assert (first.lat_deg, first.lon_deg) == (second.lat_deg, second.lon_deg)

    def test_one_antenna_carries_a_terrain_mask(self):
        masked = [s for s in default_network() if s.mask.breakpoints]
        assert len(masked) == 1
        assert masked[0].name == "PUNTA-ARENAS"

    def test_the_terrain_mask_rises_to_the_north(self):
        masked = {s.name: s for s in default_network()}["PUNTA-ARENAS"]
        north = masked.mask.min_elevation_deg(np.array([0.0]))[0]
        south = masked.mask.min_elevation_deg(np.array([180.0]))[0]
        assert north > south
