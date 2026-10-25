"""Visibility windows: boundaries, flags, invariants and the derived bound."""

from __future__ import annotations

import numpy as np
import pytest

from contactsched.ephemeris import Scenario, Spacecraft, default_fleet
from contactsched.geometry import max_pass_duration_s
from contactsched.station import GroundStation, HorizonMask
from contactsched.visibility import (
    ContactWindow,
    all_windows,
    compute_windows,
    mask_margin_deg,
    supply_seconds_by_station,
    visible_seconds,
)


class TestContactWindow:
    def test_duration_is_the_difference(self):
        window = ContactWindow("A", "S", "M", 100.0, 450.0, 30.0, 300.0, 5.0)
        assert window.duration_s == pytest.approx(350.0, rel=0.0, abs=0.0)

    def test_set_before_rise_raises(self):
        with pytest.raises(ValueError, match="not after rise_s"):
            ContactWindow("A", "S", "M", 100.0, 100.0, 30.0, 100.0, 5.0)

    def test_overlaps_is_symmetric(self):
        first = ContactWindow("A", "S", "M", 0.0, 100.0, 30.0, 50.0, 5.0)
        second = ContactWindow("A", "T", "M", 50.0, 150.0, 30.0, 100.0, 5.0)
        assert first.overlaps(second)
        assert second.overlaps(first)

    def test_touching_windows_do_not_overlap(self):
        first = ContactWindow("A", "S", "M", 0.0, 100.0, 30.0, 50.0, 5.0)
        second = ContactWindow("A", "T", "M", 100.0, 200.0, 30.0, 150.0, 5.0)
        assert not first.overlaps(second)


class TestComputeWindows:
    @pytest.fixture(scope="class")
    @staticmethod
    def scenario() -> Scenario:
        return Scenario.from_calendar(2026, 4, 30, 0, 0, 0.0, duration_s=43200.0)

    @pytest.fixture(scope="class")
    @staticmethod
    def station() -> GroundStation:
        return GroundStation("SV", 78.23, 15.39, 0.45, HorizonMask(floor_deg=5.0), 300.0)

    @pytest.fixture(scope="class")
    @staticmethod
    def spacecraft() -> Spacecraft:
        return default_fleet()[0]

    def test_finds_several_windows_over_twelve_hours(self, scenario, station, spacecraft):
        windows = compute_windows(scenario, station, spacecraft, step_s=30.0)
        assert len(windows) >= 5

    def test_windows_are_ordered_and_disjoint(self, scenario, station, spacecraft):
        windows = compute_windows(scenario, station, spacecraft, step_s=30.0)
        for first, second in zip(windows[:-1], windows[1:], strict=True):
            assert first.set_s <= second.rise_s

    def test_margin_is_near_zero_at_an_unclipped_boundary(self, scenario, station, spacecraft):
        windows = compute_windows(scenario, station, spacecraft, step_s=30.0, tolerance_s=0.25)
        interior = [w for w in windows if not w.clipped_start and not w.clipped_end]
        assert interior
        boundaries = np.array([w.rise_s for w in interior] + [w.set_s for w in interior])
        margin = mask_margin_deg(scenario, station, spacecraft, boundaries)
        # A 0.25 s tolerance at a boundary elevation rate below 0.1 deg/s bounds
        # the residual margin at 0.025 deg; measured worst case is far smaller.
        assert np.max(np.abs(margin)) < 0.025

    def test_margin_is_positive_inside_and_negative_outside(
        self, scenario, station, spacecraft
    ):
        windows = compute_windows(scenario, station, spacecraft, step_s=30.0)
        interior = [w for w in windows if not w.clipped_start and not w.clipped_end]
        midpoints = np.array([0.5 * (w.rise_s + w.set_s) for w in interior])
        assert np.all(mask_margin_deg(scenario, station, spacecraft, midpoints) > 0.0)
        outside = np.array([w.rise_s - 60.0 for w in interior if w.rise_s > 60.0])
        assert np.all(mask_margin_deg(scenario, station, spacecraft, outside) < 0.0)

    def test_peak_elevation_is_above_the_mask_floor(self, scenario, station, spacecraft):
        for window in compute_windows(scenario, station, spacecraft, step_s=30.0):
            assert window.max_elevation_deg >= station.mask.floor_deg - 1e-6

    def test_peak_is_inside_the_window(self, scenario, station, spacecraft):
        for window in compute_windows(scenario, station, spacecraft, step_s=30.0):
            assert window.rise_s <= window.max_elevation_s <= window.set_s

    def test_peak_resolution_is_reported(self, scenario, station, spacecraft):
        windows = compute_windows(scenario, station, spacecraft, step_s=30.0, peak_samples=65)
        for window in windows:
            assert window.peak_resolution_s == pytest.approx(
                window.duration_s / 64.0, rel=1e-12, abs=0.0
            )

    def test_durations_respect_the_independently_derived_bound(
        self, scenario, station, spacecraft
    ):
        # An independently derived BOUND, from closed-form spherical geometry and
        # the TLE's own mean motion, sharing no expression with the root search.
        bound = max_pass_duration_s(
            spacecraft.elements.semi_major_axis_km,
            spacecraft.elements.eccentricity,
            station.geocentric_radius_km,
            station.mask.floor_deg,
        )
        windows = compute_windows(scenario, station, spacecraft, step_s=30.0)
        interior = [w for w in windows if not w.clipped_start and not w.clipped_end]
        assert interior
        assert max(w.duration_s for w in interior) <= bound

    def test_clipped_flags_are_set_only_at_the_ends(self, scenario, station, spacecraft):
        windows = compute_windows(scenario, station, spacecraft, step_s=30.0)
        for index, window in enumerate(windows):
            if window.clipped_start:
                assert index == 0
            if window.clipped_end:
                assert index == len(windows) - 1

    def test_a_clipped_start_window_begins_at_zero(self, scenario, station, spacecraft):
        # Anchored on the TRUE value: find a real window, then start a horizon
        # 100 s inside it. The spacecraft is then already above the mask at t=0,
        # which must be reported as clipped_start with rise_s exactly 0.
        reference = compute_windows(scenario, station, spacecraft, step_s=30.0)
        interior = [w for w in reference if not w.clipped_start][0]
        offset_days = (interior.rise_s + 100.0) / 86400.0
        shifted = Scenario(
            jd=scenario.jd, fr=scenario.fr + offset_days, duration_s=7200.0
        )
        windows = compute_windows(shifted, station, spacecraft, step_s=30.0)
        assert windows[0].clipped_start
        assert windows[0].rise_s == pytest.approx(0.0, rel=0.0, abs=0.0)
        # The remaining visible time is the original window minus the 100 s cut,
        # to within the bisection tolerance on the set boundary.
        assert windows[0].duration_s == pytest.approx(
            interior.duration_s - 100.0, rel=0.0, abs=1.0
        )

    def test_a_clipped_end_window_ends_at_the_horizon(self, scenario, station, spacecraft):
        reference = compute_windows(scenario, station, spacecraft, step_s=30.0)
        interior = [w for w in reference if not w.clipped_end][0]
        duration = interior.rise_s + 0.5 * interior.duration_s
        truncated = Scenario(jd=scenario.jd, fr=scenario.fr, duration_s=duration)
        windows = compute_windows(truncated, station, spacecraft, step_s=30.0)
        assert windows[-1].clipped_end
        assert windows[-1].set_s == pytest.approx(duration, rel=0.0, abs=0.0)

    def test_a_masked_station_marks_mask_limited_windows(self, scenario, spacecraft):
        ridge = HorizonMask(
            floor_deg=5.0, breakpoints=((0.0, 25.0), (90.0, 5.0), (180.0, 5.0), (270.0, 20.0))
        )
        station = GroundStation("RIDGE", 78.23, 15.39, 0.45, ridge, 300.0)
        windows = compute_windows(scenario, station, spacecraft, step_s=30.0)
        assert any(w.mask_limited for w in windows)

    def test_a_higher_mask_never_lengthens_a_window(self, scenario, spacecraft):
        low = GroundStation("L", 78.23, 15.39, 0.45, HorizonMask(floor_deg=5.0), 300.0)
        high = GroundStation("H", 78.23, 15.39, 0.45, HorizonMask(floor_deg=15.0), 300.0)
        total_low = visible_seconds(compute_windows(scenario, low, spacecraft, step_s=30.0))
        total_high = visible_seconds(compute_windows(scenario, high, spacecraft, step_s=30.0))
        assert total_high <= total_low

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            (dict(step_s=0.0), "step_s must be positive"),
            (dict(step_s=-30.0), "step_s must be positive"),
            (dict(tolerance_s=0.0), "tolerance_s must be within"),
            (dict(tolerance_s=60.0), "tolerance_s must be within"),
            (dict(peak_samples=2), "peak_samples must be at least 3"),
        ],
    )
    def test_invalid_parameters_raise(self, scenario, station, spacecraft, kwargs, match):
        base = dict(step_s=30.0, tolerance_s=0.5, peak_samples=65)
        base.update(kwargs)
        with pytest.raises(ValueError, match=match):
            compute_windows(scenario, station, spacecraft, **base)


class TestAllWindows:
    def test_ordered_by_rise_time(self, windows):
        rises = [w.rise_s for w in windows]
        assert rises == sorted(rises)

    def test_every_station_and_spacecraft_pair_is_represented_or_absent(
        self, windows, network, fleet
    ):
        stations = {s.name for s in network}
        spacecraft = {s.name for s in fleet}
        for window in windows:
            assert window.station in stations
            assert window.spacecraft in spacecraft

    def test_mission_is_carried_from_the_spacecraft(self, windows, fleet):
        mission_of = {s.name: s.mission for s in fleet}
        for window in windows:
            assert window.mission == mission_of[window.spacecraft]

    def test_visible_seconds_is_the_sum_of_durations(self, windows):
        assert visible_seconds(list(windows)) == pytest.approx(
            sum(w.duration_s for w in windows), rel=1e-12, abs=0.0
        )

    def test_supply_is_a_union_not_a_sum(self, windows):
        supply = supply_seconds_by_station(list(windows))
        for station, union_seconds in supply.items():
            raw = sum(w.duration_s for w in windows if w.station == station)
            assert union_seconds <= raw + 1e-9

    def test_supply_is_strictly_below_the_sum_where_windows_overlap(self, windows):
        # Two antennas at the same site see overlapping passes of different
        # spacecraft, so at least one station's union must be strictly smaller
        # than its raw sum. If this ever passes trivially the fixture has stopped
        # exercising the union.
        supply = supply_seconds_by_station(list(windows))
        strictly_smaller = [
            station
            for station, union_seconds in supply.items()
            if union_seconds < sum(w.duration_s for w in windows if w.station == station) - 1e-6
        ]
        assert strictly_smaller

    def test_co_located_antennas_with_the_same_mask_see_the_same_windows(self, scenario, fleet):
        first = GroundStation("C1", 20.0, 30.0, 0.1, HorizonMask(floor_deg=6.0), 100.0)
        second = GroundStation("C2", 20.0, 30.0, 0.1, HorizonMask(floor_deg=6.0), 200.0)
        one = compute_windows(scenario, first, fleet[0], step_s=45.0)
        two = compute_windows(scenario, second, fleet[0], step_s=45.0)
        assert len(one) == len(two)
        for a, b in zip(one, two, strict=True):
            assert a.rise_s == pytest.approx(b.rise_s, rel=0.0, abs=1e-9)
            assert a.set_s == pytest.approx(b.set_s, rel=0.0, abs=1e-9)

    def test_empty_input_gives_empty_output(self, scenario):
        assert all_windows(scenario, (), ()) == []
        assert supply_seconds_by_station([]) == {}
        assert visible_seconds([]) == pytest.approx(0.0, rel=0.0, abs=0.0)
