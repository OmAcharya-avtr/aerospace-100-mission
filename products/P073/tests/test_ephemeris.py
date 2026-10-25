"""Scenario time base, propagation and the declared fleet."""

from __future__ import annotations

import numpy as np
import pytest
from sgp4.api import jday

from contactsched.ephemeris import Scenario, Spacecraft, default_fleet
from contactsched.tle import OrbitElements


def elements(**overrides) -> OrbitElements:
    base = dict(
        satnum=90001,
        epoch_year=26,
        epoch_day=120.0,
        inclination_deg=97.8,
        raan_deg=10.0,
        eccentricity=0.0012,
        argp_deg=40.0,
        mean_anomaly_deg=0.0,
        mean_motion_rev_day=15.21,
        bstar=8.0e-5,
        name="SYNTH-1",
    )
    base.update(overrides)
    return OrbitElements(**base)


class TestScenario:
    def test_from_calendar_matches_jday(self):
        scenario = Scenario.from_calendar(2026, 4, 30, 6, 30, 15.0, duration_s=600.0)
        jd, fr = jday(2026, 4, 30, 6, 30, 15.0)
        assert scenario.jd == pytest.approx(jd, rel=0.0, abs=0.0)
        assert scenario.fr == pytest.approx(fr, rel=0.0, abs=0.0)

    def test_fraction_out_of_range_raises(self):
        with pytest.raises(ValueError, match="fr must be"):
            Scenario(jd=2461000.5, fr=1.0, duration_s=100.0)

    def test_non_positive_duration_raises(self):
        with pytest.raises(ValueError, match="duration_s must be positive"):
            Scenario(jd=2461000.5, fr=0.0, duration_s=0.0)

    def test_julian_dates_keep_the_day_number_separate(self):
        scenario = Scenario(jd=2461000.0, fr=0.5, duration_s=86400.0)
        jd, fr = scenario.julian_dates(np.array([0.0, 43200.0, 86400.0]))
        assert np.allclose(jd, 2461000.0, rtol=0.0, atol=0.0)
        assert np.allclose(fr, [0.5, 1.0, 1.5], rtol=0.0, atol=1e-15)

    def test_full_julian_dates_sum_the_pair(self):
        scenario = Scenario(jd=2461000.0, fr=0.25, duration_s=86400.0)
        full = scenario.full_julian_dates(np.array([0.0, 86400.0]))
        assert np.allclose(full, [2461000.25, 2461001.25], rtol=0.0, atol=1e-9)

    def test_one_day_of_offset_advances_the_date_by_one(self):
        scenario = Scenario.from_calendar(2026, 1, 1, duration_s=172800.0)
        full = scenario.full_julian_dates(np.array([0.0, 86400.0]))
        assert full[1] - full[0] == pytest.approx(1.0, rel=0.0, abs=1e-9)


class TestSpacecraft:
    def test_repr_names_the_mission(self):
        spacecraft = Spacecraft(elements(), mission="MX")
        assert "MX" in repr(spacecraft)
        assert spacecraft.name == "SYNTH-1"

    def test_teme_radius_is_plausible_for_a_low_orbit(self, ):
        spacecraft = Spacecraft(elements())
        scenario = Scenario.from_calendar(2026, 4, 30, duration_s=6000.0)
        radius = spacecraft.radius_km(scenario, np.linspace(0.0, 6000.0, 61))
        assert np.all(radius > 6600.0)
        assert np.all(radius < 7200.0)

    def test_radius_is_frame_independent(self):
        spacecraft = Spacecraft(elements())
        scenario = Scenario.from_calendar(2026, 4, 30, duration_s=3000.0)
        times = np.linspace(0.0, 3000.0, 31)
        teme = np.linalg.norm(spacecraft.teme_km(scenario, times), axis=1)
        ecef = np.linalg.norm(spacecraft.ecef_km(scenario, times), axis=1)
        assert np.allclose(teme, ecef, rtol=0.0, atol=1e-9)

    def test_scalar_time_returns_one_row(self):
        spacecraft = Spacecraft(elements())
        scenario = Scenario.from_calendar(2026, 4, 30, duration_s=100.0)
        assert spacecraft.ecef_km(scenario, 0.0).shape == (1, 3)

    def test_radius_oscillates_over_one_revolution(self):
        # Eccentricity 0.02 at a = 6881 km gives perigee and apogee radii about
        # 275 km apart, so the radius must vary by more than 100 km over a
        # revolution. A constant radius would mean the eccentricity was lost.
        spacecraft = Spacecraft(elements(eccentricity=0.02))
        scenario = Scenario.from_calendar(2026, 4, 30, duration_s=6000.0)
        radius = spacecraft.radius_km(scenario, np.linspace(0.0, 5700.0, 201))
        assert radius.max() - radius.min() > 100.0

    def test_propagation_failure_is_raised_not_returned(self):
        # A mean motion of 25 rev/day puts the semi-major axis inside the Earth,
        # and sgp4 reports an error code rather than a position.
        spacecraft = Spacecraft(elements(mean_motion_rev_day=25.0, eccentricity=0.6))
        scenario = Scenario.from_calendar(2026, 4, 30, duration_s=86400.0)
        with pytest.raises(RuntimeError, match="sgp4 failed"):
            spacecraft.teme_km(scenario, np.linspace(0.0, 86400.0, 100))

    def test_the_error_message_names_the_time(self):
        spacecraft = Spacecraft(elements(mean_motion_rev_day=25.0, eccentricity=0.6))
        scenario = Scenario.from_calendar(2026, 4, 30, duration_s=86400.0)
        with pytest.raises(RuntimeError, match="after the scenario epoch"):
            spacecraft.teme_km(scenario, np.linspace(0.0, 86400.0, 100))


class TestDefaultFleet:
    def test_has_six_spacecraft(self):
        assert len(default_fleet()) == 6

    def test_names_are_unique(self):
        names = [s.name for s in default_fleet()]
        assert len(set(names)) == len(names)

    def test_one_mission_owns_three_spacecraft(self):
        counts: dict[str, int] = {}
        for spacecraft in default_fleet():
            counts[spacecraft.mission] = counts.get(spacecraft.mission, 0) + 1
        assert counts["POLARIS"] == 3

    def test_four_missions_are_present(self):
        assert len({s.mission for s in default_fleet()}) == 4

    def test_every_member_propagates_over_a_day(self):
        scenario = Scenario.from_calendar(2026, 4, 30, duration_s=86400.0)
        times = np.linspace(0.0, 86400.0, 97)
        for spacecraft in default_fleet():
            positions = spacecraft.ecef_km(scenario, times)
            assert positions.shape == (97, 3)
            assert np.all(np.isfinite(positions))

    def test_four_members_are_near_polar(self):
        near_polar = [s for s in default_fleet() if s.elements.inclination_deg > 90.0]
        assert len(near_polar) == 4
