"""Unit and integration tests for keepout.windows."""

import datetime as dt

import numpy as np
import pytest

from keepout import (
    EARTH_MU,
    EARTH_RADIUS_M,
    OrbitPointingProblem,
    Window,
    circular_orbit_positions,
    julian_date,
    orbital_period,
    windows_from_margin,
)

DEG = np.pi / 180.0


class TestWindow:
    def test_duration(self):
        assert Window(10.0, 25.0).duration == pytest.approx(15.0)

    def test_zero_length_allowed(self):
        assert Window(3.0, 3.0).duration == 0.0

    def test_reversed_rejected(self):
        with pytest.raises(ValueError, match="must be >="):
            Window(5.0, 1.0)

    def test_nonfinite_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            Window(0.0, np.inf)


class TestOrbitalPeriod:
    def test_leo_period(self):
        # a = 6378137 + 500000 = 6878137 m, mu = 3.986004418e14 m^3/s^2.
        # T = 2 pi sqrt(a^3/mu); a^3 = 3.25429...e20, a^3/mu = 816380 s^2,
        # sqrt = 903.54 s, T = 5677 s = 94.6 min.
        t = orbital_period(500e3)
        a = EARTH_RADIUS_M + 500e3
        assert t == pytest.approx(2 * np.pi * np.sqrt(a**3 / EARTH_MU))
        assert 94.0 < t / 60.0 < 95.0

    def test_geo_period_is_a_sidereal_day(self):
        # 42164 km semi-major axis is defined by T = 86164 s; using the standard
        # GEO altitude 35786 km must reproduce that to within a few seconds.
        assert orbital_period(35786e3) == pytest.approx(86164.0, abs=10.0)

    def test_negative_altitude(self):
        with pytest.raises(ValueError, match="must be >= 0"):
            orbital_period(-1.0)


class TestCircularOrbitPositions:
    def test_radius_is_constant(self):
        t = np.linspace(0, 20000, 97)
        r = circular_orbit_positions(t, 700e3, 51.6 * DEG, 0.3, 1.1)
        assert np.allclose(np.linalg.norm(r, axis=-1), EARTH_RADIUS_M + 700e3)

    def test_equatorial_orbit_stays_in_the_plane(self):
        t = np.linspace(0, 6000, 51)
        r = circular_orbit_positions(t, 400e3, 0.0)
        assert np.allclose(r[:, 2], 0.0, atol=1e-6)

    def test_epoch_position_at_the_node(self):
        r = circular_orbit_positions(0.0, 400e3, 30 * DEG, raan=0.0, arg_lat0=0.0)
        assert np.allclose(r, [EARTH_RADIUS_M + 400e3, 0.0, 0.0])

    def test_returns_to_start_after_one_period(self):
        p = orbital_period(500e3)
        r0 = circular_orbit_positions(0.0, 500e3, 45 * DEG, 0.7, 0.2)
        r1 = circular_orbit_positions(p, 500e3, 45 * DEG, 0.7, 0.2)
        assert np.allclose(r0, r1, atol=1e-6)

    def test_max_latitude_equals_inclination(self):
        inc = 63.4 * DEG
        t = np.linspace(0, orbital_period(800e3), 500)
        r = circular_orbit_positions(t, 800e3, inc)
        lat = np.arcsin(r[:, 2] / np.linalg.norm(r, axis=-1))
        assert lat.max() == pytest.approx(inc, abs=1e-3)

    def test_bad_inclination(self):
        with pytest.raises(ValueError, match="inclination"):
            circular_orbit_positions(0.0, 400e3, 4.0)


class TestWindowsFromMargin:
    def test_all_allowed_is_one_window(self):
        t = np.linspace(0, 10, 11)
        w = windows_from_margin(t, np.ones_like(t))
        assert len(w) == 1
        assert (w[0].start, w[0].end) == (0.0, 10.0)

    def test_none_allowed(self):
        t = np.linspace(0, 10, 11)
        assert windows_from_margin(t, -np.ones_like(t)) == []

    def test_two_windows(self):
        t = np.arange(6.0)
        m = np.array([1.0, 1.0, -1.0, -1.0, 1.0, 1.0])
        w = windows_from_margin(t, m)
        assert len(w) == 2
        assert (w[0].start, w[0].end) == (0.0, 1.0)
        assert (w[1].start, w[1].end) == (4.0, 5.0)

    def test_refinement_finds_the_exact_crossing(self):
        # margin(t) = 2.5 - t, so the boundary is exactly t = 2.5.
        def fn(x):
            return 2.5 - x

        t = np.arange(6.0)
        w = windows_from_margin(t, fn(t), fn)
        assert len(w) == 1
        assert w[0].start == 0.0
        assert w[0].end == pytest.approx(2.5, abs=1e-6)

    def test_empty_input(self):
        assert windows_from_margin(np.array([]), np.array([])) == []

    def test_non_monotone_times_rejected(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            windows_from_margin([0.0, 2.0, 1.0], [1.0, 1.0, 1.0])

    def test_shape_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            windows_from_margin([0.0, 1.0], [1.0])


class TestOrbitPointingProblem:
    @pytest.fixture
    def problem(self):
        return OrbitPointingProblem(
            epoch_jd=julian_date(dt.datetime(2026, 3, 20, 0, 0, 0)),
            altitude_m=550e3,
            inclination=97.6 * DEG,
            raan=0.4,
            sun_exclusion=45.0 * DEG,
            earth_exclusion=10.0 * DEG,
            moon_exclusion=15.0 * DEG,
        )

    def test_three_cones(self, problem):
        ks = problem.keepout_at(0.0)
        assert ks.names == ("sun", "earth", "moon")

    def test_earth_cone_half_angle(self, problem):
        # Limb reference: half-angle = arcsin(R_E/(R_E+h)) + 10 deg.
        # At 550 km, arcsin(6378137/6928137) = arcsin(0.9206...) = 67.0159 deg,
        # so the cone half-angle is 77.0159 deg.
        ks = problem.keepout_at(0.0)
        earth = [c for c in ks if c.name == "earth"][0]
        expected = np.degrees(np.arcsin(EARTH_RADIUS_M / (EARTH_RADIUS_M + 550e3))) + 10.0
        assert earth.half_angle_deg == pytest.approx(expected, abs=1e-9)
        assert earth.half_angle_deg == pytest.approx(77.0159, abs=1e-3)

    def test_earth_cone_points_at_nadir(self, problem):
        ks = problem.keepout_at(0.0)
        earth = [c for c in ks if c.name == "earth"][0]
        r = problem.position(0.0)
        assert np.allclose(earth.axis, -r / np.linalg.norm(r), atol=1e-12)

    def test_optional_bodies(self):
        p = OrbitPointingProblem(2461041.5, 400e3, 0.0, sun_exclusion=30 * DEG)
        assert p.keepout_at(0.0).names == ("sun",)

    def test_at_least_one_body_required(self):
        with pytest.raises(ValueError, match="at least one"):
            OrbitPointingProblem(2461041.5, 400e3, 0.0)

    def test_bad_reference(self):
        with pytest.raises(ValueError, match="'limb' or 'center'"):
            OrbitPointingProblem(2461041.5, 400e3, 0.0, sun_exclusion=0.1, reference="x")

    def test_period_property(self, problem):
        assert problem.period == pytest.approx(orbital_period(550e3))

    def test_margin_series_shape(self, problem):
        t = np.linspace(0.0, problem.period, 20)
        m = problem.margin_series(t, [1.0, 0.0, 0.0])
        assert m.shape == (20,)

    def test_windows_partition_the_scan(self, problem):
        # An orbit-normal target chosen so the Earth cone sweeps past it.
        target = [0.0, 0.0, 1.0]
        t = np.arange(0.0, 2 * problem.period, 20.0)
        w = problem.windows(t, target)
        assert all(x.start <= x.end for x in w)
        for a, bb in zip(w[:-1], w[1:], strict=True):
            assert a.end < bb.start

    def test_windows_agree_with_a_direct_sample_test(self, problem):
        target = [0.2, -0.9, 0.3]
        t = np.arange(0.0, problem.period, 15.0)
        w = problem.windows(t, target, refine=False)
        inside = np.zeros_like(t, dtype=bool)
        for x in w:
            inside |= (t >= x.start - 1e-9) & (t <= x.end + 1e-9)
        direct = problem.margin_series(t, target) >= 0.0
        assert np.array_equal(inside, direct)

    def test_refined_boundary_has_near_zero_margin(self, problem):
        target = [0.0, 0.0, 1.0]
        t = np.arange(0.0, 2 * problem.period, 20.0)
        w = problem.windows(t, target, refine=True)
        interior = [x for x in w if x.start > t[0] + 1e-6]
        assert interior, "expected at least one window whose start was refined"
        for x in interior:
            assert abs(problem.margin(x.start, target)) < 1e-7

    def test_sun_cone_is_not_geocentric_only(self, problem):
        # The Sun cone axis is the topocentric direction, so it must differ from
        # the geocentric direction by roughly R_orbit / 1 AU (about 0.0026 deg
        # at 550 km altitude) -- small, but not zero.
        from keepout.bodies import sun_direction_mod
        from keepout.geometry import angular_separation

        ks = problem.keepout_at(0.0)
        sun = [c for c in ks if c.name == "sun"][0]
        geo, _ = sun_direction_mod(problem.epoch_jd)
        sep_deg = np.degrees(angular_separation(sun.axis, geo))
        assert 0.0 < sep_deg < 0.01
