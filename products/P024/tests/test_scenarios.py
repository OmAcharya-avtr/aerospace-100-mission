"""Seeded synthetic scenario generation."""

from __future__ import annotations

import numpy as np
import pytest

from detumblesim.scenarios import (
    DEFAULT_TARGET_RATE_RAD_S,
    sample_scenario,
    sample_scenarios,
)


class TestSampling:
    def test_is_deterministic_in_the_seed(self):
        a, b = sample_scenario(42), sample_scenario(42)
        assert np.allclose(a.inertia, b.inertia)
        assert np.allclose(a.omega0_rad_s, b.omega0_rad_s)
        assert np.allclose(a.q0, b.q0)
        assert a.orbit == b.orbit

    def test_different_seeds_differ(self):
        assert not np.allclose(sample_scenario(1).omega0_rad_s, sample_scenario(2).omega0_rad_s)

    def test_ranges_are_respected(self):
        for s in sample_scenarios(60, 0):
            d = np.diag(s.inertia)
            assert np.all(d > 0.0)
            assert 0.6 * 0.01 <= d.min() and d.max() <= 1.6 * 0.30 + 1e-12
            assert 3.0 - 1e-9 <= np.degrees(s.rate0_rad_s) <= 20.0 + 1e-9
            assert 400.0 <= s.orbit.altitude_km <= 800.0
            assert 20.0 <= s.orbit.inclination_deg <= 100.0
            m = float(s.magnetorquer.max_dipole_am2[0])
            assert 0.05 - 1e-12 <= m <= 0.50 + 1e-12
            assert np.isclose(float(np.linalg.norm(s.q0)), 1.0)

    def test_inertia_satisfies_the_triangle_inequality(self):
        for s in sample_scenarios(60, 100):
            a, b, c = np.diag(s.inertia)
            assert a + b >= c and b + c >= a and c + a >= b

    def test_inertia_scale_is_the_mean_moment(self):
        s = sample_scenario(3)
        assert np.isclose(s.inertia_scale_kgm2, float(np.mean(np.diag(s.inertia))))

    def test_seeds_are_consecutive(self):
        assert [s.seed for s in sample_scenarios(4, 77)] == [77, 78, 79, 80]

    def test_rejects_bad_count(self):
        with pytest.raises(ValueError, match="n must be"):
            sample_scenarios(0)

    def test_to_config_defaults(self):
        cfg = sample_scenario(5).to_config()
        assert cfg.control_dt_s == 2.0
        assert cfg.stop_when_detumbled is True
        assert np.isclose(cfg.target_rate_rad_s, DEFAULT_TARGET_RATE_RAD_S)
        assert np.isclose(DEFAULT_TARGET_RATE_RAD_S, np.radians(1.0))

    def test_target_rate_is_well_above_the_orbital_rate(self):
        # B-dot cannot drive the body rate below roughly the orbital rate, so
        # the default threshold has to sit clearly above it.
        for s in sample_scenarios(20, 200):
            assert DEFAULT_TARGET_RATE_RAD_S > 8.0 * s.orbit.mean_motion_rad_s
