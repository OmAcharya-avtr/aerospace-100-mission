"""Pinned-value regression tests and a wall-clock budget check.

Every number below was produced by this repository on the machine that built
it and is pinned so that an unintended change to the dynamics, the field
model or the integrator shows up as a test failure rather than as a quietly
different plot.  Tolerances are relative and tight; they are not there to
absorb a real change.
"""

from __future__ import annotations

import time

import numpy as np

from detumblesim import (
    CircularOrbit,
    DetumbleConfig,
    FixedGainPolicy,
    Magnetorquer,
    controllability_report,
    field_magnitude_nt,
    inertia_from_diagonal,
    orbit_field_moments,
    simulate_detumble,
)

REFERENCE_ORBIT = CircularOrbit(altitude_km=500.0, inclination_deg=97.4)


def reference_config(**kw) -> DetumbleConfig:
    args = {
        "inertia": inertia_from_diagonal(0.05, 0.06, 0.04),
        "orbit": REFERENCE_ORBIT,
        "magnetorquer": Magnetorquer.isotropic(0.2),
        "omega0_rad_s": np.radians([8.0, -6.0, 5.0]),
        "duration_s": 23000.0,
        "control_dt_s": 2.0,
        "substeps": 2,
        "target_rate_rad_s": np.radians(1.0),
        "stop_when_detumbled": True,
    }
    args.update(kw)
    return DetumbleConfig(**args)


class TestPinnedValues:
    def test_reference_detumble_run(self):
        r = simulate_detumble(reference_config(), FixedGainPolicy(1.5e5))
        assert np.isclose(r.detumble_time_s, 3285.277687, rtol=1e-6)
        assert np.isclose(r.saturated_fraction, 0.221546, rtol=1e-4)
        assert np.isclose(r.actuation_cost_a2m4s, 102.575773, rtol=1e-6)

    def test_reference_field_moments(self):
        m = orbit_field_moments(REFERENCE_ORBIT, 4000, 10.0 * REFERENCE_ORBIT.period_s)
        assert np.isclose(m.rms_b_t, 3.70592007e-05, rtol=1e-8)
        assert np.isclose(m.mean_b2_t2, 1.37338436e-09, rtol=1e-8)

    def test_reference_controllability(self):
        rep = controllability_report(
            REFERENCE_ORBIT, 4000, 10.0 * REFERENCE_ORBIT.period_s
        )
        assert np.allclose(
            rep.weighted_eigenvalues, [0.45882009, 0.55563383, 0.98554608], rtol=1e-7
        )
        assert np.isclose(rep.anisotropy, 2.148001, rtol=1e-6)

    def test_equatorial_controllability_gap(self):
        eq = CircularOrbit(altitude_km=500.0, inclination_deg=0.0)
        rep = controllability_report(eq, 4000, 10.0 * eq.period_s)
        assert np.allclose(
            rep.weighted_eigenvalues, [0.06049206, 0.96836119, 0.97114675], rtol=1e-7
        )

    def test_reference_field_magnitude(self):
        assert np.isclose(field_magnitude_nt(45.0, 0.0, 500.0), 38269.3111, rtol=1e-8)


class TestComputeBudget:
    def test_reference_run_is_fast(self):
        # The whole package is sized so that a single detumble run costs a
        # fraction of a second on two cores; the sweeps depend on it.
        t0 = time.perf_counter()
        simulate_detumble(reference_config(), FixedGainPolicy(1.5e5))
        assert time.perf_counter() - t0 < 10.0

    def test_full_length_run_stays_inside_the_budget(self):
        t0 = time.perf_counter()
        simulate_detumble(
            reference_config(stop_when_detumbled=False), FixedGainPolicy(1.5e5)
        )
        assert time.perf_counter() - t0 < 30.0
