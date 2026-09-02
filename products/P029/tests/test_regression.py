"""Pinned numbers. A change here is either a bug or a deliberate model change that must
be recorded in CHANGELOG.md. Every value was produced by the validation scripts in this
repository and is quoted with the script that produced it.
"""

from __future__ import annotations

import numpy as np
import pytest

from momentummgr import (
    FixedThresholdScheduler,
    circular_state,
    averaged_controllability,
    dipole_field_eci,
    momentum_budget,
    momentum_per_orbit_eci,
    orbital_period,
    pyramid_four,
    reference_orbit,
    reference_smallsat,
    rollout,
    sample_episode,
    sun_direction_for_beta,
)

# validation/p027_cross_check.py, reference environment
REFERENCE_TOTAL_DH_NMS = np.array([-2.4484285744e-03, 2.8372528957e-03, -2.2167544874e-03])
REFERENCE_SOLAR_DH_NMS = np.array([-1.2065216122e-04, -1.2863853089e-04, 3.7137042242e-04])
REFERENCE_PERIOD_S = 5676.9780285259


def _case() -> tuple:
    sc = reference_smallsat()
    orbit = reference_orbit(500.0)
    sun = sun_direction_for_beta(orbit.inclination_rad, orbit.raan_rad, np.radians(20.0))
    return sc, orbit, sun


def test_period_regression() -> None:
    assert orbital_period(6878137.0) == pytest.approx(REFERENCE_PERIOD_S, rel=1e-12)


def test_total_momentum_regression() -> None:
    sc, orbit, sun = _case()
    dh = momentum_per_orbit_eci(sc, orbit, sun, "total")
    assert np.allclose(dh, REFERENCE_TOTAL_DH_NMS, rtol=5e-10)


def test_solar_momentum_regression() -> None:
    sc, orbit, sun = _case()
    dh = momentum_per_orbit_eci(sc, orbit, sun, "solar")
    assert np.allclose(dh, REFERENCE_SOLAR_DH_NMS, rtol=5e-10)


def test_budget_regression() -> None:
    # validation/p027_cross_check.py section 3 and the CLI budget table.
    sc, orbit, sun = _case()
    budget = momentum_budget(sc, orbit, sun)
    expected = {
        "gravity_gradient": 1.0840622384e-02,
        "aerodynamic": 6.9117755125e-03,
        "solar": 4.1112140090e-04,
        "magnetic": 2.2362180824e-03,
        "total": 4.3541712112e-03,
    }
    for source, value in expected.items():
        assert budget[source]["secular_per_orbit_nms"] == pytest.approx(value, rel=1e-9)


def test_wheel_envelope_regression() -> None:
    assert pyramid_four().guaranteed_body_envelope_nms == pytest.approx(
        0.06666666666666667, rel=1e-14
    )


def test_controllability_eigenvalue_regression() -> None:
    # validation/magnetic_controllability.py, 51.6 deg row: 0.605071, 0.616911, 0.778018
    orbit = reference_orbit(500.0)
    u = np.linspace(0.0, 2.0 * np.pi, 3601)
    r, _ = circular_state(orbit.radius_m, orbit.inclination_rad, orbit.raan_rad, u)
    t = u / (2.0 * np.pi) * orbit.period_s
    _, eig, _ = averaged_controllability(dipole_field_eci(r), t)
    assert eig[0] == pytest.approx(0.605071, abs=5e-6)
    assert eig[1] == pytest.approx(0.616911, abs=5e-6)
    assert eig[2] == pytest.approx(0.778018, abs=5e-6)


def test_episode_regression() -> None:
    # CLI: python -m momentummgr schedule --seed 5000
    ep = sample_episode(5000)
    assert ep.n_windows == 56
    assert ep.envelope_nms == pytest.approx(0.09672907, rel=1e-6)
    m = rollout(ep, FixedThresholdScheduler(0.5, 0.4).decider()).metrics
    assert m.duty_fraction == pytest.approx(0.10447, abs=2e-4)
    assert m.max_h_fraction == pytest.approx(0.58123, abs=2e-4)
    assert m.violated is False
