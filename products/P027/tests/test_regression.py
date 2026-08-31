"""Regression and benchmark tests.

The pinned numbers below were produced by this package on 2026-08-31 and reproduced in
``validation/``. They exist so that a refactor that silently changes a coefficient, a
frame convention or a quadrature default is caught. Any change to them must be a
deliberate, documented change, not a drift.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from disturbtorque import (
    Orbit,
    body_from_lvlh,
    budget,
    compute_profile,
    dipole_field_eci,
    mean_dipole_field_over_orbit,
    momentum_accumulation,
    reference_orbit,
    reference_smallsat,
    sun_direction_for_beta,
)
from disturbtorque.atmosphere import EXPONENTIAL_TABLE

# Reference case: 500 km, i = 51.6 deg, RAAN 0, beta 20 deg, 5 deg pitch and roll,
# reference smallsat, 721 samples, body frame. Matches validation/momentum_integration.py
# and validation/leo_smallsat_magnitudes.py.
PINNED_BODY = {
    "gravity_gradient": dict(peak=2.0126133057e-06, rms=2.0126133057e-06,
                             secular=2.0126133057e-06, cyclic_peak=0.0),
    "aerodynamic": dict(peak=1.2880737478e-06, rms=1.2765149587e-06,
                        secular=1.2754531559e-06, cyclic_peak=7.3203240052e-08),
    "solar": dict(peak=5.0071885317e-07, rms=2.9663533582e-07,
                  secular=9.2892095078e-08, cyclic_peak=5.3771286057e-07),
    "magnetic": dict(peak=4.1300110129e-06, rms=2.9969185233e-06,
                     secular=1.7530943303e-06, cyclic_peak=3.2427562071e-06),
    "total": dict(peak=4.2034421119e-06, rms=3.1679996976e-06,
                  secular=2.2312193670e-06, cyclic_peak=3.2203999248e-06),
}
PINNED_PERIOD_S = 5676.9780285259
PINNED_ECLIPSE_FRACTION = 0.3703190014


@pytest.fixture(scope="module")
def reference_profile():
    sc = reference_smallsat()
    orb = reference_orbit(500.0)
    sun = sun_direction_for_beta(orb.inclination_rad, orb.raan_rad, np.radians(20.0))
    return compute_profile(sc, orb, sun, n_samples=721)


def test_reference_case_period_and_eclipse(reference_profile):
    assert reference_profile.period_s == pytest.approx(PINNED_PERIOD_S, rel=1e-11)
    assert reference_profile.eclipse_fraction == pytest.approx(PINNED_ECLIPSE_FRACTION, rel=1e-9)


@pytest.mark.parametrize("source", list(PINNED_BODY))
def test_reference_case_torque_budget_regression(reference_profile, source):
    b = budget(reference_profile, "body")[source]
    pin = PINNED_BODY[source]
    assert b["peak_nm"] == pytest.approx(pin["peak"], rel=1e-9)
    assert b["rms_nm"] == pytest.approx(pin["rms"], rel=1e-9)
    assert b["secular_magnitude_nm"] == pytest.approx(pin["secular"], rel=1e-9)
    if pin["cyclic_peak"] == 0.0:
        assert b["cyclic_peak_nm"] < 1e-18
    else:
        assert b["cyclic_peak_nm"] == pytest.approx(pin["cyclic_peak"], rel=1e-9)


def test_reference_case_momentum_regression(reference_profile):
    # dh over one orbit, ECI frame, N = 721. Cross-checked in
    # validation/momentum_integration.py against a QUADPACK reference for the solar term
    # and against a spectrally converged trapezoid for the three continuous terms.
    h = momentum_accumulation(reference_profile, "total", "eci")[-1]
    assert float(np.linalg.norm(h)) == pytest.approx(4.3538152233e-03, rel=1e-8)


def test_altitude_sweep_regression():
    # Peak total torque, beta = 0, N = 361. The aerodynamic column falls four orders of
    # magnitude between 300 and 800 km; see validation/leo_smallsat_magnitudes.py.
    sc = reference_smallsat()
    expected_aero = {300.0: 4.6186e-05, 500.0: 1.2881e-06, 800.0: 2.0630e-08}
    for alt, want in expected_aero.items():
        orb = reference_orbit(alt)
        sun = sun_direction_for_beta(orb.inclination_rad, orb.raan_rad, 0.0)
        prof = compute_profile(sc, orb, sun, n_samples=361)
        assert prof.peak_magnitude("aerodynamic", "body") == pytest.approx(want, rel=1e-4)


def test_atmosphere_table_boundary_continuity_regression():
    # Every band's base density must equal the previous band's value at the shared
    # boundary. Worst mismatch above 25 km is 9.588e-05 relative; this is how a mistyped
    # digit in the 84-number table would be caught.
    worst = max(
        abs(r0 * np.exp(-(h1 - h0) / hs) - r1) / r1
        for (h0, r0, hs), (h1, r1, _) in zip(EXPONENTIAL_TABLE[1:-1], EXPONENTIAL_TABLE[2:])
    )
    assert worst == pytest.approx(9.5877e-05, rel=1e-3)
    assert len(EXPONENTIAL_TABLE) == 28


def test_orbit_averaged_field_closed_form_regression():
    r = 6878137.0
    inc, raan = np.radians(51.6), 0.4
    u = np.linspace(0.0, 2 * np.pi, 40001)
    from disturbtorque import circular_orbit_state

    r_eci, _ = circular_orbit_state(r, inc, raan, u)
    numeric = np.trapezoid(dipole_field_eci(r_eci), u, axis=0) / (2 * np.pi)
    analytic = mean_dipole_field_over_orbit(r, inc, raan)
    assert np.allclose(numeric, analytic, rtol=1e-10)


def test_body_from_lvlh_convention_regression():
    # A pure pitch of theta puts nadir at (-sin theta, 0, cos theta) in the body frame.
    theta = np.radians(30.0)
    nadir = body_from_lvlh(pitch_rad=theta) @ np.array([0.0, 0.0, 1.0])
    assert np.allclose(nadir, [-np.sin(theta), 0.0, np.cos(theta)], atol=1e-15)


def test_profile_benchmark_stays_inside_the_documented_compute_budget():
    # The documented budget is that a 721-sample orbit sweep completes in about 0.21 s on
    # two cores. The *minimum* over repeats is used, not the mean, because a shared or
    # loaded machine inflates individual timings without saying anything about the code;
    # a 20x margin on top of that still catches an order-of-magnitude regression.
    sc = reference_smallsat()
    orb = reference_orbit(500.0)
    sun = sun_direction_for_beta(orb.inclination_rad, orb.raan_rad, np.radians(20.0))
    compute_profile(sc, orb, sun, n_samples=721)  # warm up imports and caches
    best = float("inf")
    for _ in range(5):
        t0 = time.perf_counter()
        compute_profile(sc, orb, sun, n_samples=721)
        best = min(best, time.perf_counter() - t0)
    assert best < 5.0, f"fastest of 5 721-sample sweeps took {best:.3f} s"


def test_high_inclination_and_retrograde_orbits_are_handled():
    sc = reference_smallsat()
    for inc_deg in (0.0, 90.0, 97.8, 120.0):
        orb = Orbit(altitude_m=600_000.0, inclination_rad=np.radians(inc_deg),
                    pitch_rad=np.radians(5.0))
        sun = sun_direction_for_beta(orb.inclination_rad, orb.raan_rad, 0.0)
        prof = compute_profile(sc, orb, sun, n_samples=181)
        assert np.all(np.isfinite(prof.torque("total", "body")))
        assert prof.peak_magnitude("total", "body") > 0.0
