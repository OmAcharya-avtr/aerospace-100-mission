"""End-to-end integration: build a spacecraft and an orbit, sweep a full orbit, split
secular from cyclic, accumulate momentum, and drive the same path through the CLI in a
child process."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from disturbtorque import (
    SOURCES,
    Orbit,
    Spacecraft,
    beta_angle,
    body_from_lvlh,
    budget,
    compute_profile,
    momentum_accumulation,
    node_axes,
    reference_orbit,
    reference_smallsat,
    sun_direction_for_beta,
)


@pytest.fixture(scope="module")
def profile():
    sc = reference_smallsat()
    orb = reference_orbit(500.0)
    sun = sun_direction_for_beta(orb.inclination_rad, orb.raan_rad, np.radians(20.0))
    return compute_profile(sc, orb, sun, n_samples=721)


def test_full_pipeline_shapes_and_finiteness(profile):
    n = len(profile.u_rad)
    assert n == 721
    assert profile.time_s[0] == 0.0
    assert profile.time_s[-1] == pytest.approx(profile.period_s, rel=1e-14)
    for name in SOURCES:
        assert profile.torques_body[name].shape == (n, 3)
        assert profile.torques_eci[name].shape == (n, 3)
        assert np.all(np.isfinite(profile.torques_body[name]))
        assert np.all(np.isfinite(profile.torques_eci[name]))
    total_b = profile.torque("total", "body")
    assert np.allclose(total_b, sum(profile.torques_body[s] for s in SOURCES))


def test_frame_change_preserves_torque_magnitude(profile):
    for name in (*SOURCES, "total"):
        mb = np.linalg.norm(profile.torque(name, "body"), axis=1)
        me = np.linalg.norm(profile.torque(name, "eci"), axis=1)
        assert np.allclose(mb, me, rtol=1e-12)


def test_secular_plus_cyclic_reconstructs_the_torque(profile):
    for name in (*SOURCES, "total"):
        original = profile.torque(name, "body")
        rebuilt = profile.cyclic(name, "body") + profile.secular(name, "body")
        # subtracting then re-adding the mean is exact to float roundoff on the torque
        # scale, i.e. a few eps * max|T| ~ 1e-21 N m for these 1e-6 N m torques
        assert np.allclose(rebuilt, original, rtol=0.0, atol=1e-20 * np.max(np.abs(original)) / 1e-6)
        # the cyclic part integrates to zero over the orbit, by construction
        mean_cyc = np.trapezoid(profile.cyclic(name, "body"), profile.time_s, axis=0)
        assert np.max(np.abs(mean_cyc)) < 1e-12 * (
            np.max(np.abs(profile.torque(name, "body"))) * profile.period_s + 1e-30
        )


def test_momentum_matches_secular_torque_times_period(profile):
    for name in (*SOURCES, "total"):
        h = momentum_accumulation(profile, name, "eci")
        assert h.shape == profile.torque(name, "eci").shape
        assert np.allclose(h[0], 0.0)
        expected = profile.secular(name, "eci") * profile.period_s
        assert np.allclose(h[-1], expected, rtol=1e-10, atol=1e-18)


def test_gravity_gradient_is_constant_in_the_body_frame_for_nadir_pointing(profile):
    t = profile.torque("gravity_gradient", "body")
    assert np.allclose(t, t[0], rtol=0.0, atol=1e-22)
    assert profile.cyclic_peak("gravity_gradient", "body") < 1e-18


def test_eci_secular_torque_matches_the_orbit_normal_closed_form():
    # For any torque that is constant in LVLH, the ECI orbit average is
    # -(T_lvlh)_y h_hat, because the LVLH x and z axes turn through a full revolution.
    sc = reference_smallsat()
    orb = Orbit(altitude_m=600_000.0, inclination_rad=np.radians(97.8),
                raan_rad=0.9, pitch_rad=np.radians(3.0), roll_rad=np.radians(-4.0))
    sun = sun_direction_for_beta(orb.inclination_rad, orb.raan_rad, np.radians(60.0))
    prof = compute_profile(sc, orb, sun, n_samples=1441, co_rotating_atmosphere=False)
    _, _, h_hat = node_axes(orb.inclination_rad, orb.raan_rad)
    c_bl = body_from_lvlh(orb.yaw_rad, orb.pitch_rad, orb.roll_rad)
    for name in ("gravity_gradient", "aerodynamic"):
        t_lvlh = c_bl.T @ prof.torque(name, "body")[0]
        assert np.allclose(prof.secular(name, "eci"), -t_lvlh[1] * h_hat, rtol=1e-9)


def test_budget_keys_and_consistency(profile):
    b = budget(profile, "body")
    assert set(b) == {*SOURCES, "total"}
    for name, entry in b.items():
        assert entry["peak_nm"] >= entry["rms_nm"] >= 0.0
        assert entry["peak_nm"] >= entry["secular_magnitude_nm"] - 1e-18
        assert entry["secular_momentum_per_orbit_nms"] == pytest.approx(
            entry["secular_magnitude_nm"] * profile.period_s, rel=1e-12
        )
        assert np.isfinite(entry["cyclic_momentum_peak_nms"])
        assert entry["cyclic_momentum_peak_nms"] >= 0.0


def test_beta_angle_round_trip_through_the_profile():
    orb = reference_orbit(700.0)
    for beta_deg in (-70.0, 0.0, 35.0, 75.0):
        s = sun_direction_for_beta(orb.inclination_rad, orb.raan_rad, np.radians(beta_deg))
        assert np.degrees(beta_angle(s, orb.inclination_rad, orb.raan_rad)) == pytest.approx(
            beta_deg, abs=1e-9
        )


def test_no_eclipse_at_high_beta_and_no_solar_torque_ever_exceeds_the_sunlit_value():
    sc = reference_smallsat()
    orb = reference_orbit(500.0)
    sun = sun_direction_for_beta(orb.inclination_rad, orb.raan_rad, np.radians(80.0))
    prof = compute_profile(sc, orb, sun, n_samples=361)
    assert prof.eclipse_fraction == 0.0
    assert np.all(prof.illuminated)
    dark = compute_profile(
        sc, orb, sun_direction_for_beta(orb.inclination_rad, orb.raan_rad, 0.0), n_samples=361
    )
    assert 0.30 < dark.eclipse_fraction < 0.40
    assert np.any(np.linalg.norm(dark.torque("solar", "body"), axis=1) == 0.0)


def test_zero_property_spacecraft_gives_only_gravity_gradient():
    sc = Spacecraft(inertia=np.diag([4.0, 8.0, 10.0]))
    orb = reference_orbit(500.0)
    sun = sun_direction_for_beta(orb.inclination_rad, orb.raan_rad, 0.0)
    prof = compute_profile(sc, orb, sun, n_samples=61)
    for name in ("aerodynamic", "solar", "magnetic"):
        assert np.max(np.abs(prof.torque(name, "body"))) == 0.0
    assert prof.peak_magnitude("gravity_gradient", "body") > 0.0


def test_cli_budget_table_runs():
    out = subprocess.run(
        [sys.executable, "-m", "disturbtorque", "budget", "--altitude-km", "500"],
        capture_output=True, text=True, check=True,
    )
    assert "gravity_gradient" in out.stdout
    assert "aerodynamic" in out.stdout
    assert "eclipse fraction" in out.stdout


def test_cli_budget_json_matches_the_library():
    out = subprocess.run(
        [sys.executable, "-m", "disturbtorque", "budget", "--altitude-km", "500",
         "--beta-deg", "20", "--json"],
        capture_output=True, text=True, check=True,
    )
    res = json.loads(out.stdout)
    sc = reference_smallsat()
    orb = reference_orbit(500.0)
    sun = sun_direction_for_beta(orb.inclination_rad, orb.raan_rad, np.radians(20.0))
    prof = compute_profile(sc, orb, sun, n_samples=721)
    assert res["period_s"] == pytest.approx(prof.period_s, rel=1e-12)
    assert res["eclipse_fraction"] == pytest.approx(prof.eclipse_fraction, rel=1e-12)
    for name in SOURCES:
        assert res["sources"][name]["peak_nm"] == pytest.approx(
            prof.peak_magnitude(name, "body"), rel=1e-12
        )


def test_cli_sweep_and_bad_arguments():
    out = subprocess.run(
        [sys.executable, "-m", "disturbtorque", "sweep", "--altitude-km", "400", "600"],
        capture_output=True, text=True, check=True,
    )
    assert "400.0" in out.stdout and "600.0" in out.stdout
    bad = subprocess.run(
        [sys.executable, "-m", "disturbtorque", "budget", "--altitude-km", "-10"],
        capture_output=True, text=True,
    )
    assert bad.returncode != 0
