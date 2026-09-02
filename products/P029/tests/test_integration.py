"""Integration tests: the CLI in a child process, and one end-to-end sizing workflow."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from momentummgr import (
    averaged_controllability,
    momentum_budget,
    momentum_per_orbit_eci,
    pyramid_four,
    reference_orbit,
    reference_smallsat,
    sun_direction_for_beta,
    sweep_orbit,
    thruster_dump,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    """Run ``python -m momentummgr`` in a child process from the repository root."""
    return subprocess.run(
        [sys.executable, "-m", "momentummgr", *args],
        capture_output=True, text=True, cwd=ROOT, check=False,
    )


def test_cli_budget_text() -> None:
    r = run_cli("budget", "--altitude-km", "500")
    assert r.returncode == 0, r.stderr
    assert "gravity_gradient" in r.stdout
    assert "wheel array envelope" in r.stdout


def test_cli_budget_json_matches_the_library() -> None:
    r = run_cli("budget", "--altitude-km", "500", "--beta-deg", "20", "--json")
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    sc = reference_smallsat()
    orbit = reference_orbit(500.0)
    sun = sun_direction_for_beta(orbit.inclination_rad, orbit.raan_rad, np.radians(20.0))
    dh = float(np.linalg.norm(momentum_per_orbit_eci(sc, orbit, sun, "total")))
    assert payload["budget_nms"]["total"]["secular_per_orbit_nms"] == pytest.approx(dh, rel=1e-12)
    assert payload["period_s"] == pytest.approx(orbit.period_s, rel=1e-12)
    assert payload["wheel_envelope_nms"] == pytest.approx(4.0 / 3.0 * 0.05, rel=1e-12)


def test_cli_controllability_json() -> None:
    r = run_cli("controllability", "--inclination-deg", "51.6", "--json")
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    assert np.trace(np.array(payload["gramian"])) == pytest.approx(2.0, abs=1e-10)
    assert min(payload["eigenvalues"]) > 0.0


def test_cli_schedule_json() -> None:
    r = run_cli("schedule", "--seed", "5000", "--json")
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    assert 0.0 <= payload["duty_fraction"] <= 1.0
    assert payload["violated"] is False


def test_cli_rejects_bad_input_with_exit_code_two() -> None:
    r = run_cli("budget", "--altitude-km", "-100")
    assert r.returncode == 2
    assert "altitude_m must be > 0" in r.stderr


def test_end_to_end_sizing_workflow() -> None:
    """Budget -> wheel sizing -> desaturation cadence -> propellant, all consistent."""
    sc = reference_smallsat()
    orbit = reference_orbit(500.0)
    sun = sun_direction_for_beta(orbit.inclination_rad, orbit.raan_rad, np.radians(20.0))

    budget = momentum_budget(sc, orbit, sun)
    secular = budget["total"]["secular_per_orbit_nms"]
    assert secular == pytest.approx(4.3541712112e-03, rel=1e-6)

    wheels = pyramid_four(max_momentum_nms=0.05)
    envelope = wheels.guaranteed_body_envelope_nms
    orbits_to_fill = envelope / secular
    assert 10.0 < orbits_to_fill < 25.0

    dump = thruster_dump(secular * orbits_to_fill, 0.5, 220.0)
    assert dump.propellant_kg == pytest.approx(
        2.0 * envelope / 0.5 / (220.0 * 9.80665), rel=1e-9
    )

    sweep = sweep_orbit(sc, orbit, sun, n_samples=361)
    _, eig, _ = averaged_controllability(sweep.b_eci_t, sweep.time_s)
    assert float(eig[0]) > 0.5  # every direction dumpable for over half the orbit

    # the same allocation must reproduce the body momentum it was asked for
    request = np.array([1.0, -2.0, 0.5])
    request = request / np.linalg.norm(request) * 0.4 * envelope
    alloc = wheels.allocate(request, avoid_zero_speed=True, envelope_fraction=0.7)
    assert np.allclose(wheels.body_momentum(alloc.wheel_momentum_nms), request, atol=1e-15)
    assert alloc.feasible


def test_budget_sources_sum_to_the_total() -> None:
    sc = reference_smallsat()
    orbit = reference_orbit(500.0)
    sun = sun_direction_for_beta(orbit.inclination_rad, orbit.raan_rad, np.radians(20.0))
    parts = sum(
        momentum_per_orbit_eci(sc, orbit, sun, s)
        for s in ("gravity_gradient", "aerodynamic", "solar", "magnetic")
    )
    total = momentum_per_orbit_eci(sc, orbit, sun, "total")
    assert np.allclose(parts, total, rtol=1e-12)
