"""Reference target: Riccati convergence, filter algebra, controller behaviour."""

from __future__ import annotations

import math

import numpy as np
import pytest

from faultinject.target import (
    DT,
    N_STEPS,
    U_MAX,
    DoubleIntegratorPlant,
    GncController,
    SanitisingGncController,
    gain_convergence,
    kalman_gain,
    measurement_covariance,
    plant_matrices,
    process_covariance,
    reference,
    steady_state_gain,
)


def test_plant_matrices_known_answer():
    a, b = plant_matrices(0.02)
    assert a.tolist() == [[1.0, 0.02], [0.0, 1.0]]
    assert b[0, 0] == pytest.approx(0.0002, abs=1e-18)
    assert b[1, 0] == 0.02


def test_process_covariance_structure():
    q = process_covariance(0.02, 0.5)
    # Bar-Shalom et al. 2001 Sec. 6.2.2: Q = sigma_a^2 [[dt^4/4, dt^3/2],[dt^3/2, dt^2]]
    assert q[0, 0] == pytest.approx(0.25 * 0.02**4 * 0.25, rel=1e-15)
    assert q[0, 1] == q[1, 0]
    assert q[1, 1] == pytest.approx(0.25 * 0.02**2, rel=1e-15)
    assert np.all(np.linalg.eigvals(q) >= 0)


def test_measurement_covariance_diagonal():
    r = measurement_covariance(0.05, 0.02)
    assert r[0, 0] == pytest.approx(0.0025, rel=1e-15)
    assert r[1, 1] == pytest.approx(0.0004, rel=1e-15)
    assert r[0, 1] == 0.0 and r[1, 0] == 0.0


def test_riccati_converged():
    iters, change = gain_convergence()
    assert change < 1e-15
    assert 1 <= iters <= 2000


def test_riccati_raises_if_it_cannot_converge():
    with pytest.raises(RuntimeError, match="did not converge"):
        steady_state_gain(tol=1e-30, max_iter=5)


def test_error_dynamics_are_stable():
    a, _ = plant_matrices()
    k = kalman_gain()
    m = a @ (np.eye(2) - k @ np.eye(2))
    assert np.all(np.abs(np.linalg.eigvals(m)) < 1.0)


def test_kalman_gain_regression():
    # Recorded from the converged Riccati recursion in this build.
    k = kalman_gain()
    assert k[0, 0] == pytest.approx(0.00794229, abs=5e-8)
    assert k[1, 1] == pytest.approx(0.39032435, abs=5e-8)


def test_reference_is_sinusoidal():
    assert reference(0) == 0.0
    assert abs(reference(k := 62)) <= 1.0 + 1e-12
    assert k == 62


def test_controller_prediction_uses_previous_command():
    c = GncController()
    c.reset()
    c.p_hat, c.v_hat, c.u_prev = 1.0, 2.0, 3.0
    # p_pred = 1 + 0.02*2 + 0.5*0.02^2*3 = 1.0406 ; v_pred = 2 + 0.02*3 = 2.06
    c.step(0, {"pos": 1.0406, "vel": 2.06})
    step, e_p, e_v = c.innovations[0]
    assert step == 0
    assert e_p == pytest.approx(0.0, abs=1e-12)
    assert e_v == pytest.approx(0.0, abs=1e-12)


def test_zero_innovation_leaves_estimate_at_prediction():
    c = GncController()
    c.reset()
    c.p_hat, c.v_hat, c.u_prev = 0.5, 0.0, 0.0
    c.step(0, {"pos": 0.5, "vel": 0.0})
    assert c.p_hat == pytest.approx(0.5, abs=1e-15)
    assert c.v_hat == pytest.approx(0.0, abs=1e-15)


def test_invalid_measurement_skips_update():
    c = GncController()
    c.reset()
    c.step(0, {"pos": 99.0, "vel": 99.0, "valid": 0.0})
    assert c.updates == 0
    assert c.skipped == 1
    assert c.innovations == []
    assert c.p_hat == 0.0


def test_command_saturation():
    c = GncController()
    c.reset()
    c.step(0, {"pos": -1000.0, "vel": 0.0})
    assert c.u_prev == U_MAX
    c.reset()
    c.step(0, {"pos": 1000.0, "vel": 0.0})
    assert c.u_prev == -U_MAX


def test_nan_propagates_through_the_plain_controller():
    c = GncController()
    c.reset()
    out = c.step(0, {"pos": float("nan"), "vel": 0.0})
    assert math.isnan(out["u"])


def test_sanitising_controller_absorbs_nan():
    c = SanitisingGncController()
    c.reset()
    out = c.step(0, {"pos": float("nan"), "vel": 0.0})
    assert math.isfinite(out["u"])


def test_plant_advance_known_answer():
    p = DoubleIntegratorPlant()
    p.reset()
    p.v = 1.0
    p.advance(2.0, 0.0, 0.0)
    # p = 0 + 0.02*1 + 0.5*0.0004*2 = 0.0204 ; v = 1 + 0.04 = 1.04
    assert p.p == pytest.approx(0.0204, abs=1e-15)
    assert p.v == pytest.approx(1.04, abs=1e-15)


def test_plant_measure_adds_noise():
    p = DoubleIntegratorPlant()
    p.reset()
    m = p.measure(0.1, -0.2)
    assert m == {"pos": 0.1, "vel": -0.2, "valid": 1.0}


def test_defaults():
    assert DT == 0.02
    assert N_STEPS == 150
