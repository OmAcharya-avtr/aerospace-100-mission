"""Two-body propagation: conservation, round trips and refused regimes."""

from __future__ import annotations

import numpy as np
import pytest

from pointahead.kepler import (
    MU_EARTH,
    KeplerTrajectory,
    LinearTrajectory,
    propagate_two_body,
    stumpff_c2,
    stumpff_c3,
)


def test_stumpff_values_at_zero_are_the_series_limits() -> None:
    assert stumpff_c2(0.0) == pytest.approx(0.5, rel=1e-15)
    assert stumpff_c3(0.0) == pytest.approx(1.0 / 6.0, rel=1e-15)


@pytest.mark.parametrize("psi", [0.5, -0.5, 2.0, -2.0, 1.0e-2, -1.0e-2])
def test_stumpff_closed_form_agrees_where_the_closed_form_is_accurate(psi: float) -> None:
    # The reference values are built from the half-angle identities
    # 1 - cos x = 2 sin^2(x / 2) and cosh x - 1 = 2 sinh^2(x / 2), which do not
    # subtract two nearly equal numbers, so this is an independent evaluation
    # and not a restatement of the implementation.
    if psi > 0:
        root = float(np.sqrt(psi))
        c2 = 2.0 * np.sin(0.5 * root) ** 2 / psi
        c3 = (root - np.sin(root)) / root**3
    else:
        root = float(np.sqrt(-psi))
        c2 = -2.0 * np.sinh(0.5 * root) ** 2 / psi
        c3 = (np.sinh(root) - root) / root**3
    assert stumpff_c2(psi) == pytest.approx(float(c2), rel=1e-12)
    assert stumpff_c3(psi) == pytest.approx(float(c3), rel=1e-7)


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_stumpff_is_continuous_across_the_series_switch(sign: float) -> None:
    # The implementation switches from the closed form to a series at
    # |psi| = 1e-6. A discontinuity there would be a defect that no single-sided
    # test could see, and at |psi| = 1e-7 the closed form is the inaccurate one,
    # so the switch is checked for continuity rather than against a closed form.
    below = sign * 0.9e-6
    above = sign * 1.1e-6
    assert stumpff_c2(below) == pytest.approx(stumpff_c2(above), rel=1e-7)
    assert stumpff_c3(below) == pytest.approx(stumpff_c3(above), rel=1e-7)


def test_circular_orbit_returns_after_one_period(mu: float) -> None:
    # a = 7000 km, mu = 3.986004418e14 m^3/s^2:
    #   v_circ = sqrt(mu / a) = 7546.0532901... m/s
    #   T = 2 pi sqrt(a^3 / mu) = 5828.5166376... s
    a = 7.0e6
    speed = float(np.sqrt(mu / a))
    period = float(2.0 * np.pi * np.sqrt(a**3 / mu))
    assert speed == pytest.approx(7546.053290107542, rel=1e-12)
    assert period == pytest.approx(5828.516637686015, rel=1e-12)
    step = propagate_two_body([a, 0.0, 0.0], [0.0, speed, 0.0], period, mu)
    assert float(np.linalg.norm(step.position_m - np.array([a, 0.0, 0.0]))) < 1.0e-6
    assert float(np.linalg.norm(step.velocity_mps - np.array([0.0, speed, 0.0]))) < 1.0e-9


def test_circular_orbit_half_period_is_the_antipode(mu: float) -> None:
    a = 7.0e6
    speed = float(np.sqrt(mu / a))
    period = float(2.0 * np.pi * np.sqrt(a**3 / mu))
    step = propagate_two_body([a, 0.0, 0.0], [0.0, speed, 0.0], 0.5 * period, mu)
    assert float(np.linalg.norm(step.position_m - np.array([-a, 0.0, 0.0]))) < 1.0e-6


def test_propagation_round_trip(mu: float) -> None:
    r0 = np.array([7.2e6, 1.1e6, -3.0e5])
    v0 = np.array([-1.2e3, 7.1e3, 2.0e2])
    forward = propagate_two_body(r0, v0, 900.0, mu)
    back = propagate_two_body(forward.position_m, forward.velocity_mps, -900.0, mu)
    assert float(np.linalg.norm(back.position_m - r0)) < 1.0e-6
    assert float(np.linalg.norm(back.velocity_mps - v0)) < 1.0e-9


def test_specific_energy_and_angular_momentum_are_conserved(mu: float) -> None:
    r0 = np.array([8.0e6, 0.0, 1.0e6])
    v0 = np.array([0.0, 6.9e3, 1.0e3])

    def energy(r: np.ndarray, v: np.ndarray) -> float:
        return 0.5 * float(np.dot(v, v)) - mu / float(np.linalg.norm(r))

    before = energy(r0, v0)
    momentum_before = np.cross(r0, v0)
    step = propagate_two_body(r0, v0, 2400.0, mu)
    after = energy(step.position_m, step.velocity_mps)
    assert abs(after - before) / abs(before) < 1.0e-12
    momentum_after = np.cross(step.position_m, step.velocity_mps)
    assert (
        float(np.linalg.norm(momentum_after - momentum_before))
        / float(np.linalg.norm(momentum_before))
        < 1.0e-12
    )


def test_fg_residual_is_at_rounding(mu: float) -> None:
    # f * gdot - fdot * g = 1 identically in exact arithmetic; it is reported
    # rather than asserted bit-exactly, which floating point cannot promise.
    step = propagate_two_body([7.1e6, 0.0, 0.0], [0.0, 7.4e3, 1.0e2], 1200.0, mu)
    assert step.fg_residual < 1.0e-12


def test_zero_interval_returns_the_input_state(mu: float) -> None:
    r0 = np.array([7.0e6, 1.0e5, 0.0])
    v0 = np.array([0.0, 7.5e3, 0.0])
    step = propagate_two_body(r0, v0, 0.0, mu)
    assert np.array_equal(step.position_m, r0)
    assert np.array_equal(step.velocity_mps, v0)
    assert step.iterations == 0


def test_hyperbolic_orbit_propagates(mu: float) -> None:
    # Escape speed at 7000 km is sqrt(2 mu / r) = 10672.0... m/s; 1.3 times
    # that is firmly hyperbolic.
    r0 = np.array([7.0e6, 0.0, 0.0])
    escape = float(np.sqrt(2.0 * mu / 7.0e6))
    v0 = np.array([0.0, 1.3 * escape, 0.0])
    step = propagate_two_body(r0, v0, 600.0, mu)
    assert float(np.linalg.norm(step.position_m)) > 7.0e6
    assert step.fg_residual < 1.0e-10


def test_parabolic_orbit_is_refused(mu: float) -> None:
    r0 = np.array([7.0e6, 0.0, 0.0])
    escape = float(np.sqrt(2.0 * mu / 7.0e6))
    with pytest.raises(ValueError, match="near-parabolic"):
        propagate_two_body(r0, [0.0, escape, 0.0], 600.0, mu)


@pytest.mark.parametrize("bad_mu", [0.0, -1.0, np.inf])
def test_bad_gravitational_parameter_is_refused(bad_mu: float) -> None:
    with pytest.raises(ValueError, match="mu must be"):
        propagate_two_body([7.0e6, 0.0, 0.0], [0.0, 7.5e3, 0.0], 10.0, bad_mu)


def test_zero_position_is_refused() -> None:
    with pytest.raises(ValueError, match="must be non-zero"):
        propagate_two_body([0.0, 0.0, 0.0], [0.0, 7.5e3, 0.0], 10.0, MU_EARTH)


def test_non_finite_interval_is_refused() -> None:
    with pytest.raises(ValueError, match="dt_s must be finite"):
        propagate_two_body([7.0e6, 0.0, 0.0], [0.0, 7.5e3, 0.0], np.nan, MU_EARTH)


def test_linear_trajectory_is_straight_and_has_no_acceleration() -> None:
    trajectory = LinearTrajectory([1.0e6, 0.0, 0.0], [10.0, 20.0, 30.0], 5.0)
    assert np.allclose(trajectory.position(5.0), [1.0e6, 0.0, 0.0])
    assert np.allclose(trajectory.position(6.0), [1.0e6 + 10.0, 20.0, 30.0])
    assert np.allclose(trajectory.velocity(1.0e9), [10.0, 20.0, 30.0])
    assert np.allclose(trajectory.acceleration(0.0), np.zeros(3))


def test_kepler_trajectory_acceleration_matches_the_inverse_square_law(mu: float) -> None:
    trajectory = KeplerTrajectory([7.0e6, 0.0, 0.0], [0.0, 7.5e3, 0.0], 0.0, mu)
    expected = -mu / 7.0e6**2
    assert trajectory.acceleration(0.0)[0] == pytest.approx(expected, rel=1e-14)


def test_kepler_trajectory_rejects_bad_mu() -> None:
    with pytest.raises(ValueError, match="mu must be positive"):
        KeplerTrajectory([7.0e6, 0.0, 0.0], [0.0, 7.5e3, 0.0], 0.0, -1.0)


def test_trajectory_rejects_non_finite_epoch() -> None:
    with pytest.raises(ValueError, match="epoch_s must be finite"):
        LinearTrajectory([1.0, 0.0, 0.0], [0.0, 0.0, 0.0], np.nan)
