"""Light-time solutions: against the closed form, truncated, and refused."""

from __future__ import annotations

import numpy as np
import pytest

from pointahead.geometry import C_LIGHT
from pointahead.kepler import KeplerTrajectory, LinearTrajectory
from pointahead.lighttime import (
    light_time_residual,
    linear_light_time_closed_form,
    solve_light_time,
)


def test_static_remote_gives_the_geometric_light_time() -> None:
    # Remote exactly one light-second away and not moving: tau = 1 s exactly,
    # in both directions, after zero iterations.
    trajectory = LinearTrajectory([C_LIGHT, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0)
    for direction in ("transmit", "receive"):
        solution = solve_light_time(trajectory, np.zeros(3), 0.0, direction)
        assert solution.tau_s == pytest.approx(1.0, rel=1e-15)
        assert solution.converged
        assert solution.residual_s == pytest.approx(0.0, abs=1e-18)


def test_relativistic_transmit_light_time_known_answer(
    relativistic_light_time: tuple[LinearTrajectory, np.ndarray],
) -> None:
    # Remote one light-second away along +x, moving at 0.6 c along +y.
    # |[c, 0.6 c tau, 0]| = c tau  =>  c^2 + 0.36 c^2 tau^2 = c^2 tau^2
    #                             =>  tau = 1 / sqrt(1 - 0.36) = 1.25 s exactly.
    trajectory, local = relativistic_light_time
    exact = linear_light_time_closed_form(trajectory, local, 0.0, "transmit")
    assert exact == pytest.approx(1.25, rel=1e-15)
    iterated = solve_light_time(trajectory, local, 0.0, "transmit")
    assert iterated.tau_s == pytest.approx(1.25, rel=1e-13)
    # At beta = 0.6 the fixed-point iteration contracts by only about 0.6 per
    # step, so it does not reach the 1e-15 second default tolerance inside the
    # 30-iteration default cap. This is a real limit of the solver and it is
    # asserted rather than papered over: an orbital crosslink has beta near
    # 2.5e-5 and converges in three iterations, and a frame moving at 0.6 c is
    # outside anything this package is for.
    assert not iterated.converged
    assert solve_light_time(trajectory, local, 0.0, "transmit", max_iter=300).converged


def test_relativistic_receive_light_time_known_answer(
    relativistic_light_time: tuple[LinearTrajectory, np.ndarray],
) -> None:
    # The receive equation has the same algebra with the sign of d . v flipped,
    # and d . v = 0 here, so the receive light time is also 1.25 s exactly.
    trajectory, local = relativistic_light_time
    assert linear_light_time_closed_form(trajectory, local, 0.0, "receive") == pytest.approx(
        1.25, rel=1e-15
    )


def test_iterated_matches_the_closed_form_over_a_seeded_family() -> None:
    rng = np.random.default_rng(1076)
    worst = 0.0
    for _ in range(200):
        position = rng.normal(size=3) * 2.0e6 + np.array([8.0e6, 0.0, 0.0])
        velocity = rng.normal(size=3) * 4.0e3
        trajectory = LinearTrajectory(position, velocity, 0.0)
        local = rng.normal(size=3) * 2.0e6
        for direction in ("transmit", "receive"):
            exact = linear_light_time_closed_form(trajectory, local, 0.0, direction)
            iterated = solve_light_time(trajectory, local, 0.0, direction)
            assert iterated.converged
            worst = max(worst, abs(iterated.tau_s - exact) / exact)
    assert worst < 1.0e-14, f"iterated light time differs from the closed form by {worst}"


def test_zero_iterations_returns_the_geometric_guess_with_a_measured_residual() -> None:
    trajectory = LinearTrajectory([2.0e6, 0.0, 0.0], [0.0, 7.5e3, 0.0], 0.0)
    solution = solve_light_time(trajectory, np.zeros(3), 0.0, "transmit", max_iter=0)
    assert solution.iterations == 0
    assert solution.tau_s == pytest.approx(2.0e6 / C_LIGHT, rel=1e-15)
    assert not solution.converged
    assert solution.residual_s > 0.0


def test_the_residual_shrinks_monotonically_with_iterations() -> None:
    trajectory = LinearTrajectory([2.0e6, 1.0e5, 0.0], [-1.0e3, 7.5e3, 2.0e2], 0.0)
    residuals = [
        abs(solve_light_time(trajectory, np.zeros(3), 0.0, "transmit", max_iter=k).residual_s)
        for k in range(4)
    ]
    for earlier, later in zip(residuals[:-1], residuals[1:], strict=True):
        assert later <= earlier


def test_light_time_residual_is_zero_at_the_solution() -> None:
    trajectory = LinearTrajectory([3.0e6, 2.0e5, 0.0], [5.0e2, 7.0e3, 0.0], 0.0)
    tau = linear_light_time_closed_form(trajectory, np.zeros(3), 0.0, "transmit")
    # The tolerance is two units in the last place of the light time itself:
    # the range is 3.0e6 m, whose float64 rounding is 4.7e-10 m, which is
    # 1.6e-18 s of light time. Asserting an exact zero here would be asserting
    # something floating point cannot promise.
    assert light_time_residual(
        trajectory, np.zeros(3), 0.0, tau, "transmit"
    ) == pytest.approx(0.0, abs=1e-17)


def test_transmit_exceeds_receive_for_a_receding_remote() -> None:
    # A remote terminal moving away makes the outgoing path longer than the
    # incoming one, so tau_tx > tau_rx. The sign of the inequality is the
    # physics; the magnitudes are measured.
    trajectory = LinearTrajectory([2.0e6, 0.0, 0.0], [5.0e3, 0.0, 0.0], 0.0)
    transmit = solve_light_time(trajectory, np.zeros(3), 0.0, "transmit")
    receive = solve_light_time(trajectory, np.zeros(3), 0.0, "receive")
    assert transmit.tau_s > receive.tau_s


def test_closed_form_refuses_a_keplerian_trajectory() -> None:
    trajectory = KeplerTrajectory([7.0e6, 0.0, 0.0], [0.0, 7.5e3, 0.0], 0.0)
    with pytest.raises(TypeError, match="requires a LinearTrajectory"):
        linear_light_time_closed_form(trajectory, np.zeros(3), 0.0)  # type: ignore[arg-type]


def test_unknown_direction_is_refused() -> None:
    trajectory = LinearTrajectory([2.0e6, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0)
    with pytest.raises(ValueError, match="direction must be one of"):
        solve_light_time(trajectory, np.zeros(3), 0.0, "both")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_iter": -1}, "max_iter must be non-negative"),
        ({"tol_s": -1.0}, "tol_s must be non-negative"),
        ({"epoch_s": np.nan}, "epoch_s must be finite"),
    ],
)
def test_bad_solver_arguments_are_refused(kwargs: dict, match: str) -> None:
    trajectory = LinearTrajectory([2.0e6, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0)
    arguments = {"epoch_s": 0.0, **kwargs}
    with pytest.raises(ValueError, match=match):
        solve_light_time(trajectory, np.zeros(3), direction="transmit", **arguments)


def test_light_time_range_is_c_times_tau() -> None:
    trajectory = LinearTrajectory([2.0e6, 0.0, 0.0], [0.0, 7.5e3, 0.0], 0.0)
    solution = solve_light_time(trajectory, np.zeros(3), 0.0, "transmit")
    assert solution.range_m == pytest.approx(C_LIGHT * solution.tau_s, rel=1e-15)


def test_kepler_and_linear_light_times_agree_over_a_millisecond_scale_transit(
    crossing_link,
) -> None:
    # Over a 7 ms light time the difference between a two-body arc and a
    # straight line is 2e-4 m, so the two light times agree to well under a
    # picosecond. The number is measured here rather than asserted from theory.
    local = crossing_link.local.position_m
    kepler = KeplerTrajectory(
        crossing_link.remote.position_m, crossing_link.remote.velocity_mps, 0.0, crossing_link.mu
    )
    linear = LinearTrajectory(
        crossing_link.remote.position_m, crossing_link.remote.velocity_mps, 0.0
    )
    a = solve_light_time(kepler, local, 0.0, "transmit").tau_s
    b = solve_light_time(linear, local, 0.0, "transmit").tau_s
    assert abs(a - b) < 1.0e-12
