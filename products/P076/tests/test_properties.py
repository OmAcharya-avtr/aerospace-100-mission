"""Hypothesis property tests over the algebraic identities this package relies on.

Hand-written cases test the geometries the author thought of. These test the
identities, over inputs the author did not choose. On P071 of this mission
Hypothesis found six equation defects that the hand-written suite missed, so
the properties here are deliberately the load-bearing ones: the agreement of
the three exact aberration routes, the aberration group structure, the
light-time equation's own residual, and the vanishing and reversal invariants
of the point-ahead angle.

Every floating-point comparison has a tolerance on both sides, and no boolean
produced by a threshold comparison is asserted equal. Any falsifying example
Hypothesis finds is pinned as a named regression test with literal inputs in
``test_regressions.py``, because ``.hypothesis/`` is gitignored and a
counterexample found by one run is otherwise lost on a fresh clone.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from pointahead.aberration import EXACT_ROUTES, aberrate, aberration_angle
from pointahead.angle import closed_form_point_ahead, offset_difference, reference_point_ahead
from pointahead.budget import direction_noise_floor_rad
from pointahead.divergence import error_for_loss_db, pointing_loss_db, pointing_loss_fraction
from pointahead.geometry import (
    C_LIGHT,
    angle_between,
    norm,
    orthonormal_basis,
    rotation_matrix,
    transverse,
    unit,
)
from pointahead.kepler import propagate_two_body
from pointahead.lighttime import (
    light_time_residual,
    linear_light_time_closed_form,
    solve_light_time,
)
from pointahead.terminal import Crosslink, TerminalState

PROFILE = settings(
    max_examples=int(__import__("os").environ.get("POINTAHEAD_HYPOTHESIS_EXAMPLES", "200")),
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

ANGLE_FLOOR_RAD = 1.0e-15
"""Absolute floor added to every relative tolerance below, in radians.

Every angle in this package is the angle between two unit vectors, and one unit
in the last place of a float64 unit vector component is 2.2e-16. An angle of
1e-12 rad is therefore known to about four parts in ten thousand, and a
purely relative tolerance on it is a statement about float64 and not about this
code.

The first version of this module omitted the floor. Hypothesis falsified four
of its properties within two hundred examples each, every one of them at an
angle below 1e-08 rad with an absolute residual below 2e-17 rad. All four
counterexamples are pinned with their literal inputs in
``test_regressions.py``; ``.hypothesis/`` is gitignored, so a counterexample
that is not pinned is lost on a fresh clone.
"""

finite = st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)
direction_strategy = st.lists(finite, min_size=3, max_size=3)
beta_magnitude = st.floats(min_value=1.0e-12, max_value=0.98)
speed = st.floats(min_value=1.0, max_value=1.0e4)
altitude = st.floats(min_value=400.0e3, max_value=2.0e7)
angle = st.floats(min_value=0.0, max_value=2.0 * np.pi)


def _unit(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    assume(float(np.linalg.norm(array)) > 1.0e-6)
    return unit(array)


# -- geometry ----------------------------------------------------------------


@PROFILE
@given(direction_strategy, direction_strategy)
def test_transverse_component_is_orthogonal(vector: list[float], axis: list[float]) -> None:
    n = _unit(axis)
    v = np.asarray(vector, dtype=np.float64)
    assume(norm(v) > 1.0e-6)
    perpendicular = transverse(v, n)
    assert abs(float(np.dot(perpendicular, n))) <= 1.0e-12 * norm(v)


@PROFILE
@given(direction_strategy)
def test_orthonormal_basis_is_orthonormal(axis: list[float]) -> None:
    n, e1, e2 = orthonormal_basis(_unit(axis))
    for a in (n, e1, e2):
        assert abs(norm(a) - 1.0) < 1.0e-12
    assert abs(float(np.dot(n, e1))) < 1.0e-12
    assert abs(float(np.dot(n, e2))) < 1.0e-12
    assert abs(float(np.dot(e1, e2))) < 1.0e-12


@PROFILE
@given(direction_strategy, direction_strategy, direction_strategy, angle)
def test_angle_between_is_rotation_invariant(
    first: list[float], second: list[float], axis: list[float], theta: float
) -> None:
    a = _unit(first)
    b = _unit(second)
    matrix = rotation_matrix(_unit(axis), theta)
    before = angle_between(a, b)
    after = angle_between(matrix @ a, matrix @ b)
    assert abs(after - before) < 1.0e-9


# -- aberration --------------------------------------------------------------


@PROFILE
@given(direction_strategy, direction_strategy, beta_magnitude)
def test_exact_aberration_routes_agree(
    direction: list[float], axis: list[float], magnitude: float
) -> None:
    p = _unit(direction)
    beta = _unit(axis) * magnitude
    results = [aberrate(p, beta, route) for route in EXACT_ROUTES]
    for other in results[1:]:
        assert angle_between(results[0], other) < 1.0e-12


@PROFILE
@given(direction_strategy, direction_strategy, beta_magnitude)
def test_aberration_preserves_the_unit_norm(
    direction: list[float], axis: list[float], magnitude: float
) -> None:
    p = _unit(direction)
    beta = _unit(axis) * magnitude
    for route in EXACT_ROUTES:
        assert abs(norm(aberrate(p, beta, route)) - 1.0) < 1.0e-12


@PROFILE
@given(direction_strategy, direction_strategy, beta_magnitude)
def test_aberration_inverse_is_the_opposite_boost(
    direction: list[float], axis: list[float], magnitude: float
) -> None:
    p = _unit(direction)
    beta = _unit(axis) * magnitude
    for route in EXACT_ROUTES:
        assert angle_between(p, aberrate(aberrate(p, beta, route), -beta, route)) < 1.0e-10


@PROFILE
@given(direction_strategy, direction_strategy, beta_magnitude)
def test_aberration_displacement_never_exceeds_pi(
    direction: list[float], axis: list[float], magnitude: float
) -> None:
    p = _unit(direction)
    beta = _unit(axis) * magnitude
    for route in EXACT_ROUTES:
        displacement = aberration_angle(p, beta, route)
        assert 0.0 <= displacement <= np.pi + 1.0e-12


@PROFILE
@given(direction_strategy, beta_magnitude)
def test_collinear_aberration_displaces_nothing(
    axis: list[float], magnitude: float
) -> None:
    b = _unit(axis)
    beta = b * magnitude
    for route in EXACT_ROUTES:
        for sign in (1.0, -1.0):
            assert aberration_angle(sign * b, beta, route) < 1.0e-12


@PROFILE
@given(direction_strategy, beta_magnitude)
def test_transverse_aberration_is_arcsin_beta(axis: list[float], magnitude: float) -> None:
    b = _unit(axis)
    _, e1, _ = orthonormal_basis(b)
    expected = float(np.arcsin(magnitude))
    for route in EXACT_ROUTES:
        measured = aberration_angle(e1, b * magnitude, route)
        assert abs(measured - expected) <= 1.0e-10 * expected + ANGLE_FLOOR_RAD


@PROFILE
@given(
    direction_strategy,
    direction_strategy,
    st.floats(min_value=1.0e-6, max_value=0.4),
    st.floats(min_value=1.0e-6, max_value=0.4),
)
def test_collinear_boosts_compose_by_velocity_addition(
    direction: list[float], axis: list[float], first: float, second: float
) -> None:
    p = _unit(direction)
    b = _unit(axis)
    combined = (first + second) / (1.0 + first * second)
    for route in EXACT_ROUTES:
        stepwise = aberrate(aberrate(p, b * first, route), b * second, route)
        single = aberrate(p, b * combined, route)
        assert angle_between(stepwise, single) < 1.0e-10


# -- light time --------------------------------------------------------------


@PROFILE
@given(altitude, angle, angle, speed, direction_strategy)
def test_light_time_residual_is_zero_at_the_iterated_solution(
    radius: float,
    first_angle: float,
    second_angle: float,
    remote_speed: float,
    heading: list[float],
) -> None:
    local = radius * np.array(
        [float(np.cos(first_angle)), float(np.sin(first_angle)), 0.0]
    )
    remote = radius * np.array(
        [
            float(np.cos(second_angle)),
            float(np.sin(second_angle)) * 0.6,
            float(np.sin(second_angle)) * 0.8,
        ]
    )
    separation = float(np.linalg.norm(remote - local))
    assume(separation > 1.0e4)
    velocity = remote_speed * _unit(heading)
    from pointahead.kepler import LinearTrajectory

    trajectory = LinearTrajectory(remote, velocity, 0.0)
    for direction in ("transmit", "receive"):
        solution = solve_light_time(trajectory, local, 0.0, direction)
        residual = light_time_residual(trajectory, local, 0.0, solution.tau_s, direction)
        assert abs(residual) <= 1.0e-14 * max(solution.tau_s, 1.0e-6)


@PROFILE
@given(altitude, angle, angle, speed, direction_strategy)
def test_iterated_light_time_matches_the_closed_form(
    radius: float,
    first_angle: float,
    second_angle: float,
    remote_speed: float,
    heading: list[float],
) -> None:
    from pointahead.kepler import LinearTrajectory

    local = radius * np.array([float(np.cos(first_angle)), float(np.sin(first_angle)), 0.0])
    remote = radius * np.array(
        [float(np.cos(second_angle)), 0.0, float(np.sin(second_angle))]
    )
    assume(float(np.linalg.norm(remote - local)) > 1.0e4)
    trajectory = LinearTrajectory(remote, remote_speed * _unit(heading), 0.0)
    for direction in ("transmit", "receive"):
        exact = linear_light_time_closed_form(trajectory, local, 0.0, direction)
        iterated = solve_light_time(trajectory, local, 0.0, direction)
        assert abs(iterated.tau_s - exact) <= 1.0e-12 * exact


# -- point ahead -------------------------------------------------------------


def _link(radius: float, first: float, second: float, velocity: np.ndarray) -> Crosslink:
    local = radius * np.array([float(np.cos(first)), float(np.sin(first)), 0.0])
    remote = radius * np.array(
        [
            float(np.cos(second)),
            float(np.sin(second)) * 0.6,
            float(np.sin(second)) * 0.8,
        ]
    )
    assume(float(np.linalg.norm(remote - local)) > 1.0e4)
    return Crosslink(
        local=TerminalState(local, np.zeros(3), 0.0, "local"),
        remote=TerminalState(remote, velocity, 0.0, "remote"),
        trajectory_model="linear",
        label="property",
        frame="property frame",
    )


@PROFILE
@given(altitude, angle, angle, speed, direction_strategy)
def test_closed_form_matches_two_v_perp_over_c(
    radius: float, first: float, second: float, remote_speed: float, heading: list[float]
) -> None:
    link = _link(radius, first, second, remote_speed * _unit(heading))
    expected = 2.0 * norm(transverse(link.relative_velocity_mps, link.line_of_sight)) / C_LIGHT
    measured = closed_form_point_ahead(link).total_rad
    assert abs(measured - expected) <= 1.0e-9 * expected + ANGLE_FLOOR_RAD


@PROFILE
@given(altitude, angle, angle, speed)
def test_radial_relative_motion_gives_no_point_ahead(
    radius: float, first: float, second: float, radial_speed: float
) -> None:
    link = _link(radius, first, second, np.zeros(3))
    radial = Crosslink(
        local=link.local,
        remote=TerminalState(
            link.remote.position_m, radial_speed * link.line_of_sight, 0.0, "remote"
        ),
        trajectory_model="linear",
        label="radial",
        frame=link.frame,
    )
    assert closed_form_point_ahead(radial).total_rad < 1.0e-18
    # The reference builds its directions from positions, so its floor is the
    # geometry's own float64 direction resolution and not a fixed angle.
    assert reference_point_ahead(radial).total_rad <= direction_noise_floor_rad(radial)


@PROFILE
@given(altitude, angle, angle, speed, direction_strategy)
def test_negating_the_relative_velocity_negates_the_offset(
    radius: float, first: float, second: float, remote_speed: float, heading: list[float]
) -> None:
    velocity = remote_speed * _unit(heading)
    link = _link(radius, first, second, velocity)
    mirrored = Crosslink(
        local=link.local,
        remote=TerminalState(link.remote.position_m, -velocity, 0.0, "remote"),
        trajectory_model="linear",
        label="mirrored",
        frame=link.frame,
    )
    forward = closed_form_point_ahead(link)
    backward = closed_form_point_ahead(mirrored)
    assume(forward.total_rad > 1.0e-14)
    assert abs(backward.total_rad - forward.total_rad) <= 1.0e-9 * forward.total_rad
    residual = float(np.linalg.norm(backward.offset_vector + forward.offset_vector))
    assert residual <= 1.0e-8 * forward.total_rad


@PROFILE
@given(altitude, angle, angle, speed, direction_strategy)
def test_reference_agrees_with_the_closed_form_at_orbital_speeds(
    radius: float, first: float, second: float, remote_speed: float, heading: list[float]
) -> None:
    link = _link(radius, first, second, remote_speed * _unit(heading))
    baseline = closed_form_point_ahead(link)
    assume(baseline.total_rad > 1.0e-12)
    reference = reference_point_ahead(link)
    assert offset_difference(baseline, reference) <= (
        1.0e-6 * baseline.total_rad + direction_noise_floor_rad(link)
    )


@PROFILE
@given(altitude, angle, angle, speed, direction_strategy, direction_strategy, angle)
def test_point_ahead_is_rotation_invariant(
    radius: float,
    first: float,
    second: float,
    remote_speed: float,
    heading: list[float],
    axis: list[float],
    theta: float,
) -> None:
    link = _link(radius, first, second, remote_speed * _unit(heading))
    matrix = rotation_matrix(_unit(axis), theta)
    rotated = Crosslink(
        local=TerminalState(
            matrix @ link.local.position_m, matrix @ link.local.velocity_mps, 0.0, "local"
        ),
        remote=TerminalState(
            matrix @ link.remote.position_m, matrix @ link.remote.velocity_mps, 0.0, "remote"
        ),
        trajectory_model="linear",
        label="rotated",
        frame=link.frame,
    )
    before = closed_form_point_ahead(link).total_rad
    after = closed_form_point_ahead(rotated).total_rad
    assume(before > 1.0e-12)
    assert abs(after - before) <= 1.0e-9 * before + ANGLE_FLOOR_RAD


# -- divergence --------------------------------------------------------------


@PROFILE
@given(
    st.floats(min_value=1.0e-7, max_value=1.0e-3),
    st.floats(min_value=1.0e-9, max_value=1.0e-4),
)
def test_pointing_loss_round_trips_through_its_inverse(
    divergence: float, error: float
) -> None:
    loss = pointing_loss_db(error, divergence)
    assume(loss > 1.0e-12)
    recovered = error_for_loss_db(loss, divergence)
    assert abs(recovered - error) <= 1.0e-9 * error


@PROFILE
@given(
    st.floats(min_value=1.0e-7, max_value=1.0e-3),
    st.floats(min_value=0.0, max_value=1.0e-4),
)
def test_pointing_loss_fraction_is_in_the_unit_interval(
    divergence: float, error: float
) -> None:
    fraction = pointing_loss_fraction(error, divergence)
    assert 0.0 <= fraction <= 1.0
    assert pointing_loss_db(error, divergence) >= 0.0


# -- two-body propagation ----------------------------------------------------


@PROFILE
@given(
    altitude,
    st.floats(min_value=0.6, max_value=1.3),
    st.floats(min_value=0.0, max_value=0.5 * np.pi),
    st.floats(min_value=-3600.0, max_value=3600.0),
)
def test_two_body_propagation_conserves_energy(
    radius: float, factor: float, inclination: float, dt: float
) -> None:
    mu = 3.986004418e14
    circular = float(np.sqrt(mu / radius))
    assume(abs(factor - float(np.sqrt(2.0))) > 1.0e-3)
    r0 = np.array([radius, 0.0, 0.0])
    v0 = (
        factor
        * circular
        * np.array([0.0, float(np.cos(inclination)), float(np.sin(inclination))])
    )
    step = propagate_two_body(r0, v0, dt, mu)
    before = 0.5 * float(np.dot(v0, v0)) - mu / radius
    after = 0.5 * float(np.dot(step.velocity_mps, step.velocity_mps)) - mu / float(
        np.linalg.norm(step.position_m)
    )
    assert abs(after - before) <= 1.0e-9 * abs(before)


@PROFILE
@given(
    altitude,
    st.floats(min_value=0.6, max_value=1.3),
    st.floats(min_value=0.0, max_value=0.5 * np.pi),
    st.floats(min_value=1.0, max_value=3600.0),
)
def test_two_body_propagation_round_trips(
    radius: float, factor: float, inclination: float, dt: float
) -> None:
    mu = 3.986004418e14
    circular = float(np.sqrt(mu / radius))
    assume(abs(factor - float(np.sqrt(2.0))) > 1.0e-3)
    r0 = np.array([radius, 0.0, 0.0])
    v0 = (
        factor
        * circular
        * np.array([0.0, float(np.cos(inclination)), float(np.sin(inclination))])
    )
    forward = propagate_two_body(r0, v0, dt, mu)
    back = propagate_two_body(forward.position_m, forward.velocity_mps, -dt, mu)
    assert float(np.linalg.norm(back.position_m - r0)) <= 1.0e-6 * radius


def test_the_profile_actually_runs_examples() -> None:
    # A Hypothesis settings profile that collects nothing passes silently, which
    # is the same class of defect as a pytest config that collects nothing.
    assert PROFILE.max_examples >= 200
    with pytest.raises(ValueError):
        aberrate([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], "momentum_boost")
