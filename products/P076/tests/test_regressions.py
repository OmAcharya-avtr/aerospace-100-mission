"""Pinned regressions: every defect this build found, with literal inputs.

Six of these were found by Hypothesis and four by the numerical sensitivity
check in :func:`pointahead.budget.verify_sensitivities` or by a validation
script disagreeing with its own prediction. ``.hypothesis/`` is gitignored, so a
falsifying example found by one run is lost on a fresh clone; every one is
pinned here with the exact inputs that produced it.

Two of the ten were defects in the code. Eight were defects in an acceptance
criterion or a prediction -- a relative tolerance applied to a quantity at the
float64 floor, a Taylor coefficient written from memory, a velocity used in a
formula that calls for a different one. Those are recorded because the failure
mode is identical from the outside: a number that disagrees with another number,
with no indication of which one is wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from pointahead.aberration import aberrate, aberration_angle
from pointahead.angle import (
    closed_form_point_ahead,
    light_time_pair,
    offset_difference,
    reference_point_ahead,
)
from pointahead.budget import (
    direction_noise_floor_rad,
    relative_acceleration,
    relative_jerk,
    staleness_position_error_m,
    staleness_velocity_error_mps,
)
from pointahead.geometry import (
    C_LIGHT,
    norm,
    orthonormal_basis,
    rotation_matrix,
    transverse,
    unit,
)
from pointahead.kepler import propagate_two_body
from pointahead.scenario import leo_crossing_pair, leo_trailing_pair
from pointahead.terminal import Crosslink, TerminalState

MU = 3.986004418e14


def _link(radius: float, first: float, second: float, velocity: np.ndarray) -> Crosslink:
    """Rebuild the exact configuration the property-test strategies produced."""
    local = radius * np.array([float(np.cos(first)), float(np.sin(first)), 0.0])
    remote = radius * np.array(
        [
            float(np.cos(second)),
            float(np.sin(second)) * 0.6,
            float(np.sin(second)) * 0.8,
        ]
    )
    return Crosslink(
        local=TerminalState(local, np.zeros(3), 0.0, "local"),
        remote=TerminalState(remote, velocity, 0.0, "remote"),
        trajectory_model="linear",
        label="regression",
        frame="regression frame",
    )


# -- code defects ------------------------------------------------------------


def test_regression_kepler_converges_over_eighty_eight_revolutions() -> None:
    """Hypothesis, test_two_body_propagation_conserves_energy.

    Falsifying example: ``radius=400000.0, factor=0.6, inclination=0.0,
    dt=3354.9375``. The orbit has a semi-major axis of about 244 km and a
    period of about 37.9 s, so the interval is 88 revolutions, and the
    universal-anomaly Newton iteration starting from a linear guess reported
    "did not converge in 60 iterations". Fixed by reducing an elliptical
    interval modulo the period before iterating, which is exact because the
    state is periodic.
    """
    radius = 400000.0
    circular = float(np.sqrt(MU / radius))
    r0 = np.array([radius, 0.0, 0.0])
    v0 = 0.6 * circular * np.array([0.0, 1.0, 0.0])
    step = propagate_two_body(r0, v0, 3354.9375, MU)
    assert step.revolutions_removed == 89.0
    assert step.iterations <= 10
    before = 0.5 * float(np.dot(v0, v0)) - MU / radius
    after = 0.5 * float(np.dot(step.velocity_mps, step.velocity_mps)) - MU / float(
        np.linalg.norm(step.position_m)
    )
    assert abs(after - before) <= 1.0e-9 * abs(before)


def test_regression_staleness_velocity_includes_the_jerk_term() -> None:
    """verify_sensitivities, ephemeris_staleness_velocity on leo_crossing.

    The line was ``2 |a_rel_perp| dt / c`` alone. For two terminals at equal
    radius under two-body gravity ``a_rel = -(mu / r^3) d``, which is parallel
    to the line of sight, so ``a_rel_perp`` is zero to rounding and the model
    predicted 2.087e-22 rad where the numerical propagation measured
    2.381e-08 rad -- a factor of 1.1e14. The jerk term supplies the whole line
    for this geometry.
    """
    link = leo_crossing_pair()
    n = link.line_of_sight
    acceleration_perp = norm(transverse(relative_acceleration(link), n))
    jerk_perp = norm(transverse(relative_jerk(link), n))
    assert acceleration_perp / norm(relative_acceleration(link)) < 1.0e-14
    assert jerk_perp > 1.0e-3
    age = 30.0
    # The first-order-only model, kept here as the thing that was wrong.
    first_order_only = 2.0 * acceleration_perp * age / C_LIGHT
    assert first_order_only < 1.0e-20
    with_jerk = 2.0 * staleness_velocity_error_mps(link, age) / C_LIGHT
    assert with_jerk == pytest.approx(2.3809601548e-08, rel=1.0e-6)
    measured = offset_difference(
        reference_point_ahead(_stale(link, age, "velocity")), reference_point_ahead(link)
    )
    assert measured == pytest.approx(with_jerk, rel=1.0e-3)
    assert measured / max(first_order_only, 1.0e-300) > 1.0e12


def _stale(link: Crosslink, age_s: float, perturb: str) -> Crosslink:
    from pointahead.budget import _stale_link

    return _stale_link(link, age_s, perturb)


def test_regression_staleness_position_uses_dt_cubed_over_three() -> None:
    """verify_sensitivities, ephemeris_staleness_geometry on leo_crossing.

    The cubic coefficient of the stale position error was written as
    ``dt^3 / 6``, the coefficient a plain Taylor series of the position
    suggests. The correct value is ``dt^3 / 3``: carrying the stale velocity
    forward contributes ``+dt^3 / 2`` and the position expansion contributes
    ``-dt^3 / 6``. With the wrong coefficient the declared bound sat a factor
    of two below the measurement and failed its own check.
    """
    link = leo_crossing_pair()
    n = link.line_of_sight
    jerk_perp = norm(transverse(relative_jerk(link), n))
    age = 30.0
    correct = jerk_perp * age**3 / 3.0
    wrong = jerk_perp * age**3 / 6.0
    assert staleness_position_error_m(link, age) == pytest.approx(correct, rel=1.0e-12)
    assert correct / wrong == pytest.approx(2.0, rel=1.0e-12)
    bound = 2.0 * norm(link.relative_velocity_mps) * correct / (C_LIGHT * link.range_m)
    measured = offset_difference(
        reference_point_ahead(_stale(link, age, "position")), reference_point_ahead(link)
    )
    assert measured <= bound * 1.01
    assert measured > 0.5 * bound
    wrong_bound = 2.0 * norm(link.relative_velocity_mps) * wrong / (C_LIGHT * link.range_m)
    assert measured > wrong_bound


# -- acceptance-criterion and prediction defects -----------------------------


def test_regression_noise_floor_counts_components_and_the_outer_subtraction() -> None:
    """validate_budget, light_time_iteration_residual on leo_trailing at cap 0.

    The float64 direction resolution was first derived as
    ``4 eps R / rho`` by counting the four unit directions an offset difference
    is built from. That ignored the three components of each and the outer
    subtraction of two offsets, giving 3.0e-15 rad for the leo_trailing
    geometry where the measurement was 9.5e-15 rad. The worst-case constant is
    ``8 sqrt(3)``, which gives 1.0e-14 rad. The constant has not been widened
    again; the combinations that still fail at iteration cap zero are reported
    as failures in validation/VALIDATION.md.
    """
    link = leo_trailing_pair()
    scale = max(norm(link.local.position_m), norm(link.remote.position_m))
    old_floor = 4.0 * float(np.finfo(np.float64).eps) * scale / link.range_m
    new_floor = direction_noise_floor_rad(link)
    assert old_floor == pytest.approx(2.97e-15, rel=1.0e-2)
    assert new_floor == pytest.approx(1.029e-14, rel=1.0e-2)
    assert new_floor / old_floor == pytest.approx(2.0 * float(np.sqrt(3.0)), rel=1.0e-12)
    measured = offset_difference(
        reference_point_ahead(link, max_iter=0), reference_point_ahead(link)
    )
    assert measured > old_floor
    assert measured < new_floor


def test_regression_light_time_gap_uses_the_remote_not_relative_radial_velocity() -> None:
    """validate_lighttime sections 4 and 6.

    The light-time equation holds the LOCAL position fixed, because the
    point-ahead angle is defined at one event there, so the fixed-point map's
    derivative is ``(v_remote . n) / c`` and not ``(v_rel . n) / c``. The first
    version of the prediction used the relative radial velocity. On
    leo_crossing that is wrong by a factor of two; on leo_trailing the relative
    radial velocity is zero and the prediction was wrong by twelve orders of
    magnitude, while the measured gap is identical to leo_crossing's because
    the two geometries share a remote radial velocity.
    """
    for link in (leo_crossing_pair(), leo_trailing_pair()):
        receive, transmit = light_time_pair(link)
        measured = transmit.tau_s - receive.tau_s
        remote_radial = float(np.dot(link.remote.velocity_mps, link.line_of_sight))
        good = 2.0 * remote_radial * link.nominal_light_time_s / C_LIGHT
        assert measured == pytest.approx(good, rel=1.0e-8)
    crossing = leo_crossing_pair()
    trailing = leo_trailing_pair()
    receive_c, transmit_c = light_time_pair(crossing)
    receive_t, transmit_t = light_time_pair(trailing)
    assert (transmit_c.tau_s - receive_c.tau_s) == pytest.approx(
        transmit_t.tau_s - receive_t.tau_s, rel=1.0e-8
    )
    relative_radial = abs(
        float(np.dot(trailing.relative_velocity_mps, trailing.line_of_sight))
    )
    assert relative_radial < 1.0e-12


def test_regression_hand_calculated_point_ahead_literal() -> None:
    """tests/test_angle.py, test_closed_form_hand_calculated_known_answer.

    The hand-written literal in the first version of that test was
    5.003461712e-05 rad. The value ``15000 / 299792458`` actually evaluates to
    5.003461427972281e-05, so the literal was wrong in its seventh significant
    figure and the test failed on its own comment.
    """
    assert 15000.0 / C_LIGHT == 5.003461427972281e-05
    assert abs(15000.0 / C_LIGHT - 5.003461712e-05) == pytest.approx(
        2.8402771942107746e-12, rel=1.0e-12, abs=0.0
    )
    link = Crosslink(
        local=TerminalState([1.0e7, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0, "local"),
        remote=TerminalState([1.0e7, 2.0e6, 0.0], [7500.0, 0.0, 0.0], 0.0, "remote"),
        trajectory_model="linear",
        label="hand",
        frame="hand frame",
    )
    result = closed_form_point_ahead(link)
    assert float(np.linalg.norm(result.offset_rad)) == pytest.approx(
        5.003461427972281e-05, rel=1.0e-15, abs=0.0
    )
    assert result.total_rad == pytest.approx(5.0034614269284495e-05, rel=1.0e-15, abs=0.0)


def test_regression_pytest_approx_default_absolute_floor() -> None:
    """tests/test_angle.py and validation/validate_regressions.py.

    ``pytest.approx(expected, rel=r)`` carries a DEFAULT absolute tolerance of
    1e-12 and accepts a value within *either* bound. On a quantity of order
    1e-05 rad -- which is every angle this package produces -- that default
    dominates any relative tolerance tighter than 2e-08, so an assertion
    written as ``rel=1e-12`` is in fact a 2e-08 relative comparison.

    This was not hypothetical. ``test_closed_form_hand_calculated_known_answer``
    asserted the closed-form ``total_rad`` equal to ``2 v_perp / c`` at
    ``rel=1e-12`` and passed, while the two differ by a relative
    2.086218195129917e-10 -- because ``total_rad`` is the angle between two
    unit vectors offset either side of the line of sight, which is
    ``2 arctan(theta / 2) = theta - theta^3 / 12``, not ``theta``.

    Re-running the whole suite with every ``rel=`` call given ``abs=0.0`` found
    exactly three assertions of 271 that depended on the default. All three are
    now explicit. This test is the guard: it asserts the default is still what
    it was measured to be, so that a future pytest changing it is a visible
    failure rather than a silent loosening.
    """
    # The default absolute tolerance accepts this; a pure relative one does not.
    assert 1.0e-5 == pytest.approx(1.0e-5 + 1.0e-13, rel=1.0e-15)
    assert 1.0e-5 != pytest.approx(1.0e-5 + 1.0e-13, rel=1.0e-15, abs=0.0)

    link = Crosslink(
        local=TerminalState([1.0e7, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0, "local"),
        remote=TerminalState([1.0e7, 2.0e6, 0.0], [7500.0, 0.0, 0.0], 0.0, "remote"),
        trajectory_model="linear",
        label="hand",
        frame="hand frame",
    )
    theta = 15000.0 / C_LIGHT
    total = closed_form_point_ahead(link).total_rad
    deficit = (theta - total) / theta
    assert deficit == pytest.approx(2.086218195129917e-10, rel=1.0e-9, abs=0.0)
    assert deficit == pytest.approx(theta**2 / 12.0, rel=1.0e-6, abs=0.0)
    # The old, wrong assertion: passes with the default floor, fails without it.
    assert total == pytest.approx(theta, rel=1.0e-12)
    assert total != pytest.approx(theta, rel=1.0e-12, abs=0.0)


def test_regression_first_order_aberration_error_is_cubic_when_transverse() -> None:
    """tests/test_aberration.py, the first-order error scaling.

    The module docstring claimed the first-order form is in error by
    ``O(beta^2)``. For a direction exactly transverse to the boost the
    second-order term vanishes and the error is
    ``arcsin(beta) - arctan(beta) = beta^3 / 2``, so a test written for a
    quadratic ratio of 100 per decade measured 1000 and failed. Both scalings
    are now stated and both are tested.
    """
    transverse_errors = []
    oblique_errors = []
    oblique = unit([1.0, 1.0, 0.0])
    for magnitude in (1.0e-3, 1.0e-2, 1.0e-1):
        beta = np.array([magnitude, 0.0, 0.0])
        transverse_errors.append(
            aberration_angle([0.0, 1.0, 0.0], beta, "first_order")
            - aberration_angle([0.0, 1.0, 0.0], beta, "momentum_boost")
        )
        oblique_errors.append(
            float(
                np.linalg.norm(
                    aberrate(oblique, beta, "first_order")
                    - aberrate(oblique, beta, "momentum_boost")
                )
            )
        )
    transverse_ratio = abs(transverse_errors[1] / transverse_errors[0])
    oblique_ratio = oblique_errors[1] / oblique_errors[0]
    assert transverse_ratio == pytest.approx(1000.0, rel=0.02)
    assert 90.0 < oblique_ratio < 115.0


def test_regression_arcsin_aberration_at_beta_one_e_minus_twelve() -> None:
    """Hypothesis, test_transverse_aberration_is_arcsin_beta.

    Falsifying example: ``axis=[1.0, 1.0, 1.0], magnitude=1e-12``. The residual
    was 8.113064303823435e-18 rad on an expected 1e-12 rad, which is 8.1e-06
    relative and far below the 2.2e-16 resolution of a float64 unit vector. The
    criterion needed an absolute floor; the code was correct.
    """
    axis = unit([1.0, 1.0, 1.0])
    magnitude = 1.0e-12
    _, e1, _ = orthonormal_basis(axis)
    expected = float(np.arcsin(magnitude))
    for route in ("angle_relation", "momentum_boost", "event_worldline"):
        measured = aberration_angle(e1, axis * magnitude, route)
        residual = abs(measured - expected)
        assert residual < 1.0e-15
        assert residual > 1.0e-10 * expected


def test_regression_closed_form_at_four_hundred_kilometre_radius() -> None:
    """Hypothesis, test_closed_form_matches_two_v_perp_over_c.

    Falsifying example: ``radius=400000.0, first=0.0, second=1.0,
    remote_speed=1.0, heading=[0.0, 0.0, 1.0]``. Residual
    6.172500313028885e-18 rad on an angle of 4.75e-09 rad. Criterion, not code.
    """
    link = _link(400000.0, 0.0, 1.0, np.array([0.0, 0.0, 1.0]))
    expected = 2.0 * norm(transverse(link.relative_velocity_mps, link.line_of_sight)) / C_LIGHT
    measured = closed_form_point_ahead(link).total_rad
    assert expected == pytest.approx(4.7506987823907895e-09, rel=1.0e-9)
    assert abs(measured - expected) < 1.0e-15
    assert abs(measured - expected) > 1.0e-9 * expected


def test_regression_radial_motion_at_seven_hundred_thirty_eight_kilometres() -> None:
    """Hypothesis, test_radial_relative_motion_gives_no_point_ahead.

    Falsifying example: ``radius=738448.0, first=0.0, second=0.125,
    radial_speed=2.0``. The reference returned 1.2434567574317502e-15 rad
    against a fixed 1e-15 rad criterion. The geometry's own float64 direction
    resolution is 2.5e-14 rad, so the criterion was 25 times tighter than the
    arithmetic allows. Criterion, not code.
    """
    base = _link(738448.0, 0.0, 0.125, np.zeros(3))
    radial = Crosslink(
        local=base.local,
        remote=TerminalState(
            base.remote.position_m, 2.0 * base.line_of_sight, 0.0, "remote"
        ),
        trajectory_model="linear",
        label="radial",
        frame=base.frame,
    )
    total = reference_point_ahead(radial).total_rad
    assert closed_form_point_ahead(radial).total_rad < 1.0e-18
    assert total > 1.0e-15
    assert total <= direction_noise_floor_rad(radial)


def test_regression_rotation_invariance_at_four_hundred_kilometre_radius() -> None:
    """Hypothesis, test_point_ahead_is_rotation_invariant.

    Falsifying example: ``radius=400000.0, first=0.0, second=1.0,
    remote_speed=1.0, heading=[0.0, 0.0, 1.0], axis=[0.0, 0.0, 1.0],
    theta=1.0``. Residual 1.9759676410596312e-17 rad on an angle of 4.75e-09
    rad, which is 4.2e-09 relative against a 1e-09 relative criterion.
    Criterion, not code.
    """
    link = _link(400000.0, 0.0, 1.0, np.array([0.0, 0.0, 1.0]))
    matrix = rotation_matrix([0.0, 0.0, 1.0], 1.0)
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
    residual = abs(after - before)
    assert residual < 1.0e-15
    assert residual > 1.0e-9 * before


def test_regression_reference_versus_closed_form_at_a_twelve_kilometre_range() -> None:
    """Hypothesis, test_reference_agrees_with_the_closed_form_at_orbital_speeds.

    Falsifying example: ``radius=1037906.0, first=0.0, second=0.01171875,
    remote_speed=1.0, heading=[1.0, 0.0, 0.0]``. The two terminals are 12.2 km
    apart at a radius of 1038 km, so the float64 direction resolution is
    2.6e-13 rad, and the measured offset difference of 8.29631502385908e-15 rad
    is thirty times below it while being 1.2e-06 of a 6.67e-09 rad angle.
    Criterion, not code.
    """
    link = _link(1037906.0, 0.0, 0.01171875, np.array([1.0, 0.0, 0.0]))
    baseline = closed_form_point_ahead(link)
    reference = reference_point_ahead(link)
    difference = offset_difference(baseline, reference)
    floor = direction_noise_floor_rad(link)
    assert link.range_m == pytest.approx(1.2159e4, rel=1.0e-3)
    assert difference < floor
    assert difference > 1.0e-6 * baseline.total_rad
