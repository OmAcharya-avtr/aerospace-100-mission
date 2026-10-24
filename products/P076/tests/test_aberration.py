"""Aberration: four routes, their agreement, and hand-calculated known answers."""

from __future__ import annotations

import numpy as np
import pytest

from pointahead.aberration import (
    ABERRATION_ROUTES,
    EXACT_ROUTES,
    aberrate,
    aberrate_event_worldline,
    aberrate_first_order,
    aberration_angle,
    beta_from_velocity,
    lorentz_gamma,
    route_disagreement,
)
from pointahead.geometry import C_LIGHT, angle_between, norm, unit


def test_beta_from_velocity_known_answer() -> None:
    # 299792.458 m/s is exactly one thousandth of c.
    assert np.allclose(beta_from_velocity([299792.458, 0.0, 0.0]), [1.0e-3, 0.0, 0.0], atol=1e-18)


def test_lorentz_gamma_known_answer() -> None:
    # beta = 0.6 gives gamma = 1 / sqrt(1 - 0.36) = 1 / 0.8 = 1.25 exactly.
    assert lorentz_gamma(0.6) == pytest.approx(1.25, rel=1e-15)
    assert lorentz_gamma(0.0) == 1.0


@pytest.mark.parametrize("bad", [1.0, 1.5, -0.1])
def test_lorentz_gamma_rejects_out_of_range(bad: float) -> None:
    with pytest.raises(ValueError, match="beta_magnitude"):
        lorentz_gamma(bad)


@pytest.mark.parametrize("route", sorted(ABERRATION_ROUTES))
def test_every_route_rejects_a_superluminal_frame(route: str) -> None:
    with pytest.raises(ValueError, match=r"\|beta\| must be < 1"):
        aberrate([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], route)


@pytest.mark.parametrize("route", sorted(ABERRATION_ROUTES))
def test_every_route_returns_a_unit_vector(route: str) -> None:
    result = aberrate([1.0, 2.0, -3.0], [0.02, -0.03, 0.05], route)
    assert norm(result) == pytest.approx(1.0, abs=1e-15)


@pytest.mark.parametrize("route", sorted(ABERRATION_ROUTES))
def test_zero_boost_is_the_identity(route: str) -> None:
    direction = unit([1.0, -2.0, 0.5])
    assert np.allclose(aberrate(direction, np.zeros(3), route), direction, atol=1e-16)


def test_exact_transverse_aberration_known_answer() -> None:
    # Direction +y, boost 0.1 c along +x. The angle from the boost axis goes
    # from 90 degrees to arccos(-0.1) = 95.7392... degrees, so the direction
    # swings by 5.7392... degrees = arcsin(0.1) = 0.100167421161... rad.
    expected = float(np.arcsin(0.1))
    for route in EXACT_ROUTES:
        assert aberration_angle([0.0, 1.0, 0.0], [0.1, 0.0, 0.0], route) == pytest.approx(
            expected, rel=1e-14
        )


def test_first_order_transverse_aberration_known_answer() -> None:
    # The first-order form displaces +y by -0.1 x-hat, giving arctan(0.1)
    # = 0.0996686524912... rad, which is below the exact arcsin(0.1) by
    # 4.98e-4 rad -- an O(beta^3) difference at beta = 0.1.
    angle = aberration_angle([0.0, 1.0, 0.0], [0.1, 0.0, 0.0], "first_order")
    assert angle == pytest.approx(float(np.arctan(0.1)), rel=1e-14)
    assert float(np.arcsin(0.1)) - angle == pytest.approx(4.9879e-4, rel=1e-3)


def test_aberration_along_the_boost_axis_is_zero() -> None:
    for route in ABERRATION_ROUTES:
        assert aberration_angle([1.0, 0.0, 0.0], [0.3, 0.0, 0.0], route) == pytest.approx(
            0.0, abs=1e-15
        )
        assert aberration_angle([-1.0, 0.0, 0.0], [0.3, 0.0, 0.0], route) == pytest.approx(
            0.0, abs=1e-15
        )


def test_collinear_direction_is_preserved_exactly() -> None:
    for route in EXACT_ROUTES:
        assert np.allclose(aberrate([1.0, 0.0, 0.0], [0.4, 0.0, 0.0], route), [1.0, 0.0, 0.0])
        assert np.allclose(aberrate([-1.0, 0.0, 0.0], [0.4, 0.0, 0.0], route), [-1.0, 0.0, 0.0])


@pytest.mark.parametrize("route", sorted(EXACT_ROUTES))
def test_inverse_boost_recovers_the_direction(route: str) -> None:
    direction = unit([0.2, -1.0, 0.7])
    beta = np.array([0.05, 0.1, -0.02])
    recovered = aberrate(aberrate(direction, beta, route), -beta, route)
    assert angle_between(direction, recovered) < 1e-15


def test_the_three_exact_routes_agree_to_rounding() -> None:
    rng = np.random.default_rng(76)
    worst = 0.0
    for _ in range(400):
        direction = unit(rng.normal(size=3))
        beta = rng.normal(size=3)
        beta = beta / norm(beta) * float(rng.uniform(1.0e-8, 0.9))
        disagreement = route_disagreement(direction, beta)
        for name, value in disagreement.items():
            if "first_order" in name:
                continue
            worst = max(worst, value)
    assert worst < 1.0e-13, f"exact aberration routes disagree by {worst} rad"


def test_first_order_error_is_cubic_for_a_transverse_direction() -> None:
    # For a direction exactly transverse to the boost the first-order form
    # gives arctan(beta) and the exact one arcsin(beta), so the error is
    # arcsin(beta) - arctan(beta) = beta^3 / 2 + O(beta^5): one decade in beta
    # is three decades in the error.
    direction = np.array([0.0, 1.0, 0.0])
    errors = []
    for magnitude in (1.0e-3, 1.0e-2, 1.0e-1):
        beta = np.array([magnitude, 0.0, 0.0])
        errors.append(
            angle_between(
                aberrate_first_order(direction, beta), aberrate(direction, beta, "momentum_boost")
            )
        )
        assert errors[-1] == pytest.approx(0.5 * magnitude**3, rel=3.0e-3)
    assert errors[1] / errors[0] == pytest.approx(1000.0, rel=0.02)
    assert errors[2] / errors[1] == pytest.approx(1000.0, rel=0.02)


def test_first_order_error_is_quadratic_for_an_oblique_direction() -> None:
    # Away from the transverse case the second-order term survives and the
    # error is O(beta^2): one decade in beta is two decades in the error.
    direction = unit([1.0, 1.0, 0.0])
    errors = []
    for magnitude in (1.0e-3, 1.0e-2, 1.0e-1):
        beta = np.array([magnitude, 0.0, 0.0])
        errors.append(
            angle_between(
                aberrate_first_order(direction, beta), aberrate(direction, beta, "momentum_boost")
            )
        )
    assert 90.0 < errors[1] / errors[0] < 115.0
    assert 90.0 < errors[2] / errors[1] < 120.0


def test_event_worldline_baseline_cancels_exactly() -> None:
    direction = unit([0.3, 0.9, -0.2])
    beta = np.array([0.01, -0.02, 0.03])
    first = aberrate_event_worldline(direction, beta, baseline_m=1.0)
    second = aberrate_event_worldline(direction, beta, baseline_m=1.0e12)
    assert angle_between(first, second) < 1e-15


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan])
def test_event_worldline_rejects_a_bad_baseline(bad: float) -> None:
    with pytest.raises(ValueError, match="baseline_m"):
        aberrate_event_worldline([0.0, 1.0, 0.0], [0.1, 0.0, 0.0], bad)


def test_unknown_route_is_rejected_by_name() -> None:
    with pytest.raises(ValueError, match="unknown aberration route"):
        aberrate([0.0, 1.0, 0.0], [0.1, 0.0, 0.0], "astropy")


def test_orbital_speed_aberration_is_about_twenty_five_microradians() -> None:
    # A 600 km circular orbit has a speed of 7557.86... m/s; the transverse
    # aberration is v / c = 2.5210...e-05 rad. The value is computed here, not
    # quoted: the test asserts the relation, not a remembered number.
    speed = 7557.865206532812
    beta = beta_from_velocity([speed, 0.0, 0.0])
    angle = aberration_angle([0.0, 1.0, 0.0], beta)
    assert angle == pytest.approx(speed / C_LIGHT, rel=1e-9)
    assert 2.5e-5 < angle < 2.6e-5
