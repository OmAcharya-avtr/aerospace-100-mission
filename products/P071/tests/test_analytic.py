"""The three analytic Pc paths against closed forms and against each other.

Every comparison here is a tolerance on both sides. Nothing asserts bit-exact
float equality, and nothing asserts a boolean produced by a threshold.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from conjprob.analytic import (
    pc_alfano,
    pc_chan,
    pc_circular_closed_form,
    pc_foster,
    pc_small_body_approximation,
)
from conjprob.quadrature import pc_quadrature

METHODS = (pc_foster, pc_chan, pc_alfano)


def test_rayleigh_closed_form_hand_calculated():
    # sigma = 100 m, R = 20 m, zero miss: 1 - exp(-R^2 / (2 sigma^2))
    # = 1 - exp(-0.02) = 0.0198013266932447 to 15 significant figures.
    assert pc_circular_closed_form(20.0, 100.0) == pytest.approx(
        0.019801326693244697, rel=1e-13
    )


@pytest.mark.parametrize("radius_m", [1.0, 20.0, 100.0, 400.0])
def test_foster_reproduces_rayleigh_closed_form(radius_m):
    expected = pc_circular_closed_form(radius_m, 100.0)
    assert pc_foster(100.0, 100.0, 0.0, 0.0, radius_m) == pytest.approx(expected, rel=1e-10)


@pytest.mark.parametrize("radius_m", [1.0, 20.0, 100.0, 400.0])
def test_chan_reproduces_rayleigh_closed_form_at_zero_miss(radius_m):
    # At zero miss the Chan series has a single term and is exact for every
    # order, because v = 0 kills every term beyond m = 0.
    expected = pc_circular_closed_form(radius_m, 100.0)
    assert pc_chan(100.0, 100.0, 0.0, 0.0, radius_m, order=0) == pytest.approx(
        expected, rel=1e-13
    )


def test_alfano_reproduces_rayleigh_to_its_measured_accuracy():
    # The half-chord has a singular derivative at the endpoints, so Simpson's
    # rule does not reach its nominal order. The measured relative error at 201
    # nodes for this geometry is 1.4e-4, so the test allows 1e-3 and no more.
    expected = pc_circular_closed_form(20.0, 100.0)
    assert pc_alfano(100.0, 100.0, 0.0, 0.0, 20.0) == pytest.approx(expected, rel=1e-3)


def test_alfano_error_decreases_with_node_count():
    expected = pc_circular_closed_form(20.0, 100.0)
    coarse = abs(pc_alfano(100.0, 100.0, 0.0, 0.0, 20.0, nodes=51) - expected)
    fine = abs(pc_alfano(100.0, 100.0, 0.0, 0.0, 20.0, nodes=801) - expected)
    assert fine < coarse


@pytest.mark.parametrize(
    "sigma_x, sigma_y, xbar, ybar, radius",
    [
        (300.0, 54.0, 120.0, 30.0, 12.0),
        (100.0, 100.0, 250.0, 0.0, 5.0),
        (50.0, 20.0, 10.0, -8.0, 3.0),
        (1000.0, 900.0, 0.0, 1500.0, 25.0),
    ],
)
def test_foster_matches_independent_quadrature(sigma_x, sigma_y, xbar, ybar, radius):
    reference, abserr = pc_quadrature(
        np.diag([sigma_x**2, sigma_y**2]), (xbar, ybar), radius
    )
    value = pc_foster(sigma_x, sigma_y, xbar, ybar, radius)
    assert value == pytest.approx(reference, rel=1e-8, abs=10.0 * abserr)


def test_small_body_approximation_hand_calculated():
    # R = 12 m, sigma_x = 300 m, sigma_y = 54 m, miss (120, 30) m.
    # M^2 = (120/300)^2 + (30/54)^2 = 0.16 + 0.308641975308642 = 0.468641975308642
    # area term = 144 / (2 * 300 * 54) = 0.0044444444444444444
    # Pc = 0.0044444444444444444 * exp(-0.234320987654321) = 0.003516034841273
    assert pc_small_body_approximation(300.0, 54.0, 120.0, 30.0, 12.0) == pytest.approx(
        0.0035160348412730624, rel=1e-12
    )


def test_small_body_approximation_is_an_upper_bound_near_a_flat_density():
    # With the hard body well inside one standard deviation and the miss at the
    # mode, treating the density as constant over the disc overestimates Pc.
    exact = pc_foster(500.0, 500.0, 0.0, 0.0, 5.0)
    approximate = pc_small_body_approximation(500.0, 500.0, 0.0, 0.0, 5.0)
    assert approximate >= exact
    assert approximate == pytest.approx(exact, rel=1e-4)


@pytest.mark.parametrize("method", METHODS)
def test_pc_increases_with_hard_body_radius(method):
    values = [method(200.0, 80.0, 90.0, 20.0, radius) for radius in (1.0, 5.0, 20.0, 60.0)]
    assert all(b > a for a, b in zip(values, values[1:], strict=False))


@pytest.mark.parametrize("method", METHODS)
def test_pc_is_a_probability(method):
    for radius in (1.0, 100.0, 10000.0):
        value = method(50.0, 20.0, 10.0, 5.0, radius)
        assert 0.0 <= value <= 1.0


@pytest.mark.parametrize("method", METHODS)
def test_zero_radius_gives_zero(method):
    assert method(100.0, 50.0, 10.0, 10.0, 0.0) == 0.0


@pytest.mark.parametrize("method", METHODS)
def test_scale_invariance(method):
    # Scaling every length by the same factor leaves the integral unchanged in
    # exact arithmetic; the tolerance is for rounding, not for approximation.
    base = method(300.0, 54.0, 120.0, 30.0, 12.0)
    for factor in (1e-3, 0.5, 7.0, 1e4):
        scaled = method(
            300.0 * factor, 54.0 * factor, 120.0 * factor, 30.0 * factor, 12.0 * factor
        )
        assert scaled == pytest.approx(base, rel=1e-9)


@pytest.mark.parametrize("method", METHODS)
def test_sign_symmetry_of_the_miss_vector(method):
    base = method(300.0, 54.0, 120.0, 30.0, 12.0)
    for xbar, ybar in ((-120.0, 30.0), (120.0, -30.0), (-120.0, -30.0)):
        assert method(300.0, 54.0, xbar, ybar, 12.0) == pytest.approx(base, rel=1e-9)


def test_chan_series_converges_in_order_to_its_own_limit():
    # The series converges, but not to the exact answer: it converges to the
    # equivalent-area answer, which differs from the exact one whenever the
    # aspect ratio is not one. The measured residual for this geometry at
    # aspect 1.2 is 1.5e-4 relative and does not shrink with further terms.
    sigma_x, sigma_y, xbar, ybar, radius = 120.0, 100.0, 150.0, 60.0, 8.0
    reference = pc_foster(sigma_x, sigma_y, xbar, ybar, radius)
    errors = [
        abs(pc_chan(sigma_x, sigma_y, xbar, ybar, radius, order=order) - reference)
        for order in (0, 2, 6, 14, 30)
    ]
    assert errors[3] < errors[0]
    assert errors[4] == pytest.approx(errors[3], rel=1e-6)
    assert errors[4] / reference < 1e-3


def test_chan_series_is_exact_in_the_limit_of_a_circular_covariance():
    # Removing the aspect-ratio error leaves only the truncation error, which
    # does go to zero: at aspect exactly one the series converges to the exact
    # value to within rounding.
    sigma, xbar, ybar, radius = 110.0, 150.0, 60.0, 8.0
    reference = pc_foster(sigma, sigma, xbar, ybar, radius)
    assert pc_chan(sigma, sigma, xbar, ybar, radius, order=40) == pytest.approx(
        reference, rel=1e-9
    )


def test_chan_departs_from_quadrature_at_large_aspect_ratio():
    # A measured statement, not an assertion of theory: at aspect ratio 20 and
    # R / sigma_eff = 1 the equivalent-area substitution is wrong by more than
    # 10 per cent. The sweep in validation/validate_regimes.py maps the whole
    # boundary; this pins one point of it.
    sigma_x, sigma_y = 400.0, 20.0
    sigma_eff = math.sqrt(sigma_x * sigma_y)
    reference = pc_foster(sigma_x, sigma_y, sigma_x, 0.0, sigma_eff)
    chan = pc_chan(sigma_x, sigma_y, sigma_x, 0.0, sigma_eff)
    assert abs(chan / reference - 1.0) > 0.1


@pytest.mark.parametrize("method", METHODS)
def test_rejects_non_positive_sigma(method):
    with pytest.raises(ValueError, match="strictly positive"):
        method(0.0, 50.0, 1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="strictly positive"):
        method(50.0, -1.0, 1.0, 1.0, 1.0)


@pytest.mark.parametrize("method", METHODS)
def test_rejects_negative_radius(method):
    with pytest.raises(ValueError, match="non-negative"):
        method(50.0, 50.0, 1.0, 1.0, -1.0)


@pytest.mark.parametrize("method", METHODS)
def test_rejects_non_finite_input(method):
    with pytest.raises(ValueError, match="finite"):
        method(50.0, 50.0, math.inf, 1.0, 1.0)


def test_alfano_rejects_even_or_small_node_counts():
    with pytest.raises(ValueError, match="odd"):
        pc_alfano(50.0, 50.0, 1.0, 1.0, 1.0, nodes=200)
    with pytest.raises(ValueError, match="odd"):
        pc_alfano(50.0, 50.0, 1.0, 1.0, 1.0, nodes=2)


def test_chan_rejects_negative_order():
    with pytest.raises(ValueError, match="non-negative"):
        pc_chan(50.0, 50.0, 1.0, 1.0, 1.0, order=-1)


def test_circular_closed_form_rejects_bad_input():
    with pytest.raises(ValueError, match="strictly positive"):
        pc_circular_closed_form(1.0, 0.0)
    with pytest.raises(ValueError, match="non-negative"):
        pc_circular_closed_form(-1.0, 1.0)
