"""Dimensionless parameterisation and the measured regime boundaries."""

from __future__ import annotations

import math

import numpy as np
import pytest

from conjprob.regimes import (
    METHOD_NAMES,
    evaluate_methods,
    geometry_from_dimensionless,
    regime_sweep,
)


def test_dimensionless_geometry_reproduces_its_parameters():
    sigma_x, sigma_y, xbar, ybar, radius = geometry_from_dimensionless(
        0.3, 4.0, 2.5, psi_rad=0.7, sigma_effective_m=250.0
    )
    assert math.sqrt(sigma_x * sigma_y) == pytest.approx(250.0, rel=1e-13)
    assert sigma_x / sigma_y == pytest.approx(4.0, rel=1e-13)
    assert math.hypot(xbar / sigma_x, ybar / sigma_y) == pytest.approx(2.5, rel=1e-13)
    assert radius == pytest.approx(0.3 * 250.0, rel=1e-13)


def test_dimensionless_geometry_mahalanobis_is_independent_of_psi():
    for psi in (0.0, 0.3, 1.2, 2.9, 5.5):
        sigma_x, sigma_y, xbar, ybar, _ = geometry_from_dimensionless(
            0.1, 7.0, 1.75, psi_rad=psi
        )
        assert math.hypot(xbar / sigma_x, ybar / sigma_y) == pytest.approx(1.75, rel=1e-12)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"aspect_ratio": 0.5}, "aspect_ratio"),
        ({"radius_over_sigma": -1.0}, "radius_over_sigma"),
        ({"mahalanobis": -1.0}, "mahalanobis"),
        ({"sigma_effective_m": 0.0}, "sigma_effective_m"),
    ],
)
def test_dimensionless_geometry_rejects_bad_input(kwargs, message):
    defaults = {"radius_over_sigma": 0.1, "aspect_ratio": 1.0, "mahalanobis": 1.0}
    defaults.update(kwargs)
    with pytest.raises(ValueError, match=message):
        geometry_from_dimensionless(**defaults)


def test_evaluate_methods_returns_every_key():
    values = evaluate_methods(*geometry_from_dimensionless(0.2, 2.0, 1.0))
    for name in METHOD_NAMES:
        assert name in values
    assert "reference" in values
    assert "reference_abserr" in values
    assert values["reference"] > 0.0


def test_foster_is_within_quadrature_error_everywhere_on_a_small_grid():
    sweep = regime_sweep(
        np.logspace(-2.0, 0.5, 8), np.array([1.0, 10.0]), np.array([0.5, 3.0])
    )
    assert sweep.worst_error("foster") < 1e-8


def test_alfano_error_is_small_but_not_negligible_on_the_same_grid():
    # Alfano's fixed-node Simpson rule is a different quadrature of the same
    # reduction, so it agrees with Foster to the accuracy of the rule and no
    # better. The measured worst error on this grid is reported in
    # validation/validate_regimes_output.txt.
    sweep = regime_sweep(
        np.logspace(-2.0, 0.5, 8), np.array([1.0, 10.0]), np.array([0.5, 3.0])
    )
    worst = sweep.worst_error("alfano")
    assert 1e-12 < worst < 1e-2


def test_chan_and_small_body_errors_are_measurable_on_the_same_grid():
    sweep = regime_sweep(
        np.logspace(-2.0, 0.5, 8), np.array([1.0, 10.0]), np.array([0.5, 3.0])
    )
    assert sweep.worst_error("chan") > 1e-3
    assert sweep.worst_error("small_body") > 1e-3


def test_boundary_is_monotone_in_the_error_it_thresholds():
    sweep = regime_sweep(
        np.logspace(-3.0, 0.6, 30), np.array([1.0]), np.array([1.0])
    )
    loose = sweep.boundary("small_body", 1e-2)
    tight = sweep.boundary("small_body", 1e-4)
    assert tight[0, 0] < loose[0, 0]


def test_boundary_is_nan_where_the_tolerance_is_never_exceeded():
    sweep = regime_sweep(
        np.logspace(-3.0, -2.0, 6), np.array([1.0]), np.array([1.0])
    )
    assert np.isnan(sweep.boundary("foster", 1e-6)[0, 0])


def test_boundary_rejects_an_unknown_method():
    sweep = regime_sweep(np.array([0.1]), np.array([1.0]), np.array([1.0]))
    with pytest.raises(KeyError, match="unknown method"):
        sweep.boundary("nope", 1e-3)


def test_regime_sweep_rejects_empty_axes():
    with pytest.raises(ValueError, match="non-empty"):
        regime_sweep(np.array([]), np.array([1.0]), np.array([1.0]))
