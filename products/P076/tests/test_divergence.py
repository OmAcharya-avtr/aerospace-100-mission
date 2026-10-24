"""Beam divergence and pointing loss: known answers and inverses."""

from __future__ import annotations

import numpy as np
import pytest

from pointahead.divergence import (
    beam_fraction,
    error_for_loss_db,
    gaussian_divergence_half_angle,
    pointing_loss_db,
    pointing_loss_fraction,
    rayleigh_range,
)


def test_divergence_known_answer() -> None:
    # lambda = 1550 nm, w0 = 25 mm:
    #   theta = 1550e-9 / (pi * 0.025) = 1550e-9 / 0.0785398163...
    #         = 1.9735212943395024e-05 rad = 19.735 urad.
    theta = gaussian_divergence_half_angle(1550.0e-9, 0.025)
    assert theta == pytest.approx(1550.0e-9 / (np.pi * 0.025), rel=1e-15)
    assert theta == pytest.approx(1.9735212943395024e-05, rel=1e-13)


def test_divergence_scales_inversely_with_the_waist() -> None:
    narrow = gaussian_divergence_half_angle(1064.0e-9, 0.075)
    wide = gaussian_divergence_half_angle(1064.0e-9, 0.025)
    assert wide / narrow == pytest.approx(3.0, rel=1e-13)


def test_rayleigh_range_known_answer() -> None:
    # z_R = pi w0^2 / lambda = pi * 0.025^2 / 1550e-9 = 1266.7712... m
    assert rayleigh_range(1550.0e-9, 0.025) == pytest.approx(
        np.pi * 0.025**2 / 1550.0e-9, rel=1e-13
    )


@pytest.mark.parametrize(
    ("wavelength", "waist"),
    [(0.0, 0.025), (-1.0, 0.025), (1550.0e-9, 0.0), (1550.0e-9, -0.025), (np.nan, 0.025)],
)
def test_divergence_rejects_non_positive_inputs(wavelength: float, waist: float) -> None:
    with pytest.raises(ValueError, match="must be positive and finite"):
        gaussian_divergence_half_angle(wavelength, waist)


def test_divergence_rejects_a_sub_wavelength_waist() -> None:
    with pytest.raises(ValueError, match="paraxial far-field"):
        gaussian_divergence_half_angle(1550.0e-9, 1.0e-9)


def test_loss_at_boresight_is_zero() -> None:
    assert pointing_loss_db(0.0, 2.0e-5) == 0.0
    assert pointing_loss_fraction(0.0, 2.0e-5) == 1.0


def test_loss_at_one_divergence_half_angle_known_answer() -> None:
    # At theta = theta_div the normalised intensity is exp(-2) = 0.1353352832...
    # and the loss is 20 / ln(10) = 8.685889638065035 dB.
    assert pointing_loss_fraction(2.0e-5, 2.0e-5) == pytest.approx(float(np.exp(-2.0)), rel=1e-14)
    assert pointing_loss_db(2.0e-5, 2.0e-5) == pytest.approx(
        20.0 / float(np.log(10.0)), rel=1e-14
    )
    assert pointing_loss_db(2.0e-5, 2.0e-5) == pytest.approx(8.685889638065035, rel=1e-13)


def test_loss_is_quadratic_in_the_error() -> None:
    theta = 2.0e-5
    assert pointing_loss_db(0.5 * theta, theta) / pointing_loss_db(
        0.25 * theta, theta
    ) == pytest.approx(4.0, rel=1e-13)


def test_error_for_loss_round_trips() -> None:
    theta = 1.9735212943395024e-05
    for loss in (0.1, 0.5, 1.0, 3.0):
        error = error_for_loss_db(loss, theta)
        assert pointing_loss_db(error, theta) == pytest.approx(loss, rel=1e-13)


def test_affordable_error_at_one_db_known_answer() -> None:
    # theta = theta_div sqrt(L ln(10) / 20); at L = 1 dB and
    # theta_div = 19.735213 urad this is 6.696296... urad.
    theta = gaussian_divergence_half_angle(1550.0e-9, 0.025)
    expected = theta * float(np.sqrt(1.0 * np.log(10.0) / 20.0))
    assert error_for_loss_db(1.0, theta) == pytest.approx(expected, rel=1e-14)
    assert error_for_loss_db(1.0, theta) == pytest.approx(6.696296316980666e-06, rel=1e-12)


def test_beam_fraction_known_answer() -> None:
    assert beam_fraction(1.0e-5, 2.0e-5) == pytest.approx(0.5, rel=1e-15)


@pytest.mark.parametrize("bad", [-1.0e-6, np.nan])
def test_negative_error_is_refused(bad: float) -> None:
    with pytest.raises(ValueError, match="error_rad must be non-negative"):
        pointing_loss_db(bad, 2.0e-5)


@pytest.mark.parametrize("bad", [0.0, -2.0e-5, np.inf])
def test_bad_divergence_is_refused(bad: float) -> None:
    with pytest.raises(ValueError, match="divergence_rad must be positive"):
        pointing_loss_db(1.0e-6, bad)


@pytest.mark.parametrize("bad", [-1.0, np.nan])
def test_bad_loss_budget_is_refused(bad: float) -> None:
    with pytest.raises(ValueError, match="loss_db must be non-negative"):
        error_for_loss_db(bad, 2.0e-5)


def test_the_same_error_costs_very_different_losses_on_two_beams() -> None:
    # This is the product's whole premise, so it is a test. A 5 urad error on a
    # 50 urad beam and on a 5 urad beam differ by a factor measured here.
    wide = pointing_loss_db(5.0e-6, 5.0e-5)
    narrow = pointing_loss_db(5.0e-6, 5.0e-6)
    assert narrow / wide == pytest.approx(100.0, rel=1e-12)
    assert wide < 0.1
    assert narrow > 8.0
