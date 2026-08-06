"""Pointing-loss closed form vs Monte Carlo and input validation."""

import numpy as np
import pytest

from jitterscope import pointing_loss_avg, pointing_loss_avg_mc


def test_zero_jitter_gives_no_loss():
    """Known answer: sigma = 0 -> <L> = 1 exactly."""
    assert pointing_loss_avg(0.0, 10e-6) == 1.0


def test_known_answer_sigma_equals_half_divergence():
    """Hand calculation: sigma/theta_div = 0.5 ->
    <L> = 1/(1 + 4*0.25) = 1/2 exactly."""
    assert pointing_loss_avg(5e-6, 10e-6) == pytest.approx(0.5, abs=1e-15)


def test_known_answer_sigma_equals_divergence():
    """Hand calculation: sigma = theta_div -> <L> = 1/5 = 0.2."""
    assert pointing_loss_avg(10e-6, 10e-6) == pytest.approx(0.2, abs=1e-15)


@pytest.mark.parametrize("ratio", [0.1, 0.3, 0.5, 1.0, 2.0])
def test_closed_form_matches_monte_carlo(ratio: float):
    """Seeded MC (200k draws) agrees with 1/(1+4 r^2) within 1 %.

    MC standard error ~ 1/sqrt(2e5) ~ 0.22 % of the mean, so a 1 %
    relative tolerance is ~4.5 sigma — a stable seeded test.
    """
    theta_div = 12e-6
    sigma = ratio * theta_div
    closed = pointing_loss_avg(sigma, theta_div)
    mc = pointing_loss_avg_mc(sigma, theta_div, n_samples=200_000, seed=99)
    assert mc == pytest.approx(closed, rel=0.01)


def test_loss_monotone_in_jitter():
    theta_div = 10e-6
    losses = [pointing_loss_avg(s, theta_div) for s in np.linspace(0, 50e-6, 20)]
    assert all(a >= b for a, b in zip(losses, losses[1:]))
    assert all(0 < loss <= 1 for loss in losses)


class TestInputValidation:
    def test_negative_sigma_raises(self):
        with pytest.raises(ValueError, match="sigma_theta"):
            pointing_loss_avg(-1e-6, 10e-6)

    def test_nonpositive_divergence_raises(self):
        with pytest.raises(ValueError, match="theta_div"):
            pointing_loss_avg(1e-6, 0.0)

    def test_nan_raises(self):
        with pytest.raises(ValueError, match="finite"):
            pointing_loss_avg(float("nan"), 10e-6)

    def test_mc_small_n_raises(self):
        with pytest.raises(ValueError, match="n_samples"):
            pointing_loss_avg_mc(1e-6, 10e-6, n_samples=10)
