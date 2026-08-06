"""Tests for first-order uncertainty propagation and Monte Carlo cross-check."""

import numpy as np
import pytest

from linkbudgetx import LinkBudget, monte_carlo_margin, propagate_margin_sigma


def nominal(**overrides):
    params = dict(
        tx_power_dbm=20.0,
        wavelength_nm=1550.0,
        beam_divergence_rad=1.0e-3,
        range_km=10.0,
        rx_aperture_diameter_m=0.1,
        rx_sensitivity_dbm=-40.0,
        tx_optics_efficiency=0.8,
        rx_optics_efficiency=0.8,
        pointing_error_rad=0.25e-3,
        atmos_attenuation_db_per_km=0.5,
        beam_profile="gaussian",
    )
    params.update(overrides)
    return LinkBudget(**params)


class TestFirstOrder:
    def test_tx_power_partial_is_exactly_one(self):
        # margin is linear in tx_power_dbm with slope 1, so sigma_margin = sigma.
        unc = propagate_margin_sigma(nominal(), {"tx_power_dbm": 0.5})
        assert unc.partials["tx_power_dbm"] == pytest.approx(1.0, abs=1e-6)
        assert unc.sigma_margin_db == pytest.approx(0.5, rel=1e-6)

    def test_atmos_partial_is_minus_range(self):
        # margin = ... - alpha*R, so d(margin)/d(alpha) = -R = -10 dB/(dB/km).
        unc = propagate_margin_sigma(nominal(), {"atmos_attenuation_db_per_km": 0.05})
        assert unc.partials["atmos_attenuation_db_per_km"] == pytest.approx(-10.0, rel=1e-6)
        assert unc.sigma_margin_db == pytest.approx(0.5, rel=1e-6)

    def test_rss_combination(self):
        # Two independent 0.5 dB contributions -> sqrt(0.5^2 + 0.5^2).
        unc = propagate_margin_sigma(
            nominal(), {"tx_power_dbm": 0.5, "atmos_attenuation_db_per_km": 0.05}
        )
        assert unc.sigma_margin_db == pytest.approx(np.hypot(0.5, 0.5), rel=1e-6)

    def test_unknown_field_raises(self):
        with pytest.raises(ValueError):
            propagate_margin_sigma(nominal(), {"nonexistent_field": 0.1})

    def test_negative_sigma_raises(self):
        with pytest.raises(ValueError):
            propagate_margin_sigma(nominal(), {"tx_power_dbm": -0.1})

    def test_zero_sigma_contributes_nothing(self):
        unc = propagate_margin_sigma(nominal(), {"tx_power_dbm": 0.0})
        assert unc.sigma_margin_db == 0.0


class TestMonteCarloCrossCheck:
    def test_linear_case_matches_mc(self):
        # tx power only: margin is exactly linear, MC std must equal sigma.
        b = nominal()
        samples = monte_carlo_margin(b, {"tx_power_dbm": 0.5}, n_samples=8000, seed=42)
        assert samples.std(ddof=1) == pytest.approx(0.5, rel=0.05)
        assert samples.mean() == pytest.approx(b.compute().margin_db, abs=0.05)

    def test_mildly_nonlinear_case_agrees_within_tolerance(self):
        # Small sigmas on several inputs: first order should agree with MC
        # to within ~10 % (documented linearity assumption).
        b = nominal()
        sigmas = {
            "tx_power_dbm": 0.5,
            "atmos_attenuation_db_per_km": 0.05,
            "pointing_error_rad": 0.02e-3,
        }
        unc = propagate_margin_sigma(b, sigmas)
        samples = monte_carlo_margin(b, sigmas, n_samples=8000, seed=1)
        assert samples.std(ddof=1) == pytest.approx(unc.sigma_margin_db, rel=0.10)

    def test_reproducible_with_seed(self):
        b = nominal()
        s1 = monte_carlo_margin(b, {"tx_power_dbm": 0.5}, n_samples=100, seed=7)
        s2 = monte_carlo_margin(b, {"tx_power_dbm": 0.5}, n_samples=100, seed=7)
        np.testing.assert_array_equal(s1, s2)

    def test_bad_n_samples_raises(self):
        with pytest.raises(ValueError):
            monte_carlo_margin(nominal(), {"tx_power_dbm": 0.5}, n_samples=0)
