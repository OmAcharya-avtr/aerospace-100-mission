"""Unit tests for beamtwin.channel (scintillation, jitter, Monte Carlo)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from beamtwin.budget import LinkParams, compute_budget
from beamtwin.channel import (
    ChannelParams,
    build_channel_model,
    lognormal_sigma_ln,
    mean_pointing_loss_fraction,
    rytov_variance_plane_wave,
    sample_received_power_dbm,
)


class TestRytovVariance:
    def test_known_answer_10km(self):
        # sigma_R^2 = 1.23 Cn2 k^(7/6) L^(11/6); Cn2=5e-16, 1550 nm, 10 km.
        k = 2 * math.pi / 1550e-9
        expected = 1.23 * 5e-16 * k ** (7 / 6) * 10_000.0 ** (11 / 6)
        assert rytov_variance_plane_wave(5e-16, 1550e-9, 10_000.0) == pytest.approx(expected)
        # Cross-check against the value reported by the twin report (0.6782).
        assert expected == pytest.approx(0.6782, rel=1e-3)

    def test_zero_cn2_gives_zero(self):
        assert rytov_variance_plane_wave(0.0, 1550e-9, 10_000.0) == 0.0

    def test_scales_linearly_with_cn2(self):
        a = rytov_variance_plane_wave(1e-16, 1550e-9, 5000.0)
        b = rytov_variance_plane_wave(2e-16, 1550e-9, 5000.0)
        assert b == pytest.approx(2 * a)

    def test_range_exponent_is_11_over_6(self):
        a = rytov_variance_plane_wave(1e-15, 1550e-9, 1000.0)
        b = rytov_variance_plane_wave(1e-15, 1550e-9, 2000.0)
        assert b / a == pytest.approx(2 ** (11 / 6))

    def test_wavelength_exponent_is_7_over_6(self):
        # sigma_R^2 ~ k^(7/6) ~ lambda^(-7/6); halving lambda multiplies by 2^(7/6).
        a = rytov_variance_plane_wave(1e-15, 1550e-9, 5000.0)
        b = rytov_variance_plane_wave(1e-15, 775e-9, 5000.0)
        assert b / a == pytest.approx(2 ** (7 / 6))

    def test_rejects_negative_cn2(self):
        with pytest.raises(ValueError):
            rytov_variance_plane_wave(-1e-15, 1550e-9, 1000.0)


class TestLognormalSigma:
    def test_zero_index_gives_zero_sigma(self):
        assert lognormal_sigma_ln(0.0) == 0.0

    def test_known_answer(self):
        # sigma_ln = sqrt(ln(1+0.6782)) = 0.71954.
        assert lognormal_sigma_ln(0.6782) == pytest.approx(0.71954, rel=1e-4)

    def test_monotone_increasing(self):
        vals = [lognormal_sigma_ln(s) for s in (0.0, 0.1, 0.5, 1.0, 2.0)]
        assert all(b > a for a, b in zip(vals, vals[1:]))

    def test_rejects_negative(self):
        with pytest.raises(ValueError):
            lognormal_sigma_ln(-0.1)


class TestMeanPointingLoss:
    def test_zero_jitter_is_unity(self):
        assert mean_pointing_loss_fraction(0.0, 0.25) == pytest.approx(1.0)

    def test_known_answer(self):
        # E[L_p] = 1/(1+4*0.05^2/0.2475^2) = 0.859661 (validation V2b).
        assert mean_pointing_loss_fraction(0.05, 0.2475) == pytest.approx(0.859661, rel=1e-5)

    def test_monotone_decreasing_in_jitter(self):
        vals = [mean_pointing_loss_fraction(s, 0.25) for s in (0.0, 0.02, 0.05, 0.1)]
        assert all(b < a for a, b in zip(vals, vals[1:]))

    def test_in_unit_interval(self):
        for s in (0.0, 0.01, 0.1, 1.0):
            assert 0.0 < mean_pointing_loss_fraction(s, 0.25) <= 1.0

    def test_rejects_negative_sigma(self):
        with pytest.raises(ValueError):
            mean_pointing_loss_fraction(-0.01, 0.25)


class TestChannelParamsValidation:
    def test_defaults_construct(self):
        assert ChannelParams().cn2 == pytest.approx(1e-15)

    def test_rejects_negative_cn2(self):
        with pytest.raises(ValueError):
            ChannelParams(cn2=-1e-16)

    def test_rejects_absurd_cn2(self):
        with pytest.raises(ValueError, match="check units"):
            ChannelParams(cn2=1.0)

    def test_rejects_negative_jitter(self):
        with pytest.raises(ValueError):
            ChannelParams(pointing_jitter_rad=-1e-6)

    def test_rejects_nan(self):
        with pytest.raises((TypeError, ValueError)):
            ChannelParams(cn2=float("nan"))


class TestChannelModel:
    def test_weak_regime_flag_true_for_low_turbulence(self):
        m = build_channel_model(LinkParams(range_m=5000.0), ChannelParams(cn2=1e-16))
        assert m.rytov_variance < 1.0
        assert m.weak_regime_valid is True

    def test_weak_regime_flag_false_for_strong_turbulence(self):
        m = build_channel_model(LinkParams(range_m=15_000.0), ChannelParams(cn2=5e-14))
        assert m.rytov_variance >= 1.0
        assert m.weak_regime_valid is False

    def test_displacement_sigma_is_jitter_times_range(self):
        m = build_channel_model(
            LinkParams(range_m=10_000.0), ChannelParams(pointing_jitter_rad=5e-6)
        )
        assert m.sigma_disp_m == pytest.approx(0.05)


class TestMonteCarlo:
    def test_reproducible_with_same_seed(self):
        a = sample_received_power_dbm(LinkParams(), ChannelParams(), n_samples=1000, seed=3)
        b = sample_received_power_dbm(LinkParams(), ChannelParams(), n_samples=1000, seed=3)
        assert np.array_equal(a.samples_dbm, b.samples_dbm)

    def test_different_seeds_differ(self):
        a = sample_received_power_dbm(LinkParams(), ChannelParams(), n_samples=1000, seed=3)
        b = sample_received_power_dbm(LinkParams(), ChannelParams(), n_samples=1000, seed=4)
        assert not np.array_equal(a.samples_dbm, b.samples_dbm)

    def test_sample_shape_and_finiteness(self):
        r = sample_received_power_dbm(LinkParams(), ChannelParams(), n_samples=5000, seed=1)
        assert r.samples_dbm.shape == (5000,)
        assert np.all(np.isfinite(r.samples_dbm))

    def test_deterministic_when_no_randomness(self):
        # Cn2 = 0 and jitter = 0 -> every sample equals the deterministic budget.
        link = LinkParams(range_m=5000.0)
        r = sample_received_power_dbm(
            link, ChannelParams(cn2=0.0, pointing_jitter_rad=0.0), n_samples=200, seed=1
        )
        expected = compute_budget(link).received_power_dbm
        assert np.allclose(r.samples_dbm, expected)

    def test_static_bias_reproduced_without_jitter(self):
        # With jitter = 0 but a static bias, samples equal the budget including bias.
        link = LinkParams(range_m=5000.0, pointing_bias_rad=3e-6)
        r = sample_received_power_dbm(
            link, ChannelParams(cn2=0.0, pointing_jitter_rad=0.0), n_samples=100, seed=2
        )
        assert np.allclose(r.samples_dbm, compute_budget(link).received_power_dbm)

    def test_mean_irradiance_is_preserved_by_scintillation(self):
        # E[X] = 1 for the lognormal factor, so mean linear power ~= deterministic.
        link = LinkParams(range_m=5000.0)
        r = sample_received_power_dbm(
            link, ChannelParams(cn2=5e-16, pointing_jitter_rad=0.0), n_samples=400_000, seed=9
        )
        mean_w = float(np.mean(10 ** (r.samples_dbm / 10) * 1e-3))
        det_w = 10 ** (compute_budget(link).received_power_dbm / 10) * 1e-3
        assert mean_w == pytest.approx(det_w, rel=0.01)

    def test_jitter_reduces_mean_power(self):
        link = LinkParams(range_m=10_000.0)
        no_j = sample_received_power_dbm(
            link, ChannelParams(cn2=0.0, pointing_jitter_rad=0.0), n_samples=50_000, seed=1
        )
        with_j = sample_received_power_dbm(
            link, ChannelParams(cn2=0.0, pointing_jitter_rad=1e-5), n_samples=50_000, seed=1
        )
        assert float(np.mean(with_j.samples_dbm)) < float(np.mean(no_j.samples_dbm))

    def test_scintillation_increases_spread(self):
        link = LinkParams(range_m=10_000.0)
        weak = sample_received_power_dbm(
            link, ChannelParams(cn2=1e-17, pointing_jitter_rad=0.0), n_samples=50_000, seed=1
        )
        strong = sample_received_power_dbm(
            link, ChannelParams(cn2=1e-15, pointing_jitter_rad=0.0), n_samples=50_000, seed=1
        )
        assert float(np.std(strong.samples_dbm)) > float(np.std(weak.samples_dbm))

    def test_result_records_seed_and_n(self):
        r = sample_received_power_dbm(LinkParams(), ChannelParams(), n_samples=128, seed=77)
        assert r.n_samples == 128 and r.seed == 77

    @pytest.mark.parametrize("n", [0, -1, 2.5, "1000"])
    def test_invalid_n_samples_raises(self, n):
        with pytest.raises((ValueError, TypeError)):
            sample_received_power_dbm(LinkParams(), ChannelParams(), n_samples=n, seed=0)

    def test_excessive_n_samples_raises(self):
        with pytest.raises(ValueError, match="budget"):
            sample_received_power_dbm(
                LinkParams(), ChannelParams(), n_samples=50_000_000, seed=0
            )

    @pytest.mark.parametrize("seed", [-1, 1.5, "abc"])
    def test_invalid_seed_raises(self, seed):
        with pytest.raises((ValueError, TypeError)):
            sample_received_power_dbm(LinkParams(), ChannelParams(), n_samples=10, seed=seed)
