"""Tests for waveforge.atmosphere."""

from __future__ import annotations

import numpy as np
import pytest

from waveforge.atmosphere import (
    FrozenFlowAtmosphere,
    band_limited_structure_function,
    phase_screen,
    screen_psd,
    structure_function,
)
from waveforge.pupil import PupilGrid, variance


class TestScreenPSD:
    def test_kolmogorov_slope(self):
        f = np.array([1.0, 2.0])
        psd = screen_psd(f, np.zeros(2), 0.1)
        assert psd[0] / psd[1] == pytest.approx(2.0 ** (11 / 3), rel=1e-12)

    def test_zero_frequency_is_zero_for_kolmogorov(self):
        assert screen_psd(np.array(0.0), np.array(0.0), 0.1)[0] == 0.0

    def test_von_karman_is_finite_at_zero(self):
        value = screen_psd(np.array(0.0), np.array(0.0), 0.1, outer_scale_m=20.0)[0]
        assert np.isfinite(value) and value > 0.0

    def test_von_karman_matches_kolmogorov_at_high_frequency(self):
        f = np.array([50.0])
        k = screen_psd(f, np.zeros(1), 0.1)
        v = screen_psd(f, np.zeros(1), 0.1, outer_scale_m=20.0)
        assert v[0] == pytest.approx(k[0], rel=1e-5)

    def test_von_karman_below_kolmogorov_at_low_frequency(self):
        f = np.array([0.01])
        assert screen_psd(f, np.zeros(1), 0.1, 20.0)[0] < screen_psd(f, np.zeros(1), 0.1)[0]

    def test_bad_r0(self):
        with pytest.raises(ValueError, match="r0_m"):
            screen_psd(np.array(1.0), np.array(0.0), -1.0)

    def test_bad_outer_scale(self):
        with pytest.raises(ValueError, match="outer_scale_m"):
            screen_psd(np.array(1.0), np.array(0.0), 0.1, outer_scale_m=0.0)


class TestPhaseScreen:
    def test_shape_and_dtype(self):
        screen = phase_screen(32, 0.01, 0.1, rng=0)
        assert screen.shape == (32, 32)
        assert screen.dtype == np.float64

    def test_deterministic_with_seed(self):
        a = phase_screen(32, 0.01, 0.1, rng=7)
        b = phase_screen(32, 0.01, 0.1, rng=7)
        assert np.array_equal(a, b)

    def test_different_seeds_differ(self):
        a = phase_screen(32, 0.01, 0.1, rng=7)
        b = phase_screen(32, 0.01, 0.1, rng=8)
        assert not np.allclose(a, b)

    def test_generator_accepted(self):
        gen = np.random.default_rng(3)
        assert phase_screen(16, 0.01, 0.1, rng=gen).shape == (16, 16)

    def test_piston_removed_by_default(self):
        assert phase_screen(32, 0.01, 0.1, rng=1).mean() == pytest.approx(0.0, abs=1e-10)

    def test_piston_retained_when_asked(self):
        screen = phase_screen(32, 0.01, 0.1, rng=1, remove_piston=False)
        assert abs(screen.mean()) > 0.0

    def test_variance_scales_with_r0(self):
        # variance propto r0^(-5/3)
        weak = np.mean([np.var(phase_screen(64, 0.01, 0.4, rng=s)) for s in range(12)])
        strong = np.mean([np.var(phase_screen(64, 0.01, 0.2, rng=s)) for s in range(12)])
        assert strong / weak == pytest.approx(2.0 ** (5 / 3), rel=0.15)

    def test_subharmonics_increase_low_frequency_power(self):
        plain = np.mean([np.var(phase_screen(64, 0.02, 0.1, rng=s)) for s in range(12)])
        rich = np.mean(
            [np.var(phase_screen(64, 0.02, 0.1, n_subharmonics=4, rng=s)) for s in range(12)]
        )
        assert rich > plain

    def test_von_karman_reduces_variance(self):
        plain = np.mean([np.var(phase_screen(64, 0.02, 0.1, rng=s)) for s in range(8)])
        capped = np.mean(
            [np.var(phase_screen(64, 0.02, 0.1, outer_scale_m=0.5, rng=s)) for s in range(8)]
        )
        assert capped < plain

    @pytest.mark.parametrize("n_pix", [3, 0, -8, 8.5])
    def test_bad_n_pix(self, n_pix):
        with pytest.raises(ValueError, match="n_pix"):
            phase_screen(n_pix, 0.01, 0.1)

    @pytest.mark.parametrize("scale", [0.0, -0.01, float("nan")])
    def test_bad_pixel_scale(self, scale):
        with pytest.raises(ValueError, match="pixel_scale_m"):
            phase_screen(16, scale, 0.1)

    def test_bad_subharmonics(self):
        with pytest.raises(ValueError, match="n_subharmonics"):
            phase_screen(16, 0.01, 0.1, n_subharmonics=-1)


class TestStructureFunction:
    def test_matches_band_limited_prediction(self):
        # Implementation check: the measured structure function of Fourier
        # screens must match the exact discrete-spectrum expectation.
        n, d, r0 = 128, 0.02, 0.1
        acc = np.zeros(n // 8)
        for seed in range(24):
            _, measured = structure_function(
                phase_screen(n, d, r0, rng=1000 + seed), max_lag=n // 8
            )
            acc += measured
        acc /= 24
        lags = np.arange(1, n // 8 + 1)
        expected = band_limited_structure_function(n, d, r0, lags)
        assert np.max(np.abs(acc / expected - 1.0)) < 0.10

    def test_band_limited_is_below_continuous_theory(self):
        from waveforge.statistics import phase_structure_function

        lags = np.array([1, 2, 4, 8])
        band = band_limited_structure_function(64, 0.02, 0.1, lags)
        theory = phase_structure_function(lags * 0.02, 0.1)
        assert np.all(band < theory)

    def test_increases_with_lag(self):
        lags, d_phi = structure_function(phase_screen(64, 0.02, 0.1, rng=5), max_lag=16)
        assert np.all(np.diff(d_phi) > 0.0)

    def test_zero_for_constant_screen(self):
        _, d_phi = structure_function(np.ones((16, 16)), max_lag=4)
        assert np.allclose(d_phi, 0.0)

    def test_axis_zero_works(self):
        lags, d_phi = structure_function(phase_screen(32, 0.02, 0.1, rng=2), max_lag=4, axis=0)
        assert len(lags) == 4 and np.all(d_phi > 0.0)

    def test_rejects_1d(self):
        with pytest.raises(ValueError, match="2-D"):
            structure_function(np.zeros(16))

    def test_rejects_bad_axis(self):
        with pytest.raises(ValueError, match="axis"):
            structure_function(np.zeros((8, 8)), axis=2)

    def test_rejects_bad_max_lag(self):
        with pytest.raises(ValueError, match="max_lag"):
            structure_function(np.zeros((8, 8)), max_lag=8)

    def test_band_limited_rejects_negative_lags(self):
        with pytest.raises(ValueError, match="lags"):
            band_limited_structure_function(16, 0.01, 0.1, np.array([-1.0]))

    def test_band_limited_rejects_bad_grid(self):
        with pytest.raises(ValueError, match="n_pix"):
            band_limited_structure_function(2, 0.01, 0.1, np.array([1.0]))
        with pytest.raises(ValueError, match="pixel_scale_m"):
            band_limited_structure_function(16, 0.0, 0.1, np.array([1.0]))


class TestFrozenFlow:
    @pytest.fixture(scope="class")
    @staticmethod
    def atmosphere():
        return FrozenFlowAtmosphere(
            pupil=PupilGrid(16, 0.5),
            r0_m=0.1,
            wind_speed_m_s=10.0,
            frame_time_s=1e-3,
            screen_pixels=128,
            n_subharmonics=2,
            seed=4,
        )

    def test_frame_shape(self, atmosphere):
        assert atmosphere.frame(0).shape == (16, 16)

    def test_frames_stack(self, atmosphere):
        assert atmosphere.frames(5).shape == (5, 16, 16)

    def test_piston_removed(self, atmosphere):
        frame = atmosphere.frame(3)
        assert frame[atmosphere.pupil.mask].mean() == pytest.approx(0.0, abs=1e-12)

    def test_outside_mask_is_zero(self, atmosphere):
        frame = atmosphere.frame(3)
        assert np.all(frame[~atmosphere.pupil.mask] == 0.0)

    def test_shift_per_frame(self, atmosphere):
        # v T / d = 10 * 1e-3 / (0.5/16) = 0.01 / 0.03125 = 0.32 samples
        assert atmosphere.shift_per_frame_pix == pytest.approx(0.32)

    def test_consecutive_frames_are_correlated(self, atmosphere):
        a = atmosphere.frame(10)[atmosphere.pupil.mask]
        b = atmosphere.frame(11)[atmosphere.pupil.mask]
        assert np.corrcoef(a, b)[0, 1] > 0.9

    def test_distant_frames_are_less_correlated(self, atmosphere):
        a = atmosphere.frame(0)[atmosphere.pupil.mask]
        near = atmosphere.frame(2)[atmosphere.pupil.mask]
        far = atmosphere.frame(60)[atmosphere.pupil.mask]
        assert np.corrcoef(a, near)[0, 1] > np.corrcoef(a, far)[0, 1]

    def test_still_air_repeats_the_same_frame(self):
        still = FrozenFlowAtmosphere(
            pupil=PupilGrid(16, 0.5),
            r0_m=0.1,
            wind_speed_m_s=0.0,
            screen_pixels=64,
            n_subharmonics=0,
            seed=1,
        )
        assert np.array_equal(still.frame(0), still.frame(50))
        assert np.isinf(still.repeat_period_frames)
        assert np.isinf(still.max_frames)

    def test_max_frames_guard(self, atmosphere):
        limit = int(atmosphere.max_frames)
        atmosphere.frame(limit - 1)
        with pytest.raises(ValueError, match="wrap"):
            atmosphere.frame(limit)

    def test_no_guard_without_subharmonics(self):
        periodic = FrozenFlowAtmosphere(
            pupil=PupilGrid(16, 0.5),
            r0_m=0.1,
            screen_pixels=64,
            n_subharmonics=0,
            seed=1,
        )
        assert np.isinf(periodic.max_frames)
        assert periodic.frame(10_000).shape == (16, 16)

    def test_screen_too_small(self):
        with pytest.raises(ValueError, match="screen_pixels"):
            FrozenFlowAtmosphere(pupil=PupilGrid(32, 0.5), r0_m=0.1, screen_pixels=32)

    def test_bad_r0(self):
        with pytest.raises(ValueError, match="r0_m"):
            FrozenFlowAtmosphere(pupil=PupilGrid(16, 0.5), r0_m=0.0, screen_pixels=64)

    def test_bad_wind(self):
        with pytest.raises(ValueError, match="wind_speed_m_s"):
            FrozenFlowAtmosphere(
                pupil=PupilGrid(16, 0.5), r0_m=0.1, wind_speed_m_s=-1.0, screen_pixels=64
            )

    def test_bad_frame_time(self):
        with pytest.raises(ValueError, match="frame_time_s"):
            FrozenFlowAtmosphere(
                pupil=PupilGrid(16, 0.5), r0_m=0.1, frame_time_s=0.0, screen_pixels=64
            )

    def test_non_integer_screen_pixels(self):
        with pytest.raises(ValueError, match="screen_pixels"):
            FrozenFlowAtmosphere(pupil=PupilGrid(16, 0.5), r0_m=0.1, screen_pixels=64.5)

    def test_negative_frame_index(self, atmosphere):
        with pytest.raises(ValueError, match="frame index"):
            atmosphere.frame(-1)

    def test_bad_n_frames(self, atmosphere):
        with pytest.raises(ValueError, match="n_frames"):
            atmosphere.frames(0)

    def test_variance_is_positive(self, atmosphere):
        assert variance(atmosphere.frame(0), atmosphere.pupil.mask) > 0.0
