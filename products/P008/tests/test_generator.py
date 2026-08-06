"""Tests for the synthetic spot generator: known answers, validation, reproducibility."""

import numpy as np
import pytest

from centroidnet import generate_spots, snr_estimate, spot_image


class TestSpotImage:
    def test_flux_conservation_centered_spot(self):
        # Known answer: erf-integrated Gaussian with sigma=1.5 px at the centre
        # of a 16x16 window captures essentially all of S = 1000 e-
        # (window half-width 8 px = 5.3 sigma; Gaussian tail beyond 5.3 sigma
        # is < 1e-6 of the flux per axis).
        img = spot_image(0.0, 0.0, grid_size=16, sigma=1.5, signal=1000.0)
        assert img.shape == (16, 16)
        assert img.sum() == pytest.approx(1000.0, rel=1e-6)

    def test_peak_pixel_at_true_centre(self):
        # Spot at (x0, y0) = (+3, -2): brightest pixel must be at
        # column = centre + 3 = 7.5 + 3 -> col 10 or 11; with x0 exactly on
        # a half-integer grid offset use integer offset instead.
        img = spot_image(3.0, -2.0, grid_size=16, sigma=1.0, signal=1000.0)
        row, col = np.unravel_index(np.argmax(img), img.shape)
        # centre index (N-1)/2 = 7.5 -> nearest pixels to x=+3 are cols 10/11,
        # to y=-2 rows 5/6 (symmetric split across the half-integer centre).
        assert col in (10, 11)
        assert row in (5, 6)

    def test_point_sampled_mode_close_to_integrated(self):
        # For sigma >= 1.5 px point sampling approximates the pixel integral.
        a = spot_image(0.3, -0.4, sigma=2.0, pixelated=True)
        b = spot_image(0.3, -0.4, sigma=2.0, pixelated=False)
        assert np.allclose(a, b, rtol=0.02, atol=0.02 * a.max())

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError):
            spot_image(0, 0, sigma=0.0)  # sigma must be > 0
        with pytest.raises(ValueError):
            spot_image(0, 0, sigma=-1.0)
        with pytest.raises(ValueError):
            spot_image(0, 0, signal=-10.0)  # negative counts
        with pytest.raises(ValueError):
            spot_image(0, 0, grid_size=3)
        with pytest.raises(TypeError):
            spot_image(0, 0, grid_size=16.5)


class TestGenerateSpots:
    def test_shapes_and_offset_range(self):
        imgs, truths = generate_spots(20, grid_size=16, offset_range=1.5, seed=1)
        assert imgs.shape == (20, 16, 16)
        assert truths.shape == (20, 2)
        assert np.all(np.abs(truths) <= 1.5)

    def test_reproducibility_fixed_seed(self):
        a_img, a_tr = generate_spots(10, seed=1234)
        b_img, b_tr = generate_spots(10, seed=1234)
        assert np.array_equal(a_img, b_img)
        assert np.array_equal(a_tr, b_tr)

    def test_different_seeds_differ(self):
        a_img, _ = generate_spots(5, seed=1)
        b_img, _ = generate_spots(5, seed=2)
        assert not np.array_equal(a_img, b_img)

    def test_noise_free_mode_is_deterministic_clean(self):
        offsets = np.array([[0.5, -0.5]])
        imgs, _ = generate_spots(
            offsets=offsets, background=0.0, read_noise=0.0, shot_noise=False, seed=7
        )
        clean = spot_image(0.5, -0.5)
        assert np.allclose(imgs[0], clean)

    def test_explicit_offsets_used(self):
        offsets = np.array([[1.0, 2.0], [-1.0, 0.0]])
        _, truths = generate_spots(offsets=offsets, seed=0)
        assert np.array_equal(truths, offsets)

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError):
            generate_spots(0, seed=1)  # n_spots must be >= 1
        with pytest.raises(ValueError):
            generate_spots(5, background=-1.0, seed=1)  # negative counts
        with pytest.raises(ValueError):
            generate_spots(5, read_noise=-2.0, seed=1)
        with pytest.raises(ValueError):
            generate_spots(offsets=np.zeros((3, 4)), seed=1)  # wrong shape


class TestSnrEstimate:
    def test_known_answer_hand_calculated(self):
        # S=1000, B=0, R=0 -> SNR = 1000/sqrt(1000) = sqrt(1000) = 31.6228
        assert snr_estimate(1000.0, 0.0, 0.0) == pytest.approx(31.6228, abs=1e-3)
        # S=1000, B=0.5, R=2, N=16 -> var = 1000 + 256*(0.5+4) = 2152
        # SNR = 1000/sqrt(2152) = 21.5565
        assert snr_estimate(1000.0, 0.5, 2.0, 16) == pytest.approx(21.5565, abs=1e-3)

    def test_zero_signal(self):
        assert snr_estimate(0.0, 1.0, 1.0) == 0.0

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            snr_estimate(-1.0, 0.0, 0.0)
