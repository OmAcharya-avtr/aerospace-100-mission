"""Tests for classical centroid baselines: known answers, symmetry (Hypothesis),
input validation."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from scipy.special import erf

from centroidnet import cog_centroid, quadcell_centroid, spot_image

# Non-negative finite 8x8 images with at least some flux (Hypothesis strategy).
_images = arrays(
    dtype=np.float64,
    shape=(8, 8),
    elements=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
).filter(lambda a: a.sum() > 1e-9)


class TestCogKnownAnswers:
    def test_single_bright_pixel(self):
        # Hand calculation: 4x4 image, all zero except pixel [row=1, col=2]=5.
        # Centre index = (4-1)/2 = 1.5, so x = 2 - 1.5 = +0.5, y = 1 - 1.5 = -0.5.
        img = np.zeros((4, 4))
        img[1, 2] = 5.0
        x, y = cog_centroid(img)
        assert x == pytest.approx(0.5)
        assert y == pytest.approx(-0.5)

    def test_two_equal_pixels_midpoint(self):
        # Pixels [0,0] and [3,3] equal -> centroid at array centre (0, 0).
        img = np.zeros((4, 4))
        img[0, 0] = img[3, 3] = 2.0
        x, y = cog_centroid(img)
        assert x == pytest.approx(0.0)
        assert y == pytest.approx(0.0)

    def test_noise_free_gaussian_recovery(self):
        # Noise-free known-answer: CoG of an integrated Gaussian spot recovers
        # the true centre to < 1e-3 px (limited only by window truncation).
        for x0, y0 in [(0.0, 0.0), (0.7, -1.2), (-1.5, 0.4)]:
            img = spot_image(x0, y0, grid_size=16, sigma=1.5, signal=1000.0)
            x, y = cog_centroid(img)
            assert abs(x - x0) < 1e-3
            assert abs(y - y0) < 1e-3

    def test_threshold_removes_uniform_background(self):
        img = spot_image(1.0, -0.5, sigma=1.5, signal=1000.0) + 5.0
        x, y = cog_centroid(img, threshold=5.0)
        assert x == pytest.approx(1.0, abs=1e-3)
        assert y == pytest.approx(-0.5, abs=1e-3)


class TestQuadcellKnownAnswers:
    def test_centered_spot_zero_output(self):
        img = spot_image(0.0, 0.0, sigma=1.5)
        x, y = quadcell_centroid(img)
        assert x == pytest.approx(0.0, abs=1e-12)
        assert y == pytest.approx(0.0, abs=1e-12)

    def test_erf_response_gaussian_spot(self):
        # Theory (Tyler & Fried 1982): (I_R - I_L)/I_tot = erf(d/(sigma*sqrt(2))).
        sigma = 1.5
        for d in [0.2, 0.8, 1.5]:
            img = spot_image(d, 0.0, grid_size=16, sigma=sigma, signal=1000.0)
            x, y = quadcell_centroid(img)
            assert x == pytest.approx(erf(d / (sigma * np.sqrt(2.0))), abs=1e-4)
            assert y == pytest.approx(0.0, abs=1e-6)

    def test_scale_calibration_small_offset(self):
        # scale = sigma*sqrt(pi/2) linearizes the small-offset response.
        sigma = 1.5
        d = 0.1  # |d| << sigma
        img = spot_image(d, 0.0, sigma=sigma)
        x, _ = quadcell_centroid(img, scale=sigma * np.sqrt(np.pi / 2.0))
        assert x == pytest.approx(d, rel=0.01)


class TestSymmetryProperties:
    @settings(max_examples=50, deadline=None)
    @given(img=_images)
    def test_cog_mirror_x(self, img):
        # Mirroring columns negates x and preserves y.
        x, y = cog_centroid(img)
        xm, ym = cog_centroid(img[:, ::-1].copy())
        assert xm == pytest.approx(-x, abs=1e-9)
        assert ym == pytest.approx(y, abs=1e-9)

    @settings(max_examples=50, deadline=None)
    @given(img=_images)
    def test_cog_mirror_y(self, img):
        x, y = cog_centroid(img)
        xm, ym = cog_centroid(img[::-1, :].copy())
        assert xm == pytest.approx(x, abs=1e-9)
        assert ym == pytest.approx(-y, abs=1e-9)

    @settings(max_examples=50, deadline=None)
    @given(img=_images)
    def test_quadcell_mirror_x(self, img):
        x, y = quadcell_centroid(img)
        xm, ym = quadcell_centroid(img[:, ::-1].copy())
        assert xm == pytest.approx(-x, abs=1e-9)
        assert ym == pytest.approx(y, abs=1e-9)

    @settings(max_examples=50, deadline=None)
    @given(img=_images, gain=st.floats(min_value=1e-3, max_value=1e3))
    def test_cog_gain_invariance(self, img, gain):
        # CoG is invariant to a positive multiplicative gain.
        x, y = cog_centroid(img)
        xg, yg = cog_centroid(img * gain)
        assert xg == pytest.approx(x, abs=1e-6)
        assert yg == pytest.approx(y, abs=1e-6)


class TestInputValidation:
    def test_wrong_dimensionality(self):
        with pytest.raises(ValueError):
            cog_centroid(np.ones(16))
        with pytest.raises(ValueError):
            cog_centroid(np.ones((4, 4, 4)))
        with pytest.raises(ValueError):
            quadcell_centroid(np.ones(16))

    def test_all_zero_image(self):
        with pytest.raises(ValueError):
            cog_centroid(np.zeros((8, 8)))
        with pytest.raises(ValueError):
            quadcell_centroid(np.zeros((8, 8)))

    def test_all_negative_image(self):
        # Negative counts clip to zero -> no flux -> ValueError.
        with pytest.raises(ValueError):
            cog_centroid(-np.ones((8, 8)))

    def test_nan_rejected(self):
        img = np.ones((8, 8))
        img[0, 0] = np.nan
        with pytest.raises(ValueError):
            cog_centroid(img)

    def test_quadcell_odd_dimensions(self):
        with pytest.raises(ValueError):
            quadcell_centroid(np.ones((7, 8)))

    def test_threshold_above_peak(self):
        img = spot_image(0.0, 0.0, signal=100.0)
        with pytest.raises(ValueError):
            cog_centroid(img, threshold=1e9)
