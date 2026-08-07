"""Spot formation and the detector noise chain."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from shacksim import (
    LensletArray,
    extract_subapertures,
    generate_subaperture_dataset,
    simulate_frame,
    subaperture_spot,
    tilt_slopes,
)

ARRAY = LensletArray()


def _marginal_width(marginal: np.ndarray) -> float:
    """RMS width [px] of a 1-D marginal profile."""
    x = np.arange(len(marginal)) - (len(marginal) - 1) / 2.0
    w = marginal / marginal.sum()
    mu = float((w * x).sum())
    return float(np.sqrt((w * (x - mu) ** 2).sum()))


class TestSubapertureSpot:
    def test_flux_is_conserved_for_a_centred_spot(self):
        # sigma = 0.885 px in a 16 px window: >8 sigma to the edge, so the
        # truncated flux is far below 1e-9 of the total.
        stamp = subaperture_spot(ARRAY, 0.0, 0.0, 1000.0)
        assert stamp.sum() == pytest.approx(1000.0, rel=1e-9)

    def test_peak_moves_with_displacement(self):
        stamp = subaperture_spot(ARRAY, 3.0, -2.0, 1000.0)
        row, col = np.unravel_index(int(np.argmax(stamp)), stamp.shape)
        centre = (ARRAY.pixels_per_sub - 1) / 2.0
        assert col - centre == pytest.approx(3.0, abs=0.5)
        assert row - centre == pytest.approx(-2.0, abs=0.5)

    def test_shape_and_dtype(self):
        stamp = subaperture_spot(ARRAY, 0.0, 0.0)
        assert stamp.shape == (ARRAY.pixels_per_sub, ARRAY.pixels_per_sub)
        assert stamp.dtype == np.float64

    def test_zero_photons_gives_zero_image(self):
        assert np.all(subaperture_spot(ARRAY, 1.0, 1.0, 0.0) == 0.0)

    def test_elongation_widens_only_the_chosen_axis(self):
        round_spot = subaperture_spot(ARRAY, 0.0, 0.0, 1.0)
        elong = subaperture_spot(ARRAY, 0.0, 0.0, 1.0, sigma_x_px=3 * ARRAY.spot_sigma_px)
        assert elong.max() < round_spot.max()
        # Hand calculation: the binned RMS width of a Gaussian of width s is
        # sqrt(s^2 + 1/12) (Sheppard's correction). Round: s = 0.8850 px ->
        # 0.9309 px. Elongated 3x: s = 2.6549 px -> 2.6706 px. The measured
        # value comes out ~1 % low because the broad spot is truncated by the
        # 16 px window.
        sig = ARRAY.spot_sigma_px
        assert _marginal_width(round_spot.sum(axis=0)) == pytest.approx(
            np.sqrt(sig**2 + 1 / 12), rel=1e-6
        )
        assert _marginal_width(elong.sum(axis=0)) == pytest.approx(
            np.sqrt((3 * sig) ** 2 + 1 / 12), rel=0.02
        )
        assert _marginal_width(elong.sum(axis=1)) == pytest.approx(
            _marginal_width(round_spot.sum(axis=1)), rel=1e-9
        )

    def test_flux_lost_off_the_edge(self):
        far = subaperture_spot(ARRAY, 7.9, 0.0, 1000.0)
        assert far.sum() < 1000.0
        assert far.sum() > 400.0

    @settings(max_examples=40, deadline=None)
    @given(st.floats(-5.0, 5.0, allow_nan=False, allow_infinity=False))
    def test_mirror_symmetry_in_x(self, d: float):
        """A spot at +d is the x-mirror of a spot at -d (property test)."""
        a = subaperture_spot(ARRAY, d, 0.0, 1000.0)
        b = subaperture_spot(ARRAY, -d, 0.0, 1000.0)
        assert np.allclose(a, np.flip(b, axis=1), atol=1e-12)

    @settings(max_examples=40, deadline=None)
    @given(
        st.floats(-4.0, 4.0, allow_nan=False, allow_infinity=False),
        st.floats(-4.0, 4.0, allow_nan=False, allow_infinity=False),
    )
    def test_transpose_symmetry(self, dx: float, dy: float):
        """Swapping the axes of a round spot transposes the image (property test)."""
        a = subaperture_spot(ARRAY, dx, dy, 500.0)
        b = subaperture_spot(ARRAY, dy, dx, 500.0)
        assert np.allclose(a, b.T, atol=1e-12)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"photons": -1.0},
            {"sigma_x_px": 0.0},
            {"sigma_y_px": -1.0},
            {"dx_px": float("nan")},
        ],
    )
    def test_invalid_input(self, kwargs):
        args = {"dx_px": 0.0, "dy_px": 0.0}
        args.update(kwargs)
        with pytest.raises(ValueError):
            subaperture_spot(ARRAY, **args)


class TestSimulateFrame:
    def test_shape_and_subaperture_layout(self):
        slopes = tilt_slopes(ARRAY, 0.0, 0.0)
        frame = simulate_frame(ARRAY, slopes, photons=1000.0, shot_noise=False)
        assert frame.shape == (ARRAY.image_size, ARRAY.image_size)
        # The four corner subapertures are outside the pupil -> exactly zero.
        rs, cs = ARRAY.subaperture_slice(0, 0)
        assert np.all(frame[rs, cs] == 0.0)

    def test_total_flux_matches_valid_subaperture_count(self):
        slopes = tilt_slopes(ARRAY, 0.0, 0.0)
        frame = simulate_frame(ARRAY, slopes, photons=1000.0, shot_noise=False)
        assert frame.sum() == pytest.approx(1000.0 * ARRAY.n_valid, rel=1e-9)

    def test_background_added_everywhere(self):
        slopes = tilt_slopes(ARRAY, 0.0, 0.0)
        frame = simulate_frame(ARRAY, slopes, photons=0.0, background=5.0, shot_noise=False)
        assert np.all(frame == 5.0)

    def test_seed_gives_bitwise_reproducible_frames(self):
        slopes = tilt_slopes(ARRAY, 1e-3, 0.0)
        kw = {"photons": 500.0, "background": 2.0, "read_noise": 3.0}
        a = simulate_frame(ARRAY, slopes, seed=7, **kw)
        b = simulate_frame(ARRAY, slopes, seed=7, **kw)
        c = simulate_frame(ARRAY, slopes, seed=8, **kw)
        assert np.array_equal(a, b)
        assert not np.array_equal(a, c)

    def test_noise_increases_variance(self):
        slopes = tilt_slopes(ARRAY, 0.0, 0.0)
        clean = simulate_frame(ARRAY, slopes, photons=1000.0, shot_noise=False)
        noisy = simulate_frame(ARRAY, slopes, photons=1000.0, read_noise=5.0, seed=1)
        assert noisy.std() > clean.std()

    def test_per_subaperture_flux_array(self):
        flux = np.linspace(100.0, 1000.0, ARRAY.n_valid)
        frame = simulate_frame(
            ARRAY, tilt_slopes(ARRAY, 0.0, 0.0), photons=flux, shot_noise=False
        )
        assert frame.sum() == pytest.approx(flux.sum(), rel=1e-9)

    @pytest.mark.parametrize(
        "kwargs, exc",
        [
            ({"slopes": np.zeros((3, 2))}, ValueError),
            ({"slopes": np.zeros((52, 3))}, ValueError),
            ({"background": -1.0}, ValueError),
            ({"read_noise": -1.0}, ValueError),
            ({"elongation": 0.5}, ValueError),
            ({"elongation_axis": "z"}, ValueError),
            ({"photons": -5.0}, ValueError),
            ({"photons": np.zeros(3)}, ValueError),
            ({"seed": "abc"}, TypeError),
        ],
    )
    def test_invalid_input(self, kwargs, exc):
        args = {"slopes": tilt_slopes(ARRAY, 0.0, 0.0), "photons": 100.0}
        args.update(kwargs)
        with pytest.raises(exc):
            simulate_frame(ARRAY, **args)

    def test_nonfinite_slopes_rejected(self):
        slopes = tilt_slopes(ARRAY, 0.0, 0.0).copy()
        slopes[0, 0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            simulate_frame(ARRAY, slopes)


class TestExtractSubapertures:
    def test_round_trip_against_simulate_frame(self):
        slopes = tilt_slopes(ARRAY, 1e-3, -1e-3)
        frame = simulate_frame(ARRAY, slopes, photons=1000.0, shot_noise=False)
        stamps = extract_subapertures(frame, ARRAY)
        assert stamps.shape == (ARRAY.n_valid, ARRAY.pixels_per_sub, ARRAY.pixels_per_sub)
        # every illuminated stamp carries the same flux for a uniform tilt
        assert np.allclose(stamps.sum(axis=(1, 2)), stamps[0].sum(), rtol=1e-9)

    def test_wrong_shape_rejected(self):
        with pytest.raises(ValueError, match="shape"):
            extract_subapertures(np.zeros((10, 10)), ARRAY)


class TestDataset:
    def test_shapes_and_label_range(self):
        stamps, slopes = generate_subaperture_dataset(ARRAY, 50, photons=500.0, seed=1)
        assert stamps.shape == (50, ARRAY.pixels_per_sub, ARRAY.pixels_per_sub)
        assert slopes.shape == (50, 2)
        assert np.abs(slopes).max() <= 0.6 * ARRAY.max_slope

    def test_reproducible(self):
        a = generate_subaperture_dataset(ARRAY, 20, photons=(100.0, 1000.0), seed=3)
        b = generate_subaperture_dataset(ARRAY, 20, photons=(100.0, 1000.0), seed=3)
        assert np.array_equal(a[0], b[0])
        assert np.array_equal(a[1], b[1])

    def test_flux_range_is_respected(self):
        stamps, _ = generate_subaperture_dataset(
            ARRAY, 200, photons=(100.0, 200.0), shot_noise=False, seed=5
        )
        totals = stamps.sum(axis=(1, 2))
        assert totals.min() >= 99.0
        assert totals.max() <= 201.0

    @pytest.mark.parametrize(
        "kwargs, exc",
        [
            ({"n_samples": 0}, ValueError),
            ({"n_samples": 1.5}, TypeError),
            ({"slope_fraction": 0.0}, ValueError),
            ({"slope_fraction": 2.0}, ValueError),
            ({"photons": (0.0, 10.0)}, ValueError),
            ({"photons": (100.0, 10.0)}, ValueError),
            ({"elongation": (0.5, 2.0)}, ValueError),
            ({"elongation_axis": "q"}, ValueError),
            ({"background": -1.0}, ValueError),
        ],
    )
    def test_invalid_input(self, kwargs, exc):
        args = {"n_samples": 10, "photons": 100.0}
        args.update(kwargs)
        with pytest.raises(exc):
            generate_subaperture_dataset(ARRAY, **args)
