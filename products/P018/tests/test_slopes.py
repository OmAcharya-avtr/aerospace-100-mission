"""Classical slope extraction: known answers, symmetry properties, noise model."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from shacksim import (
    LensletArray,
    cog_displacement,
    cog_noise_sigma,
    cog_slopes,
    correlation_displacement,
    correlation_slopes,
    defocus_slopes,
    generate_subaperture_dataset,
    random_slopes,
    reference_template,
    simulate_frame,
    slope_rms,
    subaperture_spot,
    tilt_slopes,
)

ARRAY = LensletArray()


def _marginal_width(marginal: np.ndarray) -> float:
    """RMS width [px] of a 1-D marginal profile — an unambiguous size metric."""
    x = np.arange(len(marginal)) - (len(marginal) - 1) / 2.0
    w = marginal / marginal.sum()
    mu = float((w * x).sum())
    return float(np.sqrt((w * (x - mu) ** 2).sum()))


class TestKnownTilt:
    """The primary answer test: a global tilt must give the predicted uniform slopes."""

    def test_hand_calculated_displacement(self):
        # W = gx X with gx = 1.000e-3 rad. Spot displacement
        #   d = f * gx / p = 50e-3 m * 1e-3 / 31.25e-6 m = 1.600 px exactly.
        # The CoG of a noise-free symmetric spot returns that displacement.
        stamp = subaperture_spot(ARRAY, 1.6, 0.0, 1.0e5)
        assert cog_displacement(stamp)[0, 0] == pytest.approx(1.6, abs=1e-6)

    @pytest.mark.parametrize(
        "gx, gy", [(0.0, 0.0), (1e-3, 0.0), (0.0, -1.5e-3), (2e-3, 1e-3), (-2.5e-3, -2.5e-3)]
    )
    def test_uniform_slope_recovered_everywhere(self, gx, gy):
        truth = tilt_slopes(ARRAY, gx, gy)
        frame = simulate_frame(ARRAY, truth, photons=5.0e4, shot_noise=False, seed=0)
        est = cog_slopes(frame, ARRAY)
        assert est.shape == (ARRAY.n_valid, 2)
        # every subaperture reports the same slope, to 1e-8 rad = 1.6e-5 px
        assert np.abs(est - truth).max() < 1e-8
        assert est.std(axis=0).max() < 1e-9

    def test_zero_wavefront_gives_zero_slopes(self):
        zero = np.zeros((ARRAY.n_valid, 2))
        frame = simulate_frame(ARRAY, zero, photons=5.0e4, shot_noise=False, seed=0)
        assert np.abs(cog_slopes(frame, ARRAY)).max() < 1e-12
        assert np.abs(correlation_slopes(frame, ARRAY)).max() < 1e-12

    def test_zero_wavefront_with_noise_is_unbiased(self):
        zero = np.zeros((ARRAY.n_valid, 2))
        frame = simulate_frame(
            ARRAY, zero, photons=5000.0, background=1.0, read_noise=3.0, seed=11
        )
        est = cog_slopes(frame, ARRAY, threshold=10.0)
        # mean over 52 subapertures; per-subaperture sigma is ~1.2e-5 rad
        assert np.abs(est.mean(axis=0)).max() < 1e-5

    def test_correlation_recovers_tilt(self):
        truth = tilt_slopes(ARRAY, 1.25e-3, -0.625e-3)
        frame = simulate_frame(ARRAY, truth, photons=5.0e4, shot_noise=False, seed=0)
        est = correlation_slopes(frame, ARRAY)
        err_px = np.abs(ARRAY.slope_to_displacement(est - truth)).max()
        # 3-point parabolic interpolation carries a documented S-curve bias
        assert err_px < 0.05

    def test_defocus_gives_a_radial_slope_fan(self):
        slopes = defocus_slopes(ARRAY, 1.0)
        centres = ARRAY.valid_centres()
        # slope must be parallel to the radius vector and grow with radius
        cross = slopes[:, 0] * centres[:, 1] - slopes[:, 1] * centres[:, 0]
        assert np.abs(cross).max() < 1e-20
        assert np.corrcoef(np.hypot(*slopes.T), np.hypot(*centres.T))[0, 1] > 0.999


class TestCogProperties:
    @settings(max_examples=50, deadline=None)
    @given(st.floats(-5.0, 5.0, allow_nan=False, allow_infinity=False))
    def test_x_mirror_negates_dx(self, d: float):
        """Flipping the stamp left-right must negate dx and leave dy at zero."""
        stamp = subaperture_spot(ARRAY, d, 0.0, 1000.0)
        a = cog_displacement(stamp)[0]
        b = cog_displacement(np.flip(stamp, axis=1))[0]
        assert b[0] == pytest.approx(-a[0], abs=1e-9)
        assert b[1] == pytest.approx(a[1], abs=1e-9)

    @settings(max_examples=50, deadline=None)
    @given(
        st.floats(-4.0, 4.0, allow_nan=False, allow_infinity=False),
        st.floats(-4.0, 4.0, allow_nan=False, allow_infinity=False),
    )
    def test_rotation_by_180_negates_both_components(self, dx: float, dy: float):
        stamp = subaperture_spot(ARRAY, dx, dy, 1000.0)
        a = cog_displacement(stamp)[0]
        b = cog_displacement(np.flip(np.flip(stamp, axis=0), axis=1))[0]
        assert b == pytest.approx(-a, abs=1e-9)

    @settings(max_examples=50, deadline=None)
    @given(
        st.floats(-4.0, 4.0, allow_nan=False, allow_infinity=False),
        st.floats(0.01, 1000.0, allow_nan=False, allow_infinity=False),
    )
    def test_gain_invariance(self, d: float, gain: float):
        """Scaling every pixel by g > 0 must not move the unthresholded centroid."""
        stamp = subaperture_spot(ARRAY, d, 0.0, 1000.0)
        a = cog_displacement(stamp)[0]
        b = cog_displacement(stamp * gain)[0]
        assert b == pytest.approx(a, abs=1e-8)

    @settings(max_examples=40, deadline=None)
    @given(
        st.floats(-4.0, 4.0, allow_nan=False, allow_infinity=False),
        st.floats(-4.0, 4.0, allow_nan=False, allow_infinity=False),
    )
    def test_transpose_swaps_components(self, dx: float, dy: float):
        stamp = subaperture_spot(ARRAY, dx, dy, 1000.0)
        a = cog_displacement(stamp)[0]
        b = cog_displacement(stamp.T)[0]
        assert b[0] == pytest.approx(a[1], abs=1e-9)
        assert b[1] == pytest.approx(a[0], abs=1e-9)

    def test_empty_stamp_falls_back_to_the_centre(self):
        assert np.all(cog_displacement(np.zeros((16, 16))) == 0.0)

    def test_threshold_removes_a_uniform_pedestal(self):
        stamp = subaperture_spot(ARRAY, 3.0, 0.0, 1000.0) + 4.0
        biased = cog_displacement(stamp)[0, 0]
        corrected = cog_displacement(stamp, threshold=4.0)[0, 0]
        assert abs(corrected - 3.0) < abs(biased - 3.0)
        assert corrected == pytest.approx(3.0, abs=1e-3)


class TestBackgroundBias:
    def test_matches_the_analytic_shrinkage(self):
        # Hand calculation: S = 1000 e-, B = 2 e-/px, p^2 = 256 pixels.
        #   kappa = S / (S + B p^2) = 1000 / (1000 + 512) = 0.661376...
        #   d = 4 px  ->  x_hat = 2.645503 px
        stamp = subaperture_spot(ARRAY, 4.0, 0.0, 1000.0) + 2.0
        kappa = 1000.0 / (1000.0 + 2.0 * 256.0)
        assert kappa == pytest.approx(0.6613757, rel=1e-6)
        assert cog_displacement(stamp)[0, 0] == pytest.approx(kappa * 4.0, abs=1e-3)

    @pytest.mark.parametrize("bkg", [0.5, 2.0, 10.0, 50.0])
    def test_shrinkage_over_a_range_of_backgrounds(self, bkg):
        signal, d = 5000.0, 3.0
        stamp = subaperture_spot(ARRAY, d, 0.0, signal) + bkg
        kappa = signal / (signal + bkg * ARRAY.pixels_per_sub**2)
        assert cog_displacement(stamp)[0, 0] == pytest.approx(kappa * d, abs=1e-3)

    def test_bias_is_toward_zero(self):
        stamp = subaperture_spot(ARRAY, 4.0, 0.0, 1000.0) + 10.0
        assert 0.0 < cog_displacement(stamp)[0, 0] < 4.0


class TestNoisePropagation:
    def test_photon_limit_scales_as_one_over_sqrt_n(self):
        s1 = cog_noise_sigma(ARRAY, 1000.0)
        s2 = cog_noise_sigma(ARRAY, 4000.0)
        assert s1 / s2 == pytest.approx(2.0, rel=1e-9)

    def test_photon_limit_matches_sigma_over_sqrt_n(self):
        # Pure photon term: sqrt(M2)/sqrt(N) px, with M2 = sigma^2 + 1/12
        # (Sheppard's correction for pixel binning).
        m2 = ARRAY.spot_sigma_px**2 + 1.0 / 12.0
        pred = np.sqrt(m2 / 1000.0) * ARRAY.pixel_angle
        assert cog_noise_sigma(ARRAY, 1000.0) == pytest.approx(pred, rel=2e-3)

    def test_read_noise_term_dominates_at_low_flux(self):
        photon_only = cog_noise_sigma(ARRAY, 100.0)
        with_read = cog_noise_sigma(ARRAY, 100.0, read_noise=3.0)
        assert with_read > 10 * photon_only

    def test_displacement_increases_the_lever_arm(self):
        centred = cog_noise_sigma(ARRAY, 1000.0, 1.0, 3.0, displacement_px=0.0)
        offset = cog_noise_sigma(ARRAY, 1000.0, 1.0, 3.0, displacement_px=4.0)
        assert offset > centred

    def test_matches_a_monte_carlo_measurement(self):
        """Regression against a measured Monte Carlo error (validation section 3)."""
        stamps, slopes = generate_subaperture_dataset(
            ARRAY, 1500, photons=3000.0, background=1.0, read_noise=3.0, seed=7003
        )
        d_true = ARRAY.slope_to_displacement(slopes)
        est = cog_displacement(stamps - 1.0, threshold=0.0, clip_negative=False)
        meas = float(np.std(est[:, 0] - d_true[:, 0]))
        pred = float(
            np.sqrt(
                np.mean(
                    (
                        cog_noise_sigma(
                            ARRAY, 3000.0, 1.0, 3.0, displacement_px=d_true[:, 0]
                        )
                        / ARRAY.pixel_angle
                    )
                    ** 2
                )
            )
        )
        assert meas / pred == pytest.approx(1.0, abs=0.10)

    def test_array_displacement_returns_array(self):
        out = cog_noise_sigma(ARRAY, 1000.0, displacement_px=np.array([0.0, 1.0, 2.0]))
        assert isinstance(out, np.ndarray) and out.shape == (3,)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"photons": 0.0},
            {"photons": -1.0},
            {"background": -1.0},
            {"read_noise": -1.0},
            {"axis": "z"},
            {"displacement_px": float("inf")},
        ],
    )
    def test_invalid_input(self, kwargs):
        args = {"photons": 1000.0}
        args.update(kwargs)
        with pytest.raises(ValueError):
            cog_noise_sigma(ARRAY, **args)


class TestCorrelation:
    def test_template_is_centred_and_normalized_shape(self):
        tpl = reference_template(ARRAY)
        assert tpl.shape == (ARRAY.pixels_per_sub, ARRAY.pixels_per_sub)
        assert cog_displacement(tpl)[0] == pytest.approx([0.0, 0.0], abs=1e-9)

    def test_integer_shift_recovered_exactly(self):
        stamp = subaperture_spot(ARRAY, 3.0, -2.0, 1.0e5)
        est = correlation_displacement(stamp, reference_template(ARRAY))[0]
        assert est == pytest.approx([3.0, -2.0], abs=0.02)

    def test_mean_subtraction_makes_the_estimate_background_invariant(self):
        tpl = reference_template(ARRAY)
        low = subaperture_spot(ARRAY, 3.0, 0.0, 1000.0) + 5.0
        high = subaperture_spot(ARRAY, 3.0, 0.0, 1000.0) + 500.0
        a = correlation_displacement(low, tpl, subtract_mean=True)[0, 0]
        b = correlation_displacement(high, tpl, subtract_mean=True)[0, 0]
        assert a == pytest.approx(b, abs=1e-12)
        # without mean subtraction the pedestal shifts the answer with B
        c = correlation_displacement(low, tpl, subtract_mean=False)[0, 0]
        d = correlation_displacement(high, tpl, subtract_mean=False)[0, 0]
        assert abs(c - d) > 1e-8

    def test_elongated_template_available(self):
        tpl = reference_template(ARRAY, elongation=3.0, elongation_axis="y")
        assert _marginal_width(tpl.sum(axis=1)) > 2.5 * _marginal_width(tpl.sum(axis=0))

    @pytest.mark.parametrize(
        "kwargs, exc",
        [
            ({"template": np.zeros((8, 8))}, ValueError),
            ({"template": np.ones((16, 16)) * np.nan}, ValueError),
            ({"template": np.ones((16, 16))}, ValueError),  # zero after mean removal
        ],
    )
    def test_invalid_template(self, kwargs, exc):
        stamp = subaperture_spot(ARRAY, 0.0, 0.0, 100.0)
        with pytest.raises(exc):
            correlation_displacement(stamp, **kwargs)

    def test_reference_template_validation(self):
        with pytest.raises(ValueError):
            reference_template(ARRAY, elongation=0.5)
        with pytest.raises(ValueError):
            reference_template(ARRAY, elongation_axis="z")


class TestInputValidation:
    @pytest.mark.parametrize(
        "stamps",
        [np.zeros((2, 2)), np.zeros((4, 5)), np.zeros((2, 3, 4)), np.full((16, 16), np.nan)],
    )
    def test_bad_stamps_rejected(self, stamps):
        with pytest.raises(ValueError):
            cog_displacement(stamps)

    @pytest.mark.parametrize("threshold", [-1.0, float("nan")])
    def test_bad_threshold_rejected(self, threshold):
        with pytest.raises(ValueError):
            cog_displacement(np.ones((16, 16)), threshold=threshold)

    def test_slope_rms_shape_check(self):
        with pytest.raises(ValueError):
            slope_rms(np.zeros((5, 3)))
        assert slope_rms(np.array([[3.0, 4.0]])) == pytest.approx(np.sqrt(12.5))

    def test_wavefront_helpers_validate(self):
        with pytest.raises(ValueError):
            tilt_slopes(ARRAY, float("nan"))
        with pytest.raises(ValueError):
            defocus_slopes(ARRAY, float("inf"))
        with pytest.raises(ValueError):
            random_slopes(ARRAY, -1.0)

    def test_random_slopes_reproducible(self):
        a = random_slopes(ARRAY, 1e-4, seed=5)
        b = random_slopes(ARRAY, 1e-4, seed=5)
        assert np.array_equal(a, b)
        assert a.shape == (ARRAY.n_valid, 2)
