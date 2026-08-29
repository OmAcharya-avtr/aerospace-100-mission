"""Tests for waveforge.pupil."""

from __future__ import annotations

import numpy as np
import pytest

from waveforge.pupil import (
    PupilGrid,
    piston_removed,
    rms,
    strehl_from_field,
    variance,
)


class TestPupilGridConstruction:
    def test_defaults(self):
        grid = PupilGrid(16, 1.0)
        assert grid.n_pix == 16
        assert grid.diameter_m == 1.0
        assert grid.obscuration == 0.0

    def test_radius_is_half_diameter(self):
        assert PupilGrid(8, 0.5).radius_m == 0.25

    def test_sample_spacing(self):
        # d = D / n = 0.5 / 8 = 0.0625 m exactly
        assert PupilGrid(8, 0.5).sample_spacing_m == pytest.approx(0.0625)

    @pytest.mark.parametrize("n_pix", [1, 0, -4])
    def test_rejects_small_n_pix(self, n_pix):
        with pytest.raises(ValueError, match="n_pix"):
            PupilGrid(n_pix, 1.0)

    def test_rejects_non_integer_n_pix(self):
        with pytest.raises(ValueError, match="n_pix"):
            PupilGrid(8.5, 1.0)

    @pytest.mark.parametrize("diameter", [0.0, -1.0, float("nan"), float("inf")])
    def test_rejects_bad_diameter(self, diameter):
        with pytest.raises(ValueError, match="diameter_m"):
            PupilGrid(8, diameter)

    @pytest.mark.parametrize("obscuration", [-0.1, 1.0, 1.5])
    def test_rejects_bad_obscuration(self, obscuration):
        with pytest.raises(ValueError, match="obscuration"):
            PupilGrid(8, 1.0, obscuration)


class TestGeometry:
    def test_coords_are_symmetric_about_zero(self, small_pupil):
        x, y = small_pupil.coords_m()
        assert np.allclose(x, -x[:, ::-1])
        assert np.allclose(y, -y[::-1, :])

    def test_no_sample_exactly_at_origin(self, small_pupil):
        x, y = small_pupil.coords_m()
        assert np.min(np.hypot(x, y)) > 0.0

    def test_normalised_coords_reach_almost_one(self, small_pupil):
        xn, _ = small_pupil.normalised_coords()
        # outermost sample centre is at (n/2 - 0.5) * d, i.e. 1 - 1/n in rho
        assert xn.max() == pytest.approx(1.0 - 1.0 / small_pupil.n_pix)

    def test_polar_matches_cartesian(self, small_pupil):
        xn, yn = small_pupil.normalised_coords()
        rho, theta = small_pupil.polar()
        assert np.allclose(rho * np.cos(theta), xn)
        assert np.allclose(rho * np.sin(theta), yn)

    def test_mask_is_circular(self, small_pupil):
        rho, _ = small_pupil.polar()
        assert np.array_equal(small_pupil.mask, rho <= 1.0)

    def test_mask_area_approaches_pi_over_four(self):
        grid = PupilGrid(128, 1.0)
        # fraction of a square filled by its inscribed circle = pi/4 = 0.7854
        fraction = grid.n_valid / grid.n_pix**2
        assert fraction == pytest.approx(np.pi / 4, abs=0.01)

    def test_area_matches_pi_r_squared(self):
        grid = PupilGrid(128, 2.0)
        assert grid.area_m2 == pytest.approx(np.pi * 1.0**2, rel=0.01)

    def test_obscuration_removes_centre(self):
        grid = PupilGrid(64, 1.0, obscuration=0.3)
        rho, _ = grid.polar()
        assert not grid.mask[rho < 0.3].any()
        assert grid.n_valid < PupilGrid(64, 1.0).n_valid

    def test_n_valid_positive(self, small_pupil):
        assert small_pupil.n_valid > 0


class TestStatistics:
    def test_piston_removed_has_zero_mean(self, small_pupil, rng):
        phase = rng.normal(size=(32, 32))
        out = piston_removed(phase, small_pupil.mask)
        assert out[small_pupil.mask].mean() == pytest.approx(0.0, abs=1e-12)

    def test_piston_removed_zeroes_outside_mask(self, small_pupil, rng):
        phase = rng.normal(size=(32, 32)) + 5.0
        out = piston_removed(phase, small_pupil.mask)
        assert np.all(out[~small_pupil.mask] == 0.0)

    def test_piston_removed_without_mask(self):
        phase = np.array([[1.0, 3.0], [5.0, 7.0]])
        assert piston_removed(phase).mean() == pytest.approx(0.0)

    def test_piston_removal_is_idempotent(self, small_pupil, rng):
        phase = rng.normal(size=(32, 32))
        once = piston_removed(phase, small_pupil.mask)
        twice = piston_removed(once, small_pupil.mask)
        assert np.allclose(once, twice)

    def test_variance_known_answer(self):
        # values 1, 2, 3, 4 -> mean 2.5, variance = (2.25+0.25+0.25+2.25)/4 = 1.25
        phase = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert variance(phase) == pytest.approx(1.25)

    def test_rms_is_sqrt_variance(self, rng):
        phase = rng.normal(size=(8, 8))
        assert rms(phase) == pytest.approx(np.sqrt(variance(phase)))

    def test_variance_ignores_unmasked_samples(self):
        phase = np.array([[1.0, 1000.0], [1.0, 1.0]])
        mask = np.array([[True, False], [True, True]])
        assert variance(phase, mask) == pytest.approx(0.0)

    def test_mask_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="mask shape"):
            variance(np.zeros((4, 4)), np.zeros((3, 3), dtype=bool))

    def test_empty_mask_raises(self):
        with pytest.raises(ValueError, match="no samples"):
            variance(np.zeros((4, 4)), np.zeros((4, 4), dtype=bool))

    def test_piston_removed_mask_shape_mismatch(self):
        # piston_removed indexes with the mask, so a wrong shape must fail loudly
        with pytest.raises(IndexError):
            piston_removed(np.zeros((4, 4)), np.zeros((5, 5), dtype=bool))


class TestStrehl:
    def test_flat_wavefront_gives_unit_strehl(self, small_pupil):
        assert strehl_from_field(np.zeros((32, 32)), small_pupil.mask) == pytest.approx(1.0)

    def test_piston_only_gives_unit_strehl(self, small_pupil):
        assert strehl_from_field(np.full((32, 32), 0.7), small_pupil.mask) == pytest.approx(1.0)

    def test_strehl_in_unit_interval(self, small_pupil, rng):
        for scale in (0.1, 1.0, 3.0):
            phase = rng.normal(scale=scale, size=(32, 32))
            value = strehl_from_field(phase, small_pupil.mask)
            assert 0.0 <= value <= 1.0

    def test_strehl_decreases_with_aberration(self, small_pupil, rng):
        base = rng.normal(size=(32, 32))
        values = [strehl_from_field(scale * base, small_pupil.mask) for scale in (0.1, 0.5, 1.0)]
        assert values[0] > values[1] > values[2]

    def test_strehl_matches_marechal_for_small_gaussian_phase(self, small_pupil, rng):
        # For zero-mean Gaussian phase, S -> exp(-sigma^2). Averaged over many
        # realisations this must hold; sigma = 0.2 rad is well inside validity.
        sigma = 0.2
        values = [
            strehl_from_field(rng.normal(scale=sigma, size=(32, 32)), small_pupil.mask)
            for _ in range(200)
        ]
        assert np.mean(values) == pytest.approx(np.exp(-(sigma**2)), rel=0.02)
