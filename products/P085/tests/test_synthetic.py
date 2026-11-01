"""Synthetic residual sets: geometry, truth, outliers and determinism."""

from __future__ import annotations

import numpy as np
import pytest

from pointmodel import synthetic, terms


class TestGeometries:
    """The five declared geometries have the structure they claim."""

    def test_all_five_are_declared(self) -> None:
        assert set(synthetic.GEOMETRIES) == {
            "full_sky",
            "narrow_el",
            "single_az",
            "two_az",
            "low_el",
        }

    def test_full_sky_covers_azimuth_and_elevation(self) -> None:
        azimuth, elevation = synthetic.geometry("full_sky", 2000, seed=1)
        assert float(np.min(np.rad2deg(elevation))) > 15.0 - 1.0
        assert float(np.max(np.rad2deg(elevation))) < 80.0 + 1.0
        # 2000 uniform draws leave every 30-degree azimuth sector occupied with
        # probability indistinguishable from one.
        occupied = np.unique((np.rad2deg(azimuth) // 30).astype(int))
        assert occupied.size == 12

    def test_narrow_band_is_four_degrees_wide(self) -> None:
        _azimuth, elevation = synthetic.geometry("narrow_el", 500, seed=2)
        span = float(np.ptp(np.rad2deg(elevation)))
        assert span < 4.0

    def test_single_azimuth_is_one_value(self) -> None:
        azimuth, _elevation = synthetic.geometry("single_az", 500, seed=3)
        assert float(np.ptp(azimuth)) == pytest.approx(0.0, rel=0.0, abs=0.0)
        assert float(np.rad2deg(azimuth[0])) == pytest.approx(135.0, rel=1e-13, abs=0.0)

    def test_two_azimuths_are_two_values(self) -> None:
        azimuth, _elevation = synthetic.geometry("two_az", 500, seed=4)
        values = np.unique(np.round(np.rad2deg(azimuth), 9))
        assert values.size == 2
        assert values == pytest.approx(np.array([90.0, 270.0]), rel=1e-12, abs=0.0)

    def test_low_elevation_stays_low(self) -> None:
        _azimuth, elevation = synthetic.geometry("low_el", 500, seed=5)
        assert float(np.max(np.rad2deg(elevation))) < 25.0 + 1.0

    def test_rejects_an_unknown_geometry(self) -> None:
        with pytest.raises(ValueError, match="unknown geometry"):
            synthetic.geometry("equator", 100)

    def test_rejects_too_few_observations(self) -> None:
        with pytest.raises(ValueError, match="cannot constrain"):
            synthetic.geometry("full_sky", 4)

    def test_is_deterministic_in_the_seed(self) -> None:
        first = synthetic.geometry("full_sky", 50, seed=77)
        second = synthetic.geometry("full_sky", 50, seed=77)
        assert np.array_equal(first[0], second[0])
        assert np.array_equal(first[1], second[1])


class TestGeneration:
    """Residual sets, their truth and their shapes."""

    def test_shapes_and_blocks(self) -> None:
        residuals = synthetic.generate("full_sky", 40, sigma_arcsec=1.0, seed=9)
        assert residuals.n_observations == 40
        assert residuals.design.shape == (80, terms.N_TERMS)
        assert residuals.observed.shape == (80,)
        assert residuals.truth_residuals.shape == (80,)
        assert residuals.row_outlier_mask.shape == (80,)

    def test_noiseless_observed_equals_truth(self) -> None:
        residuals = synthetic.generate("full_sky", 30, sigma_arcsec=0.0, seed=9)
        assert residuals.observed == pytest.approx(
            residuals.truth_residuals, rel=0.0, abs=0.0
        )

    def test_exact_truth_differs_from_linear_truth_by_the_truncation(self) -> None:
        exact = synthetic.generate("full_sky", 60, sigma_arcsec=0.0, seed=12)
        linear = synthetic.generate(
            "full_sky", 60, sigma_arcsec=0.0, seed=12, linear_truth=True
        )
        assert np.array_equal(exact.azimuth_rad, linear.azimuth_rad)
        difference = np.abs(exact.truth_residuals - linear.truth_residuals)
        # Non-zero, because the exact model is not the design matrix, and small,
        # because the nominal misalignments total about 174 arcsec and the
        # truncation is of order (174/206265)^2 * 206265 = 0.147 arcsec.
        assert float(np.max(difference)) > 1e-6
        assert float(np.max(difference)) < 1.0

    def test_noise_scales_with_the_declared_sigma(self) -> None:
        low = synthetic.generate("full_sky", 400, sigma_arcsec=1.0, seed=13)
        high = synthetic.generate("full_sky", 400, sigma_arcsec=4.0, seed=13)
        low_scatter = float(np.std(low.observed - low.truth_residuals))
        high_scatter = float(np.std(high.observed - high.truth_residuals))
        # 800 samples: the sample standard deviation has a relative standard
        # error of 1/sqrt(2*800) = 0.025, so a four-sigma band on the ratio is
        # 0.14 wide around 4.0.
        assert high_scatter / low_scatter == pytest.approx(4.0, rel=0.15, abs=0.0)

    def test_a_separate_noise_seed_redraws_only_the_noise(self) -> None:
        first = synthetic.generate("full_sky", 50, sigma_arcsec=2.0, seed=21, noise_seed=1)
        second = synthetic.generate("full_sky", 50, sigma_arcsec=2.0, seed=21, noise_seed=2)
        assert np.array_equal(first.azimuth_rad, second.azimuth_rad)
        assert np.array_equal(first.design, second.design)
        assert first.truth_residuals == pytest.approx(
            second.truth_residuals, rel=0.0, abs=0.0
        )
        assert not np.array_equal(first.observed, second.observed)

    def test_custom_coefficients_are_used(self) -> None:
        residuals = synthetic.generate(
            "full_sky", 30, coefficients={"CA": 100.0}, sigma_arcsec=0.0, seed=1
        )
        assert residuals.truth_mapping()["CA"] == pytest.approx(100.0, rel=0.0, abs=0.0)
        assert residuals.truth_mapping()["IA"] == pytest.approx(0.0, rel=0.0, abs=0.0)
        # Hand calculation of the exact model for collimation alone. From
        # equation (12) with only CA non-zero the beam is
        #   Ry(-E) Rz(CA) xhat = [cos E cos CA, sin CA, sin E cos CA],
        # so the exact elevation out is arcsin(sin E cos CA) and
        #   dE = arcsin(sin E cos CA) - E = -(CA^2 / 2) tan(E) + O(CA^4)
        # in radians, while the cross-elevation residual is
        #   dX = cos(E) arctan(sin CA / (cos E cos CA)) = CA + O(CA^3).
        # So the elevation block is NOT zero: it carries a second-order term
        # of -(CA_arcsec^2 / (2 * 206264.806)) tan(E) arcsec, which at
        # CA = 100 arcsec and E = 80 degrees is -0.1375 arcsec. This is the
        # first-order truncation of pointmodel.terms made explicit.
        cross = residuals.truth_residuals[:30]
        elevation = residuals.truth_residuals[30:]
        assert cross == pytest.approx(np.full(30, 100.0), rel=0.0, abs=0.1)
        predicted = (
            -(100.0**2)
            / (2.0 * 180.0 * 3600.0 / np.pi)
            * np.tan(residuals.elevation_rad)
        )
        assert elevation == pytest.approx(predicted, rel=1e-4, abs=0.0)

    def test_accepts_a_coefficient_vector(self) -> None:
        vector = np.arange(1.0, terms.N_TERMS + 1.0)
        residuals = synthetic.generate("full_sky", 20, coefficients=vector, seed=1)
        assert residuals.coefficients == pytest.approx(vector, rel=0.0, abs=0.0)


class TestOutliers:
    """Acquisition failures, with ground truth returned."""

    def test_count_is_the_floor_of_the_fraction(self) -> None:
        residuals = synthetic.generate(
            "full_sky", 100, outlier_fraction=0.075, outlier_arcsec=300.0, seed=1
        )
        assert int(residuals.outlier_mask.sum()) == 7

    def test_both_components_of_an_observation_are_displaced(self) -> None:
        residuals = synthetic.generate(
            "full_sky",
            80,
            sigma_arcsec=0.0,
            outlier_fraction=0.1,
            outlier_arcsec=300.0,
            seed=1,
        )
        offset = residuals.observed - residuals.truth_residuals
        cross = offset[:80]
        elevation = offset[80:]
        magnitude = np.hypot(cross, elevation)
        injected = residuals.outlier_mask
        # The displacement is 300 arcsec in a random direction, so its modulus
        # is exactly 300 on every failed acquisition and exactly zero
        # elsewhere, up to rounding.
        assert magnitude[injected] == pytest.approx(
            np.full(int(injected.sum()), 300.0), rel=1e-12, abs=0.0
        )
        assert magnitude[~injected] == pytest.approx(
            np.zeros(int((~injected).sum())), rel=0.0, abs=1e-12
        )

    def test_row_mask_matches_the_observation_mask(self) -> None:
        residuals = synthetic.generate("full_sky", 50, outlier_fraction=0.2, seed=1)
        assert np.array_equal(
            residuals.row_outlier_mask,
            np.concatenate([residuals.outlier_mask, residuals.outlier_mask]),
        )

    def test_zero_fraction_gives_no_outliers(self) -> None:
        residuals = synthetic.generate("full_sky", 50, outlier_fraction=0.0, seed=1)
        assert not np.any(residuals.outlier_mask)


class TestInputValidation:
    """Every rejection path."""

    def test_rejects_a_negative_sigma(self) -> None:
        with pytest.raises(ValueError, match="sigma_arcsec must be non-negative"):
            synthetic.generate("full_sky", 20, sigma_arcsec=-1.0)

    def test_rejects_an_out_of_range_outlier_fraction(self) -> None:
        with pytest.raises(ValueError, match="outlier_fraction must lie"):
            synthetic.generate("full_sky", 20, outlier_fraction=1.0)

    def test_rejects_a_negative_outlier_magnitude(self) -> None:
        with pytest.raises(ValueError, match="outlier_arcsec must be non-negative"):
            synthetic.generate("full_sky", 20, outlier_arcsec=-1.0)

    def test_rejects_a_wrong_length_coefficient_vector(self) -> None:
        with pytest.raises(ValueError, match="coefficients vector must have shape"):
            synthetic.generate("full_sky", 20, coefficients=np.zeros(4))

    def test_rejects_a_non_finite_coefficient_vector(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            synthetic.generate(
                "full_sky", 20, coefficients=np.full(terms.N_TERMS, np.nan)
            )


class TestNominalCoefficients:
    """The declared test values, and what they are not."""

    def test_every_term_is_declared(self) -> None:
        assert set(synthetic.NOMINAL_COEFFICIENTS) == set(terms.TERM_NAMES)

    def test_the_refraction_coefficient_is_near_the_standard_value(self) -> None:
        # Bennett's formula gives 59.69 arcsec at 45 degrees under standard
        # conditions; the declared test value is 58.0, which is near it but is
        # a test value and not a measurement.
        assert 50.0 < synthetic.NOMINAL_COEFFICIENTS["RF"] < 70.0
