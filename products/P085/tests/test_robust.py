"""Robust fits and the disagreement diagnostic."""

from __future__ import annotations

import numpy as np
import pytest

from pointmodel import robust, synthetic, terms, wls


class TestHuber:
    """Equations (25) to (27)."""

    def test_agrees_with_least_squares_on_clean_data(
        self, clean_full_sky: synthetic.ResidualSet
    ) -> None:
        # With Gaussian noise and no outliers, Huber at delta = 1.345 has 95
        # per cent of the efficiency of least squares. Two consistent
        # estimators of the same quantity differ by a random amount of the
        # order of their own standard errors, not by a negligible amount, so
        # the declared criterion is one standard error per term -- the
        # measured maximum over six seeds is 0.63 -- and, in the space that is
        # identifiable whatever the geometry, a predicted sky difference below
        # a tenth of the declared noise. The measured value there is 0.072
        # arcsec against a declared sigma of 2.0.
        least = wls.fit(clean_full_sky.design, clean_full_sky.observed, 2.0)
        huber = robust.huber_fit(clean_full_sky.design, clean_full_sky.observed, 2.0)
        difference = np.abs(huber.coefficients - least.coefficients)
        assert np.all(difference < 1.0 * least.standard_errors)
        delta = robust.disagreement(
            clean_full_sky.design, least, huber, sigma_reference=2.0
        )
        assert delta.prediction_rms_arcsec < 0.1 * 2.0

    def test_resists_injected_acquisition_failures(
        self, contaminated: synthetic.ResidualSet
    ) -> None:
        # Anchored on the TRUE coefficients, not on an output. Six per cent
        # acquisition failures at 300 arcsec must pull least squares far off
        # the truth and leave Huber near it.
        truth = contaminated.coefficients
        least = wls.fit(contaminated.design, contaminated.observed, 2.0)
        huber = robust.huber_fit(contaminated.design, contaminated.observed, 2.0)
        least_error = float(np.max(np.abs(least.coefficients - truth)))
        huber_error = float(np.max(np.abs(huber.coefficients - truth)))
        assert least_error > 20.0, f"the contamination control did not fire: {least_error}"
        assert huber_error < 0.3 * least_error

    def test_downweights_the_injected_rows_and_not_many_others(
        self, contaminated: synthetic.ResidualSet
    ) -> None:
        huber = robust.huber_fit(contaminated.design, contaminated.observed, 2.0)
        injected = contaminated.row_outlier_mask
        # Every injected row must be downweighted below a half.
        assert np.all(huber.weights[injected] < 0.5)
        # Huber downweights the Gaussian tail too, by design: a row beyond
        # 1.345 robust sigmas gets a weight below one. For a standard normal
        # that is 2 * (1 - Phi(1.345)) = 0.1786 of rows, so the clean
        # downweighted fraction is expected near 0.18 and is asserted below
        # 0.30 rather than near zero.
        clean = huber.weights[~injected]
        assert float(np.mean(clean < 1.0 - 1e-12)) < 0.30

    def test_converges_and_reports_it(self, contaminated: synthetic.ResidualSet) -> None:
        huber = robust.huber_fit(contaminated.design, contaminated.observed, 2.0)
        assert huber.detail["converged"] == pytest.approx(1.0, rel=0.0, abs=0.0)
        assert 1 <= huber.iterations <= 60

    def test_standard_errors_are_positive_and_comparable_to_least_squares(
        self, clean_full_sky: synthetic.ResidualSet
    ) -> None:
        least = wls.fit(clean_full_sky.design, clean_full_sky.observed, 2.0)
        huber = robust.huber_fit(clean_full_sky.design, clean_full_sky.observed, 2.0)
        assert np.all(huber.standard_errors > 0.0)
        # Huber's asymptotic variance under Gaussian errors is about 1/0.95 of
        # the least-squares variance at delta = 1.345, so the standard errors
        # should be within about 30 per cent on a finite sample.
        ratio = huber.standard_errors / least.standard_errors
        assert np.all(ratio > 0.7)
        assert np.all(ratio < 1.4)

    def test_rejects_a_non_positive_delta(
        self, clean_full_sky: synthetic.ResidualSet
    ) -> None:
        with pytest.raises(ValueError, match="delta must be positive"):
            robust.huber_fit(clean_full_sky.design, clean_full_sky.observed, 2.0, delta=0.0)

    def test_rejects_a_zero_iteration_cap(
        self, clean_full_sky: synthetic.ResidualSet
    ) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            robust.huber_fit(
                clean_full_sky.design, clean_full_sky.observed, 2.0, max_iterations=0
            )


class TestRansac:
    """Fischler & Bolles consensus, with observation-wise sampling."""

    def test_finds_the_injected_outliers(self, contaminated: synthetic.ResidualSet) -> None:
        result = robust.ransac_fit(contaminated.design, contaminated.observed, 2.0)
        inliers = result.weights > 0.5
        injected = contaminated.row_outlier_mask
        # Recall on the injected rows: every one must be outside the consensus
        # set. This is anchored on ground truth.
        assert not np.any(inliers[injected])
        # A 3-sigma consensus threshold leaves about 0.27 per cent of clean
        # Gaussian rows outside by chance, so the clean inlier fraction is
        # expected near 0.997 and is asserted above 0.98.
        assert float(np.mean(inliers[~injected])) > 0.98

    def test_recovers_the_truth_on_contaminated_data(
        self, contaminated: synthetic.ResidualSet
    ) -> None:
        truth = contaminated.coefficients
        least = wls.fit(contaminated.design, contaminated.observed, 2.0)
        result = robust.ransac_fit(contaminated.design, contaminated.observed, 2.0)
        assert float(np.max(np.abs(result.coefficients - truth))) < 0.3 * float(
            np.max(np.abs(least.coefficients - truth))
        )

    def test_is_deterministic_in_the_seed(
        self, contaminated: synthetic.ResidualSet
    ) -> None:
        first = robust.ransac_fit(contaminated.design, contaminated.observed, 2.0, seed=5)
        second = robust.ransac_fit(contaminated.design, contaminated.observed, 2.0, seed=5)
        assert first.coefficients == pytest.approx(second.coefficients, rel=0.0, abs=0.0)
        assert np.array_equal(first.weights, second.weights)

    def test_a_different_seed_gives_a_comparable_answer(
        self, contaminated: synthetic.ResidualSet
    ) -> None:
        # Not the same answer: the consensus set differs by the chance
        # exclusions. The predicted sky must agree to well inside the noise.
        first = robust.ransac_fit(contaminated.design, contaminated.observed, 2.0, seed=5)
        second = robust.ransac_fit(contaminated.design, contaminated.observed, 2.0, seed=6)
        delta = robust.disagreement(
            contaminated.design, first, second, sigma_reference=2.0
        )
        assert delta.prediction_rms_arcsec < 2.0

    def test_works_on_a_rank_deficient_geometry(
        self, single_azimuth: synthetic.ResidualSet
    ) -> None:
        # A degenerate subset must not raise: the minimum-norm solution still
        # predicts the residual field, which is all a consensus count needs.
        result = robust.ransac_fit(single_azimuth.design, single_azimuth.observed, 2.0)
        assert float(np.mean(result.weights > 0.5)) > 0.95

    def test_rejects_an_odd_row_count(self) -> None:
        with pytest.raises(ValueError, match="even count"):
            robust.ransac_fit(np.ones((21, terms.N_TERMS)), np.zeros(21), 1.0)

    def test_rejects_an_out_of_range_subset_size(
        self, clean_full_sky: synthetic.ResidualSet
    ) -> None:
        with pytest.raises(ValueError, match="subset_size must lie"):
            robust.ransac_fit(
                clean_full_sky.design, clean_full_sky.observed, 2.0, subset_size=2
            )

    def test_rejects_a_non_positive_inlier_threshold(
        self, clean_full_sky: synthetic.ResidualSet
    ) -> None:
        with pytest.raises(ValueError, match="inlier_sigma must be positive"):
            robust.ransac_fit(
                clean_full_sky.design, clean_full_sky.observed, 2.0, inlier_sigma=0.0
            )

    def test_rejects_zero_trials(self, clean_full_sky: synthetic.ResidualSet) -> None:
        with pytest.raises(ValueError, match="trials must be at least 1"):
            robust.ransac_fit(clean_full_sky.design, clean_full_sky.observed, 2.0, trials=0)

    def test_raises_when_no_consensus_is_reachable(
        self, clean_full_sky: synthetic.ResidualSet
    ) -> None:
        # Pure noise at a thousand times the declared sigma: no subset can
        # gather nine consistent rows at three declared sigmas.
        rng = np.random.default_rng(1)
        garbage = rng.normal(0.0, 2000.0, clean_full_sky.observed.size)
        with pytest.raises(ValueError, match="no RANSAC trial reached"):
            robust.ransac_fit(clean_full_sky.design, garbage, 2.0, trials=20)


class TestDisagreement:
    """The diagnostic, which prefers neither fit."""

    def test_not_flagged_on_clean_data(self, clean_full_sky: synthetic.ResidualSet) -> None:
        least = wls.fit(clean_full_sky.design, clean_full_sky.observed, 2.0)
        huber = robust.huber_fit(clean_full_sky.design, clean_full_sky.observed, 2.0)
        delta = robust.disagreement(
            clean_full_sky.design, least, huber, sigma_reference=2.0
        )
        assert delta.flagged is False
        assert delta.prediction_rms_arcsec < 2.0

    def test_flagged_on_contaminated_data(
        self, contaminated: synthetic.ResidualSet
    ) -> None:
        least = wls.fit(contaminated.design, contaminated.observed, 2.0)
        huber = robust.huber_fit(contaminated.design, contaminated.observed, 2.0)
        delta = robust.disagreement(contaminated.design, least, huber, sigma_reference=2.0)
        assert delta.flagged is True
        assert delta.prediction_rms_arcsec > 2.0

    def test_refused_terms_are_marked(self, narrow_band: synthetic.ResidualSet) -> None:
        least = wls.fit(narrow_band.design, narrow_band.observed, 2.0)
        huber = robust.huber_fit(narrow_band.design, narrow_band.observed, 2.0)
        delta = robust.disagreement(
            narrow_band.design,
            least,
            huber,
            sigma_reference=2.0,
            unidentifiable=("IA", "CA", "NPAE"),
        )
        assert set(delta.unidentifiable) == {"IA", "CA", "NPAE"}
        assert "IA" not in delta.reportable_terms
        assert "AN" in delta.reportable_terms

    def test_rejects_an_unknown_term_name(
        self, clean_full_sky: synthetic.ResidualSet
    ) -> None:
        least = wls.fit(clean_full_sky.design, clean_full_sky.observed, 2.0)
        with pytest.raises(ValueError, match="unknown term name"):
            robust.disagreement(
                clean_full_sky.design,
                least,
                least,
                sigma_reference=2.0,
                unidentifiable=("WRONG",),
            )

    def test_rejects_a_non_positive_reference(
        self, clean_full_sky: synthetic.ResidualSet
    ) -> None:
        least = wls.fit(clean_full_sky.design, clean_full_sky.observed, 2.0)
        with pytest.raises(ValueError, match="sigma_reference must be positive"):
            robust.disagreement(clean_full_sky.design, least, least, sigma_reference=0.0)

    def test_rejects_a_non_positive_threshold(
        self, clean_full_sky: synthetic.ResidualSet
    ) -> None:
        least = wls.fit(clean_full_sky.design, clean_full_sky.observed, 2.0)
        with pytest.raises(ValueError, match="threshold_sigma must be positive"):
            robust.disagreement(
                clean_full_sky.design, least, least, sigma_reference=2.0, threshold_sigma=0.0
            )


class TestSharedValidation:
    """Input checks common to both robust fits."""

    def test_rejects_wrong_shapes(self) -> None:
        with pytest.raises(ValueError, match="design must be"):
            robust.huber_fit(np.zeros((20, 3)), np.zeros(20), 1.0)
        with pytest.raises(ValueError, match="observed must be 1-D"):
            robust.huber_fit(np.zeros((20, terms.N_TERMS)), np.zeros(19), 1.0)

    def test_rejects_bad_sigma(self) -> None:
        with pytest.raises(ValueError, match="finite and strictly positive"):
            robust.huber_fit(np.ones((20, terms.N_TERMS)), np.zeros(20), -1.0)

    def test_rejects_non_finite(self) -> None:
        values = np.zeros(20)
        values[3] = np.inf
        with pytest.raises(ValueError, match="non-finite"):
            robust.huber_fit(np.ones((20, terms.N_TERMS)), values, 1.0)
