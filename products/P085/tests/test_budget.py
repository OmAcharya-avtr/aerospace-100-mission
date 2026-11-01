"""Integration: the whole pipeline through :func:`pointmodel.decompose`."""

from __future__ import annotations

import numpy as np
import pytest

import pointmodel
from pointmodel import synthetic, terms


class TestIntegrationGoodGeometry:
    """The end-to-end path on a geometry that supports a per-term split."""

    def test_full_pipeline_recovers_the_truth_and_reports_everything(
        self, clean_full_sky: synthetic.ResidualSet
    ) -> None:
        budget = pointmodel.decompose(
            clean_full_sky.design, clean_full_sky.observed, sigma_arcsec=2.0
        )
        # Structure.
        assert len(budget.terms) == terms.N_TERMS
        assert [row.name for row in budget.terms] == list(terms.TERM_NAMES)
        assert budget.huber is not None
        assert budget.ransac is not None
        assert budget.huber_disagreement is not None
        assert budget.ransac_disagreement is not None
        # Verdict.
        assert budget.identifiability.separable is True
        assert budget.refused_terms == ()
        assert all(row.separable for row in budget.terms)
        # Recovery, anchored on the truth.
        truth = clean_full_sky.coefficients
        error = np.abs(budget.least_squares.coefficients - truth)
        assert np.all(error < 4.0 * budget.least_squares.standard_errors)
        # Fit quality.
        assert budget.fit_rms_arcsec < budget.total_rms_arcsec
        assert 0.0 < budget.explained_fraction < 1.0
        # Neither robust fit is flagged on clean data.
        assert budget.huber_disagreement.flagged is False
        assert budget.ransac_disagreement.flagged is False

    def test_quadrature_closure_departs_from_one_because_terms_correlate(
        self, clean_full_sky: synthetic.ResidualSet, narrow_band: synthetic.ResidualSet
    ) -> None:
        # The columns are not orthogonal, so the sum of squared contributions
        # is not the squared total and a reader must not add the budget rows
        # in quadrature. On the full sky the design is nearly orthogonal and
        # the closure is 1.0107, a 1.1 per cent departure; on the narrow
        # elevation band, where three cross-elevation columns are nearly
        # parallel and the fit splits a constant into enormous cancelling
        # pieces, the closure is 1.05e6. The near-one case is asserted to be
        # near one and NOT equal to one; the restricted case is asserted to be
        # nowhere near it.
        good = pointmodel.decompose(
            clean_full_sky.design, clean_full_sky.observed, sigma_arcsec=2.0, robust=False
        )
        assert abs(good.quadrature_closure - 1.0) > 1e-6
        assert abs(good.quadrature_closure - 1.0) < 0.05
        restricted = pointmodel.decompose(
            narrow_band.design, narrow_band.observed, sigma_arcsec=2.0, robust=False
        )
        assert restricted.quadrature_closure > 1e4

    def test_contributions_are_the_column_norms_of_the_fitted_terms(
        self, clean_full_sky: synthetic.ResidualSet
    ) -> None:
        budget = pointmodel.decompose(
            clean_full_sky.design, clean_full_sky.observed, sigma_arcsec=2.0
        )
        for index, row in enumerate(budget.terms):
            expected = float(
                np.sqrt(
                    np.mean(
                        (
                            clean_full_sky.design[:, index]
                            * budget.least_squares.coefficients[index]
                        )
                        ** 2
                    )
                )
            )
            assert row.contribution_rms_arcsec == pytest.approx(
                expected, rel=1e-12, abs=0.0
            )

    def test_the_refraction_contribution_is_the_largest_on_the_elevation_side(
        self, clean_full_sky: synthetic.ResidualSet
    ) -> None:
        # With the declared test coefficients the refraction coefficient is 58
        # arcsec at 45 degrees and its column reaches 5.4 at 15 degrees
        # elevation, so it dominates the elevation-side contributions.
        budget = pointmodel.decompose(
            clean_full_sky.design, clean_full_sky.observed, sigma_arcsec=2.0
        )
        by_name = {row.name: row.contribution_rms_arcsec for row in budget.terms}
        assert by_name["RF"] > by_name["IE"]
        assert by_name["RF"] > by_name["TF"]
        assert by_name["RF"] > by_name["TS"]


class TestIntegrationRestrictedGeometry:
    """The refusal path, end to end."""

    def test_narrow_band_refuses_and_names_the_combinations(
        self, narrow_band: synthetic.ResidualSet
    ) -> None:
        budget = pointmodel.decompose(
            narrow_band.design, narrow_band.observed, sigma_arcsec=2.0
        )
        assert budget.identifiability.separable is False
        refused = set(budget.refused_terms)
        assert {"IA", "CA", "NPAE"} <= refused
        assert [row.name for row in budget.terms if not row.separable] == [
            name for name in terms.TERM_NAMES if name in refused
        ]
        # The refused coefficients are reported, and they are far from the
        # truth: that is why they are refused. Anchored on the truth.
        truth = dict(zip(terms.TERM_NAMES, narrow_band.coefficients, strict=True))
        worst = max(
            abs(row.coefficient - truth[row.name])
            for row in budget.terms
            if not row.separable
        )
        assert worst > 10.0 * narrow_band.sigma_arcsec
        # And the identifiable combinations are still measured.
        determined = [c for c in budget.identifiability.combinations if c.determined]
        assert len(determined) == 6
        assert all(c.precision > 1.0 for c in determined)

    def test_the_tilt_survives_a_narrow_band(
        self, narrow_band: synthetic.ResidualSet
    ) -> None:
        # AN and AW are determined by azimuth coverage, which a narrow
        # elevation band does not restrict, so they must still be recovered.
        budget = pointmodel.decompose(
            narrow_band.design, narrow_band.observed, sigma_arcsec=2.0
        )
        truth = dict(zip(terms.TERM_NAMES, narrow_band.coefficients, strict=True))
        for name in ("AN", "AW"):
            row = next(r for r in budget.terms if r.name == name)
            assert row.separable is True
            assert abs(row.coefficient - truth[name]) < 1.0

    def test_single_azimuth_refuses_the_tilt_instead(
        self, single_azimuth: synthetic.ResidualSet
    ) -> None:
        budget = pointmodel.decompose(
            single_azimuth.design, single_azimuth.observed, sigma_arcsec=2.0
        )
        assert {"AN", "AW", "NPAE"} <= set(budget.refused_terms)
        # And the terms a single azimuth does determine are determined.
        truth = dict(zip(terms.TERM_NAMES, single_azimuth.coefficients, strict=True))
        for name in ("IA", "CA", "TF", "RF"):
            row = next(r for r in budget.terms if r.name == name)
            assert row.separable is True
            assert abs(row.coefficient - truth[name]) < 5.0


class TestIntegrationContaminated:
    """The diagnostic path, end to end."""

    def test_contamination_is_flagged_by_both_robust_fits(
        self, contaminated: synthetic.ResidualSet
    ) -> None:
        budget = pointmodel.decompose(
            contaminated.design, contaminated.observed, sigma_arcsec=2.0
        )
        assert budget.huber_disagreement is not None
        assert budget.ransac_disagreement is not None
        assert budget.huber_disagreement.flagged is True
        assert budget.ransac_disagreement.flagged is True
        # Both robust fits are nearer the truth than least squares.
        truth = contaminated.coefficients
        assert budget.huber is not None and budget.ransac is not None
        least_error = float(np.max(np.abs(budget.least_squares.coefficients - truth)))
        assert float(np.max(np.abs(budget.huber.coefficients - truth))) < least_error
        assert float(np.max(np.abs(budget.ransac.coefficients - truth))) < least_error


class TestOptionsAndValidation:
    """Switches and rejections at the top level."""

    def test_robust_can_be_skipped(self, clean_full_sky: synthetic.ResidualSet) -> None:
        budget = pointmodel.decompose(
            clean_full_sky.design, clean_full_sky.observed, sigma_arcsec=2.0, robust=False
        )
        assert budget.huber is None
        assert budget.ransac is None
        assert budget.huber_disagreement is None
        assert budget.ransac_disagreement is None

    def test_confidence_level_reaches_the_rows(
        self, clean_full_sky: synthetic.ResidualSet
    ) -> None:
        narrow = pointmodel.decompose(
            clean_full_sky.design,
            clean_full_sky.observed,
            sigma_arcsec=2.0,
            confidence_level=0.68,
            robust=False,
        )
        wide = pointmodel.decompose(
            clean_full_sky.design,
            clean_full_sky.observed,
            sigma_arcsec=2.0,
            confidence_level=0.99,
            robust=False,
        )
        for a, b in zip(narrow.terms, wide.terms, strict=True):
            assert (b.interval[1] - b.interval[0]) > (a.interval[1] - a.interval[0])

    def test_rejects_bad_inputs(self, clean_full_sky: synthetic.ResidualSet) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            pointmodel.decompose(
                clean_full_sky.design, clean_full_sky.observed, sigma_arcsec=0.0
            )
        with pytest.raises(ValueError, match="must have 9 columns"):
            pointmodel.decompose(np.zeros((40, 3)), np.zeros(40))


class TestPublicSurface:
    """The package exports what its docstring says it does."""

    def test_version(self) -> None:
        assert pointmodel.__version__ == "0.1.0"

    def test_everything_in_all_is_importable(self) -> None:
        for name in pointmodel.__all__:
            assert hasattr(pointmodel, name), name
