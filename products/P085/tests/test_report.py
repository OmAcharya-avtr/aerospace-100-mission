"""Text rendering: table shape, refusal marking, and what must be printed."""

from __future__ import annotations

import numpy as np
import pytest

import pointmodel
from pointmodel import report, synthetic, terms, wls


class TestFormatTable:
    """The one table renderer."""

    def test_columns_are_sized_to_the_widest_cell(self) -> None:
        text = report.format_table(["a", "bbbb"], [["xxxxx", "y"]])
        lines = text.splitlines()
        assert len(lines) == 3
        assert lines[0].startswith("a      bbbb")
        assert set(lines[1]) <= {"-", " "}
        assert lines[2].startswith("xxxxx  y")

    def test_rejects_a_row_of_the_wrong_width(self) -> None:
        with pytest.raises(ValueError, match="has 1 cells but there are 2 headers"):
            report.format_table(["a", "b"], [["only"]])

    def test_accepts_no_rows(self) -> None:
        text = report.format_table(["a", "b"], [])
        assert len(text.splitlines()) == 2


class TestRenderTerms:
    """The declared term set, printed."""

    def test_every_term_appears(self) -> None:
        text = report.render_terms()
        for name in terms.TERM_NAMES:
            assert name in text
        assert "arcsec" in text
        assert "rho_hat(E)" in text


class TestRenderIdentifiability:
    """The verdict comes first and the refusal is explicit."""

    def test_separable_geometry_says_so(self, clean_full_sky: synthetic.ResidualSet) -> None:
        budget = pointmodel.decompose(
            clean_full_sky.design, clean_full_sky.observed, 2.0, robust=False
        )
        text = report.render_identifiability(budget.identifiability)
        assert text.startswith("WHAT THIS GEOMETRY CANNOT DETERMINE")
        assert "No direction falls below the declared null threshold." in text
        assert "IDENTIFIABLE LINEAR COMBINATIONS" in text

    def test_restricted_geometry_names_the_refused_terms(
        self, narrow_band: synthetic.ResidualSet
    ) -> None:
        budget = pointmodel.decompose(
            narrow_band.design, narrow_band.observed, 2.0, robust=False
        )
        text = report.render_identifiability(budget.identifiability)
        assert "NOT IDENTIFIABLE" in text
        assert "terms refused:" in text
        for name in ("IA", "CA", "NPAE"):
            assert name in text
        assert "not identifiable" in text

    def test_thresholds_are_printed(self, narrow_band: synthetic.ResidualSet) -> None:
        budget = pointmodel.decompose(
            narrow_band.design, narrow_band.observed, 2.0, robust=False
        )
        text = report.render_identifiability(budget.identifiability)
        assert "declared thresholds" in text
        assert "0.001" in text


class TestRenderFit:
    """The fit, with the VIF caveat spelled out."""

    def test_carries_the_estimator_against_design_warning(
        self, clean_full_sky: synthetic.ResidualSet
    ) -> None:
        result = wls.fit(clean_full_sky.design, clean_full_sky.observed, 2.0)
        text = report.render_fit(result)
        assert "about the DESIGN" in text
        assert "ESTIMATOR" in text
        assert "chi square" in text
        for name in terms.TERM_NAMES:
            assert name in text

    def test_handles_an_infinite_variance_inflation(
        self, single_azimuth: synthetic.ResidualSet
    ) -> None:
        result = wls.fit(single_azimuth.design, single_azimuth.observed, 2.0)
        text = report.render_fit(result)
        assert "rank                : 7 of 9" in text


class TestRenderBudget:
    """The whole decomposition, in order."""

    def test_verdict_precedes_the_coefficients(
        self, narrow_band: synthetic.ResidualSet
    ) -> None:
        budget = pointmodel.decompose(
            narrow_band.design, narrow_band.observed, 2.0, robust=False
        )
        text = report.render_budget(budget)
        assert text.index("WHAT THIS GEOMETRY CANNOT DETERMINE") < text.index(
            "POINTING ERROR BUDGET"
        )

    def test_refused_rows_are_marked(self, narrow_band: synthetic.ResidualSet) -> None:
        budget = pointmodel.decompose(
            narrow_band.design, narrow_band.observed, 2.0, robust=False
        )
        text = report.render_budget(budget)
        assert "REFUSED" in text
        assert "REFUSED per-term split for:" in text

    def test_quadrature_closure_is_printed_with_its_caveat(
        self, clean_full_sky: synthetic.ResidualSet
    ) -> None:
        budget = pointmodel.decompose(
            clean_full_sky.design, clean_full_sky.observed, 2.0, robust=False
        )
        text = report.render_budget(budget)
        assert "quadrature closure" in text
        assert "only for an orthogonal design" in text

    def test_robust_sections_appear_when_requested(
        self, contaminated: synthetic.ResidualSet
    ) -> None:
        budget = pointmodel.decompose(
            contaminated.design, contaminated.observed, 2.0, robust=True
        )
        text = report.render_budget(budget)
        assert "ROBUST FIT (huber)" in text
        assert "ROBUST FIT (ransac)" in text
        assert "DISAGREEMENT: least squares against Huber" in text
        assert "DISAGREEMENT: least squares against RANSAC" in text
        assert "flagged             : YES" in text

    def test_no_robust_sections_when_skipped(
        self, clean_full_sky: synthetic.ResidualSet
    ) -> None:
        budget = pointmodel.decompose(
            clean_full_sky.design, clean_full_sky.observed, 2.0, robust=False
        )
        text = report.render_budget(budget)
        assert "ROBUST FIT" not in text


class TestNumberRendering:
    """Non-finite values are spelled out rather than printed as a float."""

    def test_nan_and_inf(self) -> None:
        text = report.format_table(
            ["v"], [[report._number(np.nan)], [report._number(np.inf)], [report._number(-np.inf)]]
        )
        assert "nan" in text
        assert "inf" in text
        assert "-inf" in text

    def test_none_renders_as_a_dash(self) -> None:
        assert report._number(None) == "-"
