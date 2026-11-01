"""The rank analysis: the refusal, the named directions, and the combinations."""

from __future__ import annotations

import numpy as np
import pytest

from pointmodel import identifiability, synthetic, terms, wls


def _report(residuals: synthetic.ResidualSet, **kwargs: float) -> tuple:
    result = wls.fit(residuals.design, residuals.observed, residuals.sigma_arcsec or 1.0)
    return result, identifiability.analyse_fit(residuals.design, result, **kwargs)


class TestVerdictByGeometry:
    """Each declared geometry gets the verdict its structure implies."""

    def test_full_sky_is_separable_but_weakly_determined(
        self, clean_full_sky: synthetic.ResidualSet
    ) -> None:
        _result, report = _report(clean_full_sky)
        assert report.numerical_rank == terms.N_TERMS
        assert report.separable is True
        assert report.null_directions == ()
        # The flexure-against-refraction direction is a property of the term
        # set, not of a bad schedule, and it is weakly determined even here.
        assert len(report.marginal_directions) >= 1
        assert report.verdict == "weakly determined"
        names = {name for name, _weight in report.marginal_directions[0].dominant(4)}
        assert names & {"IE", "TF", "TS", "RF"}

    def test_narrow_elevation_band_refuses_three_directions(
        self, narrow_band: synthetic.ResidualSet
    ) -> None:
        _result, report = _report(narrow_band)
        assert report.numerical_rank == 6
        assert report.separable is False
        assert report.verdict == "not separable"
        assert len(report.null_directions) == 3
        refused = set(report.unidentifiable_terms)
        # The cross-elevation constants IA, CA and NPAE collapse into one, and
        # so do the elevation-side IE, TF, TS and RF.
        assert {"IA", "CA", "NPAE"} <= refused
        assert {"IE", "TF", "RF"} <= refused
        # The azimuth-axis tilt is still measured: cos(A) and sin(A) are fully
        # sampled, and a narrow elevation band does not touch them.
        assert "AN" not in refused
        assert "AW" not in refused

    def test_the_collimation_direction_is_named_in_the_narrow_band(
        self, narrow_band: synthetic.ResidualSet
    ) -> None:
        _result, report = _report(narrow_band)
        found = False
        for direction in report.null_directions:
            names = {name for name, _weight in direction.dominant(3)}
            if names == {"CA", "IA", "NPAE"}:
                found = True
                weights = dict(direction.dominant(3))
                # The combination is CA against (IA + NPAE)/2-ish: over a 4
                # degree band cos(E) and sin(E) are both nearly constant, so
                # the three columns are nearly parallel and the null vector
                # puts roughly equal and opposite weight on IA and NPAE
                # against CA.
                assert abs(weights["CA"]) > 0.5
                assert weights["IA"] * weights["CA"] < 0.0
                assert weights["NPAE"] * weights["CA"] < 0.0
        assert found, (
            "the IA/CA/NPAE degeneracy was not among the named null directions: "
            + "; ".join(direction.statement() for direction in report.null_directions)
        )

    def test_single_azimuth_is_exactly_rank_deficient_by_two(
        self, single_azimuth: synthetic.ResidualSet
    ) -> None:
        _result, report = _report(single_azimuth)
        # With azimuth fixed, cos(A) and sin(A) are constants. The elevation
        # block then has IE, AN and AW all contributing a constant, and the
        # cross-elevation block has AN, AW and NPAE all contributing a
        # multiple of sin(E). Four coefficients enter through two functions,
        # so the deficiency is two.
        assert float(report.ratios[-1]) < 1e-14
        assert float(report.ratios[-2]) < 1e-14
        assert float(report.ratios[-3]) > 1e-4
        assert report.numerical_rank == 7
        refused = set(report.unidentifiable_terms)
        assert {"AN", "AW", "NPAE", "IE"} <= refused

    def test_two_azimuths_are_nearly_as_good_as_the_full_sky(
        self, clean_full_sky: synthetic.ResidualSet
    ) -> None:
        # Two opposed azimuths separate a tilt completely. This is in the
        # suite because the geometry looks as restricted as single_az and is
        # not, and a reader who judges a schedule by its pointing count will
        # get it wrong.
        two = synthetic.generate("two_az", 240, sigma_arcsec=2.0, seed=4242)
        _one_result, good = _report(clean_full_sky)
        _two_result, pair = _report(two)
        assert pair.numerical_rank == terms.N_TERMS
        assert pair.condition_number < 3.0 * good.condition_number

    def test_low_elevation_loses_the_flexure_split(self) -> None:
        low = synthetic.generate("low_el", 240, sigma_arcsec=2.0, seed=4242)
        _result, report = _report(low)
        assert report.separable is False
        assert {"IE", "TF", "TS"} <= set(report.unidentifiable_terms)


class TestDirectionsAndCombinations:
    """Equations (23) and (24)."""

    def test_a_null_direction_changes_no_prediction(
        self, narrow_band: synthetic.ResidualSet
    ) -> None:
        result, report = _report(narrow_band)
        baseline = narrow_band.design @ result.coefficients
        for direction in report.null_directions:
            # Add 1000 arcsec along the direction. The predicted residual field
            # must move by far less than the declared noise, which is what
            # "not identifiable" means operationally.
            moved = narrow_band.design @ (result.coefficients + 1000.0 * direction.weights)
            shift = float(np.sqrt(np.mean((moved - baseline) ** 2)))
            assert shift < 0.5 * narrow_band.sigma_arcsec, (
                f"direction {direction.index} moved the prediction by {shift:.6g} arcsec"
            )

    def test_a_determined_direction_does_change_the_prediction(
        self, narrow_band: synthetic.ResidualSet
    ) -> None:
        # The contrapositive control. The best-determined direction must move
        # the prediction by orders of magnitude more than a null one for the
        # same coefficient-space step.
        result, report = _report(narrow_band)
        baseline = narrow_band.design @ result.coefficients
        best = report.combinations[0]
        step = best.unit_weights / np.linalg.norm(best.unit_weights)
        moved = narrow_band.design @ (result.coefficients + 1000.0 * step)
        shift = float(np.sqrt(np.mean((moved - baseline) ** 2)))
        assert shift > 100.0 * narrow_band.sigma_arcsec

    def test_combinations_are_a_complete_reparameterisation(
        self, clean_full_sky: synthetic.ResidualSet
    ) -> None:
        result, report = _report(clean_full_sky)
        assert len(report.combinations) == terms.N_TERMS
        # Each combination's value is v_k . (S theta); stacking the v_k gives
        # an orthogonal matrix, so the values recover S theta exactly.
        right = np.stack(
            [combination.weights / report.column_scales for combination in report.combinations]
        )
        assert right @ right.T == pytest.approx(
            np.eye(terms.N_TERMS), rel=0.0, abs=1e-12
        )
        values = np.array([combination.value for combination in report.combinations])
        recovered = (right.T @ values) / report.column_scales
        assert recovered == pytest.approx(result.coefficients, rel=1e-9, abs=1e-9)

    def test_combination_standard_errors_fall_with_the_singular_value(
        self, narrow_band: synthetic.ResidualSet
    ) -> None:
        _result, report = _report(narrow_band)
        errors = [combination.standard_error for combination in report.combinations]
        assert all(b >= a for a, b in zip(errors, errors[1:], strict=False))

    def test_statements_name_the_terms(self, narrow_band: synthetic.ResidualSet) -> None:
        _result, report = _report(narrow_band)
        for direction in report.null_directions:
            text = direction.statement()
            assert any(name in text for name in terms.TERM_NAMES)
            assert "not identifiable" in text
        for combination in report.combinations:
            assert "precision" in combination.statement()


class TestThresholds:
    """The declared thresholds, and the gap that makes them robust."""

    def test_the_verdict_is_stable_across_the_gap(
        self, narrow_band: synthetic.ResidualSet
    ) -> None:
        # The narrow-band spectrum has a gap of two orders of magnitude between
        # the smallest determined ratio and the largest refused one. Any
        # threshold inside that gap gives the same verdict, so the verdict does
        # not rest on the default.
        results = []
        for null_threshold in (3e-4, 1e-3, 3e-3):
            _result, report = _report(narrow_band, null_threshold=null_threshold)
            results.append(report.numerical_rank)
        assert len(set(results)) == 1

    def test_a_tighter_threshold_refuses_less(
        self, narrow_band: synthetic.ResidualSet
    ) -> None:
        _result, loose = _report(narrow_band, null_threshold=1e-3)
        _result2, tight = _report(narrow_band, null_threshold=1e-7)
        assert tight.numerical_rank > loose.numerical_rank

    def test_rejects_inconsistent_thresholds(self) -> None:
        design = np.eye(terms.N_TERMS)
        with pytest.raises(ValueError, match="thresholds must satisfy"):
            identifiability.analyse(
                design,
                np.zeros(terms.N_TERMS),
                null_threshold=0.5,
                marginal_threshold=0.1,
            )

    def test_rejects_a_non_positive_sigma_scale(self) -> None:
        with pytest.raises(ValueError, match="sigma_scale must be positive"):
            identifiability.analyse(
                np.eye(terms.N_TERMS), np.zeros(terms.N_TERMS), sigma_scale=0.0
            )

    def test_rejects_a_wrong_shaped_design(self) -> None:
        with pytest.raises(ValueError, match="must be"):
            identifiability.analyse(np.zeros((10, 4)), np.zeros(terms.N_TERMS))

    def test_rejects_a_wrong_shaped_coefficient_vector(self) -> None:
        with pytest.raises(ValueError, match="coefficients must have shape"):
            identifiability.analyse(np.eye(terms.N_TERMS), np.zeros(3))

    def test_rejects_a_zero_column(self) -> None:
        design = np.ones((20, terms.N_TERMS))
        design[:, 0] = 0.0
        with pytest.raises(ValueError, match="identically zero"):
            identifiability.analyse(design, np.zeros(terms.N_TERMS))


class TestIdentityDesign:
    """An exactly orthonormal design is the degenerate easy case."""

    def test_identity_design_is_perfectly_conditioned(self) -> None:
        report = identifiability.analyse(np.eye(terms.N_TERMS), np.ones(terms.N_TERMS))
        assert report.condition_number == pytest.approx(1.0, rel=1e-12, abs=0.0)
        assert report.numerical_rank == terms.N_TERMS
        assert report.verdict == "separable"
        assert report.vif == pytest.approx(np.ones(terms.N_TERMS), rel=1e-12, abs=0.0)
