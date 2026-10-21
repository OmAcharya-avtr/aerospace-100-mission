"""The assembled report: disagreement, regime flags, Monte Carlo consistency."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from conjprob.dilution import DilutionWarning
from conjprob.frames import covariance_from_rtn_sigmas
from conjprob.geometry import ObjectState
from conjprob.report import ANALYTIC_METHODS, OPERATIONAL_DEFAULT, assess_conjunction


def _assess(primary, secondary, radius, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DilutionWarning)
        return assess_conjunction(primary, secondary, radius, **kwargs)


def test_report_has_no_scalar_pc_attribute(leo_primary, leo_secondary, leo_radius_m):
    report = _assess(leo_primary, leo_secondary, leo_radius_m)
    assert not hasattr(report, "pc_value")
    assert callable(report.pc)
    assert report.pc(OPERATIONAL_DEFAULT) == pytest.approx(report.analytic["foster"], rel=0.0)


def test_every_analytic_method_is_present(leo_primary, leo_secondary, leo_radius_m):
    report = _assess(leo_primary, leo_secondary, leo_radius_m)
    assert tuple(report.analytic) == ANALYTIC_METHODS
    assert report.pc("quadrature") == pytest.approx(report.quadrature_pc, rel=0.0)


def test_unknown_method_raises(leo_primary, leo_secondary, leo_radius_m):
    report = _assess(leo_primary, leo_secondary, leo_radius_m)
    with pytest.raises(KeyError, match="unknown method"):
        report.pc("nope")


def test_disagreement_fields_are_consistent(leo_primary, leo_secondary, leo_radius_m):
    report = _assess(leo_primary, leo_secondary, leo_radius_m)
    disagreement = report.disagreement
    values = np.array(list(disagreement.values.values()))
    assert disagreement.max_absolute_spread == pytest.approx(
        float(values.max() - values.min()), rel=1e-12
    )
    assert disagreement.reference_method == "quadrature"
    assert disagreement.max_relative_spread == pytest.approx(
        max(abs(v) for v in disagreement.relative_errors.values()), rel=1e-12
    )
    assert disagreement.worst_method in disagreement.values


def test_foster_agrees_with_quadrature_to_quadrature_accuracy(
    leo_primary, leo_secondary, leo_radius_m
):
    report = _assess(leo_primary, leo_secondary, leo_radius_m)
    assert abs(report.disagreement.relative_errors["foster"]) < 1e-8


def test_regime_flags_match_the_geometry(leo_primary, leo_secondary, leo_radius_m):
    report = _assess(leo_primary, leo_secondary, leo_radius_m)
    assert report.regime.radius_over_sigma == pytest.approx(
        leo_radius_m / report.geometry.sigma_effective_m, rel=1e-12
    )
    assert report.regime.aspect_ratio == pytest.approx(
        report.geometry.frame.aspect_ratio, rel=1e-12
    )
    assert report.regime.mahalanobis == pytest.approx(report.geometry.mahalanobis, rel=1e-12)
    assert report.regime.notes


def test_dilution_note_appears_when_the_mahalanobis_distance_is_small(
    leo_primary, leo_secondary, leo_radius_m
):
    report = _assess(leo_primary, leo_secondary, leo_radius_m)
    assert report.geometry.mahalanobis < np.sqrt(2.0)
    assert any("dilution branch" in note for note in report.regime.notes)


def test_dilution_warning_reaches_the_caller(leo_primary, leo_secondary, leo_radius_m):
    with pytest.warns(DilutionWarning):
        assess_conjunction(leo_primary, leo_secondary, leo_radius_m)


def test_dilution_warning_can_be_suppressed(leo_primary, leo_secondary, leo_radius_m):
    with warnings.catch_warnings():
        warnings.simplefilter("error", DilutionWarning)
        assess_conjunction(leo_primary, leo_secondary, leo_radius_m, warn_on_dilution=False)


def test_monte_carlo_is_optional_and_consistent(leo_primary, leo_secondary, leo_radius_m):
    without = _assess(leo_primary, leo_secondary, leo_radius_m)
    assert without.monte_carlo is None
    assert without.monte_carlo_consistent == ()
    with_mc = _assess(
        leo_primary,
        leo_secondary,
        leo_radius_m,
        monte_carlo_samples=400_000,
        monte_carlo_seed=7,
    )
    assert with_mc.monte_carlo is not None
    assert set(with_mc.monte_carlo_consistent) | set(with_mc.monte_carlo_inconsistent) == set(
        with_mc.disagreement.values
    )
    assert "foster" in with_mc.monte_carlo_consistent


def test_summary_mentions_every_method_and_the_refusal(
    leo_primary, leo_secondary, leo_radius_m
):
    report = _assess(
        leo_primary, leo_secondary, leo_radius_m, monte_carlo_samples=50_000
    )
    text = report.summary()
    for name in ANALYTIC_METHODS:
        assert name in text
    assert "quadrature" in text
    assert "monte_carlo" in text
    assert "disagreement (the primary output)" in text
    assert "not an input to a manoeuvre decision" in text


def test_large_hard_body_case_shows_a_real_disagreement():
    # A declared case with the hard body comparable to the uncertainty. This is
    # the regime the product exists for: the methods differ by far more than
    # rounding and the report says which one is wrong about what.
    position_a = np.array([7078137.0, 0.0, 0.0])
    velocity_a = np.array([0.0, 7504.0, 0.0])
    position_b = np.array([7078147.0, 18.0, 6.0])
    velocity_b = np.array([0.0, 5306.934315511, 5306.934315511])
    primary = ObjectState(
        position_m=position_a,
        velocity_mps=velocity_a,
        covariance_m2=covariance_from_rtn_sigmas(position_a, velocity_a, [8.0, 30.0, 8.0]),
    )
    secondary = ObjectState(
        position_m=position_b,
        velocity_mps=velocity_b,
        covariance_m2=covariance_from_rtn_sigmas(position_b, velocity_b, [6.0, 22.0, 6.0]),
    )
    report = _assess(primary, secondary, 30.0)
    assert report.regime.radius_over_sigma > 1.0
    assert report.disagreement.max_relative_spread > 0.1
    assert report.disagreement.worst_method == "small_body"


def test_assess_rejects_a_negative_radius(leo_primary, leo_secondary):
    with pytest.raises(ValueError, match="non-negative"):
        assess_conjunction(leo_primary, leo_secondary, -1.0)
