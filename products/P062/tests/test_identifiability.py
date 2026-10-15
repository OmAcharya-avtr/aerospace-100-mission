"""Identifiability tests, with two exactly known null directions."""

from __future__ import annotations

import numpy as np
import pytest

from twincalib.identifiability import (
    DEFAULT_RANK_TOL,
    DEFAULT_WEAK_RATIO,
    analyse,
    collinearity_index,
    worst_subsets,
)
from twincalib.scenarios import (
    LINEAR_THETA_TRUE,
    build_inputs,
    calibration_model,
    design,
    linear_design,
    linear_model,
)
from twincalib.sensitivity import log_sensitivity


def _thermal(name: str):
    d = design(name)
    sens = log_sensitivity(
        calibration_model(),
        d.theta_true,
        build_inputs(d),
        d.initial_state,
        d.channels,
        d.sigma_k,
        d.sample_stride,
    )
    return d, sens, analyse(sens)


def _linear():
    model = linear_model()
    sens = log_sensitivity(
        model, LINEAR_THETA_TRUE, linear_design(121, 25.0), [300.0], [0], 0.3
    )
    return sens, analyse(sens)


def test_linear_model_null_direction_is_the_known_one():
    # Known answer, proved in LinearNodeModel's docstring: T(t) depends on
    # (C, k, q) only through q/k and k/C, so scaling all three by one factor
    # leaves every prediction unchanged. The null direction in log-parameter
    # coordinates is therefore exactly (1, 1, 1)/sqrt(3) = 0.5773502692 each.
    _sens, report = _linear()
    assert report.rank == 2
    assert not report.structurally_identifiable
    nulls = report.null_combinations
    assert len(nulls) == 1
    vector = nulls[0].vector
    expected = np.full(3, 1.0 / np.sqrt(3.0))
    # The singular vector is determined up to sign.
    aligned = vector if float(vector @ expected) > 0.0 else -vector
    assert np.max(np.abs(aligned - expected)) < 1e-6


def test_linear_model_null_direction_leaves_predictions_unchanged():
    # The direct consequence: moving along the null direction by a factor of 3
    # must not move the trajectory by more than the integrator's own error.
    model = linear_model()
    inputs = linear_design(121, 25.0)
    base = model.observe(LINEAR_THETA_TRUE, inputs, [300.0], [0])
    scaled = model.observe(
        np.asarray(LINEAR_THETA_TRUE, dtype=float) * 3.0, inputs, [300.0], [0]
    )
    assert np.max(np.abs(base - scaled)) < 1e-6


def test_full_sun_null_direction_is_the_known_heat_load_combination():
    # Known answer, proved in ThermalTwinModel's docstring: with s(t) = 1 the
    # two loads enter only through q_sun + q_par, so the null direction in log
    # coordinates is proportional to (q_par, -q_sun) = (18, -55) on those two
    # coordinates and zero on the other four. Normalised: (0.31103906, -0.95039713).
    d, _sens, report = _thermal("full-sun")
    assert report.rank == 5
    nulls = report.null_combinations
    assert len(nulls) == 1
    vector = nulls[0].vector
    expected = np.zeros(6)
    expected[4] = d.theta_true[5]
    expected[5] = -d.theta_true[4]
    expected /= np.linalg.norm(expected)
    aligned = vector if float(vector @ expected) > 0.0 else -vector
    assert np.max(np.abs(aligned - expected)) < 1e-5
    assert expected[4] == pytest.approx(0.31103906, abs=1e-7)
    assert expected[5] == pytest.approx(-0.95039713, abs=1e-7)


def test_full_sun_null_direction_leaves_predictions_unchanged():
    d = design("full-sun")
    model = calibration_model()
    inputs = build_inputs(d)
    base = model.observe(
        d.theta_true, inputs, d.initial_state, d.channels, d.sample_stride
    )
    # Move 5 W from the solar load to the parasitic load: the sum is unchanged.
    moved = list(d.theta_true)
    moved[4] -= 5.0
    moved[5] += 5.0
    other = model.observe(
        moved, inputs, d.initial_state, d.channels, d.sample_stride
    )
    assert np.max(np.abs(base - other)) < 1e-9


def test_baseline_design_is_full_rank_and_well_conditioned():
    _d, _sens, report = _thermal("baseline-eclipse")
    assert report.rank == 6
    assert report.structurally_identifiable
    assert report.weak_combinations == ()
    assert report.condition_number < 1e4
    assert np.all(np.isfinite(report.relative_sd))
    assert not report.pseudo_inverse_used


def test_weak_excitation_is_full_rank_but_reported_as_near_null():
    # The design the rank test is not enough for: full numerical rank, and a
    # near-null direction several decades below the weak-reporting cut.
    _d, _sens, report = _thermal("weak-excitation")
    assert report.rank == 6
    assert report.structurally_identifiable
    assert len(report.weak_combinations) >= 1
    assert report.condition_number > 1e4
    assert float(np.max(report.relative_sd)) > 1.0
    assert "near-null" in report.verdict()


def test_verdict_text_distinguishes_the_three_outcomes():
    _d, _s, baseline = _thermal("baseline-eclipse")
    _d, _s, fullsun = _thermal("full-sun")
    _d, _s, weak = _thermal("weak-excitation")
    assert "no near-null direction found" in baseline.verdict()
    assert fullsun.verdict().startswith("NOT IDENTIFIABLE")
    assert "near-null" in weak.verdict()


def test_singular_values_descend_and_ratios_are_consistent():
    _d, _sens, report = _thermal("baseline-eclipse")
    sv = report.singular_values
    assert np.all(np.diff(sv) <= 1e-12 * sv[0])
    for c in report.combinations:
        assert c.ratio == pytest.approx(c.singular_value / float(sv[0]), rel=1e-12)
        assert np.linalg.norm(c.vector) == pytest.approx(1.0, abs=1e-12)


def test_combination_text_names_the_parameters():
    d, _sens, report = _thermal("full-sun")
    text = report.null_combinations[0].as_text()
    assert "q_sun" in text
    assert "q_par" in text
    assert "C1" not in text
    terms = report.null_combinations[0].terms()
    assert {name for name, _w in terms} == {"q_sun", "q_par"}
    assert abs(terms[0][1]) >= abs(terms[1][1])
    assert d.name == "full-sun"


def test_combination_text_handles_an_empty_selection():
    _d, _sens, report = _thermal("baseline-eclipse")
    assert "no component" in report.combinations[0].as_text(min_weight=1.5)


def test_collinearity_index_of_a_single_column_is_one():
    _d, sens, _report = _thermal("baseline-eclipse")
    assert collinearity_index(sens, [0]) == pytest.approx(1.0, rel=1e-9)


def test_collinearity_index_is_infinite_for_a_degenerate_pair():
    _d, sens, _report = _thermal("full-sun")
    assert collinearity_index(sens, [4, 5]) > 1e6


def test_collinearity_index_rejects_an_empty_subset():
    _d, sens, _report = _thermal("baseline-eclipse")
    with pytest.raises(ValueError, match="at least one column"):
        collinearity_index(sens, [])


def test_collinearity_index_is_infinite_for_a_zero_column():
    _d, sens, _report = _thermal("baseline-eclipse")
    patched = type(sens)(
        jacobian_log=sens.jacobian_log,
        weighted=np.column_stack(
            [np.zeros(sens.n_observations), sens.weighted[:, 1:]]
        ),
        prediction=sens.prediction,
        sigma=sens.sigma,
        theta=sens.theta,
        parameter_names=sens.parameter_names,
        log_step=sens.log_step,
    )
    assert collinearity_index(patched, [0, 1]) == float("inf")


def test_worst_subsets_finds_the_degenerate_pair_first():
    _d, sens, _report = _thermal("full-sun")
    ranked = worst_subsets(sens, size=2, top=3)
    assert set(ranked[0][0]) == {"q_sun", "q_par"}
    assert ranked[0][1] > ranked[1][1]
    assert len(ranked) == 3


def test_worst_subsets_validation():
    _d, sens, _report = _thermal("baseline-eclipse")
    with pytest.raises(ValueError, match="size must lie"):
        worst_subsets(sens, size=1)
    with pytest.raises(ValueError, match="size must lie"):
        worst_subsets(sens, size=7)
    with pytest.raises(ValueError, match="top"):
        worst_subsets(sens, size=2, top=0)


def test_analyse_threshold_validation():
    _d, sens, _report = _thermal("baseline-eclipse")
    for bad in (0.0, -1.0, 1.5, np.nan):
        with pytest.raises(ValueError, match="rank_tol"):
            analyse(sens, rank_tol=bad)
        with pytest.raises(ValueError, match="weak_ratio"):
            analyse(sens, weak_ratio=bad)


def test_analyse_rejects_a_zero_sensitivity_matrix():
    _d, sens, _report = _thermal("baseline-eclipse")
    patched = type(sens)(
        jacobian_log=np.zeros_like(sens.jacobian_log),
        weighted=np.zeros_like(sens.weighted),
        prediction=sens.prediction,
        sigma=sens.sigma,
        theta=sens.theta,
        parameter_names=sens.parameter_names,
        log_step=sens.log_step,
    )
    with pytest.raises(ValueError, match="identically zero"):
        analyse(patched)


def test_correlation_matrix_shape_and_diagonal():
    _d, _sens, report = _thermal("baseline-eclipse")
    corr = report.correlation
    assert corr.shape == (6, 6)
    assert np.allclose(np.diag(corr), 1.0, rtol=0.0, atol=1e-12)
    finite = corr[np.isfinite(corr)]
    assert float(np.max(np.abs(finite))) <= 1.0 + 1e-9


def test_default_thresholds_are_the_documented_values():
    assert DEFAULT_RANK_TOL == pytest.approx(1e-8)
    assert DEFAULT_WEAK_RATIO == pytest.approx(1e-3)


def test_rank_verdict_does_not_balance_on_the_tolerance():
    # The shipped verdicts must not flip when the rank tolerance moves by two
    # decades either way: the structural null is 1e-11 and the smallest
    # full-rank ratio is 2.5e-3.
    _d, sens_full, _r = _thermal("full-sun")
    _d, sens_base, _r2 = _thermal("baseline-eclipse")
    for tol in (1e-6, 1e-8, 1e-10):
        assert analyse(sens_full, rank_tol=tol).rank == 5
        assert analyse(sens_base, rank_tol=tol).rank == 6
