"""Renderer tests. Every renderer returns a string and prints nothing."""

from __future__ import annotations

import numpy as np
import pytest

from twincalib.identifiability import analyse
from twincalib.profile import profile_likelihood
from twincalib.report import (
    render_correlation,
    render_design,
    render_fit,
    render_identifiability,
    render_profiles,
    render_sensitivity,
)
from twincalib.scenarios import DESIGN_NAMES, calibration_model, design, designs


def test_render_design_covers_every_shipped_design():
    for name in DESIGN_NAMES:
        text = render_design(designs()[name])
        assert name in text
        assert "thermometer noise" in text
        assert "observations" in text


def test_render_design_marks_the_constant_profiles():
    assert "CONSTANT at 1" in render_design(design("full-sun"))
    assert "CONSTANT at 24 W" in render_design(design("weak-excitation"))
    assert "square wave" in render_design(design("baseline-eclipse"))
    assert "parameter override" in render_design(design("lumped-nodes"))


def test_render_sensitivity_lists_all_parameters(exact_baseline_fit):
    _problem, result, _d = exact_baseline_fit
    text = render_sensitivity(result.sensitivity)
    for name in result.parameter_names:
        assert name in text
    assert "delta_msqr" in text


def test_render_identifiability_puts_the_verdict_first(exact_baseline_fit):
    _problem, result, _d = exact_baseline_fit
    text = render_identifiability(analyse(result.sensitivity))
    assert text.startswith("VERDICT:")
    assert "singular spectrum" in text
    assert "Cramer-Rao" in text
    assert "no parameter combination" in text


def test_render_identifiability_writes_the_unconstrained_combination(exact_full_sun):
    from twincalib.fitting import fit

    d, _data, problem = exact_full_sun
    result = fit(problem, np.asarray(d.theta_true, dtype=float))
    text = render_identifiability(analyse(result.sensitivity))
    assert "NOT IDENTIFIABLE" in text
    assert "UNCONSTRAINED" in text
    assert "d ln q_sun" in text
    assert "d ln q_par" in text
    assert "pseudo-inverse" in text


def test_render_correlation_shape_and_flagging():
    names = ["a", "b", "c"]
    corr = np.array([[1.0, 0.99, 0.1], [0.99, 1.0, 0.2], [0.1, 0.2, 1.0]])
    text = render_correlation(names, corr, flag_above=0.95)
    assert "a" in text and "b" in text
    assert "+0.990000" in text
    quiet = render_correlation(names, corr, flag_above=0.999)
    assert "no pair reaches" in quiet


def test_render_correlation_rejects_a_mismatched_matrix():
    with pytest.raises(ValueError, match="correlation must be"):
        render_correlation(["a", "b"], np.eye(3))


def test_render_fit_shows_caveats_above_the_numbers(truth_baseline):
    from twincalib.fitting import fit

    d, _data, problem = truth_baseline
    result = fit(problem, np.asarray(d.theta_true, dtype=float))
    text = render_fit(
        result, units=calibration_model().parameter_units, theta_true=d.theta_true
    )
    assert text.startswith("CAVEATS")
    assert text.index("CAVEATS") < text.index("weighted least-squares estimate")
    assert "reduced chi-squared" in text
    assert "LOCAL QUADRATIC" in text
    assert "J/K" in text


def test_render_fit_without_truth_omits_the_truth_columns(exact_baseline_fit):
    _problem, result, _d = exact_baseline_fit
    text = render_fit(result, units=calibration_model().parameter_units)
    assert "true" not in text.split("\n")[1]


def test_render_fit_rejects_a_bad_unit_list(exact_baseline_fit):
    _problem, result, _d = exact_baseline_fit
    with pytest.raises(ValueError, match="units must have one entry"):
        render_fit(result, units=["J/K"])


def test_render_profiles_table_and_open_side(exact_baseline_fit):
    problem, result, _d = exact_baseline_fit
    curves = [
        profile_likelihood(problem, result, i, n_points=2, refine_steps=1)
        for i in (0, 2)
    ]
    text = render_profiles(curves)
    assert "profile-likelihood intervals" in text
    assert "width ratio" in text
    assert "every profile closed" in text


def test_render_profiles_names_the_open_parameters(exact_full_sun):
    from twincalib.fitting import fit

    d, _data, problem = exact_full_sun
    result = fit(problem, np.asarray(d.theta_true, dtype=float))
    curves = [profile_likelihood(problem, result, 4, n_points=2, refine_steps=2)]
    text = render_profiles(curves)
    assert "OPEN" in text
    assert "NOT determined" in text
    assert "q_sun: open below; searched" in text


def test_render_profiles_rejects_an_empty_list():
    with pytest.raises(ValueError, match="at least one profile"):
        render_profiles([])


def test_renderers_return_strings_and_print_nothing(capsys, exact_baseline_fit):
    _problem, result, d = exact_baseline_fit
    outputs = [
        render_design(d),
        render_sensitivity(result.sensitivity),
        render_identifiability(analyse(result.sensitivity)),
        render_correlation(result.parameter_names, result.correlation),
        render_fit(result, units=calibration_model().parameter_units),
    ]
    assert all(isinstance(o, str) and o for o in outputs)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
