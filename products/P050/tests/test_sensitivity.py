"""Stability intervals: boundaries, knife-edge detection, honest flags."""

from __future__ import annotations

import numpy as np
import pytest

from coderateopt import (
    EmpiricalFade,
    GammaGammaFade,
    LognormalFade,
    RateProblem,
    illustrative_modcod_table,
    margin_sensitivity,
    scintillation_sensitivity,
    select_rate,
    stability_interval,
    target_sensitivity,
)


def _problem(**overrides) -> RateProblem:
    kwargs = {
        "modcods": illustrative_modcod_table(),
        "fade": LognormalFade(0.2),
        "margin_db": 12.0,
        "availability_target": 0.99,
        "max_entries": 1,
        "mode": "per_interval",
    }
    kwargs.update(overrides)
    return RateProblem(**kwargs)


def test_support_is_unchanged_across_the_reported_interval():
    problem = _problem()
    report = scintillation_sensitivity(problem)
    base = select_rate(problem).support
    for value in np.linspace(report.lower + 1e-9, report.upper - 1e-9, 15):
        assert select_rate(problem.with_fade(LognormalFade(float(value)))).support == base


def test_support_changes_just_outside_a_resolved_boundary():
    problem = _problem()
    report = scintillation_sensitivity(problem)
    base = select_rate(problem).support
    assert not report.bounded_above_by_scan
    assert not report.bounded_below_by_scan
    # 1e-3 past each boundary is outside the bisection tolerance of 1e-4*0.2.
    assert select_rate(problem.with_fade(LognormalFade(report.upper + 1e-3))).support != base
    assert select_rate(problem.with_fade(LognormalFade(report.lower - 1e-3))).support != base
    assert report.boundary_signature_above != base
    assert report.boundary_signature_below != base


def test_margin_sensitivity_brackets_the_nominal_margin():
    report = margin_sensitivity(_problem())
    assert report.lower < 12.0 < report.upper
    assert report.parameter == "margin_db"
    assert report.upward_headroom > 0.0
    assert report.downward_headroom > 0.0


def test_target_sensitivity_finds_the_feasibility_boundary():
    # Raising the target far enough makes every MODCOD fail, which the sweep
    # must treat as a change of answer rather than as an exception.
    report = target_sensitivity(_problem(margin_db=8.0))
    assert report.upper < 1.0
    assert report.parameter == "availability_target"


def test_infeasibility_counts_as_a_changed_signature():
    # At margin 8 dB and target 0.995 the lowest MODCOD just closes at
    # scintillation index 0.2; raise the scintillation enough and nothing in
    # the table closes. The sweep must record that boundary as a change of
    # answer with signature None, not raise out of the sweep.
    problem = _problem(margin_db=8.0, availability_target=0.995)
    report = scintillation_sensitivity(problem, span=8.0)
    assert report.signature is not None
    assert not report.bounded_above_by_scan
    assert report.boundary_signature_above is None
    with pytest.raises(Exception, match="unreachable"):
        select_rate(problem.with_fade(LognormalFade(report.upper + 1e-2)))


def test_flat_decision_reports_the_scan_bound_honestly():
    # A one-entry table has nothing to switch to, so the decision cannot
    # change anywhere; is_flat must be True and both endpoints must be
    # flagged as scan bounds rather than resolved boundaries.
    single = illustrative_modcod_table().entries[0:1]
    problem = _problem(
        modcods=type(illustrative_modcod_table())(single),
        availability_target=0.5,
    )
    report = scintillation_sensitivity(problem, span=3.0)
    assert report.is_flat
    assert report.bounded_below_by_scan and report.bounded_above_by_scan
    assert not report.is_knife_edge
    assert report.lower == pytest.approx(0.2 / 3.0)
    assert report.upper == pytest.approx(0.2 * 3.0)


def test_knife_edge_flag_agrees_with_a_direct_ten_percent_probe():
    problem = _problem()
    report = scintillation_sensitivity(problem, knife_edge_at=0.10)
    base = select_rate(problem).support
    direct = any(
        select_rate(problem.with_fade(LognormalFade(0.2 * f))).support != base
        for f in (0.9, 1.1)
    )
    assert report.is_knife_edge == direct
    # And the flag must be consistent with the interval it reports.
    if report.is_knife_edge:
        assert min(report.downward_headroom, report.upward_headroom) < 0.10 + 1e-9
    else:
        assert min(report.downward_headroom, report.upward_headroom) >= 0.10 - 1e-9


def test_a_knife_edge_instance_is_detectable():
    # Place the margin so that the chosen MODCOD is a hair from its boundary.
    problem = _problem(margin_db=12.0, availability_target=0.99)
    flat = scintillation_sensitivity(problem)
    boundary = flat.upper  # the scintillation index at which the answer flips
    knife = scintillation_sensitivity(problem.with_fade(LognormalFade(boundary * 0.999)))
    assert knife.upward_headroom < 0.01
    assert knife.is_knife_edge
    assert not flat.is_knife_edge
    assert knife.relative_width < flat.relative_width


def test_gamma_gamma_sensitivity_preserves_the_alpha_beta_ratio():
    problem = _problem(
        fade=GammaGammaFade.from_scintillation(0.3, ratio=0.5),
        availability_target=0.9,
    )
    report = scintillation_sensitivity(problem, grid_points=12)
    assert report.nominal == pytest.approx(0.3, rel=1e-9)
    assert report.lower < 0.3 < report.upper


def test_sensitivity_rejects_a_fade_model_without_a_scintillation_index():
    problem = _problem(fade=EmpiricalFade(np.linspace(-12.0, 2.0, 500)))
    with pytest.raises(TypeError, match="scintillation_index"):
        scintillation_sensitivity(problem)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"nominal": 5.0, "lower_bound": 6.0, "upper_bound": 7.0}, "strictly inside"),
        ({"nominal": 5.0, "lower_bound": 1.0, "upper_bound": 5.0}, "strictly inside"),
        ({"nominal": 5.0, "lower_bound": 1.0, "upper_bound": 9.0, "grid_points": 1}, "grid_points"),
        (
            {"nominal": 5.0, "lower_bound": 1.0, "upper_bound": 9.0, "knife_edge_at": 0.0},
            "knife_edge_at",
        ),
        (
            {"nominal": 5.0, "lower_bound": 1.0, "upper_bound": 9.0, "knife_edge_at": 1.0},
            "knife_edge_at",
        ),
    ],
)
def test_stability_interval_validation(kwargs, match):
    problem = _problem()
    with pytest.raises(ValueError, match=match):
        stability_interval(problem.with_margin, parameter="margin_db", **kwargs)


def test_narrower_tolerance_tightens_the_boundary_monotonically():
    problem = _problem()
    coarse = scintillation_sensitivity(problem, rel_tol=1e-2)
    fine = scintillation_sensitivity(problem, rel_tol=1e-6)
    # Bisection only ever keeps points where the signature is unchanged, so
    # refining can only move each endpoint outward, never inward.
    assert fine.upper >= coarse.upper - 1e-12
    assert fine.lower <= coarse.lower + 1e-12
    assert fine.upper - fine.lower >= coarse.upper - coarse.lower - 1e-12
