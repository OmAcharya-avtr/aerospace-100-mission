"""RateProblem construction, validation and derived quantities."""

from __future__ import annotations

import numpy as np
import pytest

from coderateopt import EmpiricalFade, LognormalFade, RateProblem, illustrative_modcod_table


def _problem(**overrides):
    kwargs = {
        "modcods": illustrative_modcod_table(),
        "fade": LognormalFade(0.2),
        "margin_db": 12.0,
        "availability_target": 0.99,
    }
    kwargs.update(overrides)
    return RateProblem(**kwargs)


def test_availabilities_and_goodputs_line_up_with_the_table(monotone_problem):
    # Empirical survival on [-6,-4,-2,0] at margin 10 dB:
    #   m1 thr 4  -> level -6 -> 4/4 = 1.00, goodput 1.0 * 1.00 = 1.0
    #   m2 thr 6  -> level -4 -> 3/4 = 0.75, goodput 2.0 * 0.75 = 1.5
    #   m3 thr 8  -> level -2 -> 2/4 = 0.50, goodput 4.0 * 0.50 = 2.0
    #   m4 thr 10 -> level  0 -> 1/4 = 0.25, goodput 8.0 * 0.25 = 2.0
    assert list(monotone_problem.availabilities()) == [1.0, 0.75, 0.5, 0.25]
    assert list(monotone_problem.goodputs()) == [1.0, 1.5, 2.0, 2.0]


def test_allowed_mask_depends_on_mode(monotone_problem):
    assert list(monotone_problem.allowed_mask()) == [True, True, False, False]
    relaxed = RateProblem(
        modcods=monotone_problem.modcods,
        fade=monotone_problem.fade,
        margin_db=10.0,
        availability_target=0.7,
        mode="long_run",
    )
    assert list(relaxed.allowed_mask()) == [True] * 4


@pytest.mark.parametrize(
    "overrides, exc, match",
    [
        ({"margin_db": float("nan")}, ValueError, "margin_db"),
        ({"availability_target": 0.0}, ValueError, "availability_target"),
        ({"availability_target": 1.0}, ValueError, "availability_target"),
        ({"availability_target": 1.5}, ValueError, "availability_target"),
        ({"max_entries": 0}, ValueError, "max_entries"),
        ({"max_entries": -3}, ValueError, "max_entries"),
        ({"max_entries": 2.0}, TypeError, "max_entries"),
        ({"max_entries": True}, TypeError, "max_entries"),
        ({"mode": "whatever"}, ValueError, "mode"),
        ({"min_dwell_fraction": 1.0}, ValueError, "min_dwell_fraction"),
        ({"min_dwell_fraction": -0.1}, ValueError, "min_dwell_fraction"),
        ({"modcods": "not a set"}, TypeError, "ModcodSet"),
        ({"fade": object()}, TypeError, "AvailabilityModel"),
    ],
)
def test_validation(overrides, exc, match):
    with pytest.raises(exc, match=match):
        _problem(**overrides)


def test_effective_k_is_capped_by_table_size_and_dwell():
    assert _problem(max_entries=50).effective_k == 9
    assert _problem(max_entries=4, min_dwell_fraction=0.3).effective_k == 3
    assert _problem(max_entries=9, min_dwell_fraction=0.9).effective_k == 1
    assert _problem(max_entries=3).effective_k == 3


def test_with_helpers_preserve_every_other_field():
    base = _problem(max_entries=3, mode="long_run", min_dwell_fraction=0.05)
    for variant in (
        base.with_fade(LognormalFade(0.5)),
        base.with_margin(20.0),
        base.with_target(0.5),
    ):
        assert variant.max_entries == 3
        assert variant.mode == "long_run"
        assert variant.min_dwell_fraction == 0.05
        assert variant.modcods is base.modcods
    assert base.with_margin(20.0).margin_db == 20.0
    assert base.with_target(0.5).availability_target == 0.5
    assert base.with_fade(LognormalFade(0.5)).fade.scintillation_index == 0.5


def test_empirical_fade_problem_is_constructible():
    problem = _problem(fade=EmpiricalFade(np.linspace(-10.0, 2.0, 1000)))
    assert problem.n_modcods == 9
    assert np.all(np.diff(problem.availabilities()) <= 0.0)
