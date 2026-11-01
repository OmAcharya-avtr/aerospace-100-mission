"""Unit and input-validation tests for the plain-text renderers."""

from __future__ import annotations

import numpy as np
import pytest

from transfergap.bounds import hoeffding_mean_upper, normal_tolerance_upper
from transfergap.exchangeability import permutation_test
from transfergap.plan import plan_existence, plan_for_slack
from transfergap.report import (
    format_table,
    render_bounds,
    render_coverage,
    render_exchangeability,
    render_plan,
)
from transfergap.scoring import CoverageScore


def test_format_table_pads_columns_to_their_widest_cell():
    text = format_table(["a", "bbbb"], [["xxxxx", "y"]])
    lines = text.splitlines()
    assert lines[0].startswith("a      bbbb")
    assert lines[1] == "-----  ----"
    assert lines[2] == "xxxxx  y   "


def test_format_table_rejects_a_row_of_the_wrong_length():
    with pytest.raises(ValueError, match="row 0 has 1 cells"):
        format_table(["a", "b"], [["only"]])


def test_format_table_handles_no_rows():
    text = format_table(["a", "b"], [])
    assert text.splitlines() == ["a  b", "-  -"]


def test_render_bounds_marks_a_non_distribution_free_row(field_sample):
    bounds = [
        hoeffding_mean_upper(field_sample.gaps, field_sample.bound_range, 0.95, "kN.m"),
        normal_tolerance_upper(field_sample.gaps, field_sample.bound_range, 0.95, 0.95, "kN.m"),
    ]
    text = render_bounds(bounds, "kN.m")
    assert "dist-free" in text
    assert "NO" in text
    assert "not comparable" in text


def test_render_exchangeability_states_both_directions():
    tests = [permutation_test(np.arange(40.0), "trend", permutations=199, seed=1)]
    text = render_exchangeability(tests)
    assert "REJECTED" in text
    assert "evidence FOR exchangeability" in text


def test_render_plan_prints_a_note_per_row(small_field):
    plans = plan_for_slack(small_field, 2.0, 0.95) + plan_existence(small_field, 0.95, 0.95)
    text = render_plan(plans)
    for plan in plans:
        assert f"{plan.method}:" in text
    assert "MORE" in text


def test_render_coverage_puts_coverage_before_width():
    score = CoverageScore(
        method="hoeffding", target="mean", n=40, repetitions=300, declared=0.95,
        coverage=0.9833, coverage_se=0.0074, tolerance_rate=None, tolerance_rate_se=None,
        mean_upper=13.5, mean_width=7.74, failures=0, unit="kN.m", distribution_free=True,
    )
    text = render_coverage([score])
    header = text.splitlines()[0]
    assert header.index("coverage") < header.index("mean upper")
    assert "covers" in text
    assert "COVERAGE ONLY" in text


def test_render_coverage_flags_an_under_covering_method():
    score = CoverageScore(
        method="learned_quantile", target="quantile", n=40, repetitions=300,
        declared=0.95, coverage=0.84, coverage_se=0.005, tolerance_rate=0.02,
        tolerance_rate_se=0.008, mean_upper=9.0, mean_width=9.0, failures=0,
        unit="kN.m", distribution_free=False,
    )
    text = render_coverage([score])
    assert "UNDER-COVERS" in text
    assert score.covers is False


def test_coverage_score_verdict_allows_two_standard_errors():
    """A rate of 0.94 against 0.95 with se 0.0126 is not measurably below.

    0.95 - 2 * 0.0126 = 0.9248, and 0.94 is above it. The allowance exists
    because a strict comparison of a Monte Carlo rate against a threshold is a
    boolean produced by a threshold comparison on a random quantity.
    """
    score = CoverageScore(
        method="x", target="mean", n=40, repetitions=300, declared=0.95, coverage=0.94,
        coverage_se=0.0126, tolerance_rate=None, tolerance_rate_se=None, mean_upper=1.0,
        mean_width=1.0, failures=0, unit="kN.m", distribution_free=True,
    )
    assert score.covers is True
