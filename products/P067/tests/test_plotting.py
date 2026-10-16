"""Plotting: the figures are produced, on the Agg backend, with no display."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

from acqsearch import (
    coverage_curve,
    optimal_dwell_time,
    score_policy,
    standard_paths,
)
from acqsearch.plotting import plot_coverage_curves, plot_dwell_tradeoff, plot_policy_comparison
from acqsearch.policy import EpisodeResult


def test_the_agg_backend_is_in_use():
    assert matplotlib.get_backend().lower() == "agg"


def test_coverage_figure_is_written(tmp_path, grid, uncertainty, geometry):
    curves = {
        name: coverage_curve(path, uncertainty, geometry.footprint_radius_rad, 3000, seed=1)
        for name, path in standard_paths(grid).items()
    }
    out = plot_coverage_curves(curves, tmp_path / "coverage.png", closed_form=(50, 0.8))
    assert out.exists()
    assert out.stat().st_size > 10000


def test_dwell_figure_is_written(tmp_path, grid, spiral_order, geometry, detector, uncertainty):
    optimum = optimal_dwell_time(
        grid, spiral_order, geometry, detector, uncertainty, np.geomspace(4e-4, 4e-3, 7), 0.5
    )
    out = plot_dwell_tradeoff(optimum, tmp_path / "dwell.png")
    assert out.exists()
    assert out.stat().st_size > 10000


def test_policy_figure_is_written(tmp_path):
    scores = {
        "a": score_policy("a", [EpisodeResult(1.0, True, 4), EpisodeResult(2.0, True, 8)]),
        "b": score_policy("b", [EpisodeResult(3.0, True, 9), EpisodeResult(4.0, False, 11)]),
    }
    out = plot_policy_comparison(scores, tmp_path / "policies.png")
    assert out.exists()
    assert out.stat().st_size > 10000


def test_plotting_creates_the_parent_directory(tmp_path, grid, uncertainty, geometry):
    curve = coverage_curve(
        standard_paths(grid)["spiral"], uncertainty, geometry.footprint_radius_rad, 1000, seed=2
    )
    out = plot_coverage_curves({"spiral": curve}, tmp_path / "nested" / "deep" / "c.png")
    assert out.exists()


def test_coverage_figure_handles_an_empty_curve_set(tmp_path):
    out = plot_coverage_curves({}, tmp_path / "empty.png")
    assert out.exists()


def test_policy_figure_rejects_nothing_but_still_needs_scores(tmp_path):
    with pytest.raises(ValueError):
        score_policy("x", [])
