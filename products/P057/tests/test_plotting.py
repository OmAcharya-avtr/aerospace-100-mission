"""Plotting tests. Figures are written to a temporary directory, never shown."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

from conformalband.audit import breaking_point_sweep, coverage_audit, stratified_coverage
from conformalband.plotting import (
    METHOD_STYLE,
    plot_breaking_point,
    plot_coverage_audit,
    plot_coverage_bound,
    plot_interval_width,
    plot_model_comparison,
    plot_stratified_coverage,
    result_methods,
)

SMALL = {"replicates": 2, "n_fit": 400, "n_calibration": 120, "n_test": 120, "seed": 57901}


def test_backend_is_agg():
    assert matplotlib.get_backend().lower() == "agg"


def test_method_style_covers_every_method():
    from conformalband.audit import METHODS

    assert set(METHOD_STYLE) == set(METHODS)


@pytest.fixture(scope="module")
def audit():
    return coverage_audit(severities=(0.0, 2.0), **SMALL)


def test_result_methods_preserves_order(audit):
    from conformalband.audit import METHODS

    assert result_methods(audit) == list(METHODS)


def test_plot_coverage_audit_writes_a_png(tmp_path, audit):
    path = plot_coverage_audit(audit, tmp_path / "coverage.png")
    assert path.exists()
    assert path.stat().st_size > 5000


def test_plot_interval_width_writes_a_png(tmp_path, audit):
    path = plot_interval_width(audit, tmp_path / "width.png")
    assert path.exists()
    assert path.stat().st_size > 5000


def test_plot_coverage_audit_single_model(tmp_path):
    result = coverage_audit(severities=(0.0,), models=("physics",), **SMALL)
    path = plot_coverage_audit(result, tmp_path / "single.png")
    assert path.exists()


def test_plot_breaking_point_writes_a_png(tmp_path):
    sweep = breaking_point_sweep(
        true_severity=2.0,
        fractions=(0.0, 0.5, 1.0, 1.5),
        replicates=2,
        n_fit=400,
        n_calibration=120,
        n_test=120,
        seed=57902,
        model="physics",
    )
    path = plot_breaking_point(sweep, tmp_path / "breaking.png")
    assert path.exists()
    assert path.stat().st_size > 5000


def test_plot_stratified_coverage_writes_a_png(tmp_path):
    tally = stratified_coverage(
        replicates=1, n_fit=400, n_calibration=120, n_test=150, seed=57903, severity=2.0
    )
    path = plot_stratified_coverage(tally, tmp_path / "strata.png")
    assert path.exists()
    assert path.stat().st_size > 5000


def test_plot_coverage_bound_writes_a_png(tmp_path):
    path = plot_coverage_bound(tmp_path / "bound.png", alpha=0.1, max_n=60)
    assert path.exists()
    assert path.stat().st_size > 5000


def test_plot_model_comparison_writes_a_png(tmp_path):
    severities = np.array([0.0, 1.0, 2.0])
    rmse = {"physics": np.array([0.2, 0.21, 0.23]), "learned": np.array([0.19, 0.22, 0.27])}
    path = plot_model_comparison(severities, rmse, tmp_path / "models.png")
    assert path.exists()


def test_plot_creates_missing_directories(tmp_path):
    path = plot_coverage_bound(tmp_path / "nested" / "deeper" / "bound.png", max_n=40)
    assert path.exists()
