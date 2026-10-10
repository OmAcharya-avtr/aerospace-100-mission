"""Input-validation tests: every public entry point refuses bad input loudly."""

from __future__ import annotations

import numpy as np
import pytest

from conformalband.audit import breaking_point_sweep, coverage_audit
from conformalband.conformal import (
    MondrianConformal,
    SplitConformal,
    WeightedConformal,
    absolute_residual_score,
    conformal_rank,
    weighted_quantile,
)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.01, 2.0])
@pytest.mark.parametrize("cls", [SplitConformal, MondrianConformal, WeightedConformal])
def test_conformal_classes_reject_bad_alpha(cls, alpha):
    with pytest.raises(ValueError, match="alpha"):
        cls(alpha)


def test_absolute_residual_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="same shape"):
        absolute_residual_score(np.zeros(4), np.zeros(5))


def test_absolute_residual_rejects_empty():
    with pytest.raises(ValueError, match="non-empty"):
        absolute_residual_score(np.array([]), np.array([]))


def test_split_calibrate_refuses_too_few_points():
    with pytest.raises(ValueError, match="too small"):
        SplitConformal(0.1).calibrate(np.zeros(5), np.arange(5.0))


def test_mondrian_rejects_mismatched_bins():
    with pytest.raises(ValueError, match="bins must have shape"):
        MondrianConformal(0.1).calibrate(np.zeros(20), np.arange(20.0), np.zeros(19, dtype=int))


def test_mondrian_interval_rejects_mismatched_bins():
    model = MondrianConformal(0.1).calibrate(np.zeros(20), np.arange(20.0), np.zeros(20, dtype=int))
    with pytest.raises(ValueError, match="bins must have shape"):
        model.interval(np.zeros(3), np.zeros(2, dtype=int))


def test_weighted_rejects_mismatched_weights():
    with pytest.raises(ValueError, match="weights must have shape"):
        WeightedConformal(0.1).calibrate(np.zeros(20), np.arange(20.0), np.ones(19))


def test_weighted_rejects_negative_weights():
    with pytest.raises(ValueError, match="non-negative"):
        WeightedConformal(0.1).calibrate(np.zeros(20), np.arange(20.0), -np.ones(20))


def test_weighted_rejects_non_finite_weights():
    weights = np.ones(20)
    weights[3] = np.inf
    with pytest.raises(ValueError, match="finite"):
        WeightedConformal(0.1).calibrate(np.zeros(20), np.arange(20.0), weights)


def test_weighted_rejects_all_zero_weights():
    with pytest.raises(ValueError, match="all be zero"):
        WeightedConformal(0.1).calibrate(np.zeros(20), np.arange(20.0), np.zeros(20))


def test_weighted_rejects_negative_test_weights():
    model = WeightedConformal(0.1).calibrate(np.zeros(20), np.arange(20.0), np.ones(20))
    with pytest.raises(ValueError, match="non-negative"):
        model.quantiles(np.array([-1.0]))


def test_weighted_rejects_non_finite_test_weights():
    model = WeightedConformal(0.1).calibrate(np.zeros(20), np.arange(20.0), np.ones(20))
    with pytest.raises(ValueError, match="finite"):
        model.quantiles(np.array([np.nan]))


def test_weighted_interval_rejects_shape_mismatch():
    model = WeightedConformal(0.1).calibrate(np.zeros(20), np.arange(20.0), np.ones(20))
    with pytest.raises(ValueError, match="must agree"):
        model.interval(np.zeros(3), np.ones(2))


def test_weighted_quantile_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        weighted_quantile(np.zeros(3), np.ones(4), 0.9)


def test_weighted_quantile_rejects_empty():
    with pytest.raises(ValueError, match="non-empty"):
        weighted_quantile(np.array([]), np.array([]), 0.9)


def test_weighted_quantile_rejects_negative_weights():
    with pytest.raises(ValueError, match="non-negative"):
        weighted_quantile(np.ones(3), -np.ones(3), 0.9)


def test_weighted_quantile_rejects_negative_tail_weight():
    with pytest.raises(ValueError, match="tail_weight"):
        weighted_quantile(np.ones(3), np.ones(3), 0.9, tail_weight=-1.0)


@pytest.mark.parametrize("level", [0.0, 1.0, -0.5, 1.5])
def test_weighted_quantile_rejects_bad_level(level):
    with pytest.raises(ValueError, match="level"):
        weighted_quantile(np.ones(3), np.ones(3), level)


def test_weighted_quantile_rejects_zero_total_weight():
    with pytest.raises(ValueError, match="total weight"):
        weighted_quantile(np.ones(3), np.zeros(3), 0.9)


@pytest.mark.parametrize("n", [0, -5])
def test_conformal_rank_rejects_bad_n(n):
    with pytest.raises(ValueError, match="n_calibration"):
        conformal_rank(n, 0.1)


def test_conformal_rank_rejects_bad_alpha():
    with pytest.raises(ValueError, match="alpha"):
        conformal_rank(100, 1.0)


def test_coverage_audit_rejects_zero_replicates():
    with pytest.raises(ValueError, match="replicates"):
        coverage_audit(replicates=0)


def test_coverage_audit_rejects_unknown_method():
    with pytest.raises(ValueError, match="unknown methods"):
        coverage_audit(replicates=1, methods=("nope",))


def test_coverage_audit_rejects_unknown_model():
    with pytest.raises(ValueError, match="unknown models"):
        coverage_audit(replicates=1, models=("nope",))


def test_coverage_audit_rejects_tiny_calibration():
    with pytest.raises(ValueError, match="too small"):
        coverage_audit(replicates=1, n_calibration=5)


def test_breaking_point_rejects_zero_severity():
    with pytest.raises(ValueError, match="true_severity"):
        breaking_point_sweep(true_severity=0.0)


def test_breaking_point_rejects_negative_fraction():
    with pytest.raises(ValueError, match="fractions"):
        breaking_point_sweep(true_severity=1.0, fractions=(-0.1, 0.5))
