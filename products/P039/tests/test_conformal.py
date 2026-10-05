"""Split-conformal calibration."""

from __future__ import annotations

import math

import numpy as np
import pytest

from latencynet.conformal import ConformalPredictor, conformal_radius, coverage_quantisation
from latencynet.dataset import log_target
from latencynet.linear import LinearTailPredictor
from latencynet.predictors import AnalyticTailPredictor


def test_radius_known_answer():
    # Ten residuals 1..10, level 0.9: k = ceil(11 * 0.9) = ceil(9.9) = 10, so
    # the radius is the 10th smallest residual, which is 10.
    r = np.arange(1.0, 11.0)
    assert conformal_radius(r, 0.9) == 10.0
    # level 0.5: k = ceil(5.5) = 6 -> the 6th smallest, 6.
    assert conformal_radius(r, 0.5) == 6.0


def test_radius_is_infinite_when_calibration_is_too_small():
    # m = 4, level 0.9: k = ceil(5 * 0.9) = 5 > 4, so no finite radius can be
    # certified and the honest answer is infinity.
    assert math.isinf(conformal_radius(np.arange(1.0, 5.0), 0.9))


def test_coverage_quantisation_known_answer():
    # 1 / (m + 1): m = 60 -> 1/61 = 0.016393442...
    assert coverage_quantisation(60) == pytest.approx(1.0 / 61.0, rel=1e-14)
    assert coverage_quantisation(1) == 0.5


def test_conformal_coverage_on_held_out_data():
    # The guarantee is MARGINAL: coverage averaged over calibration draws as
    # well as test draws lies in [0.9, 0.9 + 1/(m+1)]. Conditional on one
    # calibration draw the coverage has a standard deviation of about
    # sqrt(0.9 * 0.1 / (m + 1)) = 0.021 at m = 200, so a single draw says
    # little. Averaging 80 independent calibration draws of 400 test points
    # each reduces the standard error of the mean to about 0.003, and the
    # measured mean must then sit on the nominal value within a few of those.
    rng = np.random.default_rng(7)
    coverages = []
    for _ in range(80):
        d = conformal_radius(np.abs(rng.normal(size=200)), 0.9)
        coverages.append(float(np.mean(np.abs(rng.normal(size=400)) <= d)))
    assert float(np.mean(coverages)) == pytest.approx(0.9, abs=0.015)


def test_wrapper_end_to_end(tiny_dataset):
    base = LinearTailPredictor()
    wrapper = ConformalPredictor(base, level=0.9)
    wrapper.fit(tiny_dataset.train, 0.99).calibrate(tiny_dataset.calibration)
    assert wrapper.name == "linear_ols+conformal"
    assert wrapper.n_calibration == len(tiny_dataset.calibration)
    assert wrapper.radius is not None and wrapper.radius > 0.0
    pred = wrapper.predict_log_interval(tiny_dataset.test)
    # A conformal interval has the same width everywhere by construction.
    assert np.allclose(pred.log_width, pred.log_width[0], rtol=1e-14)
    # Point predictions are the base model's, unchanged.
    assert np.array_equal(
        wrapper.predict_log(tiny_dataset.test), base.predict_log(tiny_dataset.test)
    )


def test_wrapper_works_on_the_analytic_baseline(tiny_dataset):
    wrapper = ConformalPredictor(AnalyticTailPredictor(), level=0.8)
    wrapper.fit(tiny_dataset.train, 0.99).calibrate(tiny_dataset.calibration)
    pred = wrapper.predict_log_interval(tiny_dataset.test)
    truth = log_target(tiny_dataset.test, 0.99)
    assert pred.level == 0.8
    assert np.all(np.isfinite(pred.log_lower))
    # Not asserting a coverage rate on 14 test pipelines; that is what
    # validation/validate_interval_coverage.py is for.
    assert truth.shape == pred.log_point.shape


def test_from_fitted(tiny_dataset):
    base = AnalyticTailPredictor().fit(tiny_dataset.train, 0.99)
    wrapper = ConformalPredictor.from_fitted(base, 0.99, level=0.9)
    wrapper.calibrate(tiny_dataset.calibration)
    assert wrapper.radius is not None


def test_ordering_requirements(tiny_dataset):
    wrapper = ConformalPredictor(LinearTailPredictor(), level=0.9)
    with pytest.raises(RuntimeError, match="before calibrate"):
        wrapper.calibrate(tiny_dataset.calibration)
    wrapper.fit(tiny_dataset.train, 0.99)
    with pytest.raises(RuntimeError, match="before predict_log_interval"):
        wrapper.predict_log_interval(tiny_dataset.test)
    wrapper.calibrate(tiny_dataset.calibration)
    with pytest.raises(ValueError, match="calibrated for level"):
        wrapper.predict_log_interval(tiny_dataset.test, 0.5)


def test_validation(tiny_dataset):
    with pytest.raises(TypeError, match="must be a TailPredictor"):
        ConformalPredictor("not a model")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="level"):
        ConformalPredictor(LinearTailPredictor(), level=0.0)
    with pytest.raises(ValueError, match="non-empty"):
        conformal_radius(np.array([]), 0.9)
    with pytest.raises(ValueError, match="non-negative"):
        conformal_radius(np.array([-1.0]), 0.9)
    with pytest.raises(ValueError, match="level"):
        conformal_radius(np.array([1.0]), 1.0)
    with pytest.raises(ValueError, match="n_calibration"):
        coverage_quantisation(0)
    wrapper = ConformalPredictor(LinearTailPredictor(), level=0.9).fit(tiny_dataset.train, 0.99)
    with pytest.raises(ValueError, match="non-empty"):
        wrapper.calibrate(())
