"""The acquisition-time distribution: two independent moment paths, Monte Carlo, bounds."""

from __future__ import annotations

import numpy as np
import pytest

from handoverloss.acquisition import AcquisitionModel


@pytest.fixture
def model() -> AcquisitionModel:
    return AcquisitionModel(
        median_s=20.0, log_sigma=0.5, abort_probability=0.2, attempt_window_s=100.0
    )


def test_closed_form_and_quadrature_moments_agree(model):
    # Two paths to the same two numbers. The closed form uses the lognormal
    # moment identity and the NORMAL DISTRIBUTION function; the quadrature uses
    # the lognormal DENSITY on a Gauss-Legendre grid. They share the
    # distribution family and nothing else, so agreement is evidence about the
    # algebra rather than a restatement of it.
    closed = model.moments()
    quadrature = model.moments_quadrature(512)
    assert closed[0] == pytest.approx(quadrature[0], rel=1e-9, abs=0.0)
    assert closed[1] == pytest.approx(quadrature[1], rel=1e-9, abs=0.0)


def test_moments_agree_with_monte_carlo(model, rng):
    # A third path: 200 000 draws. The agreement tolerance is the Monte Carlo
    # standard error of the mean, 4 sigma wide, not an arbitrary constant.
    draws = model.sample(200_000, rng)
    mean, sd = model.moments()
    standard_error = sd / np.sqrt(draws.size)
    assert abs(float(draws.mean()) - mean) <= 4.0 * standard_error
    # The standard deviation's own error is larger; 8 per cent is a loose but
    # declared band and the measured discrepancy is reported in
    # validation/validate_acquisition_model.py.
    assert float(draws.std()) == pytest.approx(sd, rel=0.08, abs=0.0)


def test_zero_abort_probability_gives_the_truncated_body_alone():
    # With p = 0 the geometric count is identically zero, so the mean is the
    # truncated body's own mean and must lie strictly below the window.
    model = AcquisitionModel(
        median_s=20.0, log_sigma=0.5, abort_probability=0.0, attempt_window_s=100.0
    )
    mean, _ = model.moments()
    assert 0.0 < mean < 100.0


def test_abort_probability_raises_the_mean_monotonically():
    previous = 0.0
    for probability in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        model = AcquisitionModel(
            median_s=20.0,
            log_sigma=0.5,
            abort_probability=probability,
            attempt_window_s=100.0,
        )
        mean, _ = model.moments()
        assert mean > previous
        previous = mean


def test_samples_are_non_negative_and_reproducible(model):
    first = model.sample(5000, np.random.default_rng(11))
    second = model.sample(5000, np.random.default_rng(11))
    assert np.all(first >= 0.0)
    # Same seed, same draws. This is an identity of the generator, not of a
    # floating-point pipeline, so bit-exact equality is the right assertion.
    assert np.array_equal(first, second)


def test_successful_attempt_cannot_exceed_its_window(model, rng):
    # Every draw is k * w + L with L <= w, so draws modulo w never exceed w and
    # the minimum draw is strictly positive.
    draws = model.sample(20_000, rng)
    assert float(draws.min()) > 0.0
    residual = draws % model.attempt_window_s
    assert float(residual.max()) <= model.attempt_window_s


def test_quantile_is_monotone(model):
    low = model.quantile(0.5, size=40_000)
    high = model.quantile(0.95, size=40_000)
    assert low < high


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (dict(median_s=0.0), "median_s"),
        (dict(median_s=-1.0), "median_s"),
        (dict(log_sigma=0.0), "log_sigma"),
        (dict(abort_probability=1.0), "abort_probability"),
        (dict(abort_probability=-0.1), "abort_probability"),
        (dict(attempt_window_s=20.0), "attempt_window_s"),
        (dict(attempt_window_s=5.0), "attempt_window_s"),
    ],
)
def test_invalid_parameters_raise(kwargs, message):
    base = dict(
        median_s=20.0, log_sigma=0.5, abort_probability=0.2, attempt_window_s=100.0
    )
    base.update(kwargs)
    with pytest.raises(ValueError, match=message):
        AcquisitionModel(**base)


def test_non_finite_parameter_raises():
    with pytest.raises(ValueError, match="finite"):
        AcquisitionModel(
            median_s=float("nan"),
            log_sigma=0.5,
            abort_probability=0.2,
            attempt_window_s=100.0,
        )


def test_negative_sample_size_raises(model, rng):
    with pytest.raises(ValueError, match="non-negative"):
        model.sample(-1, rng)


def test_quadrature_point_floor(model):
    with pytest.raises(ValueError, match="at least 16"):
        model.moments_quadrature(8)


def test_quantile_level_bounds(model):
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        model.quantile(0.0)
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        model.quantile(1.0)
