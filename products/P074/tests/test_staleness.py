"""The staleness verdict, its refusals, and what it says when it has no backing."""

from __future__ import annotations

import numpy as np
import pytest

from orbitdrift.errors import CovarianceRefused, InsufficientDataError
from orbitdrift.features import FEATURE_NAMES, ObjectFeatures
from orbitdrift.staleness import StalenessVerdict, _first_crossing, staleness_verdict


class _PowerLawPredictor:
    """``drift = amplitude * t^exponent`` with a fixed multiplicative band."""

    name = "test_power_law"

    def __init__(self, amplitude: float = 1.0, exponent: float = 1.5, band: float = 2.0) -> None:
        self.amplitude = amplitude
        self.exponent = exponent
        self.band = band

    def predict_median_km(self, x: np.ndarray) -> np.ndarray:
        ages = np.atleast_2d(np.asarray(x, dtype=float))[:, -1]
        return self.amplitude * ages**self.exponent

    def predict_interval_km(self, x: np.ndarray, coverage: float = 0.90):
        del coverage
        centre = self.predict_median_km(x)
        return centre / self.band, centre * self.band


def _features(perigee_km: float = 500.0) -> ObjectFeatures:
    values = dict.fromkeys(FEATURE_NAMES, 0.0)
    values["perigee_altitude_km"] = perigee_km
    values["apogee_altitude_km"] = perigee_km + 20.0
    values["eccentricity"] = 0.002
    values["inclination_deg"] = 51.6
    values["bstar"] = 1e-4
    values["log10_abs_bstar"] = -4.0
    values["bstar_sign"] = 1.0
    values["period_min"] = 95.0
    values["log10_perigee_density"] = -12.0
    values["log10_drag_parameter"] = -15.0
    return ObjectFeatures(**values)


def test_first_crossing_interpolates_linearly() -> None:
    """Hand-calculated: between (1, 2) and (2, 6) the value 4 is at 1.5."""
    ages = np.array([1.0, 2.0, 3.0])
    values = np.array([2.0, 6.0, 12.0])
    assert _first_crossing(ages, values, 4.0) == pytest.approx(1.5, rel=1e-14, abs=0.0)
    assert _first_crossing(ages, values, 2.0) == 1.0
    assert _first_crossing(ages, values, 100.0) == float("inf")


def test_conservative_horizon_is_never_later_than_the_median_horizon() -> None:
    """The upper interval bound reaches any threshold first, by construction.

    ``upper >= median`` pointwise, and both are increasing in age, so the
    conservative horizon cannot be the later of the two. If it ever were, the
    verdict would be telling a user an element set is good for longer than the
    point prediction allows.
    """
    verdict = staleness_verdict(_PowerLawPredictor(), _features(), tolerance_km=5.0)
    assert verdict.conservative_horizon_days <= verdict.median_horizon_days
    assert np.isfinite(verdict.conservative_horizon_days)


def test_horizons_against_a_hand_solved_power_law() -> None:
    """``t^1.5 = 5`` gives ``t = 5^(2/3) = 2.924018...`` days.

    With a band factor of 2 the upper bound reaches 5 when
    ``2 t^1.5 = 5``, so ``t = 2.5^(2/3) = 1.842015...`` days. The grid is
    0.05 day, so the interpolated crossing must match to about that spacing
    squared times the curvature; 2e-3 is comfortable and is still a real
    check on the interpolation.
    """
    verdict = staleness_verdict(
        _PowerLawPredictor(amplitude=1.0, exponent=1.5, band=2.0),
        _features(),
        tolerance_km=5.0,
    )
    assert verdict.median_horizon_days == pytest.approx(5.0 ** (2.0 / 3.0), rel=0.0, abs=2e-3)
    assert verdict.conservative_horizon_days == pytest.approx(
        2.5 ** (2.0 / 3.0), rel=0.0, abs=2e-3
    )


def test_still_good_is_none_when_no_age_was_given_rather_than_false() -> None:
    """A missing input is not a negative answer."""
    verdict = staleness_verdict(_PowerLawPredictor(), _features(), tolerance_km=5.0)
    assert verdict.still_good is None
    assert "element set age" not in verdict.describe()


def test_still_good_decides_against_the_conservative_horizon() -> None:
    predictor = _PowerLawPredictor(amplitude=1.0, exponent=1.5, band=2.0)
    inside = staleness_verdict(predictor, _features(), 5.0, age_days=1.0)
    outside = staleness_verdict(predictor, _features(), 5.0, age_days=4.0)
    assert inside.still_good is True
    assert outside.still_good is False
    assert "WITHIN" in inside.describe()
    assert "OUTSIDE" in outside.describe()


def test_describe_says_the_horizon_is_unbacked_when_no_coverage_was_measured() -> None:
    """The default is to admit there is no measured coverage, not to omit it."""
    verdict = staleness_verdict(_PowerLawPredictor(), _features(), 5.0)
    text = verdict.describe()
    assert "not supplied" in text
    assert "must not be relied on" in text


def test_describe_reports_a_measured_under_coverage_as_a_warning() -> None:
    verdict = staleness_verdict(
        _PowerLawPredictor(), _features(), 5.0, coverage=0.90, measured_coverage=0.72
    )
    text = verdict.describe()
    assert "0.7200 measured" in text
    assert "UNDER-COVERS" in text


def test_describe_does_not_warn_when_the_measured_coverage_meets_the_claim() -> None:
    verdict = staleness_verdict(
        _PowerLawPredictor(), _features(), 5.0, coverage=0.90, measured_coverage=0.91
    )
    assert "UNDER-COVERS" not in verdict.describe()


def test_a_conjunction_purpose_is_refused() -> None:
    with pytest.raises(CovarianceRefused, match="P071"):
        staleness_verdict(
            _PowerLawPredictor(), _features(), 5.0, purpose="covariance for conjunction screening"
        )


def test_a_legitimate_purpose_is_accepted() -> None:
    """The negative control for the purpose screen."""
    verdict = staleness_verdict(
        _PowerLawPredictor(), _features(), 5.0, purpose="pass prediction for a ground station"
    )
    assert np.isfinite(verdict.median_horizon_days)


def test_as_covariance_always_raises() -> None:
    verdict = staleness_verdict(_PowerLawPredictor(), _features(), 5.0)
    with pytest.raises(CovarianceRefused, match="not a state covariance"):
        verdict.as_covariance()


def test_a_non_positive_tolerance_is_refused() -> None:
    for tolerance in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(InsufficientDataError, match="positive and finite"):
            staleness_verdict(_PowerLawPredictor(), _features(), tolerance)


def test_an_unreachable_tolerance_gives_an_infinite_horizon() -> None:
    """A tolerance the drift never reaches within 14 days must report infinity.

    Not the grid endpoint: reporting 14 days would be a claim that the
    element set goes stale then, which the computation did not establish.
    """
    verdict = staleness_verdict(
        _PowerLawPredictor(amplitude=1e-6, exponent=1.0), _features(), tolerance_km=1000.0
    )
    assert verdict.median_horizon_days == float("inf")
    assert verdict.conservative_horizon_days == float("inf")


def test_verdict_carries_the_predictor_name_and_the_nominal_level() -> None:
    verdict = staleness_verdict(_PowerLawPredictor(), _features(), 5.0, coverage=0.80)
    assert verdict.predictor_name == "test_power_law"
    assert verdict.nominal_coverage == 0.80
    assert isinstance(verdict, StalenessVerdict)


def test_a_lower_tolerance_gives_an_earlier_horizon() -> None:
    """Monotonicity: a tighter requirement cannot be satisfied for longer."""
    predictor = _PowerLawPredictor()
    horizons = [
        staleness_verdict(predictor, _features(), tolerance).conservative_horizon_days
        for tolerance in (1.0, 5.0, 20.0)
    ]
    assert horizons[0] < horizons[1] < horizons[2]
