"""The assembled report, and the two methods that exist only to refuse."""

from __future__ import annotations

import numpy as np
import pytest
from sgp4.api import Satrec

from orbitdrift import build_report, model_divergence
from orbitdrift.errors import CovarianceRefused
from orbitdrift.guard import COVARIANCE_KEY_PATTERN


@pytest.fixture(scope="module")
def report(moderate_drag_satrec: Satrec):
    curve = model_divergence(
        moderate_drag_satrec,
        span_days=1.5,
        steps_per_rev=120,
        report_every_hours=6.0,
        sample_every=4,
    )
    return build_report(moderate_drag_satrec, curve)


def test_report_carries_the_object_and_its_curve(report) -> None:
    assert report.satnum == 6251
    assert report.curve.source == "sgp4-vs-numerical"
    assert report.features.perigee_altitude_km > 300.0
    assert report.curve.ages_days.size == report.curve.drift_km.size


def test_lower_bound_is_half_the_drift(report) -> None:
    assert np.allclose(report.lower_bound_km, 0.5 * report.curve.drift_km)


def test_best_form_is_one_of_the_fitted_forms(report) -> None:
    assert report.best_form_name in report.fits
    assert len(report.fits) >= 3


def test_as_covariance_always_raises_and_names_p071(report) -> None:
    with pytest.raises(CovarianceRefused) as caught:
        report.as_covariance()
    assert "P071" in str(caught.value)
    assert "not a state covariance" in str(caught.value)


def test_to_dict_has_no_covariance_shaped_key_at_any_depth(report) -> None:
    payload = report.to_dict()

    def walk(mapping, prefix=""):
        for key, value in mapping.items():
            assert not COVARIANCE_KEY_PATTERN.search(str(key)), f"{prefix}{key}"
            if isinstance(value, dict):
                walk(value, f"{prefix}{key}.")

    walk(payload)


def test_to_dict_states_what_the_quantity_is_and_is_not(report) -> None:
    """The serialised payload must carry its own disclaimer.

    A report that travels without the sentence saying it is not a covariance
    is a report that can be mistaken for one.
    """
    payload = report.to_dict()
    assert "not a covariance" in payload["quantity"]
    assert "lower bound" in payload["quantity"]
    assert len(payload["drift_km"]) == len(payload["ages_days"])
    assert payload["best_form"] in report.fits


def test_to_dict_audit_fires_if_a_covariance_key_is_ever_added(report) -> None:
    """The guard is live, not decorative.

    ``to_dict`` audits its own output before returning it, so a future field
    named ``position_covariance`` would raise here. This test demonstrates the
    audit on the report's own payload with such a key inserted, which is the
    negative control for the test above.
    """
    from orbitdrift.guard import audit_mapping_for_covariance_keys

    payload = report.to_dict()
    payload["position_covariance"] = [[1.0, 0.0], [0.0, 1.0]]
    with pytest.raises(CovarianceRefused, match="covariance-shaped key"):
        audit_mapping_for_covariance_keys(payload, where="DriftReport")


def test_summary_names_units_on_every_row(report) -> None:
    text = report.summary()
    assert "epoch age  model divergence" in text
    assert "lower bound on larger model error" in text
    assert "1/earth-radii" in text
    assert "best fitting form" in text
    assert text.count("\n") > 10


def test_summary_includes_the_verdict_when_there_is_one(
    moderate_drag_satrec: Satrec, report
) -> None:
    from orbitdrift.features import features_from_satrec
    from orbitdrift.staleness import staleness_verdict

    class _Flat:
        name = "flat"

        def predict_median_km(self, x):
            return np.atleast_2d(np.asarray(x, dtype=float))[:, -1] * 2.0

        def predict_interval_km(self, x, coverage=0.90):
            del coverage
            centre = self.predict_median_km(x)
            return centre * 0.5, centre * 2.0

    verdict = staleness_verdict(
        _Flat(), features_from_satrec(moderate_drag_satrec), 10.0, measured_coverage=0.88
    )
    with_verdict = build_report(moderate_drag_satrec, report.curve, verdict=verdict)
    text = with_verdict.summary()
    assert "conservative horizon" in text
    assert "0.8800 measured" in text
    payload = with_verdict.to_dict()
    assert payload["staleness"]["measured_coverage"] == 0.88
