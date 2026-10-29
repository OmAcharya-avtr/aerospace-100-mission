"""Robust element-history outlier tests, their estimators and their refusals."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tlecheck.elements import format_tle, parse_tle
from tlecheck.history import (
    LEVEL_ELEMENTS,
    MAD_CONSISTENCY,
    MINIMUM_HISTORY,
    QUANTISATION,
    TREND_ELEMENTS,
    UNTESTED_ELEMENTS,
    check,
    modified_z,
    robust_scale,
    scale_floor,
    theil_sen,
    unwrap_degrees,
)

from conftest import REFERENCE_FIELDS


def _set(**overrides):
    fields = dict(REFERENCE_FIELDS)
    fields.update(overrides)
    return parse_tle(*format_tle(**fields))


def _history(count: int = 8, **drift):
    out = []
    for index in range(count):
        fields = dict(REFERENCE_FIELDS)
        fields["epoch_day"] = 300.0 + 1.5 * index
        for name, rate in drift.items():
            fields[name] = fields[name] + rate * index
        out.append(parse_tle(*format_tle(**fields)))
    return out


def test_consistency_constant_is_the_published_one():
    # Rousseeuw and Croux (1993) section 2: 1 / Phi^{-1}(0.75) = 1.4826.
    assert MAD_CONSISTENCY == 1.4826
    assert MAD_CONSISTENCY == pytest.approx(
        1.0 / 0.6744897501960817, rel=2.0e-5, abs=0.0
    )


def test_robust_scale_known_answer():
    # Sample 1, 2, 3, 4, 100: median 3, absolute deviations 2, 1, 0, 1, 97,
    # median of those 1, so the scale is 1.4826 exactly.
    assert robust_scale(np.array([1.0, 2.0, 3.0, 4.0, 100.0])) == pytest.approx(
        1.4826, rel=0.0, abs=1.0e-12
    )


def test_robust_scale_of_a_constant_is_zero():
    assert robust_scale(np.array([5.0, 5.0, 5.0])) == 0.0


def test_robust_scale_is_nan_for_empty():
    assert math.isnan(robust_scale(np.array([])))


def test_robust_scale_ignores_non_finite():
    assert robust_scale(np.array([1.0, 2.0, 3.0, np.nan])) == pytest.approx(
        robust_scale(np.array([1.0, 2.0, 3.0])), rel=0.0, abs=1.0e-12
    )


def test_modified_z_known_answer():
    # Against 1, 2, 3, 4, 100 the scale is 1.4826 and the median 3, so the
    # value 10 has z = 7 / 1.4826 = 4.72144... .
    assert modified_z(10.0, np.array([1.0, 2.0, 3.0, 4.0, 100.0])) == pytest.approx(
        7.0 / 1.4826, rel=0.0, abs=1.0e-12
    )


def test_modified_z_infinite_on_zero_scale():
    assert modified_z(6.0, np.array([5.0, 5.0, 5.0])) == math.inf


def test_modified_z_zero_on_zero_scale_and_no_residual():
    assert modified_z(5.0, np.array([5.0, 5.0, 5.0])) == 0.0


def test_modified_z_floor_tames_a_zero_scale():
    assert modified_z(5.001, np.array([5.0, 5.0, 5.0]), floor=0.001) == pytest.approx(
        1.0, rel=1.0e-9, abs=0.0
    )


def test_theil_sen_exact_on_a_line():
    times = np.arange(7.0)
    values = 2.0 + 3.0 * times
    slope, intercept = theil_sen(times, values)
    assert slope == pytest.approx(3.0, rel=0.0, abs=1.0e-12)
    assert intercept == pytest.approx(2.0, rel=0.0, abs=1.0e-12)


def test_theil_sen_resists_one_outlier():
    times = np.arange(9.0)
    values = 2.0 + 3.0 * times
    values[4] = 1000.0
    slope, _intercept = theil_sen(times, values)
    assert slope == pytest.approx(3.0, rel=0.0, abs=1.0e-9)


def test_theil_sen_needs_two_distinct_abscissae():
    slope, intercept = theil_sen(np.array([1.0, 1.0]), np.array([2.0, 3.0]))
    assert math.isnan(slope) and math.isnan(intercept)


def test_theil_sen_single_point():
    slope, intercept = theil_sen(np.array([1.0]), np.array([2.0]))
    assert math.isnan(slope) and math.isnan(intercept)


def test_unwrap_degrees_removes_a_wrap():
    out = unwrap_degrees(np.array([350.0, 355.0, 2.0, 8.0]))
    assert out[2] == pytest.approx(362.0, rel=0.0, abs=1.0e-12)
    assert out[3] == pytest.approx(368.0, rel=0.0, abs=1.0e-12)


def test_unwrap_degrees_is_identity_without_wraps():
    values = np.array([10.0, 20.0, 30.0])
    assert np.allclose(unwrap_degrees(values), values, rtol=0.0, atol=0.0)


def test_unwrap_degrees_empty():
    assert unwrap_degrees(np.array([])).size == 0


def test_raan_is_a_trend_element_not_a_level_one():
    # The node regresses secularly. A level test against the median of a
    # history gives a modified z-score between 1.3787 and 1.5246 on every one of
    # 217 uncorrupted element sets, a spread of 0.1459: an offset and not
    # information. Measured by validation/validate_baseline_history.py.
    assert "raan_deg" in TREND_ELEMENTS
    assert "raan_deg" not in LEVEL_ELEMENTS


def test_untested_elements_are_the_circulating_ones():
    assert set(UNTESTED_ELEMENTS) == {"arg_perigee_deg", "mean_anomaly_deg"}


def test_quantisation_steps_match_the_text_format():
    assert QUANTISATION["inclination_deg"] == 1.0e-4
    assert QUANTISATION["eccentricity"] == 1.0e-7
    assert QUANTISATION["mean_motion_rev_per_day"] == 1.0e-8


def test_scale_floor_for_bstar_is_relative():
    values = np.array([4.0e-5, 4.1e-5, 3.9e-5])
    assert scale_floor("bstar", values) == pytest.approx(4.0e-10, rel=1.0e-9, abs=0.0)


def test_scale_floor_for_zero_bstar_is_the_absolute_floor():
    assert scale_floor("bstar", np.array([0.0, 0.0, 0.0])) == 1.0e-12


def test_scale_floor_unknown_element_is_zero():
    assert scale_floor("nothing", np.array([1.0])) == 0.0


def test_check_refuses_with_too_little_history():
    verdict = check(_set(), _history(MINIMUM_HISTORY - 1))
    assert verdict.usable is False
    assert "no history test was run" in verdict.reason
    assert verdict.findings == ()


def test_refusal_is_not_a_pass():
    verdict = check(_set(eccentricity=0.5), _history(2))
    assert verdict.usable is False
    assert verdict.findings == ()


def test_check_runs_with_enough_history():
    verdict = check(_set(), _history(8))
    assert verdict.usable is True
    assert verdict.reason == ""
    assert set(verdict.z_scores) == set(LEVEL_ELEMENTS + TREND_ELEMENTS)


def test_clean_candidate_has_no_findings():
    history = _history(10)
    candidate = parse_tle(history[-1].line1, history[-1].line2)
    verdict = check(candidate, history[:-1])
    assert verdict.findings == ()


def test_sign_flipped_bstar_is_an_outlier():
    history = _history(10, bstar=1.0e-7)
    verdict = check(_set(bstar=-3.8792e-05), history)
    names = {finding.element for finding in verdict.findings}
    assert "bstar" in names


def test_implausible_eccentricity_is_an_outlier():
    history = _history(10, eccentricity=1.0e-6)
    verdict = check(_set(eccentricity=0.5), history)
    names = {finding.element for finding in verdict.findings}
    assert "eccentricity" in names


def test_quantisation_floor_prevents_an_enormous_z():
    # Without the floor, a history whose mean motion is constant to the last
    # written digit produces a scale of order 1e-8 and z-scores of order 1e7.
    history = _history(12)
    verdict = check(_set(mean_motion_rev_per_day=15.50103473), history)
    assert verdict.z_scores["mean_motion_rev_per_day"] < 10.0


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_check_rejects_bad_threshold(bad):
    with pytest.raises(ValueError, match="threshold"):
        check(_set(), _history(8), threshold=bad)


def test_check_rejects_history_of_another_object():
    other = dict(REFERENCE_FIELDS)
    other["satnum"] = 11111
    stranger = parse_tle(*format_tle(**other))
    with pytest.raises(ValueError, match="history screen compares an object"):
        check(_set(), [*_history(8), stranger])


def test_findings_carry_the_threshold_they_used():
    history = _history(10, bstar=1.0e-7)
    verdict = check(_set(bstar=-3.8792e-05), history, threshold=3.0)
    assert all(finding.threshold == 3.0 for finding in verdict.findings)


def test_history_length_is_reported():
    verdict = check(_set(), _history(9))
    assert verdict.history_length == 9
