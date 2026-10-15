"""The analytic model, the generator, and the feature encoding."""

from __future__ import annotations

import math

import numpy as np
import pytest

from activityplan.features import (
    ACTIVITY_TYPES,
    FEATURE_NAMES,
    MINIMUM_DURATION_S,
    MINIMUM_POWER_W,
    REFERENCE_TEMPERATURE_C,
    TYPE_CONSTANTS,
    ActivityRequest,
    analytic_duration_s,
    analytic_energy_wh,
    analytic_mean_power_w,
    feature_matrix,
    feature_vector,
    realise,
    sample_requests,
    to_activity,
)


def _request(**kwargs: object) -> ActivityRequest:
    defaults: dict[str, object] = {
        "name": "a",
        "activity_type": "image",
        "slew_angle_deg": 45.0,
        "data_mbit": 240.0,
        "instrument_temperature_c": 20.0,
        "gain": 1.0,
    }
    defaults.update(kwargs)
    return ActivityRequest(**defaults)  # type: ignore[arg-type]


def test_analytic_duration_known_answer() -> None:
    # image: setup 20 s + slew 45 / 0.80 = 56.25 s + settle 15 s
    #        + readout 240 / 2.40 = 100 s  ->  191.25 s.
    assert analytic_duration_s(_request()) == pytest.approx(191.25, rel=1e-13)


def test_analytic_mean_power_known_answer() -> None:
    # Data rate 240 / 191.25 = 1.254901960784... Mbit/s.
    # Power = 28 + 6 * 1.254901960784 = 35.529411764705... W.
    power = analytic_mean_power_w(_request())
    assert power == pytest.approx(28.0 + 6.0 * (240.0 / 191.25), rel=1e-13)
    assert power == pytest.approx(35.52941176470588, rel=1e-12)


def test_analytic_energy_known_answer() -> None:
    # 35.52941176470588 W for 191.25 s. The product 35.52941176470588 * 191.25
    # is 6795.0 exactly in exact arithmetic, because 6 * 240 / 2.4 contributes
    # an exact 600 W*s and 28 * 191.25 = 5355 W*s, giving 6795 W*s = 1.8875 W*h.
    energy = analytic_energy_wh(_request())
    assert energy == pytest.approx((28.0 * 191.25 + 6.0 * 240.0) / 3600.0, rel=1e-13)
    assert energy == pytest.approx(1.8875, rel=1e-12)


def test_analytic_model_ignores_temperature_and_gain_by_construction() -> None:
    warm = _request(instrument_temperature_c=30.0, gain=1.0)
    cold = _request(instrument_temperature_c=-25.0, gain=1.7)
    assert analytic_duration_s(warm) == pytest.approx(analytic_duration_s(cold), rel=1e-15)
    assert analytic_mean_power_w(warm) == pytest.approx(analytic_mean_power_w(cold), rel=1e-15)


def test_generator_adds_the_two_terms_the_analytic_model_omits() -> None:
    # With both noise scales at zero the generator is deterministic and the
    # difference from the analytic model is exactly the two omitted terms.
    constants = TYPE_CONSTANTS["spectro"]
    request = _request(activity_type="spectro", instrument_temperature_c=-20.0, gain=1.5)
    realised = realise(
        request,
        np.random.default_rng(0),
        duration_noise_fraction=0.0,
        power_noise_fraction=0.0,
    )
    cold = REFERENCE_TEMPERATURE_C - (-20.0)
    expected_duration = (
        analytic_duration_s(request)
        + constants["cold_settle_s_per_10c"] * cold / 10.0
        + constants["gain_penalty_s_per_mbit"] * 0.25 * request.data_mbit
    )
    expected_power = (
        analytic_mean_power_w(request) + constants["heater_w_per_10c"] * cold / 10.0
    )
    assert realised.duration_s == pytest.approx(expected_duration, rel=1e-12)
    assert realised.mean_power_w == pytest.approx(expected_power, rel=1e-12)
    assert realised.energy_wh == pytest.approx(
        expected_power * expected_duration / 3600.0, rel=1e-12
    )


def test_generator_adds_nothing_above_the_reference_temperature_at_unit_gain() -> None:
    request = _request(instrument_temperature_c=25.0, gain=1.0)
    realised = realise(
        request,
        np.random.default_rng(0),
        duration_noise_fraction=0.0,
        power_noise_fraction=0.0,
    )
    assert realised.duration_s == pytest.approx(analytic_duration_s(request), rel=1e-12)
    assert realised.mean_power_w == pytest.approx(analytic_mean_power_w(request), rel=1e-12)


def test_generator_noise_is_heteroscedastic_in_the_analytic_duration() -> None:
    short = _request(data_mbit=20.0)
    long = _request(data_mbit=780.0)
    spreads = []
    for request in (short, long):
        rng = np.random.default_rng(5)
        draws = [realise(request, rng).duration_s for _ in range(400)]
        spreads.append(float(np.std(draws)))
    assert spreads[1] > 3.0 * spreads[0]


def test_generator_is_reproducible_from_the_seed() -> None:
    request = _request()
    first = realise(request, np.random.default_rng(11)).duration_s
    second = realise(request, np.random.default_rng(11)).duration_s
    assert first == second


def test_generator_floors_duration_and_power() -> None:
    # A huge noise draw can go very negative; the floors catch it.
    request = _request(data_mbit=10.0, slew_angle_deg=0.0)
    realised = realise(
        request,
        np.random.default_rng(3),
        duration_noise_fraction=50.0,
        power_noise_fraction=50.0,
    )
    assert realised.duration_s >= MINIMUM_DURATION_S
    assert realised.mean_power_w >= MINIMUM_POWER_W


def test_generator_rejects_negative_noise_fractions() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        realise(_request(), np.random.default_rng(0), duration_noise_fraction=-1.0)


def test_feature_vector_layout_and_one_hot() -> None:
    request = _request(activity_type="radar")
    vector = feature_vector(request)
    assert vector.shape == (len(FEATURE_NAMES),)
    assert vector[: len(ACTIVITY_TYPES)].tolist() == [0.0, 0.0, 1.0, 0.0]
    assert vector[4] == pytest.approx(45.0, rel=1e-14)
    assert vector[5] == pytest.approx(240.0, rel=1e-14)
    assert vector[8] == pytest.approx(analytic_duration_s(request), rel=1e-14)
    assert vector[9] == pytest.approx(analytic_mean_power_w(request), rel=1e-14)


def test_feature_matrix_handles_the_empty_case() -> None:
    assert feature_matrix([]).shape == (0, len(FEATURE_NAMES))
    assert feature_matrix([_request(), _request(name="b")]).shape == (2, len(FEATURE_NAMES))


def test_request_validation() -> None:
    with pytest.raises(ValueError, match="activity_type"):
        _request(activity_type="not-a-type")
    with pytest.raises(ValueError, match="slew_angle_deg"):
        _request(slew_angle_deg=-1.0)
    with pytest.raises(ValueError, match="data_mbit"):
        _request(data_mbit=-1.0)
    with pytest.raises(ValueError, match="gain"):
        _request(gain=0.0)
    with pytest.raises(ValueError, match="priority"):
        _request(priority=0.0)


def test_sample_requests_stays_inside_the_declared_ranges() -> None:
    requests = sample_requests(300, np.random.default_rng(2))
    assert len(requests) == 300
    assert all(0.0 <= r.slew_angle_deg <= 90.0 for r in requests)
    assert all(10.0 <= r.data_mbit <= 800.0 for r in requests)
    assert all(-30.0 <= r.instrument_temperature_c <= 35.0 for r in requests)
    assert all(0.6 <= r.gain <= 1.8 for r in requests)
    assert len({r.name for r in requests}) == 300
    assert {r.activity_type for r in requests} == set(ACTIVITY_TYPES)


def test_sample_requests_narrowed_ranges_are_checked() -> None:
    requests = sample_requests(
        50, np.random.default_rng(2), data_mbit_range=(40.0, 420.0)
    )
    assert all(40.0 <= r.data_mbit <= 420.0 for r in requests)
    with pytest.raises(ValueError, match="data_mbit_range"):
        sample_requests(5, np.random.default_rng(0), data_mbit_range=(5.0, 100.0))
    with pytest.raises(ValueError, match="slew_angle_range"):
        sample_requests(5, np.random.default_rng(0), slew_angle_range=(0.0, 120.0))
    with pytest.raises(ValueError, match="non-negative"):
        sample_requests(-1, np.random.default_rng(0))


def test_to_activity_carries_the_energy_exactly() -> None:
    request = _request(data_mbit=300.0, priority=2.5, earliest_start_s=100.0)
    activity = to_activity(request, 250.0, 4.0)
    assert activity.duration_s == pytest.approx(250.0, rel=1e-14)
    assert activity.energy_wh == pytest.approx(4.0, rel=1e-13)
    assert activity.mean_power_w == pytest.approx(4.0 * 3600.0 / 250.0, rel=1e-13)
    assert activity.data_mbit == pytest.approx(300.0, rel=1e-14)
    assert activity.priority == pytest.approx(2.5, rel=1e-14)
    assert activity.earliest_start_s == pytest.approx(100.0, rel=1e-14)
    assert math.isinf(activity.latest_finish_s)
    assert len(activity.power_profile) == 1


def test_to_activity_rejects_bad_estimates() -> None:
    with pytest.raises(ValueError, match="duration_s"):
        to_activity(_request(), 0.0, 1.0)
    with pytest.raises(ValueError, match="energy_wh"):
        to_activity(_request(), 10.0, -1.0)
