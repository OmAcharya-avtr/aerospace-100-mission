"""The gap policy: equations (3), (3'), (4), (8), (9), (16), (17) and every gate."""

from __future__ import annotations

import math

import numpy as np
import pytest

from telemgap import classify, dataset
from telemgap.policy import (
    DEFAULT_COVERAGE_K,
    DEFAULT_HALF_WINDOW,
    DEFAULT_MAX_GAP_SLOTS,
    DEFAULT_RESOLVABILITY_THRESHOLD,
    MEASURABILITY_FACTOR,
    METHODS,
    ChannelTolerance,
    CurvatureEstimate,
    bernstein_curvature,
    channel_curvatures,
    decide_gap,
    estimate_curvature,
    estimate_noise_sigma,
    estimator_bandwidth_fraction,
    estimator_transfer,
    half_normal_quantile,
    high_frequency_variance_fraction,
    hold_error_bound,
    linear_error_bound,
    plan,
    quadratic_second_derivative_coefficients,
)
from telemgap.screens import Gap


def make_gap(duration_s: float, cadence_s: float = 1.0) -> Gap:
    """A gap of the given duration, with consistent frame deficits."""
    slots = int(round(duration_s / cadence_s))
    return Gap(
        after_index=0,
        before_index=1,
        start_time=0.0,
        end_time=duration_s,
        duration_s=duration_s,
        unobserved_s=max(0.0, duration_s - cadence_s),
        missing_frames_from_time=max(0, slots - 1),
        missing_frames_from_counter=max(0, slots - 1),
        agrees=True,
    )


def declared(value: float, sigma: float = 0.0) -> CurvatureEstimate:
    """A declared curvature estimate, as :func:`channel_curvatures` builds one."""
    return CurvatureEstimate(
        value=value,
        source="declared",
        noise_floor=0.0,
        sigma=sigma,
        sigma_source="declared",
        measurable=True,
        bandwidth_hz=float("inf"),
        half_window=DEFAULT_HALF_WINDOW,
    )


@pytest.mark.parametrize(
    ("curvature", "duration", "expected"),
    [
        # Hand-calculated from equation (3): M2 h^2 / 8.
        (8.0, 1.0, 1.0),
        (8.0, 2.0, 4.0),
        (0.25, 4.0, 0.5),
        (0.0, 100.0, 0.0),
        (1.0, 0.0, 0.0),
    ],
)
def test_linear_bound_known_answers(
    curvature: float, duration: float, expected: float
) -> None:
    assert linear_error_bound(curvature, duration) == pytest.approx(
        expected, rel=1e-15, abs=0.0
    )


@pytest.mark.parametrize(
    ("slew", "duration", "expected"),
    [(2.0, 3.0, 6.0), (0.5, 0.5, 0.25), (0.0, 10.0, 0.0)],
)
def test_hold_bound_known_answers(slew: float, duration: float, expected: float) -> None:
    assert hold_error_bound(slew, duration) == pytest.approx(expected, rel=1e-15, abs=0.0)


def test_bounds_reject_negative_arguments() -> None:
    with pytest.raises(ValueError, match="curvature"):
        linear_error_bound(-1.0, 1.0)
    with pytest.raises(ValueError, match="duration_s"):
        linear_error_bound(1.0, -1.0)
    with pytest.raises(ValueError, match="slew"):
        hold_error_bound(-1.0, 1.0)
    with pytest.raises(ValueError, match="duration_s"):
        hold_error_bound(1.0, -1.0)


def test_bernstein_known_answer() -> None:
    # Equation (8) at k = 2 with A = 1 and B = 1 / (2 pi): (2 pi B)^2 A = 1.
    assert bernstein_curvature(1.0, 1.0 / (2.0 * math.pi)) == pytest.approx(
        1.0, rel=1e-14, abs=0.0
    )
    assert bernstein_curvature(2.0, 1.0) == pytest.approx(
        2.0 * (2.0 * math.pi) ** 2, rel=1e-15, abs=0.0
    )
    with pytest.raises(ValueError, match="amplitude"):
        bernstein_curvature(-1.0, 1.0)
    with pytest.raises(ValueError, match="bandwidth"):
        bernstein_curvature(1.0, -1.0)


def test_bernstein_bounds_the_analytic_curvature_of_every_model() -> None:
    # A cross-check from different mathematics: the Bernstein bound of
    # equation (8) must dominate the term-by-term bound of equation (12) for
    # any model whose amplitude is concentrated at its highest frequency, and
    # must not be below it for a single-term model.
    for model in dataset.MODELS:
        if len(model.terms) == 1:
            bernstein = bernstein_curvature(model.amplitude_bound, model.bandwidth_hz)
            assert bernstein >= model.curvature_bound * (1.0 - 1e-12)


def test_half_normal_quantile_known_answers() -> None:
    # P(|X| <= z) = q for a standard normal: q = 0.95 -> 1.959964,
    # q = 0.99 -> 2.575829, q = 0.682689 -> 1.0.
    assert half_normal_quantile(0.95) == pytest.approx(1.959963985, rel=1e-8, abs=0.0)
    assert half_normal_quantile(0.99) == pytest.approx(2.575829304, rel=1e-8, abs=0.0)
    assert half_normal_quantile(0.6826894921) == pytest.approx(1.0, rel=1e-8, abs=0.0)
    assert half_normal_quantile(0.0) == pytest.approx(0.0, rel=0.0, abs=1e-9)
    with pytest.raises(ValueError, match="q must lie"):
        half_normal_quantile(1.0)
    with pytest.raises(ValueError, match="q must lie"):
        half_normal_quantile(-0.1)


def test_quadratic_coefficients_known_answer() -> None:
    # For m = 1 the quadratic fit through three points is exact, so
    # a_2 = (v_-1 - 2 v_0 + v_1) / 2 and c = (0.5, -1.0, 0.5). m = 1 is below
    # the supported minimum, so check m = 2 against the standard
    # Savitzky-Golay quadratic coefficients (1/7)(2, -1, -2, -1, 2).
    coefficients = quadratic_second_derivative_coefficients(2)
    expected = np.array([2.0, -1.0, -2.0, -1.0, 2.0]) / 14.0
    assert np.allclose(coefficients, expected, rtol=1e-12, atol=0.0)
    assert coefficients.sum() == pytest.approx(0.0, rel=0.0, abs=1e-15)
    with pytest.raises(ValueError, match="at least 2"):
        quadratic_second_derivative_coefficients(1)


def test_quadratic_coefficients_are_exact_on_a_parabola() -> None:
    # The estimator 2 a_2 must recover the exact second derivative of a
    # quadratic for any window size, since the fit is exact.
    for half_window in (2, 5, 7, 12):
        offsets = np.arange(-half_window, half_window + 1, dtype=float)
        values = 3.0 * offsets**2 - 2.0 * offsets + 7.0
        coefficients = quadratic_second_derivative_coefficients(half_window)
        assert 2.0 * float(coefficients @ values) == pytest.approx(6.0, rel=1e-10, abs=0.0)


def test_estimator_transfer_tends_to_one_at_low_frequency() -> None:
    assert estimator_transfer(7, 1e-5) == pytest.approx(1.0, rel=1e-6, abs=0.0)
    assert estimator_transfer(7, 0.0) == pytest.approx(1.0, rel=0.0, abs=0.0)


def test_estimator_transfer_attenuates_near_nyquist() -> None:
    # The closed-form reason the estimated curvature of a near-Nyquist channel
    # is wrong. The number is computed, not quoted.
    attenuation = estimator_transfer(DEFAULT_HALF_WINDOW, 0.35)
    assert attenuation < 0.01
    assert attenuation > 0.0


def test_estimator_bandwidth_is_between_zero_and_nyquist() -> None:
    fraction = estimator_bandwidth_fraction(DEFAULT_HALF_WINDOW)
    assert 0.0 < fraction < 0.5
    assert estimator_transfer(DEFAULT_HALF_WINDOW, fraction) == pytest.approx(
        0.5, rel=1e-6, abs=0.0
    )
    # A wider window has a lower bandwidth.
    assert estimator_bandwidth_fraction(12) < estimator_bandwidth_fraction(4)


def test_noise_sigma_estimator_recovers_white_noise() -> None:
    # Equation (16) on pure noise of known standard deviation.
    rng = np.random.default_rng(99)
    sigma = 0.37
    estimate = estimate_noise_sigma(rng.normal(0.0, sigma, size=20_000))
    assert estimate == pytest.approx(sigma, rel=0.05, abs=0.0)


def test_noise_sigma_estimator_is_confounded_by_fast_signal() -> None:
    # A measured limitation, asserted as a limitation: a sinusoid at 0.35 of
    # the sampling frequency is read as noise several times its true scale.
    times = np.arange(4000, dtype=float)
    clean = 0.08 * np.sin(2.0 * np.pi * 0.35 * times)
    estimate = estimate_noise_sigma(clean)
    assert estimate > 10.0 * 1e-3


def test_noise_sigma_of_short_series_is_zero() -> None:
    assert estimate_noise_sigma(np.array([1.0, 2.0])) == pytest.approx(0.0, rel=0.0, abs=0.0)


def test_estimate_curvature_recovers_a_slow_sinusoid() -> None:
    # Noiseless sinusoid at f T = 0.002: equation (9) must land within a few
    # per cent of the analytic A (2 pi f)^2.
    times = np.arange(6000, dtype=float)
    amplitude, frequency = 2.0, 0.002
    values = amplitude * np.sin(2.0 * np.pi * frequency * times)
    analytic = amplitude * (2.0 * np.pi * frequency) ** 2
    estimate = estimate_curvature(values, 1.0, span=8.0, declared_sigma=1e-9)
    assert estimate.value == pytest.approx(analytic, rel=0.05, abs=0.0)
    assert estimate.source == "estimated"
    assert estimate.measurable
    assert estimate.resolvable


def test_estimate_curvature_under_predicts_a_fast_sinusoid() -> None:
    # The headline failure mode, reproduced as a unit test. The ratio is
    # compared against the closed-form transfer function, so the test asserts
    # the mechanism and not just the symptom.
    times = np.arange(6000, dtype=float)
    amplitude, frequency = 0.08, 0.35
    values = amplitude * np.sin(2.0 * np.pi * frequency * times)
    analytic = amplitude * (2.0 * np.pi * frequency) ** 2
    estimate = estimate_curvature(values, 1.0, span=1.0, declared_sigma=1e-9)
    expected = analytic * estimator_transfer(DEFAULT_HALF_WINDOW, frequency)
    assert estimate.value == pytest.approx(expected, rel=0.25, abs=0.0)
    assert estimate.value < analytic / 100.0
    assert estimate.resolvable is False


def test_estimate_curvature_declares_noise_unmeasurable() -> None:
    rng = np.random.default_rng(3)
    values = rng.normal(0.0, 0.01, size=4000)
    estimate = estimate_curvature(values, 1.0, span=1.0, declared_sigma=0.01)
    assert estimate.measurable is False
    assert estimate.noise_floor > 0.0
    assert estimate.sigma_source == "declared"


def test_estimate_curvature_estimates_sigma_when_undeclared() -> None:
    rng = np.random.default_rng(4)
    values = rng.normal(0.0, 0.01, size=2000)
    estimate = estimate_curvature(values, 1.0, span=1.0)
    assert estimate.sigma_source == "estimated"
    assert estimate.sigma == pytest.approx(0.01, rel=0.15, abs=0.0)


def test_estimate_curvature_handles_a_short_series() -> None:
    estimate = estimate_curvature(np.zeros(3), 1.0, span=1.0, declared_sigma=1.0)
    assert estimate.value == pytest.approx(0.0, rel=0.0, abs=0.0)
    assert estimate.measurable is False
    assert estimate.resolvable is False


def test_estimate_curvature_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="cadence_s"):
        estimate_curvature(np.zeros(100), 0.0, span=1.0)
    with pytest.raises(ValueError, match="quantile"):
        estimate_curvature(np.zeros(100), 1.0, span=1.0, quantile=1.0)


def test_resolvability_can_be_disabled() -> None:
    times = np.arange(4000, dtype=float)
    values = 0.08 * np.sin(2.0 * np.pi * 0.35 * times)
    assert estimate_curvature(
        values, 1.0, span=1.0, declared_sigma=1e-9, resolvability_threshold=None
    ).resolvable
    assert not estimate_curvature(values, 1.0, span=1.0, declared_sigma=1e-9).resolvable


def test_high_frequency_fraction_known_answers() -> None:
    times = np.arange(4000, dtype=float)
    slow = np.sin(2.0 * np.pi * 0.002 * times)
    fast = np.sin(2.0 * np.pi * 0.35 * times)
    assert high_frequency_variance_fraction(slow, 1.0, 0.064, 2.0) < 0.01
    assert high_frequency_variance_fraction(fast, 1.0, 0.064, 2.0) > 0.9
    assert high_frequency_variance_fraction(np.zeros(4000), 1.0, 0.064, 2.0) == pytest.approx(
        0.0, rel=0.0, abs=0.0
    )
    assert high_frequency_variance_fraction(np.zeros(5), 1.0, 0.064, 2.0) == pytest.approx(
        0.0, rel=0.0, abs=0.0
    )


def test_high_frequency_fraction_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="cadence_s"):
        high_frequency_variance_fraction(np.zeros(400), 0.0, 0.1, 1.0)
    with pytest.raises(ValueError, match="cutoff_hz"):
        high_frequency_variance_fraction(np.zeros(400), 1.0, 0.0, 1.0)


def test_coverage_term_enters_the_prediction() -> None:
    # Equation (3'): with M2 = 0 the whole prediction is k sigma.
    decision = decide_gap(
        gap=make_gap(2.0),
        gap_index=0,
        channel_index=0,
        channel_name="probe",
        unit_tolerance=1.0,
        curvature=declared(0.0, sigma=0.1),
        slew=0.0,
        cadence_s=1.0,
        coverage_k=3.0,
    )
    assert decision.coverage == pytest.approx(0.3, rel=1e-12, abs=0.0)
    assert decision.predicted_linear == pytest.approx(0.3, rel=1e-12, abs=0.0)
    assert decision.predicted_hold == pytest.approx(0.3, rel=1e-12, abs=0.0)


def test_nyquist_gate_refuses_a_long_gap() -> None:
    decision = decide_gap(
        gap=make_gap(100.0),
        gap_index=0,
        channel_index=0,
        channel_name="probe",
        unit_tolerance=1e9,
        curvature=declared(1e-30, sigma=0.0),
        slew=0.0,
        cadence_s=1.0,
    )
    assert decision.refused
    assert "nyquist_slots" in decision.failed_gates


def test_tolerance_below_noise_is_refused() -> None:
    decision = decide_gap(
        gap=make_gap(1.5),
        gap_index=0,
        channel_index=0,
        channel_name="probe",
        unit_tolerance=0.01,
        curvature=declared(0.0, sigma=0.012),
        slew=0.0,
        cadence_s=1.0,
    )
    assert decision.refused
    assert "tolerance_above_noise" in decision.failed_gates
    assert "measurement noise" in decision.reason


def test_unmeasurable_curvature_blocks_linear_but_not_hold() -> None:
    curvature = CurvatureEstimate(
        value=1.0,
        source="estimated",
        noise_floor=10.0,
        sigma=0.0,
        sigma_source="declared",
        measurable=False,
        bandwidth_hz=0.064,
        half_window=7,
    )
    decision = decide_gap(
        gap=make_gap(2.0),
        gap_index=0,
        channel_index=0,
        channel_name="probe",
        unit_tolerance=1.0,
        curvature=curvature,
        slew=0.1,
        cadence_s=1.0,
    )
    assert not decision.refused
    assert decision.method == "hold"
    assert "curvature_measurable" in decision.failed_gates


def test_unresolvable_curvature_blocks_linear() -> None:
    curvature = CurvatureEstimate(
        value=1e-3,
        source="estimated",
        noise_floor=1e-6,
        sigma=0.0,
        sigma_source="declared",
        measurable=True,
        bandwidth_hz=0.064,
        half_window=7,
        high_frequency_fraction=0.2,
        resolvable=False,
    )
    decision = decide_gap(
        gap=make_gap(2.0),
        gap_index=0,
        channel_index=0,
        channel_name="probe",
        unit_tolerance=1.0,
        curvature=curvature,
        slew=1e9,
        cadence_s=1.0,
    )
    assert decision.refused
    assert "curvature_resolvable" in decision.failed_gates


def test_linear_preferred_when_both_permitted() -> None:
    decision = decide_gap(
        gap=make_gap(2.0),
        gap_index=0,
        channel_index=0,
        channel_name="probe",
        unit_tolerance=10.0,
        curvature=declared(1e-6, sigma=0.0),
        slew=1.0,
        cadence_s=1.0,
    )
    assert decision.method == "linear"
    assert decision.predicted_linear < decision.predicted_hold
    assert decision.margin == pytest.approx(
        10.0 - decision.predicted_linear, rel=1e-12, abs=0.0
    )


def test_decide_gap_rejects_negative_coverage() -> None:
    with pytest.raises(ValueError, match="coverage_k"):
        decide_gap(
            gap=make_gap(1.0),
            gap_index=0,
            channel_index=0,
            channel_name="probe",
            unit_tolerance=1.0,
            curvature=declared(1.0),
            slew=1.0,
            cadence_s=1.0,
            coverage_k=-1.0,
        )


def test_channel_tolerance_validation() -> None:
    ChannelTolerance("probe", 1.0)
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive and finite"):
            ChannelTolerance("probe", bad)


def test_plan_requires_a_tolerance_for_every_channel(generated_small) -> None:
    report = classify.classify(generated_small.record)
    with pytest.raises(ValueError, match="must be declared for every channel"):
        plan(generated_small.record, report, {"battery_voltage": 0.05})
    with pytest.raises(ValueError, match="max_gap_slots"):
        plan(
            generated_small.record,
            report,
            dict.fromkeys(generated_small.record.schema.channel_names, 1.0),
            max_gap_slots=0.0,
        )


def test_plan_covers_every_gap_and_channel(generated_small) -> None:
    record = generated_small.record
    report = classify.classify(record)
    tolerances = dict.fromkeys(record.schema.channel_names, 1.0)
    built = plan(record, report, tolerances)
    assert len(built.decisions) == len(report.findings.gaps) * record.n_channels
    assert set(built.curvatures) == set(record.schema.channel_names)
    summary = built.summary()
    assert summary["decisions"] == summary["interpolate"] + summary["refuse"]
    assert built.coverage_k == pytest.approx(DEFAULT_COVERAGE_K, rel=0.0, abs=0.0)
    assert built.max_gap_slots == pytest.approx(DEFAULT_MAX_GAP_SLOTS, rel=0.0, abs=0.0)
    assert built.for_channel("battery_voltage")


def test_channel_curvatures_uses_declared_where_declared(generated_small) -> None:
    estimates = channel_curvatures(generated_small.record)
    assert estimates["battery_voltage"].source == "declared"
    assert estimates["tank_pressure"].source == "declared"
    assert estimates["valve_current"].source == "estimated"
    assert estimates["gyro_rate"].source == "estimated"
    assert estimates["battery_voltage"].value == pytest.approx(
        dataset.MODELS[0].curvature_bound, rel=1e-12, abs=0.0
    )


def test_constants_are_the_documented_defaults() -> None:
    assert METHODS == ("linear", "hold")
    assert DEFAULT_COVERAGE_K == pytest.approx(3.0, rel=0.0, abs=0.0)
    assert DEFAULT_MAX_GAP_SLOTS == pytest.approx(4.0, rel=0.0, abs=0.0)
    assert MEASURABILITY_FACTOR == pytest.approx(2.0, rel=0.0, abs=0.0)
    assert DEFAULT_RESOLVABILITY_THRESHOLD == pytest.approx(0.01, rel=0.0, abs=0.0)
    assert DEFAULT_HALF_WINDOW == 7
