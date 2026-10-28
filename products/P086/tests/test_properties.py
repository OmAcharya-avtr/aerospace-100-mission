"""Property-based tests over the algebraic identities this package relies on.

Every float comparison here carries a tolerance on both sides, and no boolean
produced by a strict threshold comparison is asserted equal to another. Both
rules exist because this mission has shipped the defect: an exact coverage
identity that failed because ``1 - 0.18`` evaluates to 0.8200000000000001, and
a CUSUM alarm boolean that survived three independent verifications and was
falsified by the release gate one ulp from its threshold.

A falsifying example found by Hypothesis is pinned as an explicit regression
test in ``test_regressions.py`` with its literal inputs, because the
``.hypothesis`` directory is not committed and a counterexample found by one
run is otherwise lost on a fresh clone.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from telemgap.benchmark import confusion_matrix
from telemgap.classify import AnomalyClass, classify
from telemgap.interpolate import nearest_indices
from telemgap.learned import dispersion_error_auc, expected_calibration_error
from telemgap.policy import (
    estimate_curvature,
    half_normal_quantile,
    high_frequency_variance_fraction,
    hold_error_bound,
    linear_error_bound,
    quadratic_second_derivative_coefficients,
)
from telemgap.reparse import as_signed, byte_reverse, raw_as_read
from telemgap.screens import counter_delta, robust_local_statistics

WORD16 = st.integers(min_value=0, max_value=65535)
MODULUS = st.integers(min_value=2, max_value=65536)
POSITIVE = st.floats(min_value=1e-12, max_value=1e9, allow_nan=False, allow_infinity=False)
NON_NEGATIVE = st.floats(min_value=0.0, max_value=1e9, allow_nan=False, allow_infinity=False)


@given(a=st.integers(0, 1 << 20), b=st.integers(0, 1 << 20), modulus=MODULUS)
def test_counter_delta_is_in_range_and_antisymmetric(a: int, b: int, modulus: int) -> None:
    left = counter_delta(a % modulus, b % modulus, modulus)
    right = counter_delta(b % modulus, a % modulus, modulus)
    assert 0 <= left < modulus
    assert 0 <= right < modulus
    # Exact integer identity: the two deltas sum to zero modulo the modulus.
    assert (left + right) % modulus == 0


@given(word=WORD16)
def test_byte_reverse_is_an_involution(word: int) -> None:
    assert byte_reverse(byte_reverse(word, 16), 16) == word


@given(word=WORD16)
def test_as_signed_round_trips_through_the_word_width(word: int) -> None:
    signed = as_signed(word, 16)
    assert -32768 <= signed <= 32767
    assert signed % (1 << 16) == word


@given(word=WORD16, signed=st.booleans())
def test_raw_as_read_agrees_with_its_definition(word: int, signed: bool) -> None:
    import dataclasses

    from telemgap.record import ChannelSpec

    channel = ChannelSpec(
        name="p",
        unit="u",
        scale=1.0,
        offset=0.0,
        low=-1e9,
        high=1e9,
        max_slew=1.0,
        max_curvature=1.0,
    )
    channel = dataclasses.replace(channel, signed=signed)
    expected = as_signed(word, 16) if signed else word
    assert raw_as_read(word, channel) == expected


@given(curvature=NON_NEGATIVE, duration=st.floats(0.0, 1e4, allow_nan=False))
def test_linear_bound_scales_as_the_square(curvature: float, duration: float) -> None:
    single = linear_error_bound(curvature, duration)
    doubled = linear_error_bound(curvature, 2.0 * duration)
    assert doubled == pytest.approx(4.0 * single, rel=1e-12, abs=1e-300)


@given(slew=NON_NEGATIVE, duration=st.floats(0.0, 1e4, allow_nan=False))
def test_hold_bound_is_linear_in_the_duration(slew: float, duration: float) -> None:
    single = hold_error_bound(slew, duration)
    doubled = hold_error_bound(slew, 2.0 * duration)
    assert doubled == pytest.approx(2.0 * single, rel=1e-12, abs=1e-300)


@given(q=st.floats(0.0, 0.999999, allow_nan=False))
def test_half_normal_quantile_inverts_the_error_function(q: float) -> None:
    z = half_normal_quantile(q)
    assert z >= 0.0
    assert math.erf(z / math.sqrt(2.0)) == pytest.approx(q, rel=0.0, abs=1e-8)


@given(a=st.floats(0.0, 0.99, allow_nan=False), b=st.floats(0.0, 0.99, allow_nan=False))
def test_half_normal_quantile_is_monotone(a: float, b: float) -> None:
    low, high = min(a, b), max(a, b)
    assert half_normal_quantile(low) <= half_normal_quantile(high) + 1e-9


@given(half_window=st.integers(2, 20))
def test_quadratic_coefficients_satisfy_their_moment_conditions(half_window: int) -> None:
    # A least-squares quadratic fit's second-order coefficient must annihilate
    # constants and linear terms and return 1 on k^2. Exact in exact
    # arithmetic; compared with a tolerance because it is not.
    coefficients = quadratic_second_derivative_coefficients(half_window)
    offsets = np.arange(-half_window, half_window + 1, dtype=float)
    approx = pytest.approx
    assert float(coefficients.sum()) == approx(0.0, rel=0.0, abs=1e-12)
    assert float(coefficients @ offsets) == approx(0.0, rel=0.0, abs=1e-12)
    assert float(coefficients @ offsets**2) == approx(1.0, rel=1e-10, abs=0.0)


@given(
    values=st.lists(
        st.floats(-100.0, 100.0, allow_nan=False, allow_infinity=False),
        min_size=3,
        max_size=60,
    ),
    window=st.integers(2, 10),
    span=POSITIVE,
)
def test_local_statistics_lie_inside_their_neighbourhood(
    values: list[float], window: int, span: float
) -> None:
    series = np.asarray(values, dtype=np.float64)
    medians, scaled = robust_local_statistics(series, window, span)
    assert medians.shape == series.shape
    assert np.all(np.isfinite(medians))
    assert np.all(scaled > 0.0)
    assert np.all(medians >= series.min() - 1e-9)
    assert np.all(medians <= series.max() + 1e-9)


@given(
    times=st.lists(st.floats(-1e3, 1e3, allow_nan=False), min_size=1, max_size=40),
    queries=st.lists(st.floats(-2e3, 2e3, allow_nan=False), min_size=1, max_size=20),
)
def test_nearest_indices_are_nearest(times: list[float], queries: list[float]) -> None:
    sorted_times = np.sort(np.asarray(times, dtype=np.float64))
    query = np.asarray(queries, dtype=np.float64)
    chosen = nearest_indices(sorted_times, query)
    distances = np.abs(query[:, None] - sorted_times[None, :])
    best = distances.min(axis=1)
    picked = distances[np.arange(query.size), chosen]
    assert np.all(picked <= best + 1e-12)


@given(
    truth=st.lists(st.integers(0, 2), min_size=1, max_size=200),
    predicted=st.lists(st.integers(-1, 2), min_size=1, max_size=200),
)
def test_confusion_matrix_rows_sum_to_the_support(
    truth: list[int], predicted: list[int]
) -> None:
    size = min(len(truth), len(predicted))
    truth_array = np.asarray(truth[:size], dtype=np.int64)
    predicted_array = np.asarray(predicted[:size], dtype=np.int64)
    matrix = confusion_matrix(truth_array, predicted_array)
    for code in range(3):
        assert int(matrix[code].sum()) == int((truth_array == code).sum())
    assert int(matrix.sum()) == size


@given(
    probability=st.lists(st.floats(0.0, 1.0, allow_nan=False), min_size=1, max_size=200),
    seed=st.integers(0, 10_000),
)
def test_expected_calibration_error_is_a_probability(
    probability: list[float], seed: int
) -> None:
    rng = np.random.default_rng(seed)
    correct = rng.random(len(probability)) < 0.5
    ece, _, confidence, accuracy = expected_calibration_error(
        np.asarray(probability), correct
    )
    assert 0.0 <= ece <= 1.0 + 1e-12
    finite = np.isfinite(confidence)
    assert np.all((accuracy[finite] >= -1e-12) & (accuracy[finite] <= 1.0 + 1e-12))


@given(
    dispersion=st.lists(st.floats(0.0, 10.0, allow_nan=False), min_size=2, max_size=120),
    seed=st.integers(0, 10_000),
)
def test_dispersion_auc_is_symmetric_under_negation(
    dispersion: list[float], seed: int
) -> None:
    rng = np.random.default_rng(seed)
    wrong = rng.random(len(dispersion)) < 0.5
    values = np.asarray(dispersion)
    forward = dispersion_error_auc(values, wrong)
    reverse = dispersion_error_auc(-values, wrong)
    if np.isnan(forward):
        assert np.isnan(reverse)
        return
    assert 0.0 <= forward <= 1.0 + 1e-12
    assert forward + reverse == pytest.approx(1.0, rel=0.0, abs=1e-9)


@given(
    amplitude=st.floats(1e-6, 10.0, allow_nan=False),
    frequency=st.floats(1e-4, 0.2, allow_nan=False),
    seed=st.integers(0, 1000),
)
@settings(deadline=None, max_examples=25)
def test_estimated_curvature_is_non_negative_and_bounded(
    amplitude: float, frequency: float, seed: int
) -> None:
    rng = np.random.default_rng(seed)
    times = np.arange(600, dtype=float)
    values = amplitude * np.sin(2.0 * np.pi * frequency * times) + rng.normal(
        0.0, 1e-6, size=times.size
    )
    estimate = estimate_curvature(values, 1.0, span=4.0 * amplitude, declared_sigma=1e-6)
    assert estimate.value >= 0.0
    assert estimate.noise_floor >= 0.0
    # The estimator never reports more curvature than the analytic bound plus
    # its own noise floor, because the quadratic fit is an attenuating filter.
    analytic = amplitude * (2.0 * np.pi * frequency) ** 2
    assert estimate.value <= analytic * 1.25 + 10.0 * estimate.noise_floor + 1e-12


@given(
    frequency=st.floats(0.002, 0.45, allow_nan=False),
    seed=st.integers(0, 500),
)
@settings(deadline=None, max_examples=20)
def test_high_frequency_fraction_is_a_fraction(frequency: float, seed: int) -> None:
    rng = np.random.default_rng(seed)
    times = np.arange(800, dtype=float)
    values = np.sin(2.0 * np.pi * frequency * times) + rng.normal(0.0, 1e-3, size=times.size)
    fraction = high_frequency_variance_fraction(values, 1.0, 0.064, 4.0)
    assert 0.0 <= fraction <= 1.0 + 1e-12


@given(
    raws=st.lists(st.integers(24, 120), min_size=25, max_size=60),
)
@settings(deadline=None, max_examples=25)
def test_missing_verdicts_equal_the_gap_count(raws: list[int]) -> None:
    import numpy as numpy_module

    from telemgap.record import ChannelSpec, RecordSchema, TelemetryRecord

    channel = ChannelSpec(
        name="probe",
        unit="K",
        scale=0.5,
        offset=10.0,
        low=12.0,
        high=60.0,
        max_slew=1.0,
        max_curvature=0.25,
        noise_sigma=0.5,
    )
    schema = RecordSchema(
        cadence_s=1.0,
        jitter_s=0.1,
        counter_modulus=256,
        channels=(channel,),
        has_validity_flag=True,
        has_raw_words=True,
    )
    words = numpy_module.asarray(raws, dtype=numpy_module.uint64)
    record = TelemetryRecord(
        schema=schema,
        timestamps=numpy_module.arange(words.size, dtype=float),
        counters=numpy_module.arange(words.size, dtype=numpy_module.int64) % 256,
        values=channel.decode(words.astype(numpy_module.int64)).reshape(-1, 1),
        raw_words=words.reshape(-1, 1),
        validity=numpy_module.ones(words.size, dtype=bool),
        provenance={"time_source": "test"},
    )
    report = classify(record)
    missing = [
        anomaly for anomaly in report.anomalies if anomaly.verdict is AnomalyClass.MISSING
    ]
    assert len(missing) == len(report.findings.gaps)
    assert sum(report.counts.values()) == len(report.anomalies)
    assert all(
        anomaly.verdict is AnomalyClass.UNDECIDED or not anomaly.reasons_undecided
        for anomaly in report.anomalies
    )


@given(
    tolerance=st.floats(1e-6, 1e3, allow_nan=False),
    factor=st.floats(1.0, 1e3, allow_nan=False),
)
def test_a_larger_tolerance_never_turns_permission_into_refusal(
    tolerance: float, factor: float
) -> None:
    # The gates compare a tolerance-independent prediction against the
    # tolerance, so widening the tolerance can only move a decision from
    # refuse to interpolate. No arithmetic is done on the tolerance itself, so
    # the comparison is monotone exactly and not merely to rounding.
    from telemgap.policy import CurvatureEstimate, decide_gap
    from telemgap.screens import Gap

    gap = Gap(
        after_index=0,
        before_index=1,
        start_time=0.0,
        end_time=2.0,
        duration_s=2.0,
        unobserved_s=1.0,
        missing_frames_from_time=1,
        missing_frames_from_counter=1,
        agrees=True,
    )
    curvature = CurvatureEstimate(
        value=1e-3,
        source="declared",
        noise_floor=0.0,
        sigma=1e-4,
        sigma_source="declared",
        measurable=True,
        bandwidth_hz=float("inf"),
        half_window=7,
    )
    narrow = decide_gap(
        gap=gap,
        gap_index=0,
        channel_index=0,
        channel_name="probe",
        unit_tolerance=tolerance,
        curvature=curvature,
        slew=1e-3,
        cadence_s=1.0,
    )
    wide = decide_gap(
        gap=gap,
        gap_index=0,
        channel_index=0,
        channel_name="probe",
        unit_tolerance=tolerance * factor,
        curvature=curvature,
        slew=1e-3,
        cadence_s=1.0,
    )
    if not narrow.refused:
        assert not wide.refused
