"""Certificate tests, including the deliberate bound-violation negative control.

The negative control is the test that makes the positive result mean anything.
A certificate that always says HOLDS is worthless; these tests inject
violations of several kinds and require the certificate to catch each one and
to name the right sample.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from boundcompress.certificate import MAX_LISTED_VIOLATIONS, certify, digest_bytes
from boundcompress.codec import encode


def test_negative_control_single_injected_violation(signals: dict[str, np.ndarray]) -> None:
    """Move one sample outside the bound. The certificate must fail and name it."""
    values = signals["battery_voltage"]
    bound = 1e-3
    result = encode(values, "abs", bound, predictor="linear")
    assert result.certificate.bound_holds

    corrupted = result.reconstruction.copy()
    index = 517
    corrupted[index] += 2.5 * bound
    certificate = certify(values, corrupted, "abs", bound)

    assert certificate.bound_holds is False
    assert certificate.violations == 1
    assert certificate.violation_indices == (index,)
    assert certificate.argmax_index == index
    assert certificate.margin < 0.0
    assert certificate.max_absolute_error > bound


def test_negative_control_at_one_ulp_past_the_bound() -> None:
    """The sharpest control available: exactly at the bound passes, one ulp past fails.

    This is not a floating-point equality assertion. It is an assertion that
    the comparison carries no hidden tolerance, which is the property the whole
    product rests on.
    """
    values = np.array([1.0, 2.0, 3.0, 4.0])
    bound = 0.25
    at_bound = values + bound
    assert certify(values, at_bound, "abs", bound).bound_holds
    just_past = values.copy() + bound
    just_past[2] = np.nextafter(just_past[2], np.inf)
    certificate = certify(values, just_past, "abs", bound)
    assert certificate.bound_holds is False
    assert certificate.violation_indices == (2,)


def test_negative_control_relative_mode() -> None:
    """A violation of 1.5x the declared relative bound on one sample."""
    values = np.array([10.0, 20.0, 40.0, 80.0])
    bound = 1e-3
    corrupted = values.copy()
    corrupted[1] = values[1] * (1.0 + 1.5 * bound)
    certificate = certify(values, corrupted, "rel", bound)
    assert certificate.bound_holds is False
    assert certificate.violation_indices == (1,)
    assert certificate.max_relative_error == pytest.approx(1.5 * bound, rel=1e-9)


def test_negative_control_many_violations_are_counted_and_truncated() -> None:
    """Thirty violations: all counted, the first sixteen listed."""
    values = np.linspace(1.0, 10.0, 200)
    bound = 0.01
    corrupted = values.copy()
    corrupted[:30] += 5.0 * bound
    certificate = certify(values, corrupted, "abs", bound)
    assert certificate.violations == 30
    assert len(certificate.violation_indices) == MAX_LISTED_VIOLATIONS
    assert certificate.violation_indices == tuple(range(MAX_LISTED_VIOLATIONS))


def test_negative_control_non_finite_reconstruction_fails() -> None:
    """A nan reconstruction is not inside any bound, and the certificate says so."""
    values = np.array([1.0, 2.0, 3.0])
    corrupted = np.array([1.0, float("nan"), 3.0])
    certificate = certify(values, corrupted, "abs", 0.5)
    assert certificate.bound_holds is False
    assert certificate.violation_indices == (1,)

    infinite = np.array([1.0, float("inf"), 3.0])
    assert certify(values, infinite, "abs", 0.5).bound_holds is False


def test_certify_is_independent_of_the_encoder(signals: dict[str, np.ndarray]) -> None:
    """A reconstruction the encoder never produced is certified on its own terms."""
    values = signals["gyro_rate"]
    bound = 1e-5
    rounded = np.round(values / (2.0 * bound)) * (2.0 * bound)
    certificate = certify(values, rounded, "abs", bound)
    assert certificate.samples == values.size
    assert certificate.encoded_bytes == 0
    assert certificate.bits_per_sample == 0.0
    assert certificate.compression_ratio == 0.0
    assert certificate.bound_holds == bool(np.all(np.abs(values - rounded) <= bound))


def test_shape_mismatch_is_refused() -> None:
    with pytest.raises(ValueError, match="reconstruction has shape"):
        certify(np.array([1.0, 2.0]), np.array([1.0]), "abs", 0.1)


def test_non_finite_original_is_refused() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        certify(np.array([1.0, float("nan")]), np.array([1.0, 1.0]), "abs", 0.1)


def test_relative_mode_absolute_bound_is_nan() -> None:
    values = np.array([2.0, 4.0])
    certificate = certify(values, values.copy(), "rel", 0.01)
    assert math.isnan(certificate.effective_absolute_bound)
    assert certificate.bound_holds


def test_rate_fields_are_consistent(signals: dict[str, np.ndarray]) -> None:
    values = signals["magnetometer"]
    result = encode(values, "abs", 0.01)
    certificate = result.certificate
    assert certificate.original_bytes == 8 * values.size
    assert certificate.bits_per_sample == pytest.approx(
        8.0 * certificate.encoded_bytes / values.size, rel=1e-15
    )
    assert certificate.compression_ratio == pytest.approx(
        8.0 * values.size / certificate.encoded_bytes, rel=1e-15
    )


def test_digest_is_stable_and_changes_with_the_bytes() -> None:
    assert digest_bytes(b"abc") == digest_bytes(b"abc")
    assert digest_bytes(b"abc") != digest_bytes(b"abd")
    assert len(digest_bytes(b"")) == 64


def test_rate_below_entropy_reference_is_reported_not_asserted(
    signals: dict[str, np.ndarray]
) -> None:
    """The flag exists; whether it is set is a measurement, not a requirement.

    The first-order entropy is not a floor for a correlated source, so a rate
    below it is legitimate. This test only requires the flag to agree with the
    two numbers it is derived from.
    """
    values = signals["panel_temperature"]
    certificate = encode(values, "abs", 1e-3).certificate
    assert certificate.rate_below_entropy_reference == (
        certificate.bits_per_sample < certificate.residual_entropy_bits
    )
