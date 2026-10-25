"""The covariance prohibition. Every refusal fires, and every legitimate call passes.

An assertion that something raises is worthless if it would raise on
everything, so each refusal here is paired with a negative control: a request
that must be accepted. Without the pairing a guard that raised
unconditionally would pass the whole file.
"""

from __future__ import annotations

import pytest

from orbitdrift import guard
from orbitdrift.errors import CovarianceRefused


def test_refuse_covariance_always_raises_and_names_p071() -> None:
    with pytest.raises(CovarianceRefused) as caught:
        guard.refuse_covariance()
    message = str(caught.value)
    assert "P071" in message
    assert "ConjProb" in message
    assert "orbit determination" in message


def test_refuse_covariance_prefixes_the_detail() -> None:
    with pytest.raises(CovarianceRefused, match="something specific"):
        guard.refuse_covariance("something specific")


@pytest.mark.parametrize(
    "purpose",
    [
        "covariance",
        "need a covariance for screening",
        "conjunction assessment",
        "collision probability",
        "compute Pc",
        "CDM generation",
        "close approach report",
        "miss distance estimate",
        "manoeuvre screening",
        "encounter geometry",
        "COVARIANCE FOR CONJUNCTION",
    ],
)
def test_forbidden_purposes_are_refused(purpose: str) -> None:
    with pytest.raises(CovarianceRefused):
        guard.check_purpose(purpose)


@pytest.mark.parametrize(
    "purpose",
    [
        "ephemeris staleness check",
        "decide whether to re-task tracking",
        "pass prediction for a ground station",
        "is this TLE too old for my pointing budget",
        "pc12345 is an identifier, not the word",
        "deciding when to request a fresh element set",
    ],
)
def test_legitimate_purposes_pass_unchanged(purpose: str) -> None:
    """The negative control. Without this, an unconditional raise would pass."""
    assert guard.check_purpose(purpose) == purpose


def test_pc_is_matched_as_a_whole_word_only() -> None:
    """``pc`` must refuse on its own and must not refuse inside another word."""
    with pytest.raises(CovarianceRefused):
        guard.check_purpose("estimate pc")
    assert guard.check_purpose("spacecraft operations") == "spacecraft operations"


def test_check_purpose_rejects_a_non_string() -> None:
    with pytest.raises(TypeError, match="must be a string"):
        guard.check_purpose(42)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "key",
    [
        "covariance",
        "position_covariance",
        "cov",
        "cov_matrix",
        "sigma_matrix",
        "p_matrix",
        "state_covariance",
        "COVARIANCE",
    ],
)
def test_covariance_shaped_keys_are_refused_in_a_mapping(key: str) -> None:
    with pytest.raises(CovarianceRefused, match="covariance-shaped key"):
        guard.audit_mapping_for_covariance_keys({key: [1.0, 2.0]}, where="test")


def test_a_nested_covariance_key_is_found() -> None:
    payload = {"outer": {"inner": {"state_covariance": 1.0}}}
    with pytest.raises(CovarianceRefused, match=r"test\.outer\.inner\.state_covariance"):
        guard.audit_mapping_for_covariance_keys(payload, where="test")


def test_an_ordinary_payload_passes_the_audit() -> None:
    """The negative control for the key audit.

    ``log_residual_std`` and ``coverage`` contain neither pattern, and
    ``recovery`` contains the letters of ``cov`` but not as a whole word,
    which the pattern must not match.
    """
    payload = {
        "drift_km": [1.0, 2.0],
        "log_residual_std": 0.1,
        "coverage": 0.9,
        "recovery": "not a covariance key",
        "nested": {"ages_days": [1.0], "interval_score": 3.0},
    }
    guard.audit_mapping_for_covariance_keys(payload, where="test")


def test_covariance_refused_is_both_an_orbitdrift_error_and_a_permission_error() -> None:
    """The exception type is part of the contract: callers may catch either."""
    from orbitdrift.errors import OrbitDriftError

    assert issubclass(CovarianceRefused, OrbitDriftError)
    assert issubclass(CovarianceRefused, PermissionError)


def test_forbidden_terms_list_is_not_empty_and_contains_the_core_three() -> None:
    assert "covariance" in guard.FORBIDDEN_PURPOSE_TERMS
    assert "conjunction" in guard.FORBIDDEN_PURPOSE_TERMS
    assert "collision" in guard.FORBIDDEN_PURPOSE_TERMS
