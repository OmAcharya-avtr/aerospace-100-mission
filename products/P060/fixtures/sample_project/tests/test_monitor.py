"""Fixture test suite for the sample project.

This file is INPUT DATA for traceaudit, not a test suite of this repository.
It is deliberately defective: one test is skipped, one is xfailed, one fails,
one asserts nothing, and one claims a requirement that is not declared. Those
defects are the point. The repository's pytest configuration collects only
`tests/`, so nothing here runs as part of this package's own suite.
"""

import pytest


def _check_nonneg(value):
    """Assertion helper, deliberately placed outside the test functions."""
    assert value >= 0


@pytest.mark.verifies("REQ-001")
def test_rejects_out_of_range_speed():
    with pytest.raises(ValueError):
        raise ValueError("wheel speed 9000 rad/s outside declared range")


@pytest.mark.verifies("REQ-002")
def test_accumulates_momentum():
    assert 0.1 + 0.2 == pytest.approx(0.3)


@pytest.mark.verifies("REQ-003")
@pytest.mark.skip(reason="simulator fixture not available in this environment")
def test_desaturation_threshold():
    assert 1.0 > 0.5


@pytest.mark.verifies("REQ-004")
@pytest.mark.xfail(reason="hysteresis band off by one sample, open defect")
def test_hysteresis_band():
    assert 0.4 > 0.5


@pytest.mark.verifies("REQ-006")
def test_request_counter():
    counter = {"requests": 3}
    _check_nonneg(counter["requests"])


@pytest.mark.verifies("REQ-007")
def test_reset_clears_momentum():
    assert {"momentum": 0.0}["momentum"] == 0.0


@pytest.mark.verifies("REQ-008")
def test_saturation_ceiling():
    assert min(12.0, 10.0) == 11.0


@pytest.mark.verifies("REQ-042")
def test_claims_an_undeclared_requirement():
    assert True is not False
