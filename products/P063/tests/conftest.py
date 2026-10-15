"""Shared fixtures."""

from __future__ import annotations

import pytest

from safemodetree import (
    Fault,
    Mode,
    ObservabilityModel,
    ObservationChannel,
    Precondition,
    Response,
    SafeModeSpec,
    load_spec,
    packaged_spec_path,
)

MINIMAL_YAML: str = """
name: minimal
initial_mode: ops
modes:
  - id: ops
  - id: hold
    safe: true
faults:
  - id: f_one
    effect: something
    severity: "II"
    detection_latency_s: 5.0
responses:
  - id: r_one
    fault: f_one
    to_mode: hold
    cost_min: 7.5
channels:
  - id: c_one
    unit: deg
    nominal: 0.0
    threshold: 1.0
    faulted: 2.0
    direction: above
    noise_sd: 0.25
signatures:
  f_one:
    c_one: 1
"""


@pytest.fixture(scope="session")
def fmea() -> SafeModeSpec:
    """The shipped reaction-wheel declaration."""
    return load_spec(packaged_spec_path("reaction_wheel_fmea"))


@pytest.fixture(scope="session")
def planted() -> SafeModeSpec:
    """The shipped planted-unreachable declaration."""
    return load_spec(packaged_spec_path("planted_unreachable"))


@pytest.fixture
def minimal() -> SafeModeSpec:
    """A two-mode, one-fault declaration built in Python, not YAML."""
    return SafeModeSpec(
        name="minimal",
        modes=(Mode(id="ops"), Mode(id="hold", safe=True)),
        faults=(Fault(id="f_one", effect="something", severity="II", detection_latency_s=5.0),),
        responses=(
            Response(
                id="r_one",
                fault="f_one",
                to_mode="hold",
                cost_min=7.5,
                precondition=Precondition(()),
            ),
        ),
        observability=ObservabilityModel(
            channels=(
                ObservationChannel(
                    id="c_one",
                    unit="deg",
                    nominal=0.0,
                    threshold=1.0,
                    faulted=2.0,
                    direction="above",
                    noise_sd=0.25,
                ),
            ),
            signatures={"f_one": (1,)},
        ),
        initial_mode="ops",
    )


@pytest.fixture
def minimal_yaml() -> str:
    """A minimal valid declaration as YAML text."""
    return MINIMAL_YAML
