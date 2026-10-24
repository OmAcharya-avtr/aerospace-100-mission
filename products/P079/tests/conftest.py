"""Shared fixtures. Nothing here reaches a device, because no device exists."""

from __future__ import annotations

from pathlib import Path

import pytest

from timetransfer.acquire import acquire
from timetransfer.backends import SimulatedBackend, SimulationSpec
from timetransfer.clockmodel import ClockSpec
from timetransfer.exchange import ExchangeSet
from timetransfer.linkmodel import LinkSpec

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "fixtures" / "recorded_exchange.json"


def make_spec(
    *,
    seed: int = 4242,
    asymmetry_s: float = 0.0,
    skew_s_per_s: float = 2.0e-6,
    drift_per_s: float = 0.0,
    loss_probability: float = 0.0,
    timestamp_noise_s: float = 0.0,
    resolution_ns: int = 1,
    jitter_scale_s: float = 2.0e-4,
    q1: float = 0.0,
    q2: float = 0.0,
    period_s: float = 1.0,
) -> SimulationSpec:
    """A simulation spec with every knob named, defaulting to a noiseless clock."""
    return SimulationSpec(
        clock=ClockSpec(
            offset0_s=1.0e-3,
            skew_s_per_s=skew_s_per_s,
            drift_per_s=drift_per_s,
            q1_s2_per_s=q1,
            q2_per_s=q2,
        ),
        link=LinkSpec(
            floor_ab_s=5.0e-3 + 0.5 * asymmetry_s,
            floor_ba_s=5.0e-3 - 0.5 * asymmetry_s,
            jitter_scale_ab_s=jitter_scale_s,
            jitter_scale_ba_s=jitter_scale_s,
            loss_probability=loss_probability,
        ),
        period_s=period_s,
        timestamp_noise_s=timestamp_noise_s,
        resolution_ns=resolution_ns,
        seed=seed,
    )


@pytest.fixture
def spec() -> SimulationSpec:
    """A default simulation spec: symmetric link, 2 ppm skew, no clock noise."""
    return make_spec()


@pytest.fixture
def exchanges(spec: SimulationSpec) -> ExchangeSet:
    """200 exchanges from the default spec."""
    backend = SimulatedBackend(spec)
    backend.open()
    report = acquire(backend, 200, min_exchanges=8)
    backend.close()
    return report.exchanges


@pytest.fixture
def recorded() -> ExchangeSet:
    """The committed dry-run recording."""
    if not FIXTURE.exists():
        pytest.skip(
            "fixtures/recorded_exchange.json is missing; regenerate it with "
            "python validation/regenerate_fixture.py"
        )
    return ExchangeSet.read_json(FIXTURE)


@pytest.fixture
def recorded_path() -> Path:
    """Path to the committed dry-run recording."""
    if not FIXTURE.exists():
        pytest.skip("fixtures/recorded_exchange.json is missing")
    return FIXTURE
