"""Dry-run mode: proven by a counting stub, not by reading the code.

The Level 3 validation requirement is "dry-run mode provably issues no write
(verified by a counting stub, not by inspection)". Three layers of evidence:

1. The wrapped actuator's ``write_count`` is still zero after the run.
2. A :class:`~hilforge.backends.stubs.CountingActuator` with
   ``forbid_writes=True`` underneath raises if anything reaches it, so a leak
   fails the test rather than being missed.
3. The plant's held torque is unchanged, so nothing moved.
"""

from __future__ import annotations

import numpy as np
import pytest

from hilforge.backends import make_backend_pair
from hilforge.backends.stubs import CountingActuator
from hilforge.dryrun import DryRunActuator
from hilforge.errors import DryRunViolationError
from hilforge.hal import ChannelSpec
from hilforge.loop import HilLoop, LoopConfig
from hilforge.timing import PeriodSpec

PERIOD = 0.010
N = 200


def _dry_cfg(n=N, dry_run=True):
    return LoopConfig(
        period=PeriodSpec(period_s=PERIOD),
        n_iterations=n,
        dry_run=dry_run,
        injected_durations_s=tuple(np.full(n, 0.004)),
    )


def test_dry_run_leaves_the_actuator_write_count_at_zero():
    sim, _ = make_backend_pair()
    sim.open()
    actuator = sim.actuators["torque"]
    record = HilLoop(sim, _dry_cfg()).run()
    assert actuator.write_count == 0
    assert record.rehearsed_writes == N
    assert record.writes_issued == 0
    sim.close()


def test_normal_run_does_write_so_the_zero_above_means_something():
    sim, _ = make_backend_pair()
    sim.open()
    actuator = sim.actuators["torque"]
    record = HilLoop(sim, _dry_cfg(dry_run=False)).run()
    assert actuator.write_count == N
    assert record.writes_issued == N
    assert record.rehearsed_writes == 0
    sim.close()


def test_dry_run_leaves_the_plant_torque_untouched():
    sim, _ = make_backend_pair()
    sim.open()
    before = sim.plant.snapshot()["torque"]
    HilLoop(sim, _dry_cfg(n=50)).run()
    assert sim.plant.snapshot()["torque"] == before
    sim.close()


def test_forbidding_stub_proves_no_write_reaches_the_channel():
    spec = ChannelSpec(name="torque", length=1, units="N*m", lower=-1.5, upper=1.5)
    stub = CountingActuator(spec, forbid_writes=True)
    guard = DryRunActuator(stub)
    for _ in range(100):
        ack = guard.write(np.array([0.2]))
        assert ack.channel == "torque"
    assert stub.attempts == 0
    assert stub.write_count == 0
    assert guard.rehearsed == 100
    assert guard.write_count == 0


def test_forbidding_stub_raises_if_written_directly():
    spec = ChannelSpec(name="torque", length=1, units="N*m")
    stub = CountingActuator(spec, forbid_writes=True)
    with pytest.raises(DryRunViolationError):
        stub.write(np.array([0.0]))
    assert stub.attempts == 1
    assert stub.write_count == 0


def test_counting_stub_counts_when_allowed():
    spec = ChannelSpec(name="torque", length=1, units="N*m")
    stub = CountingActuator(spec)
    for i in range(5):
        ack = stub.write(np.array([float(i)]))
        assert ack.sequence == i + 1
    assert stub.write_count == 5
    assert stub.attempts == 5
    assert stub.last_command()[0] == 4.0


def test_dry_run_still_validates_and_clips():
    spec = ChannelSpec(name="torque", length=1, units="N*m", lower=-0.5, upper=0.5)
    stub = CountingActuator(spec, forbid_writes=True)
    guard = DryRunActuator(stub)
    ack = guard.write(np.array([2.0]))
    assert ack.applied[0] == pytest.approx(0.5)
    assert ack.saturated is True
    assert guard.last_command()[0] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        guard.write(np.array([np.nan]))
    with pytest.raises(ValueError):
        guard.write(np.array([0.1, 0.2]))
    assert stub.attempts == 0


def test_dry_run_exposes_the_inner_channel_for_assertions():
    spec = ChannelSpec(name="torque", length=1, units="N*m")
    stub = CountingActuator(spec)
    guard = DryRunActuator(stub)
    assert guard.inner is stub
    assert guard.spec is spec
    assert guard.last_command() is None


def test_dry_run_trajectory_diverges_because_nothing_was_actuated():
    """A dry run's commands diverge from a real run's — that is the point.

    The first iteration is computed from the same sensor sample, so the first
    command is identical. From the second iteration on the real run's plant has
    responded to a torque and the dry run's has not, so the two trajectories
    separate. The divergence is positive evidence that no torque was applied in
    the dry run: if a write had leaked through, the two would agree.
    """
    sim_a, _ = make_backend_pair()
    sim_b, _ = make_backend_pair()
    dry = HilLoop(sim_a, _dry_cfg(n=60)).run()
    wet = HilLoop(sim_b, _dry_cfg(n=60, dry_run=False)).run()
    dry_cmd = dry.signal_matrix()[:, 3]
    wet_cmd = wet.signal_matrix()[:, 3]
    assert dry_cmd[0] == wet_cmd[0]
    assert abs(dry_cmd[-1] - wet_cmd[-1]) > 0.1
    # The real run drove the attitude toward zero; the dry run did not.
    assert abs(wet.iterations[-1].theta_hat_rad) < abs(dry.iterations[-1].theta_hat_rad)
