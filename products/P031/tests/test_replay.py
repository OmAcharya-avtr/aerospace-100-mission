"""Record, store and replay a run; deterministic seeded replay."""

from __future__ import annotations

import numpy as np
import pytest

from hilforge.backends import make_backend_pair
from hilforge.errors import ConfigurationError
from hilforge.hal import ChannelSpec
from hilforge.loop import HilLoop, LoopConfig
from hilforge.replay import (
    RecordingActuator,
    ReplayBackend,
    ReplaySensor,
    load_samples,
    replay_run,
    save_run,
)
from hilforge.timing import PeriodSpec

PERIOD = 0.010


def _run(n=150, seed=20261004):
    sim, _ = make_backend_pair(seed=seed, sample_dt_s=PERIOD)
    cfg = LoopConfig(
        period=PeriodSpec(period_s=PERIOD),
        n_iterations=n,
        injected_durations_s=tuple(np.linspace(0.003, 0.012, n)),
    )
    return HilLoop(sim, cfg).run()


def test_replay_reproduces_the_data_digest_exactly():
    original = _run()
    again = replay_run(original)
    assert again.data_digest() == original.data_digest()
    assert np.array_equal(again.signal_matrix(), original.signal_matrix())


def test_replay_reproduces_the_overrun_accounting():
    original = _run()
    again = replay_run(original)
    assert again.overruns.direct_indices == original.overruns.direct_indices
    assert again.overruns.cascade_indices == original.overruns.cascade_indices


def test_replay_backend_identity():
    original = _run(n=40)
    again = replay_run(original)
    assert again.backend_kind == "replay"
    assert again.backend_driver == "recorded"
    assert again.is_hardware is False


def test_replay_with_different_gains_changes_the_commands():
    from hilforge.plant import PDGains

    original = _run(n=80)
    cfg = LoopConfig(
        period=PeriodSpec(period_s=PERIOD),
        n_iterations=original.n_completed,
        gains=PDGains(kp=24.0, kd=16.8),
        injected_durations_s=tuple(original.durations_s()),
    )
    again = replay_run(original, cfg)
    # Same samples in, different control law, so the commands differ.
    assert not np.array_equal(again.signal_matrix()[:, 3], original.signal_matrix()[:, 3])
    assert np.array_equal(again.signal_matrix()[:, 0:2], original.signal_matrix()[:, 0:2])


def test_save_and_load_round_trip(tmp_path):
    original = _run(n=60)
    path = save_run(original, tmp_path / "run.npz")
    assert path.exists()
    samples = load_samples(path)
    assert samples.shape == (60, 2)
    assert np.array_equal(samples, original.signal_matrix()[:, 0:2])


def test_save_rejects_an_empty_run(tmp_path):
    original = _run(n=10)
    original.iterations.clear()
    with pytest.raises(ValueError):
        save_run(original, tmp_path / "empty.npz")


def test_replay_sensor_exhaustion_is_an_explicit_error():
    spec = ChannelSpec(name="ahrs", length=2, units="rad, rad/s")
    sensor = ReplaySensor(np.zeros((3, 2)), spec)
    for _ in range(3):
        sensor.read()
    with pytest.raises(IndexError, match="exhausted"):
        sensor.read()


def test_replay_sensor_validates_shape():
    spec = ChannelSpec(name="ahrs", length=2, units="rad, rad/s")
    with pytest.raises(ConfigurationError):
        ReplaySensor(np.zeros((3, 3)), spec)
    with pytest.raises(ConfigurationError):
        ReplaySensor(np.zeros(3), spec)


def test_recording_actuator_stores_every_command():
    spec = ChannelSpec(name="torque", length=1, units="N*m", lower=-1.0, upper=1.0)
    actuator = RecordingActuator(spec)
    assert actuator.last_command() is None
    for i in range(4):
        actuator.write(np.array([0.1 * i]))
    assert actuator.write_count == 4
    assert len(actuator.commands) == 4
    assert actuator.commands[2][0] == pytest.approx(0.2)


def test_replay_backend_lifecycle_and_timebase():
    backend = ReplayBackend(np.zeros((5, 2)))
    assert backend.is_open is False
    backend.open()
    assert backend.is_open is True
    assert backend.info.detail["n_samples"] == "5"
    t0 = backend.timebase.now()
    backend.advance(0.01)
    assert backend.timebase.now() == pytest.approx(t0 + 0.01, abs=1e-12)
    backend.close()
    backend.close()
    assert backend.is_open is False


def test_replay_of_a_measured_run_reinjects_its_durations():
    sim, _ = make_backend_pair(sample_dt_s=PERIOD)
    cfg = LoopConfig(period=PeriodSpec(period_s=PERIOD), n_iterations=40)
    measured = HilLoop(sim, cfg).run()
    assert measured.deterministic_timing is False
    again = replay_run(measured)
    assert again.deterministic_timing is True
    assert np.allclose(again.durations_s(), measured.durations_s())
