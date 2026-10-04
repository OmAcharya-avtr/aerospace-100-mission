"""Backend behaviour and simulation/device parity under the same test suite.

Every test in :class:`TestBothBackends` is parametrised over the simulated
backend and the loopback device backend, which is the "identical test suite
run against both backends" requirement.
"""

from __future__ import annotations

import numpy as np
import pytest

from hilforge.backends import (
    DeviceBackend,
    LoopbackDriver,
    SimulatedBackend,
    make_backend_pair,
)
from hilforge.backends.device import AbsentDriver
from hilforge.errors import DeviceAbsentError, DeviceDisconnectedError, WriteRejectedError
from hilforge.plant import PlantConfig

SEED = 20261004


def _pair():
    return make_backend_pair(seed=SEED)


@pytest.fixture(params=["simulated", "device"])
def backend(request):
    sim, dev = _pair()
    chosen = sim if request.param == "simulated" else dev
    chosen.open()
    yield chosen
    chosen.close()


class TestBothBackends:
    """The same assertions against both backends."""

    def test_channels_and_specs(self, backend):
        assert set(backend.sensors) == {"ahrs"}
        assert set(backend.actuators) == {"torque"}
        assert backend.sensors["ahrs"].spec.length == 2
        assert backend.sensors["ahrs"].spec.units == "rad, rad/s"
        assert backend.actuators["torque"].spec.units == "N*m"

    def test_is_hardware_is_false(self, backend):
        assert backend.info.is_hardware is False

    def test_read_returns_two_finite_values(self, backend):
        sample = backend.sensors["ahrs"].read()
        assert sample.shape == (2,)
        assert np.all(np.isfinite(sample))

    def test_write_acknowledges_and_counts(self, backend):
        act = backend.actuators["torque"]
        assert act.write_count == 0
        assert act.last_command() is None
        ack = act.write(np.array([0.1]))
        assert ack.sequence == 1 and ack.channel == "torque"
        assert act.write_count == 1
        assert act.last_command()[0] == pytest.approx(0.1)

    def test_write_outside_range_is_rejected(self, backend):
        with pytest.raises(WriteRejectedError) as exc:
            backend.actuators["torque"].write(np.array([99.0]))
        assert exc.value.reason == "out_of_range"
        assert backend.actuators["torque"].write_count == 0

    def test_timebase_advances_only_when_advanced(self, backend):
        t0 = backend.timebase.now()
        backend.advance(0.01)
        assert backend.timebase.now() == pytest.approx(t0 + 0.01, abs=1e-12)

    def test_snapshot_restore_round_trip(self, backend):
        snap = backend.snapshot()
        for _ in range(10):
            backend.sensors["ahrs"].read()
            backend.actuators["torque"].write(np.array([0.05]))
            backend.advance(0.01)
        backend.restore(snap)
        assert backend.actuators["torque"].write_count == snap["write_count"]
        assert backend.sensors["ahrs"].read_count == snap["read_count"]

    def test_close_is_idempotent(self, backend):
        backend.close()
        backend.close()
        assert backend.is_open is False


def test_simulated_and_device_reads_are_bit_identical():
    sim, dev = _pair()
    sim.open()
    dev.open()
    for _ in range(200):
        a = sim.sensors["ahrs"].read()
        b = dev.sensors["ahrs"].read()
        assert a.tobytes() == b.tobytes()
        sim.actuators["torque"].write(np.array([0.03]))
        dev.actuators["torque"].write(np.array([0.03]))
        sim.advance(0.01)
        dev.advance(0.01)
    assert sim.plant.state.tobytes() == dev.driver.plant.state.tobytes()


def test_absent_device_raises_on_open():
    backend = DeviceBackend(AbsentDriver(detail="SPI bus 0 silent"))
    with pytest.raises(DeviceAbsentError, match="SPI bus 0 silent"):
        backend.open()
    assert backend.is_open is False


def test_loopback_driver_refuses_reads_before_connect():
    driver = LoopbackDriver()
    with pytest.raises(DeviceDisconnectedError):
        driver.read_channel("ahrs", 0.01)


def test_device_backend_reports_driver_name_and_detail():
    backend = DeviceBackend(LoopbackDriver())
    info = backend.info
    assert info.kind == "device"
    assert info.driver == "loopback"
    assert info.detail["is_hardware"] == "False"


def test_simulated_backend_info_carries_the_seed_and_model():
    backend = SimulatedBackend(seed=4321)
    assert backend.info.detail["seed"] == "4321"
    assert "double integrator" in backend.info.detail["model"]


def test_actuator_saturation_is_reported_through_the_ack():
    cfg = PlantConfig(torque_limit_nm=0.1)
    sim = SimulatedBackend(cfg, seed=1)
    sim.open()
    ack = sim.actuators["torque"].write(np.array([0.05]))
    assert ack.saturated is False
    sim.close()


def test_loopback_connect_counter():
    driver = LoopbackDriver()
    backend = DeviceBackend(driver)
    backend.open()
    backend.close()
    backend.open()
    assert driver.connect_calls == 2
    backend.close()
