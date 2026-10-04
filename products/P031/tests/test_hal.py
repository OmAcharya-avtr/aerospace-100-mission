"""HAL contract: channel specs, sample and command validation, backend info."""

from __future__ import annotations

import numpy as np
import pytest

from hilforge.backends import LoopbackDriver, make_backend_pair
from hilforge.backends.device import AbsentDriver, DeviceBackend
from hilforge.errors import ConfigurationError
from hilforge.hal import (
    ActuatorChannel,
    BackendInfo,
    ChannelSpec,
    SensorChannel,
    WriteAck,
    validate_command,
    validate_sample,
)


def test_channel_spec_validation():
    spec = ChannelSpec(name="gyro", length=3, units="rad/s", lower=-10.0, upper=10.0)
    assert spec.length == 3 and spec.units == "rad/s"
    with pytest.raises(ConfigurationError):
        ChannelSpec(name="", length=1, units="-")
    with pytest.raises(ConfigurationError):
        ChannelSpec(name="x", length=0, units="-")
    with pytest.raises(ConfigurationError):
        ChannelSpec(name="x", length=1, units="-", lower=1.0, upper=0.0)


def test_validate_sample_accepts_in_range_and_reports_units():
    spec = ChannelSpec(name="ahrs", length=2, units="rad, rad/s", lower=-1.0, upper=1.0)
    out = validate_sample(spec, [0.1, -0.2])
    assert out.dtype == np.float64
    assert np.array_equal(out, np.array([0.1, -0.2]))
    with pytest.raises(ValueError, match="rad, rad/s"):
        validate_sample(spec, [0.1])
    with pytest.raises(ValueError, match="non-finite"):
        validate_sample(spec, [np.nan, 0.0])
    with pytest.raises(ValueError, match=r"outside \[-1.0, 1.0\]"):
        validate_sample(spec, [2.0, 0.0])


def test_validate_command_checks_shape_and_finiteness():
    spec = ChannelSpec(name="torque", length=1, units="N*m", lower=-1.0, upper=1.0)
    assert validate_command(spec, [0.5])[0] == 0.5
    with pytest.raises(ValueError, match="expected shape"):
        validate_command(spec, [0.5, 0.5])
    with pytest.raises(ValueError, match="non-finite"):
        validate_command(spec, [np.inf])


def test_validate_rejects_non_numeric():
    spec = ChannelSpec(name="x", length=1, units="-")
    with pytest.raises(TypeError):
        validate_command(spec, ["not a number"])


def test_write_ack_fields():
    ack = WriteAck(
        channel="torque", applied=np.array([0.2]), saturated=False, sequence=1
    )
    assert ack.channel == "torque" and ack.sequence == 1 and ack.saturated is False


def test_backend_info_is_hardware_is_false_for_stubs():
    assert BackendInfo(kind="simulated", name="s").is_hardware is False
    assert BackendInfo(kind="device", name="d", driver="loopback").is_hardware is False
    assert BackendInfo(kind="device", name="d", driver="absent").is_hardware is False
    assert BackendInfo(kind="device", name="d", driver="").is_hardware is False
    # Only a named real driver counts, and no such driver ships in this repo.
    assert BackendInfo(kind="device", name="d", driver="orin-nano-spi").is_hardware is True


def test_both_backends_satisfy_the_protocols():
    sim, dev = make_backend_pair()
    for backend in (sim, dev):
        backend.open()
        assert isinstance(backend.sensors["ahrs"], SensorChannel)
        assert isinstance(backend.actuators["torque"], ActuatorChannel)
        backend.close()


def test_absent_driver_backend_reports_absent_and_has_no_channels():
    backend = DeviceBackend(AbsentDriver())
    assert backend.info.is_hardware is False
    assert backend.sensors == {}
    assert backend.actuators == {}


def test_loopback_driver_rejects_unknown_channels():
    driver = LoopbackDriver()
    driver.connect()
    with pytest.raises(KeyError):
        driver.read_channel("magnetometer", 0.01)
    with pytest.raises(KeyError):
        driver.write_channel("thruster", np.array([0.0]))
