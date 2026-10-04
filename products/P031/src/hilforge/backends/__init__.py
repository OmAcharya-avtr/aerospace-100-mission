"""HAL backends: a simulated one, a device one, and the stubs that test them."""

from __future__ import annotations

from .device import AbsentDriver, DeviceBackend, DriverShim, LoopbackDriver
from .simulated import SimulatedBackend
from .stubs import (
    CountingActuator,
    DisconnectingDriver,
    DroppingDriver,
    RejectingDriver,
    make_backend_pair,
)

__all__ = [
    "AbsentDriver",
    "CountingActuator",
    "DeviceBackend",
    "DisconnectingDriver",
    "DriverShim",
    "DroppingDriver",
    "LoopbackDriver",
    "RejectingDriver",
    "SimulatedBackend",
    "make_backend_pair",
]
