"""Backends: the simulated one, the device one, and the stubs behind it.

:func:`make_backend_pair` is the entry point the parity check uses. It builds a
simulated backend and a device backend over a loopback driver from the *same*
:class:`~timetransfer.backends.simulated.SimulationSpec`, both on a virtual
timebase, so the only thing that differs between them is the code path. Any
difference in their output is therefore a defect in the HAL layering and not a
difference in the physics.
"""

from __future__ import annotations

from .device import DeviceBackend
from .simulated import ExchangeEngine, SimulatedBackend, SimulationSpec
from .stubs import AbsentDriver, GuardedDriver, LoopbackDriver, RecordedDriver

__all__ = [
    "AbsentDriver",
    "DeviceBackend",
    "ExchangeEngine",
    "GuardedDriver",
    "LoopbackDriver",
    "RecordedDriver",
    "SimulatedBackend",
    "SimulationSpec",
    "make_backend_pair",
]


def make_backend_pair(spec: SimulationSpec) -> tuple[SimulatedBackend, DeviceBackend]:
    """A simulated backend and a loopback device backend from one spec.

    Both are returned closed. Both use a virtual timebase at the spec's
    resolution, so their outputs are comparable byte for byte.

    Returns
    -------
    tuple
        ``(simulated, device)``. ``device.info.is_hardware`` is ``False``; the
        driver is a loopback stub.
    """
    simulated = SimulatedBackend(spec, name="simulated-clock")
    device = DeviceBackend(
        LoopbackDriver(spec),
        name="device-link",
        period_s=float(spec.period_s),
        virtual_timebase=True,
        resolution_ns=int(spec.resolution_ns),
    )
    return simulated, device
