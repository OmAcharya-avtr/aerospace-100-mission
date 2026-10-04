"""A declared thermal-throttle model, for the throttled-device failure mode.

A budget that passes at the nominal clock can fail once the SoC hits its
thermal limit and drops frequency. This module expresses that as an explicit,
*declared* derating of a :class:`~edgeinfer.roofline.DeviceModel`, so that a
throttled estimate can be produced and checked against the same budget.

What this is not
----------------
It is **not a thermal model and not a measurement**. There is no junction
temperature, no thermal resistance, no power trace and no device here. The
throttle factor is a number the caller declares, exactly as the peaks of a
:class:`DeviceModel` are declared, and every object produced carries the
factor in its ``source`` string so that a throttled figure cannot be read as
a nominal one.

A real throttle curve for a Jetson Orin Nano would come from the device: its
DVFS governor's frequency table and the thermal zones in
``/sys/class/thermal``, logged under load. That is part of what is still
missing for a hardware-measured validation level, and it is listed as such in
``README.md``.

The scaling itself assumes performance is proportional to clock frequency,
which holds for the compute roof and does **not** hold for the memory roof on
most SoCs, where DRAM frequency throttles on a separate step schedule.
:func:`throttled_device` therefore takes independent compute and bandwidth
factors and defaults the bandwidth factor to 1.0 rather than silently
derating a quantity it has no basis to derate.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from edgeinfer.roofline import DeviceModel

__all__ = ["ThrottleState", "throttled_device"]


@dataclass(frozen=True)
class ThrottleState:
    """A declared throttle condition.

    Attributes
    ----------
    label
        Name of the condition, e.g. ``"sustained load, declared 60 % clock"``.
    compute_factor
        Multiplier on peak compute [dimensionless], in (0, 1].
    bandwidth_factor
        Multiplier on peak memory bandwidth [dimensionless], in (0, 1].
    basis
        Where the factors came from. Must be non-empty; the point of the field
        is that a throttle figure cannot travel without its provenance.
    """

    label: str
    compute_factor: float
    bandwidth_factor: float = 1.0
    basis: str = "declared by caller; not measured"

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("ThrottleState.label must be non-empty")
        if not self.basis:
            raise ValueError(
                "ThrottleState.basis must be non-empty: a throttle factor without a "
                "stated basis cannot be reported"
            )
        for field_name, value in (
            ("compute_factor", self.compute_factor),
            ("bandwidth_factor", self.bandwidth_factor),
        ):
            if not np.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(
                    f"{field_name} must be finite and lie in (0, 1], got {value}"
                )

    @property
    def is_nominal(self) -> bool:
        """True when nothing is derated."""
        return self.compute_factor == 1.0 and self.bandwidth_factor == 1.0


def throttled_device(device: DeviceModel, state: ThrottleState) -> DeviceModel:
    """Apply a declared :class:`ThrottleState` to a device model.

    Per-node dispatch overhead is left unchanged: it is dominated by driver
    and runtime work whose scaling with SoC clock this module has no basis to
    assert.

    Returns
    -------
    DeviceModel
        With ``name`` suffixed by the throttle label and ``source`` recording
        both factors and their basis, so a throttled latency is always
        labelled as one.
    """
    return replace(
        device,
        name=f"{device.name} [{state.label}]",
        peak_flops=device.peak_flops * state.compute_factor,
        peak_bandwidth_bytes_s=device.peak_bandwidth_bytes_s * state.bandwidth_factor,
        source=(
            f"{device.source}; THROTTLED: compute x{state.compute_factor:.4g}, "
            f"bandwidth x{state.bandwidth_factor:.4g} ({state.basis})"
        ),
    )
