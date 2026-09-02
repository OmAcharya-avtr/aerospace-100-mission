"""Fault models injected into the GNC loop.

Seven fault modes plus the fault-free case, covering the two places a GNC loop
can break: the measurement path and the actuation path.  The taxonomy and the
additive/multiplicative distinction follow the model-based FDI literature
(Gertler, *Fault Detection and Diagnosis in Engineering Systems*, Marcel
Dekker, 1998, ch. 1; Chen and Patton, *Robust Model-Based Fault Diagnosis for
Dynamic Systems*, Kluwer, 1999, ch. 1; Isermann, *Fault-Diagnosis Systems*,
Springer, 2006, ch. 1).

Sensor faults act on the measurement ``z_k`` after the true measurement is
formed:

===================  =========================================================
``SENSOR_BIAS``      ``z_i -> z_i + b``, ``b`` constant from onset.  Additive.
``SENSOR_DRIFT``     ``z_i -> z_i + a (t - t_0)``, a ramp.  Additive.
``SENSOR_STUCK``     ``z_i -> z_i(t_0)``, frozen at the onset sample.
``SENSOR_DROPOUT``   ``z_i -> 0`` (a dead channel reporting zero).
===================  =========================================================

Actuator faults act on the commanded torque before it reaches the plant, and
the *filter is not told*, so the filter propagates the commanded torque while
the plant receives the faulted one:

============================  ================================================
``ACTUATOR_LOSS_OF_EFFECT``   ``u -> (1 - l) u``, ``0 < l <= 1``.  Multiplicative.
``ACTUATOR_STUCK``            ``u -> u(t_0)``, frozen at the onset sample.
``ACTUATOR_RUNAWAY``          ``u -> u + c (t - t_0)``, ramping to the torque limit.
============================  ================================================

Detectability
-------------
Two of these are structurally hard and the package says so rather than hiding
it.  ``ACTUATOR_STUCK`` is invisible whenever the healthy command happens to
sit near the frozen value, which is exactly what a settled controller does; a
loss of effectiveness is invisible when the command is near zero, because
``(1 - l) * 0 == 0``.  Both are quantified in ``validation/VALIDATION.md``.

Units
-----
Angle-channel faults are in [rad], rate-channel faults in [rad/s], actuator
faults in [N m] (or dimensionless for the loss-of-effectiveness fraction), and
drift/runaway rates carry an extra ``1/s``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "FaultType",
    "FaultSpec",
    "FAULT_CLASSES",
    "ACTUATOR_FAULTS",
    "SENSOR_FAULTS",
    "class_index",
    "apply_sensor_fault",
    "apply_actuator_fault",
]


class FaultType(Enum):
    """Fault taxonomy.  ``NONE`` is the fault-free hypothesis ``H0``."""

    NONE = "none"
    SENSOR_BIAS = "sensor_bias"
    SENSOR_DRIFT = "sensor_drift"
    SENSOR_STUCK = "sensor_stuck"
    SENSOR_DROPOUT = "sensor_dropout"
    ACTUATOR_LOSS_OF_EFFECT = "actuator_loss_of_effectiveness"
    ACTUATOR_STUCK = "actuator_stuck"
    ACTUATOR_RUNAWAY = "actuator_runaway"


#: Canonical class order used by every confusion matrix and by the classifier.
FAULT_CLASSES: tuple[FaultType, ...] = tuple(FaultType)

SENSOR_FAULTS: frozenset[FaultType] = frozenset(
    {
        FaultType.SENSOR_BIAS,
        FaultType.SENSOR_DRIFT,
        FaultType.SENSOR_STUCK,
        FaultType.SENSOR_DROPOUT,
    }
)

ACTUATOR_FAULTS: frozenset[FaultType] = frozenset(
    {
        FaultType.ACTUATOR_LOSS_OF_EFFECT,
        FaultType.ACTUATOR_STUCK,
        FaultType.ACTUATOR_RUNAWAY,
    }
)


def class_index(fault: FaultType) -> int:
    """Index of ``fault`` in :data:`FAULT_CLASSES`."""
    return FAULT_CLASSES.index(fault)


@dataclass(frozen=True)
class FaultSpec:
    """One injected fault.

    Parameters
    ----------
    kind : FaultType
        Which fault.  ``FaultType.NONE`` disables injection entirely and all
        other fields are ignored.
    onset_step : int
        Index ``k0`` of the first faulted sample.  Must be non-negative.
    magnitude : float
        Fault size.  Units depend on ``kind``: [rad] or [rad/s] for a sensor
        bias, [rad/s] or [rad/s^2] for a sensor drift *rate*, dimensionless in
        ``(0, 1]`` for loss of effectiveness, [N m/s] for a runaway ramp rate.
        Ignored by the stuck and dropout modes, which have no free magnitude.
    channel : int
        Sensor channel the fault acts on: 0 = star tracker (angle),
        1 = rate gyro.  Ignored by actuator faults.

    Raises
    ------
    ValueError
        Negative ``onset_step``, a channel outside ``{0, 1}``, a non-finite
        magnitude, or a loss-of-effectiveness fraction outside ``(0, 1]``.
    """

    kind: FaultType = FaultType.NONE
    onset_step: int = 0
    magnitude: float = 0.0
    channel: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.kind, FaultType):
            raise TypeError(f"kind must be a FaultType, got {type(self.kind).__name__}")
        if int(self.onset_step) < 0:
            raise ValueError(f"onset_step must be >= 0, got {self.onset_step}")
        if self.channel not in (0, 1):
            raise ValueError(f"channel must be 0 (angle) or 1 (rate), got {self.channel}")
        if not np.isfinite(self.magnitude):
            raise ValueError(f"magnitude must be finite, got {self.magnitude}")
        if self.kind is FaultType.ACTUATOR_LOSS_OF_EFFECT and not (
            0.0 < float(self.magnitude) <= 1.0
        ):
            raise ValueError(
                "loss-of-effectiveness magnitude must lie in (0, 1], "
                f"got {self.magnitude}"
            )
        if self.kind in (FaultType.SENSOR_BIAS, FaultType.SENSOR_DRIFT) and (
            float(self.magnitude) == 0.0
        ):
            raise ValueError(f"{self.kind.value} needs a non-zero magnitude")

    @property
    def is_active_class(self) -> bool:
        """True unless this is the fault-free hypothesis."""
        return self.kind is not FaultType.NONE


def apply_sensor_fault(
    z: NDArray[np.float64],
    spec: FaultSpec,
    step: int,
    dt_s: float,
    frozen_value: float | None,
) -> tuple[NDArray[np.float64], float | None]:
    """Corrupt a measurement vector in place-safe fashion.

    Parameters
    ----------
    z : ndarray, shape (2,)
        Healthy measurement ``[angle_rad, rate_rad_s]``.
    spec : FaultSpec
        Fault description.  Non-sensor faults return ``z`` unchanged.
    step : int
        Current sample index.
    dt_s : float
        Sample period [s], used by the drift ramp.
    frozen_value : float or None
        Latched value for ``SENSOR_STUCK``; pass the value returned by the
        previous call, or ``None`` on the first faulted sample.

    Returns
    -------
    (z_faulted, frozen_value) : tuple
        The corrupted measurement and the updated latch.
    """
    out = np.asarray(z, dtype=float).copy()
    if spec.kind not in SENSOR_FAULTS or step < spec.onset_step:
        return out, frozen_value
    i = spec.channel
    if spec.kind is FaultType.SENSOR_BIAS:
        out[i] += float(spec.magnitude)
    elif spec.kind is FaultType.SENSOR_DRIFT:
        out[i] += float(spec.magnitude) * (step - spec.onset_step) * float(dt_s)
    elif spec.kind is FaultType.SENSOR_STUCK:
        if frozen_value is None:
            frozen_value = float(out[i])
        out[i] = frozen_value
    elif spec.kind is FaultType.SENSOR_DROPOUT:
        out[i] = 0.0
    return out, frozen_value


def apply_actuator_fault(
    u_nm: float,
    spec: FaultSpec,
    step: int,
    dt_s: float,
    frozen_value: float | None,
) -> tuple[float, float | None]:
    """Corrupt the commanded torque before it reaches the plant.

    Parameters
    ----------
    u_nm : float
        Healthy commanded torque [N m].
    spec : FaultSpec
        Fault description.  Non-actuator faults return ``u_nm`` unchanged.
    step : int
        Current sample index.
    dt_s : float
        Sample period [s], used by the runaway ramp.
    frozen_value : float or None
        Latched command for ``ACTUATOR_STUCK``.

    Returns
    -------
    (u_faulted_nm, frozen_value) : tuple
        The torque the plant actually receives [N m] and the updated latch.
    """
    u = float(u_nm)
    if spec.kind not in ACTUATOR_FAULTS or step < spec.onset_step:
        return u, frozen_value
    if spec.kind is FaultType.ACTUATOR_LOSS_OF_EFFECT:
        u = (1.0 - float(spec.magnitude)) * u
    elif spec.kind is FaultType.ACTUATOR_STUCK:
        if frozen_value is None:
            frozen_value = u
        u = frozen_value
    elif spec.kind is FaultType.ACTUATOR_RUNAWAY:
        u = u + float(spec.magnitude) * (step - spec.onset_step) * float(dt_s)
    return u, frozen_value
