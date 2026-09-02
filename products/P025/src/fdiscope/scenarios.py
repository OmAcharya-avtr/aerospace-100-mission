"""Seeded synthetic scenario generation.

A scenario is one closed-loop run: a noise seed, a fault class, a magnitude, a
sensor channel and an onset sample.  Everything is drawn from a
`numpy.random.Generator` seeded by an explicit integer, so
``sample_scenarios(n, seed0)`` is reproducible bit for bit and the whole
dataset is regenerable from ``data/generate_dataset.py``.

Class balance is exact by construction: the ``i``-th scenario of a set takes
class ``FAULT_CLASSES[i % 8]``.  A confusion matrix over an unbalanced set is
much harder to read, and the mission asks for the confusion matrix in full.

Magnitude ranges
----------------
Sensor fault sizes are quoted in units of the corresponding measurement noise
standard deviation, so they mean the same thing for both channels:
``sigma_angle = sqrt(7.6154e-7) = 8.727e-4 rad`` (0.05 deg 1-sigma) and
``sigma_rate = sqrt(1.2185e-7) = 3.491e-4 rad/s`` (0.02 deg/s 1-sigma).  These
are engineering choices covering a plausible coarse-pointing sensor suite;
they are **not** fitted to any measured hardware population and no claim is
made that they are representative of any real sensor.

Actuator fault sizes are quoted against the wheel torque limit
``max_torque_nm``.

Undetectable-by-construction cases are kept
-------------------------------------------
The sampler does **not** filter out scenarios that no detector can see -- a
stuck actuator latched near the value the healthy controller was about to
command, or a loss of effectiveness during a low-torque phase.  Dropping them
would inflate every reported detection rate.  They stay in, and the
``never detected`` counts in ``validation/VALIDATION.md`` are what they cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .faults import FAULT_CLASSES, FaultSpec, FaultType
from .plant import ControllerGains, PlantConfig
from .simulate import LoopConfig

__all__ = [
    "Scenario",
    "ScenarioSet",
    "MagnitudeRanges",
    "DEFAULT_RANGES",
    "sample_scenario",
    "sample_scenarios",
    "DEFAULT_N_STEPS",
]

#: Samples per scenario.  At ``dt = 0.1 s`` this is 200 s of flight.
DEFAULT_N_STEPS: int = 2000

#: Onset is drawn uniformly from this window of sample indices, leaving a
#: fault-free stretch at the start for the false-alarm measurement and at
#: least 700 faulted samples at the end for the detection measurement.
_ONSET_LO: int = 600
_ONSET_HI: int = 1300


@dataclass(frozen=True)
class Scenario:
    """One simulated run.

    Attributes
    ----------
    index : int
        Position within its set.
    seed : int
        Seed of the scenario's own random draws *and* of the loop noise.
    fault : FaultSpec
        The injected fault (possibly ``FaultType.NONE``).
    n_steps : int
        Samples to simulate.
    label : FaultType
        Ground-truth class, equal to ``fault.kind``.
    """

    index: int
    seed: int
    fault: FaultSpec
    n_steps: int = DEFAULT_N_STEPS

    @property
    def label(self) -> FaultType:
        """Ground-truth fault class."""
        return self.fault.kind

    @property
    def onset_step(self) -> int:
        """First faulted sample; still defined for fault-free scenarios, where
        it marks the start of the segment used for the false-alarm count."""
        return int(self.fault.onset_step)

    def config(
        self,
        plant: PlantConfig | None = None,
        gains: ControllerGains | None = None,
        noise: bool = True,
    ) -> LoopConfig:
        """Loop configuration for this scenario."""
        return LoopConfig(
            plant=plant if plant is not None else PlantConfig(),
            gains=gains if gains is not None else ControllerGains(),
            n_steps=int(self.n_steps),
            seed=int(self.seed),
            noise=bool(noise),
        )


@dataclass(frozen=True)
class MagnitudeRanges:
    """Sampling ranges, all inclusive of both ends.

    Attributes
    ----------
    bias_sigma : tuple of float
        Sensor bias in units of that channel's measurement sigma.
    drift_angle_sigma_per_s : tuple of float
        Attitude-channel drift rate in sigma per second.  Wider than the rate
        channel's because an angle drift is largely absorbed by the estimator
        (see :func:`fdiscope.analytic.innovation_dc_gain`), so the same
        sigma-per-second produces a far smaller residual.
    drift_rate_sigma_per_s : tuple of float
        Gyro-channel drift rate in sigma per second.
    loss_fraction : tuple of float
        Loss-of-effectiveness fraction, dimensionless.
    runaway_nm_per_s : tuple of float
        Actuator runaway ramp rate [N m/s].
    """

    bias_sigma: tuple[float, float] = (1.0, 8.0)
    drift_angle_sigma_per_s: tuple[float, float] = (0.4, 4.0)
    drift_rate_sigma_per_s: tuple[float, float] = (0.02, 0.30)
    loss_fraction: tuple[float, float] = (0.20, 1.00)
    runaway_nm_per_s: tuple[float, float] = (1.0e-5, 2.0e-4)


DEFAULT_RANGES: MagnitudeRanges = MagnitudeRanges()


def _channel_sigma(plant: PlantConfig, channel: int) -> float:
    return float(
        np.sqrt(plant.attitude_var_rad2 if channel == 0 else plant.gyro_var_rad2_s2)
    )


def sample_scenario(
    seed: int,
    index: int = 0,
    plant: PlantConfig | None = None,
    ranges: MagnitudeRanges | None = None,
    n_steps: int = DEFAULT_N_STEPS,
    fault_class: FaultType | None = None,
) -> Scenario:
    """Draw one scenario.

    Parameters
    ----------
    seed : int
        Seed for this scenario's draws and for the loop noise.
    index : int
        Position in the set; selects the class when ``fault_class`` is None.
    plant : PlantConfig, optional
        Used only to convert sigma-relative magnitudes into physical units.
    ranges : MagnitudeRanges, optional
        Sampling ranges; defaults to :data:`DEFAULT_RANGES`.
    n_steps : int
        Samples to simulate.
    fault_class : FaultType, optional
        Force a class instead of using ``index % 8``.

    Returns
    -------
    Scenario
    """
    p = plant if plant is not None else PlantConfig()
    rg = ranges if ranges is not None else DEFAULT_RANGES
    rng = np.random.default_rng(int(seed))
    if fault_class is not None:
        kind = fault_class
    else:
        kind = FAULT_CLASSES[int(index) % len(FAULT_CLASSES)]
    channel = int(rng.integers(0, 2))
    onset = int(rng.integers(_ONSET_LO, _ONSET_HI + 1))
    sigma = _channel_sigma(p, channel)

    if kind is FaultType.NONE:
        magnitude = 0.0
    elif kind is FaultType.SENSOR_BIAS:
        magnitude = float(rng.uniform(*rg.bias_sigma)) * sigma * float(rng.choice([-1.0, 1.0]))
    elif kind is FaultType.SENSOR_DRIFT:
        span = rg.drift_angle_sigma_per_s if channel == 0 else rg.drift_rate_sigma_per_s
        magnitude = float(rng.uniform(*span)) * sigma * float(rng.choice([-1.0, 1.0]))
    elif kind is FaultType.ACTUATOR_LOSS_OF_EFFECT:
        magnitude = float(rng.uniform(*rg.loss_fraction))
    elif kind is FaultType.ACTUATOR_RUNAWAY:
        magnitude = float(rng.uniform(*rg.runaway_nm_per_s)) * float(rng.choice([-1.0, 1.0]))
    else:
        magnitude = 0.0

    return Scenario(
        index=int(index),
        seed=int(seed),
        fault=FaultSpec(kind=kind, onset_step=onset, magnitude=magnitude, channel=channel),
        n_steps=int(n_steps),
    )


def sample_scenarios(
    n: int,
    seed0: int,
    plant: PlantConfig | None = None,
    ranges: MagnitudeRanges | None = None,
    n_steps: int = DEFAULT_N_STEPS,
) -> list[Scenario]:
    """Draw ``n`` scenarios with seeds ``seed0 .. seed0 + n - 1``.

    Classes are cycled so the set is exactly balanced when ``n`` is a multiple
    of eight.

    Raises
    ------
    ValueError
        If ``n < 1``.
    """
    if int(n) < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    return [
        sample_scenario(int(seed0) + i, index=i, plant=plant, ranges=ranges, n_steps=n_steps)
        for i in range(int(n))
    ]


@dataclass(frozen=True)
class ScenarioSet:
    """A named list of scenarios, for readable provenance in reports."""

    name: str
    scenarios: list[Scenario] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.scenarios)

    def labels(self) -> list[FaultType]:
        """Ground-truth labels in order."""
        return [s.label for s in self.scenarios]
