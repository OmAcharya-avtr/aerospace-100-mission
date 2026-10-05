"""Fault taxonomy for GNC and communications software.

The taxonomy is the enumerable backbone of the package: a fault is a
``(kind, channel, parameter-vector, start, duration)`` tuple, and the coverage
metric of :mod:`faultinject.coverage` is defined over a discretisation of that
tuple.  Nothing here executes anything; this module is pure description.

Fault classes and kinds
-----------------------
The five classes and sixteen kinds follow the standard fault-model vocabulary
used in the dependability literature, in particular the fault/error/failure
taxonomy of

  A. Avizienis, J.-C. Laprie, B. Randell and C. Landwehr, "Basic Concepts and
  Taxonomy of Dependable and Secure Computing", *IEEE Transactions on
  Dependable and Secure Computing* **1**(1), 11-33 (2004),

and, for the sensor-fault kinds specifically, the fault-signature catalogue of

  I. Hwang, S. Kim, Y. Kim and C. E. Seah, "A Survey of Fault Detection,
  Isolation, and Reconfiguration Methods", *IEEE Transactions on Control
  Systems Technology* **18**(3), 636-653 (2010), Sec. II,

which lists bias, drift, scaling and "hard failure" (stuck/frozen) as the
sensor-fault models used in aerospace FDIR work.  The bus and timing kinds
(delay, reorder, loss, late sample, overrun) are the standard asynchronous
message-passing fault set; the numerical kinds follow IEEE 754-2019 Sec. 7
exception classes (invalid, overflow, underflow/subnormal).

Units
-----
Sensor and numerical parameters are in *channel units*: metres for the ``pos``
channel, metres per second for ``vel``, metres per second squared for ``u``.
Durations and delays are in loop steps (dimensionless integers); the step
period is a property of the target, not of the taxonomy.

Validity
--------
Parameter ranges below are the ranges over which the injection code in
:mod:`faultinject.faults` has been exercised by the test suite.  Values outside
a declared range are rejected by :func:`FaultSpec.validate` rather than
silently clipped.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum


class FaultClass(StrEnum):
    """Top-level fault class."""

    SENSOR = "sensor"
    ACTUATOR = "actuator"
    BUS = "bus"
    TIMING = "timing"
    NUMERICAL = "numerical"


class FaultKind(StrEnum):
    """The sixteen fault kinds of the taxonomy."""

    SENSOR_BIAS = "sensor_bias"
    SENSOR_DRIFT = "sensor_drift"
    SENSOR_STUCK = "sensor_stuck"
    SENSOR_DROPOUT = "sensor_dropout"
    SENSOR_QUANT_COLLAPSE = "sensor_quant_collapse"
    ACTUATOR_LOSS_EFFECTIVENESS = "actuator_loss_effectiveness"
    ACTUATOR_STUCK = "actuator_stuck"
    ACTUATOR_RUNAWAY = "actuator_runaway"
    BUS_DELAY = "bus_delay"
    BUS_REORDER = "bus_reorder"
    BUS_LOSS = "bus_loss"
    TIMING_LATE_SAMPLE = "timing_late_sample"
    TIMING_OVERRUN = "timing_overrun"
    NUMERICAL_NAN = "numerical_nan"
    NUMERICAL_DENORMAL = "numerical_denormal"
    NUMERICAL_OVERFLOW = "numerical_overflow"


class Scale(StrEnum):
    """Binning scale for a parameter."""

    LINEAR = "linear"
    LOG = "log"


@dataclass(frozen=True)
class ParamSpec:
    """One scalar fault parameter with units, range and a coverage binning.

    Parameters
    ----------
    name
        Parameter name as it appears in a :class:`~faultinject.campaign.FaultCase`.
    unit
        Physical unit string. ``"channel units"`` means the unit of the signal
        the fault is attached to; ``"step"`` means loop steps (dimensionless).
    lo, hi
        Inclusive range of admissible values.
    n_bins
        Number of coverage bins spanning ``[lo, hi]``.
    scale
        ``LINEAR`` bins are equal width in the value; ``LOG`` bins are equal
        width in ``log10`` of the value and require ``lo > 0``.
    integer
        If true, sampled values are rounded to the nearest integer.
    """

    name: str
    unit: str
    lo: float
    hi: float
    n_bins: int
    scale: Scale = Scale.LINEAR
    integer: bool = False

    def __post_init__(self) -> None:
        if self.n_bins < 1:
            raise ValueError(f"{self.name}: n_bins must be >= 1, got {self.n_bins}")
        if not self.hi > self.lo:
            raise ValueError(f"{self.name}: require hi > lo, got lo={self.lo} hi={self.hi}")
        if self.scale is Scale.LOG and self.lo <= 0.0:
            raise ValueError(f"{self.name}: log scale requires lo > 0, got {self.lo}")

    def edges(self) -> tuple[float, ...]:
        """Bin edges, length ``n_bins + 1``, in value units."""
        if self.scale is Scale.LOG:
            a, b = math.log10(self.lo), math.log10(self.hi)
            return tuple(10.0 ** (a + (b - a) * i / self.n_bins) for i in range(self.n_bins + 1))
        return tuple(
            self.lo + (self.hi - self.lo) * i / self.n_bins for i in range(self.n_bins + 1)
        )

    def bin_of(self, value: float) -> int:
        """Index of the coverage bin containing ``value``.

        Values outside ``[lo, hi]`` are clamped to the end bins; use
        :meth:`FaultSpec.validate` first if out-of-range must be an error.
        """
        if self.scale is Scale.LOG:
            if value <= self.lo:
                return 0
            frac = (math.log10(value) - math.log10(self.lo)) / (
                math.log10(self.hi) - math.log10(self.lo)
            )
        else:
            frac = (value - self.lo) / (self.hi - self.lo)
        idx = int(math.floor(frac * self.n_bins))
        return max(0, min(self.n_bins - 1, idx))

    def representative(self, bin_index: int) -> float:
        """A value inside bin ``bin_index`` (geometric or arithmetic midpoint)."""
        if not 0 <= bin_index < self.n_bins:
            raise ValueError(f"{self.name}: bin_index {bin_index} outside 0..{self.n_bins - 1}")
        e = self.edges()
        lo, hi = e[bin_index], e[bin_index + 1]
        if self.scale is Scale.LOG:
            # Geometric mean taken in log space: lo * hi overflows for the top
            # bin of a parameter whose range reaches 1e300.
            mid = 10.0 ** (0.5 * (math.log10(lo) + math.log10(hi)))
        else:
            mid = 0.5 * (lo + hi)
        return round(mid) if self.integer else mid

    def sample(self, u: float) -> float:
        """Map ``u`` in ``[0, 1)`` to a value in ``[lo, hi]`` on this scale."""
        if not 0.0 <= u <= 1.0:
            raise ValueError(f"{self.name}: u must be in [0, 1], got {u}")
        if self.scale is Scale.LOG:
            a, b = math.log10(self.lo), math.log10(self.hi)
            v = 10.0 ** (a + (b - a) * u)
        else:
            v = self.lo + (self.hi - self.lo) * u
        return float(round(v)) if self.integer else float(v)

    def normalise(self, value: float) -> float:
        """Map ``value`` to ``[0, 1]`` on this scale (clamped). Feature encoding."""
        if self.scale is Scale.LOG:
            if value <= self.lo:
                return 0.0
            frac = (math.log10(value) - math.log10(self.lo)) / (
                math.log10(self.hi) - math.log10(self.lo)
            )
        else:
            frac = (value - self.lo) / (self.hi - self.lo)
        return max(0.0, min(1.0, frac))


# Universal coverage dimensions, shared by every kind.  ``start_frac`` is the
# fraction of the run at which injection begins; ``duration_frac`` is the
# fraction of the remaining run over which it is active.  Two bins each keeps
# the cross product hand-enumerable (see validation/validate_coverage.py).
START_FRAC = ParamSpec("start_frac", "fraction of run", 0.0, 1.0, 2)
DURATION_FRAC = ParamSpec("duration_frac", "fraction of run", 0.05, 1.0, 2)

SENSOR_CHANNELS: tuple[str, ...] = ("pos", "vel")
ACTUATOR_CHANNELS: tuple[str, ...] = ("u",)
TRANSPORT_CHANNELS: tuple[str, ...] = ("bus",)
NUMERICAL_CHANNELS: tuple[str, ...] = ("pos", "vel", "u")


@dataclass(frozen=True)
class FaultSpec:
    """Static description of one fault kind."""

    kind: FaultKind
    fault_class: FaultClass
    channels: tuple[str, ...]
    params: tuple[ParamSpec, ...]
    description: str
    reference: str

    def param(self, name: str) -> ParamSpec:
        for p in self.params:
            if p.name == name:
                return p
        raise KeyError(f"{self.kind.value}: no parameter named {name!r}")

    @property
    def n_cells(self) -> int:
        """Number of coverage cells this kind contributes to the cross product."""
        n = len(self.channels) * START_FRAC.n_bins * DURATION_FRAC.n_bins
        for p in self.params:
            n *= p.n_bins
        return n

    def validate(self, channel: str, params: Mapping[str, float]) -> None:
        """Raise ``ValueError`` if ``channel``/``params`` are not admissible.

        Checks channel membership, parameter presence, absence of unknown
        parameters, and that each value is finite and inside its declared range.
        """
        if channel not in self.channels:
            raise ValueError(
                f"{self.kind.value}: channel {channel!r} not in {self.channels}"
            )
        expected = {p.name for p in self.params}
        got = set(params)
        if got != expected:
            missing = sorted(expected - got)
            extra = sorted(got - expected)
            raise ValueError(
                f"{self.kind.value}: parameter mismatch (missing={missing}, unexpected={extra})"
            )
        for p in self.params:
            v = float(params[p.name])
            if not math.isfinite(v):
                raise ValueError(f"{self.kind.value}.{p.name}: must be finite, got {v}")
            if not p.lo - 1e-12 <= v <= p.hi + 1e-12:
                raise ValueError(
                    f"{self.kind.value}.{p.name}: {v} outside declared range "
                    f"[{p.lo}, {p.hi}] {p.unit}"
                )


_SPECS: tuple[FaultSpec, ...] = (
    FaultSpec(
        FaultKind.SENSOR_BIAS,
        FaultClass.SENSOR,
        SENSOR_CHANNELS,
        (ParamSpec("offset", "channel units", 0.1, 10.0, 3, Scale.LOG),),
        "Additive constant offset on the measurement while active.",
        "Hwang et al. 2010, IEEE T-CST 18(3), Sec. II (additive sensor bias)",
    ),
    FaultSpec(
        FaultKind.SENSOR_DRIFT,
        FaultClass.SENSOR,
        SENSOR_CHANNELS,
        (ParamSpec("rate", "channel units per step", 0.001, 0.5, 3, Scale.LOG),),
        "Additive ramp on the measurement, accumulating from the start step.",
        "Hwang et al. 2010, IEEE T-CST 18(3), Sec. II (sensor drift)",
    ),
    FaultSpec(
        FaultKind.SENSOR_STUCK,
        FaultClass.SENSOR,
        SENSOR_CHANNELS,
        (),
        "Measurement frozen at its value on the step before injection began.",
        "Hwang et al. 2010, IEEE T-CST 18(3), Sec. II (hard sensor failure)",
    ),
    FaultSpec(
        FaultKind.SENSOR_DROPOUT,
        FaultClass.SENSOR,
        SENSOR_CHANNELS,
        (ParamSpec("dropout_prob", "probability", 0.05, 1.0, 3),),
        "Measurement marked invalid with the given per-step probability.",
        "Avizienis et al. 2004, IEEE TDSC 1(1), Sec. 3 (omission fault)",
    ),
    FaultSpec(
        FaultKind.SENSOR_QUANT_COLLAPSE,
        FaultClass.SENSOR,
        SENSOR_CHANNELS,
        (ParamSpec("lsb", "channel units", 0.05, 20.0, 3, Scale.LOG),),
        "Measurement quantised to a coarse LSB (ADC resolution collapse).",
        "IEEE 754-2019 Sec. 4 rounding; quantisation-noise model, Widrow 1961",
    ),
    FaultSpec(
        FaultKind.ACTUATOR_LOSS_EFFECTIVENESS,
        FaultClass.ACTUATOR,
        ACTUATOR_CHANNELS,
        (ParamSpec("retained", "fraction", 0.0, 0.9, 3),),
        "Commanded actuation multiplied by a retained-effectiveness factor.",
        "Hwang et al. 2010, IEEE T-CST 18(3), Sec. II (loss of effectiveness)",
    ),
    FaultSpec(
        FaultKind.ACTUATOR_STUCK,
        FaultClass.ACTUATOR,
        ACTUATOR_CHANNELS,
        (),
        "Actuation frozen at its value on the step before injection began.",
        "Hwang et al. 2010, IEEE T-CST 18(3), Sec. II (stuck/lock-in-place)",
    ),
    FaultSpec(
        FaultKind.ACTUATOR_RUNAWAY,
        FaultClass.ACTUATOR,
        ACTUATOR_CHANNELS,
        (ParamSpec("slew", "channel units per step", 0.1, 20.0, 3, Scale.LOG),),
        "Actuation ramps away from the command at a fixed slew rate.",
        "Hwang et al. 2010, IEEE T-CST 18(3), Sec. II (hard-over/runaway)",
    ),
    FaultSpec(
        FaultKind.BUS_DELAY,
        FaultClass.BUS,
        TRANSPORT_CHANNELS,
        (ParamSpec("delay_steps", "step", 1.0, 8.0, 3, integer=True),),
        "Whole measurement frame delivered N steps late (FIFO, order preserved).",
        "Avizienis et al. 2004, IEEE TDSC 1(1), Sec. 3 (timing fault, transport)",
    ),
    FaultSpec(
        FaultKind.BUS_REORDER,
        FaultClass.BUS,
        TRANSPORT_CHANNELS,
        (ParamSpec("swap_prob", "probability", 0.1, 1.0, 3),),
        "Adjacent measurement frames swapped with the given probability.",
        "Avizienis et al. 2004, IEEE TDSC 1(1), Sec. 3 (ordering fault)",
    ),
    FaultSpec(
        FaultKind.BUS_LOSS,
        FaultClass.BUS,
        TRANSPORT_CHANNELS,
        (ParamSpec("loss_prob", "probability", 0.05, 1.0, 3),),
        "Whole measurement frame dropped with the given per-step probability.",
        "Avizienis et al. 2004, IEEE TDSC 1(1), Sec. 3 (omission fault)",
    ),
    FaultSpec(
        FaultKind.TIMING_LATE_SAMPLE,
        FaultClass.TIMING,
        TRANSPORT_CHANNELS,
        (ParamSpec("late_steps", "step", 1.0, 4.0, 2, integer=True),),
        "Sample arrives after the controller has read the port, so the stale "
        "value from N steps earlier is used.",
        "Avizienis et al. 2004, IEEE TDSC 1(1), Sec. 3 (timing fault)",
    ),
    FaultSpec(
        FaultKind.TIMING_OVERRUN,
        FaultClass.TIMING,
        TRANSPORT_CHANNELS,
        (ParamSpec("overrun_prob", "probability", 0.1, 1.0, 3),),
        "Controller task overruns its period; the previous command is held and "
        "the step's update is skipped.",
        "Buttazzo 2011, Hard Real-Time Computing Systems, 3rd ed., Sec. 4 (deadline miss)",
    ),
    FaultSpec(
        FaultKind.NUMERICAL_NAN,
        FaultClass.NUMERICAL,
        NUMERICAL_CHANNELS,
        (),
        "Quiet NaN written to the channel (IEEE 754 invalid-operation result).",
        "IEEE 754-2019 Sec. 6.2 (NaN) and Sec. 7.2 (invalid operation)",
    ),
    FaultSpec(
        FaultKind.NUMERICAL_DENORMAL,
        FaultClass.NUMERICAL,
        NUMERICAL_CHANNELS,
        (),
        "Smallest positive subnormal (5e-324) written to the channel.",
        "IEEE 754-2019 Sec. 3.4 (subnormal) and Sec. 7.5 (underflow)",
    ),
    FaultSpec(
        FaultKind.NUMERICAL_OVERFLOW,
        FaultClass.NUMERICAL,
        NUMERICAL_CHANNELS,
        (ParamSpec("magnitude", "channel units", 1e8, 1e300, 3, Scale.LOG),),
        "Large magnitude written to the channel so downstream products overflow "
        "to infinity.",
        "IEEE 754-2019 Sec. 7.4 (overflow)",
    ),
)

TAXONOMY: Mapping[FaultKind, FaultSpec] = {s.kind: s for s in _SPECS}


def kinds() -> tuple[FaultKind, ...]:
    """All sixteen fault kinds, in taxonomy order."""
    return tuple(TAXONOMY)


def spec(kind: FaultKind | str) -> FaultSpec:
    """Static description of ``kind``.

    Raises ``KeyError`` with the list of valid names for an unknown kind.
    """
    try:
        key = kind if isinstance(kind, FaultKind) else FaultKind(kind)
    except ValueError as exc:
        raise KeyError(
            f"unknown fault kind {kind!r}; valid kinds: {[k.value for k in TAXONOMY]}"
        ) from exc
    return TAXONOMY[key]


def kinds_of_class(fault_class: FaultClass | str) -> tuple[FaultKind, ...]:
    """Kinds belonging to ``fault_class``."""
    fc = fault_class if isinstance(fault_class, FaultClass) else FaultClass(fault_class)
    return tuple(k for k, s in TAXONOMY.items() if s.fault_class is fc)


def total_cells(subset: Sequence[FaultKind] | None = None) -> int:
    """Size of the coverage cross product over ``subset`` (default: all kinds)."""
    ks = tuple(subset) if subset is not None else kinds()
    return sum(TAXONOMY[k].n_cells for k in ks)


def iter_param_bins(kind: FaultKind) -> Iterator[tuple[int, ...]]:
    """Every combination of parameter bin indices for ``kind``."""
    sp = TAXONOMY[kind]
    counts = [p.n_bins for p in sp.params]
    idx = [0] * len(counts)
    if not counts:
        yield ()
        return
    while True:
        yield tuple(idx)
        i = len(counts) - 1
        while i >= 0:
            idx[i] += 1
            if idx[i] < counts[i]:
                break
            idx[i] = 0
            i -= 1
        if i < 0:
            return
