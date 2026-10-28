"""Telemetry record data model, and the declared schema the screens are run against.

A record here is a *decoded* telemetry stream: one row per received frame, in
the order it was received, with engineering-unit channel values already
produced by a reader. The point of the package is that the reader may be wrong,
so the record optionally retains two further columns that make that question
answerable:

* ``raw_words`` -- the integer words the engineering values were decoded from.
  Without them, a misparse cannot be separated from a corruption, because the
  only evidence that distinguishes them is whether a *different* reading of the
  same bits is physically plausible.
* ``validity`` -- the frame-level integrity flag a transfer-frame or
  space-packet decoder produces (CCSDS 132.0-B-3 §4.1.4.2 Frame Error Control
  Field, or CCSDS 133.0-B-2 §4.1.3.5 Packet Error Control Field). Without it, a
  bit error cannot be separated from a reader fault, because both present as a
  value outside its declared range.

Those two absences are not caveats. They are computed and returned by
:mod:`telemgap.determinability` as a first-class output, before any
classification is reported.

Units
-----
``timestamps`` are seconds. Nothing in this package assumes an epoch; all
arithmetic is on differences, so TAI, GPS, spacecraft elapsed time or UNIX time
are equally acceptable as long as one of them is used throughout. ``counters``
are dimensionless integers modulo ``RecordSchema.counter_modulus``. Channel
values are in each :class:`ChannelSpec`'s declared ``unit``.

Conventions
-----------
Engineering value from raw word, the affine decommutation every ground system
applies (see e.g. CCSDS 522.1-B-1 §3 on parameter calibration):

    value = raw * scale + offset                                        (1)

with ``raw`` the integer word as read, ``scale`` in units per count and
``offset`` in units. Equation (1) is the only decode this package models; a
polynomial or table calibration is out of scope and is named in README.md under
Limitations.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

#: Raw word widths this package will accept. Byte-swap reinterpretation is
#: defined only for an even number of bytes, which is why 8 is excluded.
SUPPORTED_WORD_BITS = (16, 32)


@dataclass(frozen=True)
class ChannelSpec:
    """Declared properties of one telemetry channel.

    Parameters
    ----------
    name
        Channel identifier. Must be non-empty and unique within a schema.
    unit
        Engineering unit string, carried through reports. Not parsed.
    scale
        Units per raw count in equation (1). Must be finite and non-zero.
    offset
        Additive term of equation (1), in ``unit``.
    low, high
        Declared valid engineering range, inclusive, in ``unit``. ``low`` must
        be strictly below ``high``. A value outside it is the trigger for the
        range screen; it is evidence of *something* wrong and on its own does
        not say what.
    max_slew
        Declared bound on ``|d value / d t|``, in ``unit`` per second. Used by
        the zero-order-hold arm of the interpolation policy, equation (4).
    max_curvature
        Declared bound on ``|d^2 value / d t^2|``, in ``unit`` per second
        squared. Used by the linear-interpolation arm of the policy,
        equation (3). This is the quantity an operator most often does not
        have; :mod:`telemgap.policy` can estimate it from the record instead
        and reports which of the two it used. A value of zero means *not
        declared*, which the policy treats as a question to be answered from
        the record rather than as a claim that the signal is a straight line.
    noise_sigma
        Declared measurement noise standard deviation of the channel, in
        ``unit``. Zero means not declared. It enters the policy through the
        coverage term of equation (3'), because an interpolation anchored on a
        noisy sample inherits that sample's noise and equations (3) and (4)
        bound only the interpolation component. A sensor's noise is normally on
        its datasheet, while its curvature bound normally is not, and the
        measured consequence of that asymmetry is the main finding of
        ``validation/validate_interpolation_policy.py``.
    word_bits
        Raw word width, one of :data:`SUPPORTED_WORD_BITS`.
    signed
        Whether the reader treated the raw word as two's complement.
    """

    name: str
    unit: str
    scale: float
    offset: float
    low: float
    high: float
    max_slew: float
    max_curvature: float
    noise_sigma: float = 0.0
    word_bits: int = 16
    signed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("ChannelSpec.name must be a non-empty string")
        for attr in (
            "scale",
            "offset",
            "low",
            "high",
            "max_slew",
            "max_curvature",
            "noise_sigma",
        ):
            value = getattr(self, attr)
            if not np.isfinite(value):
                raise ValueError(f"ChannelSpec.{attr} must be finite, got {value!r}")
        if self.scale == 0.0:
            raise ValueError("ChannelSpec.scale must be non-zero; equation (1) is not invertible")
        if not self.low < self.high:
            raise ValueError(
                f"ChannelSpec.low must be strictly below high, got low={self.low!r} "
                f"high={self.high!r}"
            )
        if self.max_slew < 0.0:
            raise ValueError("ChannelSpec.max_slew must be non-negative (units per second)")
        if self.max_curvature < 0.0:
            raise ValueError(
                "ChannelSpec.max_curvature must be non-negative (units per second squared)"
            )
        if self.noise_sigma < 0.0:
            raise ValueError("ChannelSpec.noise_sigma must be non-negative (channel units)")
        if self.word_bits not in SUPPORTED_WORD_BITS:
            raise ValueError(
                f"ChannelSpec.word_bits must be one of {SUPPORTED_WORD_BITS}, "
                f"got {self.word_bits!r}"
            )

    @property
    def span(self) -> float:
        """Width of the declared valid range, in ``unit``."""
        return float(self.high - self.low)

    @property
    def mid(self) -> float:
        """Midpoint of the declared valid range, in ``unit``."""
        return 0.5 * float(self.high + self.low)

    def decode(self, raw: Any) -> Any:
        """Apply equation (1) to raw counts, returning engineering units."""
        return np.asarray(raw, dtype=np.float64) * self.scale + self.offset

    def in_range(self, value: Any) -> Any:
        """Elementwise membership of the declared inclusive range."""
        array = np.asarray(value, dtype=np.float64)
        return (array >= self.low) & (array <= self.high)


@dataclass(frozen=True)
class RecordSchema:
    """Declared properties of a telemetry record as a whole.

    Parameters
    ----------
    cadence_s
        Nominal inter-frame interval, seconds. Must be strictly positive.
    jitter_s
        Tolerance on the departure of an inter-frame interval from
        ``cadence_s`` before the cadence screen fires, seconds. A record whose
        real jitter exceeds this will produce cadence findings that are not
        anomalies; that trade is measured in ``validation/VALIDATION.md`` §1.
    counter_modulus
        Frame counter wraps at this value, so counters run ``0 .. modulus-1``.
        CCSDS transfer frames use 256 (8-bit) and space packets 16384 (14-bit);
        both are accepted, as is any integer above 1.
    channels
        The channel specifications, in the column order of
        :attr:`TelemetryRecord.values`.
    has_validity_flag
        Whether the record carries a per-frame integrity flag. Declared
        separately from the array being present so that a caller can state
        honestly that the flag column is a placeholder.
    has_raw_words
        Whether the record retains the raw words of equation (1).
    reorder_horizon
        Counter deltas strictly greater than ``counter_modulus -
        reorder_horizon`` are read as *backward* motion -- a reorder -- rather
        than as a forward gap of nearly a whole counter cycle. The default is
        half the modulus, which is the conventional choice and the one that
        makes the ambiguity symmetric. The ambiguity itself is irreducible and
        is reported by :mod:`telemgap.determinability`.
    """

    cadence_s: float
    jitter_s: float
    counter_modulus: int
    channels: tuple[ChannelSpec, ...]
    has_validity_flag: bool = False
    has_raw_words: bool = False
    reorder_horizon: int | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.cadence_s) or self.cadence_s <= 0.0:
            raise ValueError(f"RecordSchema.cadence_s must be positive, got {self.cadence_s!r}")
        if not np.isfinite(self.jitter_s) or self.jitter_s < 0.0:
            raise ValueError(f"RecordSchema.jitter_s must be non-negative, got {self.jitter_s!r}")
        if self.jitter_s >= 0.5 * self.cadence_s:
            raise ValueError(
                "RecordSchema.jitter_s must be below half the cadence, otherwise a missing "
                f"frame is indistinguishable from jitter; got jitter_s={self.jitter_s!r} "
                f"cadence_s={self.cadence_s!r}"
            )
        if not isinstance(self.counter_modulus, (int, np.integer)) or self.counter_modulus < 2:
            raise ValueError(
                f"RecordSchema.counter_modulus must be an integer above 1, "
                f"got {self.counter_modulus!r}"
            )
        if not self.channels:
            raise ValueError("RecordSchema.channels must contain at least one ChannelSpec")
        names = [channel.name for channel in self.channels]
        if len(set(names)) != len(names):
            raise ValueError(f"RecordSchema channel names must be unique, got {names}")
        if self.reorder_horizon is not None:
            if not 1 <= self.reorder_horizon <= self.counter_modulus // 2:
                raise ValueError(
                    "RecordSchema.reorder_horizon must lie in 1..counter_modulus//2, got "
                    f"{self.reorder_horizon!r} for modulus {self.counter_modulus!r}"
                )

    @property
    def effective_reorder_horizon(self) -> int:
        """Reorder horizon actually used, defaulting to half the counter modulus."""
        if self.reorder_horizon is not None:
            return int(self.reorder_horizon)
        return max(1, int(self.counter_modulus) // 2)

    @property
    def channel_names(self) -> tuple[str, ...]:
        """Channel names in column order."""
        return tuple(channel.name for channel in self.channels)

    def channel_index(self, name: str) -> int:
        """Column index of ``name``, raising ``KeyError`` if absent."""
        try:
            return self.channel_names.index(name)
        except ValueError as error:
            raise KeyError(
                f"no channel named {name!r}; record declares {list(self.channel_names)}"
            ) from error


@dataclass
class TelemetryRecord:
    """Frames in the order received, with the schema they are screened against.

    Attributes
    ----------
    schema
        The :class:`RecordSchema`.
    timestamps
        ``(n,)`` float64 seconds. Not required to be monotonic; a decrease is
        exactly what the clock screen looks for.
    counters
        ``(n,)`` int64 frame counters, each in ``0 .. modulus-1``.
    values
        ``(n, n_channels)`` float64 engineering values in schema column order.
    raw_words
        ``(n, n_channels)`` uint64 raw words, or ``None``. Present only if
        ``schema.has_raw_words``.
    validity
        ``(n,)`` bool per-frame integrity flag, or ``None``. ``True`` means the
        frame's error-control field checked out. Present only if
        ``schema.has_validity_flag``.
    """

    schema: RecordSchema
    timestamps: np.ndarray
    counters: np.ndarray
    values: np.ndarray
    raw_words: np.ndarray | None = None
    validity: np.ndarray | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.timestamps = np.ascontiguousarray(self.timestamps, dtype=np.float64)
        self.counters = np.ascontiguousarray(self.counters, dtype=np.int64)
        self.values = np.ascontiguousarray(self.values, dtype=np.float64)
        if self.timestamps.ndim != 1:
            raise ValueError(
                f"timestamps must be one-dimensional, got shape {self.timestamps.shape}"
            )
        n = self.timestamps.size
        if n == 0:
            raise ValueError("TelemetryRecord must contain at least one frame")
        if self.counters.shape != (n,):
            raise ValueError(
                f"counters must have shape ({n},) to match timestamps, got {self.counters.shape}"
            )
        if self.values.ndim == 1:
            self.values = self.values.reshape(n, 1)
        if self.values.shape != (n, len(self.schema.channels)):
            raise ValueError(
                f"values must have shape ({n}, {len(self.schema.channels)}), "
                f"got {self.values.shape}"
            )
        if not np.all(np.isfinite(self.timestamps)):
            raise ValueError("timestamps must all be finite")
        if np.any(self.counters < 0) or np.any(self.counters >= self.schema.counter_modulus):
            raise ValueError(
                f"counters must lie in 0..{self.schema.counter_modulus - 1}; got range "
                f"{int(self.counters.min())}..{int(self.counters.max())}"
            )
        if self.raw_words is not None:
            self.raw_words = np.ascontiguousarray(self.raw_words, dtype=np.uint64)
            if self.raw_words.ndim == 1:
                self.raw_words = self.raw_words.reshape(n, 1)
            if self.raw_words.shape != self.values.shape:
                raise ValueError(
                    f"raw_words must have the same shape as values {self.values.shape}, "
                    f"got {self.raw_words.shape}"
                )
            for column, channel in enumerate(self.schema.channels):
                limit = np.uint64((1 << channel.word_bits) - 1)
                if np.any(self.raw_words[:, column] > limit):
                    raise ValueError(
                        f"raw_words column {column!r} ({channel.name}) exceeds "
                        f"{channel.word_bits}-bit range"
                    )
        if self.validity is not None:
            self.validity = np.ascontiguousarray(self.validity, dtype=bool)
            if self.validity.shape != (n,):
                raise ValueError(
                    f"validity must have shape ({n},), got {self.validity.shape}"
                )
        if self.schema.has_raw_words and self.raw_words is None:
            raise ValueError("schema declares has_raw_words but no raw_words array was given")
        if self.schema.has_validity_flag and self.validity is None:
            raise ValueError("schema declares has_validity_flag but no validity array was given")

    def __len__(self) -> int:
        return int(self.timestamps.size)

    @property
    def n_channels(self) -> int:
        """Number of declared channels."""
        return len(self.schema.channels)

    def channel(self, name: str) -> np.ndarray:
        """Engineering values of one channel, as a view in received order."""
        return self.values[:, self.schema.channel_index(name)]

    def without_validity(self) -> TelemetryRecord:
        """Copy with the integrity flag withheld, for determinability experiments.

        Used by ``validation/validate_determinability.py``: the same record is
        classified twice, once with the flag and once without, and the change
        in the undecided count is the measured value of the flag.
        """
        return TelemetryRecord(
            schema=replace(self.schema, has_validity_flag=False),
            timestamps=self.timestamps.copy(),
            counters=self.counters.copy(),
            values=self.values.copy(),
            raw_words=None if self.raw_words is None else self.raw_words.copy(),
            validity=None,
            provenance=dict(self.provenance),
        )

    def without_raw_words(self) -> TelemetryRecord:
        """Copy with the raw words withheld, for determinability experiments."""
        return TelemetryRecord(
            schema=replace(self.schema, has_raw_words=False),
            timestamps=self.timestamps.copy(),
            counters=self.counters.copy(),
            values=self.values.copy(),
            raw_words=None,
            validity=None if self.validity is None else self.validity.copy(),
            provenance=dict(self.provenance),
        )
