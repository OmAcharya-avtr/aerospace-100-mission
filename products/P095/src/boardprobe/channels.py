"""Channel declarations: the board's sources described as data, not as code.

A hardware abstraction layer whose knowledge of the board is spread through
``if`` statements cannot be reviewed, and cannot be retargeted without editing
logic. Everything this package knows about a board is a :class:`Channel`
record in a :class:`BoardDeclaration` table -- a name, a kind, a unit, an
access method, a locator and the reference the locator came from. The three
backends in ``boardprobe.sources`` dispatch on :class:`Access` alone, so each
backend is a thin adapter and the board table is the only thing that changes
when the board does.

``locator`` is stored **without a leading separator** so that this package's
own error messages quote a relative path. An absolute path in an error message
reaches committed validation output, which is then neither reproducible for a
reader with a cold clone nor acceptable to this mission's release gate.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


class ChannelKind(enum.Enum):
    """The four source classes this product's specification names."""

    CLOCK = "clock"
    DIE_TEMPERATURE = "die-temperature"
    RAIL_POWER = "rail-power"
    PERF_COUNTER = "performance-counter"

    def __str__(self) -> str:
        return self.value


class Access(enum.Enum):
    """How a live backend would obtain a raw value for a channel.

    Each member names one documented kernel interface. The backend dispatches
    on this and nothing else.
    """

    #: ``clock_gettime(2)`` with ``CLOCK_MONOTONIC_RAW``; raw unit nanoseconds.
    CLOCK_SYSCALL = "clock_gettime"
    #: A sysfs file holding an integer. Raw unit is whatever the ABI says.
    SYSFS_INTEGER = "sysfs-integer"
    #: ``perf_event_open(2)``, read as a 64-bit counter. Raw unit: events.
    PERF_EVENT_OPEN = "perf_event_open"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class Channel:
    """One declared source on a board.

    Attributes
    ----------
    name:
        Stable identifier used by every API in this package.
    kind:
        One of the four source classes in :class:`ChannelKind`.
    unit:
        Unit of the value a backend returns, AFTER ``scale`` is applied.
    access:
        The documented interface a live backend would use.
    locator:
        A sysfs path relative to the filesystem root (no leading separator), a
        clock identifier, or a ``perf_event_open`` type/config pair rendered as
        ``type:config``. Never absolute: see the module docstring.
    scale:
        Multiplier from the raw interface value to ``unit``. For a thermal zone
        this is 1e-3, because the kernel thermal ABI reports millidegrees
        Celsius.
    raw_unit:
        Unit of the value the interface itself returns, before ``scale``.
    reference:
        Where the locator, the raw unit and the scale were taken FROM. Every
        channel in this package cites a kernel ABI document, a manual page or a
        vendor developer guide.
    verified_on_hardware:
        ``False`` for every channel shipped in this release. Nothing in this
        table was read from a Jetson Orin Nano, because no board was reachable
        from the build environment. A ``True`` here would be a claim, and it is
        not made.
    """

    name: str
    kind: ChannelKind
    unit: str
    access: Access
    locator: str
    scale: float
    raw_unit: str
    reference: str
    verified_on_hardware: bool = False

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("channel name must be a non-empty string")
        if not self.unit:
            raise ValueError(f"channel {self.name!r}: unit must be a non-empty string")
        if type(self.kind) is not ChannelKind:
            raise TypeError(
                f"channel {self.name!r}: kind must be a ChannelKind member, got "
                f"{type(self.kind).__name__}"
            )
        if type(self.access) is not Access:
            raise TypeError(
                f"channel {self.name!r}: access must be an Access member, got "
                f"{type(self.access).__name__}"
            )
        if not (self.scale > 0.0):
            raise ValueError(
                f"channel {self.name!r}: scale must be strictly positive, got {self.scale!r}; "
                "a non-positive scale would silently invert or annihilate every reading"
            )
        if self.locator.startswith(("/", "\\")):
            raise ValueError(
                f"channel {self.name!r}: locator {self.locator!r} must be relative to the "
                "filesystem root and must not start with a separator, so that this "
                "package's error messages and committed validation output never quote an "
                "absolute host path"
            )
        if not self.reference:
            raise ValueError(
                f"channel {self.name!r}: reference must name where the locator, raw unit "
                "and scale came from; an undocumented path is not a declaration"
            )


@dataclass(frozen=True, slots=True)
class BoardDeclaration:
    """A named board and its channel table."""

    model: str
    channels: tuple[Channel, ...]
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("board model must be a non-empty string")
        if not self.channels:
            raise ValueError(f"board {self.model!r} declares no channels")
        seen: set[str] = set()
        for channel in self.channels:
            if type(channel) is not Channel:
                raise TypeError(
                    f"board {self.model!r}: every entry of channels must be a Channel, got "
                    f"{type(channel).__name__}"
                )
            if channel.name in seen:
                raise ValueError(
                    f"board {self.model!r}: duplicate channel name {channel.name!r}; "
                    "channel names are the keys every API in this package uses"
                )
            seen.add(channel.name)

    @property
    def names(self) -> tuple[str, ...]:
        """Every declared channel name, in table order."""
        return tuple(channel.name for channel in self.channels)

    def channel(self, name: str) -> Channel:
        """The channel called ``name``, or ``KeyError`` listing what exists."""
        for channel in self.channels:
            if channel.name == name:
                return channel
        raise KeyError(
            f"board {self.model!r} declares no channel {name!r}; declared channels are "
            + ", ".join(self.names)
        )

    def of_kind(self, kind: ChannelKind) -> tuple[Channel, ...]:
        """Every declared channel of one :class:`ChannelKind`."""
        if type(kind) is not ChannelKind:
            raise TypeError(f"kind must be a ChannelKind member, got {type(kind).__name__}")
        return tuple(channel for channel in self.channels if channel.kind is kind)

    @property
    def verified_count(self) -> int:
        """How many channels were verified against real hardware. Zero in this release."""
        return sum(1 for channel in self.channels if channel.verified_on_hardware)
