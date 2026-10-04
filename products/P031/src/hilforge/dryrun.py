"""Dry-run mode: the real command path, no write.

What dry run is for
-------------------
Before a deployment moves an actuator, you want the whole command path
exercised — sensor read, estimation, control law, command validation,
serialisation, deadline accounting, latency measurement — with the final bus
write suppressed. That is a rehearsal: the timing is real, the arithmetic is
real, the hardware does not move.

How it is enforced, and how that is proven
------------------------------------------
:class:`DryRunActuator` wraps a real actuator channel. It performs
:func:`hilforge.hal.validate_command` and the range check, exactly as the real
channel does, then **returns an acknowledgement without calling the wrapped
channel's** ``write``. The wrapped channel's ``write_count`` is therefore the
proof: in a dry run it must still be zero afterwards.

The validation does not take the wrapper's word for it. The channel underneath
is a :class:`hilforge.backends.stubs.CountingActuator` configured with
``forbid_writes=True``, which raises
:class:`~hilforge.errors.DryRunViolationError` if anything reaches it. So a
leak fails loudly rather than being caught by reading the code.
"""

from __future__ import annotations

import numpy as np

from .hal import ActuatorChannel, ChannelSpec, WriteAck, validate_command

__all__ = ["DryRunActuator"]


class DryRunActuator:
    """Actuator wrapper that validates a command and then discards it.

    Parameters
    ----------
    inner:
        The actuator that would have been written to. It is never written to.

    Attributes
    ----------
    rehearsed:
        Number of commands validated and discarded.
    """

    def __init__(self, inner: ActuatorChannel) -> None:
        self._inner = inner
        self.rehearsed = 0
        self._last: np.ndarray | None = None

    @property
    def inner(self) -> ActuatorChannel:
        """The wrapped actuator, for asserting its ``write_count`` is zero."""
        return self._inner

    @property
    def spec(self) -> ChannelSpec:
        return self._inner.spec

    @property
    def write_count(self) -> int:
        """Always 0: this wrapper never writes."""
        return 0

    def last_command(self) -> np.ndarray | None:
        """Last command that *would* have been written."""
        return None if self._last is None else self._last.copy()

    def write(self, command: np.ndarray) -> WriteAck:
        """Validate ``command`` and discard it.

        Returns a :class:`~hilforge.hal.WriteAck` whose ``applied`` is the
        command clipped to the channel's declared range, so a caller logging
        acknowledgements sees the same shape of data it would on a real run.
        ``saturated`` reflects that clipping. The wrapped channel is untouched.
        """
        cmd = validate_command(self.spec, command)
        clipped = np.clip(cmd, self.spec.lower, self.spec.upper)
        self.rehearsed += 1
        self._last = clipped.copy()
        return WriteAck(
            channel=self.spec.name,
            applied=clipped,
            saturated=bool(not np.array_equal(clipped, cmd)),
            sequence=self.rehearsed,
        )
