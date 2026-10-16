"""Exception types raised by watchdogspec.

Every parse failure names the YAML key path that caused it, so that a
specification author sees ``tasks[2].heartbeat_period_ms`` rather than a
stack trace.
"""

from __future__ import annotations


class WatchdogSpecError(Exception):
    """Base class for every error this package raises deliberately."""


class SpecFormatError(WatchdogSpecError):
    """A specification document is malformed, incomplete or self-contradictory.

    Parameters
    ----------
    message:
        What is wrong, in a form an author can act on.
    path:
        Dotted or bracketed key path into the YAML document, for example
        ``tasks[0].escalation``. ``None`` when the fault is in the document
        as a whole.
    """

    def __init__(self, message: str, path: str | None = None) -> None:
        self.path = path
        self.message = message
        super().__init__(f"{path}: {message}" if path else message)
