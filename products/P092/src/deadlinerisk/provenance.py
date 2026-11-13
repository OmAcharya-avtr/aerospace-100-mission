"""Where a number's input came from, carried with the data rather than implied.

This product reports two kinds of figure and the distinction is the point of it.
A MEASURED figure comes from the hardware campaign in ``data/traces/``. A
SYNTHETIC figure comes from the committed deterministic generator in
``tests/fixtures/synthetic_trace.py`` and is a check of the code, not a
measurement of any board. Every loaded campaign carries its origin so that a
report cannot print one while meaning the other.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Origin(Enum):
    """Origin of a loaded trace set."""

    MEASURED = "measured"
    SYNTHETIC = "synthetic"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Provenance:
    """Origin plus the directory label a report should print.

    Parameters
    ----------
    origin
        :class:`Origin` of the data.
    directory
        Directory label, always relative. An absolute path in committed output
        is not reproducible for a reader with a cold clone and publishes the
        build machine's filesystem layout.
    runs
        Run labels actually loaded, in sorted order.
    """

    origin: Origin
    directory: str
    runs: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.origin, Origin):
            raise TypeError(f"origin must be an Origin, got {type(self.origin).__name__}")

    @property
    def is_measured(self) -> bool:
        """True when these traces came off the board."""
        return self.origin is Origin.MEASURED

    def banner(self) -> str:
        """One line naming the origin, for the head of a report."""
        return (
            f"INPUT ORIGIN: {self.origin} -- {len(self.runs)} run(s) from "
            f"{self.directory}/"
        )
