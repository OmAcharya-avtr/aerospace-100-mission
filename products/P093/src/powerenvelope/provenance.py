"""Where a number's input came from, carried with the data rather than implied.

This product reports two kinds of figure and keeping them apart is most of the
point of it. A MEASURED figure comes from the hardware campaign under
``data/traces/``, written by ``data/campaign.py`` on the board. A SYNTHETIC
figure comes from the committed deterministic generator in
:mod:`powerenvelope.synthetic` and is a check of the arithmetic, not a
measurement of any board.

Every loaded campaign carries its origin, so a report cannot print one while
meaning the other, and so a reader of a committed output file can tell which
they are looking at without tracing the call.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Origin(Enum):
    """Origin of a loaded campaign."""

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
        is not reproducible for a reader with a cold clone and it publishes the
        build machine's filesystem layout.
    cells
        Cell labels actually loaded, in sorted order.
    """

    origin: Origin
    directory: str
    cells: tuple[str, ...]

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
            f"INPUT ORIGIN: {self.origin} -- {len(self.cells)} cell(s) from "
            f"{self.directory}/"
        )
