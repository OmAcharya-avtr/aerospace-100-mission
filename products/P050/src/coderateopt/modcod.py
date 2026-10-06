"""MODCOD descriptions and the illustrative reference table.

A MODCOD here is three numbers and a name: the information rate it carries,
the demodulation threshold it needs, and (optionally) a per-use cost such as
framing overhead. Nothing in this package models a modem; it consumes the
threshold you give it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Modcod", "ModcodSet", "illustrative_modcod_table"]


@dataclass(frozen=True)
class Modcod:
    """One entry of a modulation-and-coding table.

    Attributes
    ----------
    name
        Label, non-empty.
    rate_bits_per_symbol
        Information rate delivered when the frame is received, bits/symbol,
        strictly positive. Code rate times modulation order in bits/symbol,
        net of any overhead you have already removed.
    threshold_db
        Demodulation threshold: the received SNR (or received power, on
        whatever scale the link margin uses) at or above which a frame is
        assumed to be delivered, dB.
    overhead_fraction
        Fraction of ``rate_bits_per_symbol`` consumed by framing and
        signalling that the user does not receive, in [0, 1). Default 0.
    """

    name: str
    rate_bits_per_symbol: float
    threshold_db: float
    overhead_fraction: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Modcod.name must be a non-empty string")
        r = self.rate_bits_per_symbol
        if not np.isfinite(r) or r <= 0.0:
            raise ValueError(
                f"{self.name}: rate_bits_per_symbol must be finite and > 0, got {r!r}"
            )
        if not np.isfinite(self.threshold_db):
            raise ValueError(f"{self.name}: threshold_db must be finite, got {self.threshold_db!r}")
        if not 0.0 <= self.overhead_fraction < 1.0:
            raise ValueError(
                f"{self.name}: overhead_fraction must be in [0, 1), "
                f"got {self.overhead_fraction!r}"
            )

    @property
    def net_rate(self) -> float:
        """Rate after framing overhead, bits/symbol."""
        return self.rate_bits_per_symbol * (1.0 - self.overhead_fraction)


@dataclass(frozen=True)
class ModcodSet:
    """An ordered, deduplicated MODCOD table.

    Entries are stored in a canonical order -- ascending ``threshold_db``,
    then descending ``net_rate``, then ascending ``name`` -- so that every
    tie-break downstream is reproducible across runs and across machines. The
    canonical order is part of the published behaviour, not an accident of
    dict ordering.
    """

    entries: tuple[Modcod, ...]

    def __post_init__(self) -> None:
        if not self.entries:
            raise ValueError("ModcodSet requires at least one Modcod")
        names = [m.name for m in self.entries]
        if len(set(names)) != len(names):
            dup = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"duplicate MODCOD names: {dup}")
        ordered = tuple(
            sorted(self.entries, key=lambda m: (m.threshold_db, -m.net_rate, m.name))
        )
        object.__setattr__(self, "entries", ordered)

    @classmethod
    def from_rows(cls, rows: list[tuple[str, float, float]]) -> ModcodSet:
        """Build from ``(name, rate_bits_per_symbol, threshold_db)`` triples."""
        return cls(tuple(Modcod(n, r, t) for n, r, t in rows))

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    def __getitem__(self, index: int) -> Modcod:
        return self.entries[index]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(m.name for m in self.entries)

    @property
    def rates(self) -> np.ndarray:
        """Net rates in canonical order, bits/symbol."""
        return np.array([m.net_rate for m in self.entries], dtype=float)

    @property
    def thresholds_db(self) -> np.ndarray:
        """Thresholds in canonical order, dB."""
        return np.array([m.threshold_db for m in self.entries], dtype=float)

    def index(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError as exc:
            raise KeyError(f"no MODCOD named {name!r}; have {list(self.names)}") from exc


def illustrative_modcod_table() -> ModcodSet:
    """A small MODCOD table for examples and tests.

    **These are round illustrative constants, not measurements of any
    receiver and not the values of any standard.** They were chosen so that
    thresholds increase with rate at a plausible 1 to 3 dB per step and so that
    the table exercises the optimiser's tie and non-monotone paths. Replace
    them with your own modem's measured thresholds before reading anything
    into the output.
    """
    return ModcodSet.from_rows(
        [
            ("ook-r1/4", 0.250, 2.0),
            ("ook-r1/2", 0.500, 4.5),
            ("ook-r2/3", 0.667, 6.0),
            ("ook-r3/4", 0.750, 6.8),
            ("ook-r5/6", 0.833, 7.8),
            ("ook-r9/10", 0.900, 8.6),
            ("qpsk-r1/2", 1.000, 10.5),
            ("qpsk-r3/4", 1.500, 13.2),
            ("qam16-r3/4", 3.000, 19.0),
        ]
    )
