"""The four-timestamp exchange record and the set of them an acquisition yields.

Notation
--------
One two-way exchange produces four timestamps, following the convention of
Mills, "Internet time synchronization: the Network Time Protocol", *IEEE
Transactions on Communications* 39(10):1482-1493 (1991), section III, and of
IEEE Std 1588-2019 (delay request/response):

* ``t1`` -- local clock reading when the request left the local node
* ``t2`` -- remote clock reading when the request arrived at the remote node
* ``t3`` -- remote clock reading when the reply left the remote node
* ``t4`` -- local clock reading when the reply arrived at the local node

All four are **integer nanoseconds**. ``t1`` and ``t4`` are on the local
timescale; ``t2`` and ``t3`` are on the remote one, which is the timescale
whose error is being estimated.

Derived quantities, all exact in integer arithmetic:

* forward dwell  ``t2 - t1 = d_ab + theta``
* reverse dwell  ``t4 - t3 = d_ba - theta``
* round trip     ``(t4 - t1) - (t3 - t2) = d_ab + d_ba``, free of ``theta``
* remote hold    ``t3 - t2``, the turnaround, which cancels exactly

The round-trip identity is the reason the two-way exchange is used at all: it
measures the path without needing the clocks to agree.

Storage
-------
:class:`ExchangeSet` stores the four timestamp arrays as ``int64`` nanoseconds.
``int64`` nanoseconds span about 292 years, which is ample for an epoch-relative
acquisition and exact for every difference the estimators take. Truth columns
(``true_offset_s``, ``true_asymmetry_s``) are present only for sets produced by
a simulated or loopback backend and are ``None`` otherwise; an estimator never
reads them, and ``validation`` scripts are the only consumers.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .errors import ConfigurationError
from .timebase import NS_PER_S

__all__ = ["Exchange", "ExchangeSet"]


@dataclass(frozen=True)
class Exchange:
    """One completed four-timestamp exchange, in integer nanoseconds.

    Attributes
    ----------
    seq:
        Sequence number assigned by the acquisition, starting at 0 and
        including lost exchanges, so a gap in ``seq`` is a visible loss.
    t1_ns, t2_ns, t3_ns, t4_ns:
        The four timestamps [ns]; see the module docstring.
    """

    seq: int
    t1_ns: int
    t2_ns: int
    t3_ns: int
    t4_ns: int

    def __post_init__(self) -> None:
        if self.t4_ns < self.t1_ns:
            raise ConfigurationError(
                f"exchange {self.seq}: reply arrived before the request left "
                f"(t4={self.t4_ns} < t1={self.t1_ns} ns)"
            )
        if self.t3_ns < self.t2_ns:
            raise ConfigurationError(
                f"exchange {self.seq}: remote replied before it received "
                f"(t3={self.t3_ns} < t2={self.t2_ns} ns)"
            )

    @property
    def forward_dwell_ns(self) -> int:
        """``t2 - t1`` [ns] = ``d_ab + theta``."""
        return self.t2_ns - self.t1_ns

    @property
    def reverse_dwell_ns(self) -> int:
        """``t4 - t3`` [ns] = ``d_ba - theta``."""
        return self.t4_ns - self.t3_ns

    @property
    def round_trip_ns(self) -> int:
        """``(t4 - t1) - (t3 - t2)`` [ns] = ``d_ab + d_ba``, free of ``theta``."""
        return (self.t4_ns - self.t1_ns) - (self.t3_ns - self.t2_ns)

    @property
    def midpoint_ns(self) -> int:
        """Local-clock midpoint ``(t1 + t4) // 2`` [ns], the epoch of the estimate.

        ``//`` is floor division, so for a negative sum it rounds down rather
        than toward zero. Timestamps are epoch-relative and the timestamping
        noise can make the first one slightly negative, so the distinction is
        real; the resulting error is at most one nanosecond, which is below the
        quantisation of every timebase in this package.
        """
        return (self.t1_ns + self.t4_ns) // 2


@dataclass
class ExchangeSet:
    """A set of exchanges from one acquisition, as arrays.

    Parameters
    ----------
    t1_ns, t2_ns, t3_ns, t4_ns:
        Equal-length ``int64`` arrays of timestamps [ns].
    seq:
        Sequence numbers of the surviving exchanges.
    n_lost:
        Number of exchanges attempted that did not complete.
    backend_kind, backend_driver:
        Recorded from the backend that produced the set, so a stored set can
        never be mistaken for a hardware measurement.
    is_hardware:
        ``True`` only for a device backend not driven by a stub or a recording.
    true_offset_s, true_asymmetry_s:
        Simulation truth [s] at each exchange midpoint, or ``None``.
    """

    t1_ns: np.ndarray
    t2_ns: np.ndarray
    t3_ns: np.ndarray
    t4_ns: np.ndarray
    seq: np.ndarray
    n_lost: int = 0
    backend_kind: str = "unknown"
    backend_driver: str = ""
    is_hardware: bool = False
    true_offset_s: np.ndarray | None = None
    true_asymmetry_s: np.ndarray | None = None

    def __post_init__(self) -> None:
        arrays = {
            "t1_ns": self.t1_ns,
            "t2_ns": self.t2_ns,
            "t3_ns": self.t3_ns,
            "t4_ns": self.t4_ns,
            "seq": self.seq,
        }
        converted = {}
        for name, value in arrays.items():
            arr = np.asarray(value, dtype=np.int64).reshape(-1)
            converted[name] = arr
        lengths = {name: arr.size for name, arr in converted.items()}
        if len(set(lengths.values())) != 1:
            raise ConfigurationError(f"timestamp arrays have unequal lengths: {lengths}")
        self.t1_ns = converted["t1_ns"]
        self.t2_ns = converted["t2_ns"]
        self.t3_ns = converted["t3_ns"]
        self.t4_ns = converted["t4_ns"]
        self.seq = converted["seq"]
        if np.any(self.t4_ns < self.t1_ns):
            bad = int(np.argmax(self.t4_ns < self.t1_ns))
            raise ConfigurationError(
                f"exchange at index {bad}: t4 ({self.t4_ns[bad]} ns) precedes "
                f"t1 ({self.t1_ns[bad]} ns)"
            )
        if np.any(self.t3_ns < self.t2_ns):
            bad = int(np.argmax(self.t3_ns < self.t2_ns))
            raise ConfigurationError(
                f"exchange at index {bad}: t3 ({self.t3_ns[bad]} ns) precedes "
                f"t2 ({self.t2_ns[bad]} ns)"
            )
        if int(self.n_lost) < 0:
            raise ConfigurationError(f"n_lost must be >= 0, got {self.n_lost!r}")
        for name in ("true_offset_s", "true_asymmetry_s"):
            value = getattr(self, name)
            if value is not None:
                arr = np.asarray(value, dtype=np.float64).reshape(-1)
                if arr.size != self.n:
                    raise ConfigurationError(
                        f"{name} has {arr.size} entries, expected {self.n}"
                    )
                setattr(self, name, arr)

    @property
    def n(self) -> int:
        """Number of completed exchanges."""
        return int(self.t1_ns.size)

    @property
    def n_attempted(self) -> int:
        """Completed plus lost."""
        return self.n + int(self.n_lost)

    @property
    def loss_fraction(self) -> float:
        """Lost divided by attempted; ``0.0`` for an empty set."""
        return float(self.n_lost) / self.n_attempted if self.n_attempted else 0.0

    def forward_dwell_ns(self) -> np.ndarray:
        """``t2 - t1`` [ns], exact."""
        return self.t2_ns - self.t1_ns

    def reverse_dwell_ns(self) -> np.ndarray:
        """``t4 - t3`` [ns], exact."""
        return self.t4_ns - self.t3_ns

    def round_trip_ns(self) -> np.ndarray:
        """``(t4 - t1) - (t3 - t2)`` [ns], exact and free of ``theta``."""
        return (self.t4_ns - self.t1_ns) - (self.t3_ns - self.t2_ns)

    def midpoint_ns(self) -> np.ndarray:
        """Local-clock midpoints ``(t1 + t4) // 2`` [ns]."""
        return (self.t1_ns + self.t4_ns) // 2

    def midpoint_s(self) -> np.ndarray:
        """Local-clock midpoints [s], as ``float64``."""
        return self.midpoint_ns().astype(np.float64) / NS_PER_S

    def span_s(self) -> float:
        """Local time from the first ``t1`` to the last ``t4`` [s]."""
        if self.n == 0:
            return 0.0
        return float(self.t4_ns[-1] - self.t1_ns[0]) / NS_PER_S

    def subset(self, index: np.ndarray | slice) -> ExchangeSet:
        """A new set holding the selected exchanges, truth columns included."""
        return ExchangeSet(
            t1_ns=self.t1_ns[index],
            t2_ns=self.t2_ns[index],
            t3_ns=self.t3_ns[index],
            t4_ns=self.t4_ns[index],
            seq=self.seq[index],
            n_lost=self.n_lost,
            backend_kind=self.backend_kind,
            backend_driver=self.backend_driver,
            is_hardware=self.is_hardware,
            true_offset_s=None if self.true_offset_s is None else self.true_offset_s[index],
            true_asymmetry_s=(
                None if self.true_asymmetry_s is None else self.true_asymmetry_s[index]
            ),
        )

    def timestamp_bytes(self) -> bytes:
        """The four timestamp arrays as raw little-endian ``int64`` bytes.

        Used for parity comparison. Byte equality of this value is a strong
        statement because the arrays are integers: there is no rounding to
        excuse a difference.
        """
        parts = [
            np.ascontiguousarray(arr.astype("<i8")).tobytes()
            for arr in (self.t1_ns, self.t2_ns, self.t3_ns, self.t4_ns, self.seq)
        ]
        return b"".join(parts)

    def digest(self) -> str:
        """SHA-256 over the timestamps, loss count and backend identity."""
        hasher = hashlib.sha256()
        hasher.update(self.timestamp_bytes())
        hasher.update(
            json.dumps(
                {
                    "n_lost": int(self.n_lost),
                    "backend_kind": self.backend_kind,
                    "backend_driver": self.backend_driver,
                    "is_hardware": bool(self.is_hardware),
                },
                sort_keys=True,
            ).encode()
        )
        return hasher.hexdigest()

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable form; timestamps stay integers."""
        payload: dict[str, object] = {
            "format": "timetransfer.exchangeset/1",
            "t1_ns": [int(v) for v in self.t1_ns],
            "t2_ns": [int(v) for v in self.t2_ns],
            "t3_ns": [int(v) for v in self.t3_ns],
            "t4_ns": [int(v) for v in self.t4_ns],
            "seq": [int(v) for v in self.seq],
            "n_lost": int(self.n_lost),
            "backend_kind": self.backend_kind,
            "backend_driver": self.backend_driver,
            "is_hardware": bool(self.is_hardware),
        }
        if self.true_offset_s is not None:
            payload["true_offset_s"] = [float(v) for v in self.true_offset_s]
        if self.true_asymmetry_s is not None:
            payload["true_asymmetry_s"] = [float(v) for v in self.true_asymmetry_s]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> ExchangeSet:
        """Inverse of :meth:`to_dict`, with the format tag checked."""
        tag = payload.get("format")
        if tag != "timetransfer.exchangeset/1":
            raise ConfigurationError(
                f"unknown exchange-set format {tag!r}; expected "
                f"'timetransfer.exchangeset/1'"
            )
        return cls(
            t1_ns=np.asarray(payload["t1_ns"], dtype=np.int64),
            t2_ns=np.asarray(payload["t2_ns"], dtype=np.int64),
            t3_ns=np.asarray(payload["t3_ns"], dtype=np.int64),
            t4_ns=np.asarray(payload["t4_ns"], dtype=np.int64),
            seq=np.asarray(payload["seq"], dtype=np.int64),
            n_lost=int(payload.get("n_lost", 0)),  # type: ignore[arg-type]
            backend_kind=str(payload.get("backend_kind", "unknown")),
            backend_driver=str(payload.get("backend_driver", "")),
            is_hardware=bool(payload.get("is_hardware", False)),
            true_offset_s=payload.get("true_offset_s"),  # type: ignore[arg-type]
            true_asymmetry_s=payload.get("true_asymmetry_s"),  # type: ignore[arg-type]
        )

    def write_json(self, path: str | Path) -> Path:
        """Write :meth:`to_dict` to ``path`` as JSON and return the path."""
        target = Path(path)
        target.write_text(json.dumps(self.to_dict(), indent=1, sort_keys=True) + "\n")
        return target

    @classmethod
    def read_json(cls, path: str | Path) -> ExchangeSet:
        """Read a set written by :meth:`write_json`."""
        return cls.from_dict(json.loads(Path(path).read_text()))
