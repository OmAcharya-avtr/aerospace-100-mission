"""Driver stubs. None of these is hardware, and each says so in its name.

Three stubs, three jobs
-----------------------
* :class:`LoopbackDriver` wraps the **same** :class:`ExchangeEngine` the
  simulated backend uses, behind the :class:`~timetransfer.hal.TimestampDriver`
  interface. It exists so that the device code path can be exercised, and
  compared against the simulated one, with no board present. It is the reason
  ``validation/validate_parity.py`` can run in a container.
* :class:`RecordedDriver` replays a stored
  :class:`~timetransfer.exchange.ExchangeSet`. It is what dry-run mode uses:
  the full acquisition and estimation path runs against a recording and the
  driver's ``touch_count`` for any device operation stays at zero because there
  is no device underneath it.
* :class:`AbsentDriver` refuses to open. It is the negative control for the
  pre-flight device check, so that "the device check passes" is a statement
  with a falsifiable counterpart.

:class:`GuardedDriver` wraps any of them and can be configured to forbid
exchanges, which is how dry-run mode proves by counting rather than by
inspection that nothing was driven.
"""

from __future__ import annotations

from ..errors import ConfigurationError, DeviceAbsentError, ExchangeLostError
from ..exchange import ExchangeSet
from ..hal import ExchangeTruth
from .simulated import ExchangeEngine, SimulationSpec

__all__ = ["AbsentDriver", "GuardedDriver", "LoopbackDriver", "RecordedDriver"]


class LoopbackDriver:
    """Driver shim over the simulation engine. NOT hardware.

    Parameters
    ----------
    spec:
        Simulation parameters, identical to the ones the simulated backend is
        given when parity is being checked.
    """

    name = "loopback"

    def __init__(self, spec: SimulationSpec) -> None:
        self._engine = ExchangeEngine(spec)
        self._open = False
        self._touches = 0
        self._last_truth: ExchangeTruth | None = None

    @property
    def engine(self) -> ExchangeEngine:
        """The engine behind the shim."""
        return self._engine

    @property
    def touch_count(self) -> int:
        return self._touches

    @property
    def last_truth(self) -> ExchangeTruth | None:
        """Truth for the most recent completed exchange, or ``None``."""
        return self._last_truth

    def open(self) -> None:
        self._open = True

    def close(self) -> None:
        self._open = False

    def probe(self) -> dict[str, str]:
        """Identity read back from the stub, including that it is a stub."""
        return {
            "driver": self.name,
            "is_hardware": "false",
            "note": "loopback over the simulation engine; no device is present",
            "seed": str(int(self._engine.spec.seed)),
        }

    def exchange(self, seq: int) -> tuple[int, int, int, int]:
        """One exchange; raises ``ExchangeLostError`` when the model drops it."""
        if not self._open:
            raise DeviceAbsentError("loopback driver used before open()")
        self._touches += 1
        outcome = self._engine.step(int(seq))
        if outcome.lost or outcome.exchange is None:
            self._last_truth = None
            raise ExchangeLostError(f"exchange {seq} lost in the link model")
        self._last_truth = outcome.truth
        ex = outcome.exchange
        return ex.t1_ns, ex.t2_ns, ex.t3_ns, ex.t4_ns

    def snapshot(self) -> dict[str, object]:
        """Driver state, including the engine's generator states."""
        return {"engine": self._engine.snapshot(), "touches": self._touches, "open": self._open}

    def restore(self, snapshot: dict[str, object]) -> None:
        """Put the driver back into a captured state."""
        self._engine.restore(snapshot["engine"])  # type: ignore[arg-type]
        self._touches = int(snapshot["touches"])  # type: ignore[arg-type]
        self._open = bool(snapshot["open"])


class RecordedDriver:
    """Driver shim that replays a stored exchange set. NOT hardware.

    Gaps in the recorded sequence numbers are replayed as losses, so a
    recording made through an outage reproduces the outage.

    Parameters
    ----------
    recorded:
        The exchange set to replay.
    """

    name = "recorded"

    def __init__(self, recorded: ExchangeSet) -> None:
        if recorded.n == 0:
            raise ConfigurationError("recorded driver needs at least one exchange")
        self._recorded = recorded
        self._index = {int(s): i for i, s in enumerate(recorded.seq)}
        self._open = False
        self._touches = 0
        self._last_truth: ExchangeTruth | None = None

    @property
    def recorded(self) -> ExchangeSet:
        """The set being replayed."""
        return self._recorded

    @property
    def touch_count(self) -> int:
        return self._touches

    @property
    def last_truth(self) -> ExchangeTruth | None:
        return self._last_truth

    def open(self) -> None:
        self._open = True

    def close(self) -> None:
        self._open = False

    def probe(self) -> dict[str, str]:
        return {
            "driver": self.name,
            "is_hardware": "false",
            "note": "replay of a recorded exchange set; no device is present",
            "n_recorded": str(self._recorded.n),
            "source_backend": f"{self._recorded.backend_kind}/{self._recorded.backend_driver}",
        }

    def exchange(self, seq: int) -> tuple[int, int, int, int]:
        if not self._open:
            raise DeviceAbsentError("recorded driver used before open()")
        self._touches += 1
        i = self._index.get(int(seq))
        if i is None:
            self._last_truth = None
            raise ExchangeLostError(f"exchange {seq} is not in the recording")
        rec = self._recorded
        if rec.true_offset_s is not None and rec.true_asymmetry_s is not None:
            self._last_truth = ExchangeTruth(
                offset_s=float(rec.true_offset_s[i]),
                asymmetry_s=float(rec.true_asymmetry_s[i]),
                forward_delay_s=float("nan"),
                reverse_delay_s=float("nan"),
            )
        else:
            self._last_truth = None
        return (
            int(rec.t1_ns[i]),
            int(rec.t2_ns[i]),
            int(rec.t3_ns[i]),
            int(rec.t4_ns[i]),
        )

    def snapshot(self) -> dict[str, object]:
        return {"touches": self._touches, "open": self._open}

    def restore(self, snapshot: dict[str, object]) -> None:
        self._touches = int(snapshot["touches"])  # type: ignore[arg-type]
        self._open = bool(snapshot["open"])


class AbsentDriver:
    """Driver that refuses to open. The negative control for the device check."""

    name = "absent"

    def __init__(self, *, reason: str = "no device answered") -> None:
        self._reason = str(reason)
        self._touches = 0

    @property
    def touch_count(self) -> int:
        return self._touches

    @property
    def last_truth(self) -> ExchangeTruth | None:
        return None

    def open(self) -> None:
        raise DeviceAbsentError(self._reason)

    def close(self) -> None:
        return None

    def probe(self) -> dict[str, str]:
        return {"driver": self.name, "is_hardware": "false", "reason": self._reason}

    def exchange(self, seq: int) -> tuple[int, int, int, int]:
        raise DeviceAbsentError(f"{self._reason} (exchange {seq} not attempted)")

    def snapshot(self) -> dict[str, object]:
        return {"touches": self._touches}

    def restore(self, snapshot: dict[str, object]) -> None:
        self._touches = int(snapshot["touches"])  # type: ignore[arg-type]


class GuardedDriver:
    """Wrapper that counts, and can forbid, every driver operation.

    Parameters
    ----------
    inner:
        The driver to wrap.
    allow_exchange:
        When ``False``, :meth:`exchange` raises instead of reaching ``inner``
        and the attempt is counted in :attr:`refused_exchanges`. This is the
        negative control that makes the dry-run claim falsifiable: a run that
        was supposed to touch nothing would be caught here.
    allow_open:
        When ``False``, :meth:`open` raises.
    """

    def __init__(
        self, inner: object, *, allow_exchange: bool = True, allow_open: bool = True
    ) -> None:
        self._inner = inner
        self._allow_exchange = bool(allow_exchange)
        self._allow_open = bool(allow_open)
        self.refused_exchanges = 0
        self.refused_opens = 0

    @property
    def name(self) -> str:
        inner_name = getattr(self._inner, "name", "unknown")
        return f"guarded-{inner_name}"

    @property
    def inner(self) -> object:
        """The wrapped driver."""
        return self._inner

    @property
    def touch_count(self) -> int:
        return int(getattr(self._inner, "touch_count", 0))

    @property
    def last_truth(self) -> ExchangeTruth | None:
        return getattr(self._inner, "last_truth", None)

    def open(self) -> None:
        if not self._allow_open:
            self.refused_opens += 1
            raise DeviceAbsentError("guarded driver: open() is forbidden in this mode")
        self._inner.open()  # type: ignore[attr-defined]

    def close(self) -> None:
        self._inner.close()  # type: ignore[attr-defined]

    def probe(self) -> dict[str, str]:
        detail = dict(self._inner.probe())  # type: ignore[attr-defined]
        detail["guard_allow_exchange"] = "true" if self._allow_exchange else "false"
        detail["guard_allow_open"] = "true" if self._allow_open else "false"
        return detail

    def exchange(self, seq: int) -> tuple[int, int, int, int]:
        if not self._allow_exchange:
            self.refused_exchanges += 1
            raise DeviceAbsentError(
                f"guarded driver: exchange({seq}) is forbidden in this mode"
            )
        return self._inner.exchange(int(seq))  # type: ignore[attr-defined]

    def snapshot(self) -> dict[str, object]:
        return {
            "inner": self._inner.snapshot(),  # type: ignore[attr-defined]
            "refused_exchanges": self.refused_exchanges,
            "refused_opens": self.refused_opens,
        }

    def restore(self, snapshot: dict[str, object]) -> None:
        self._inner.restore(snapshot["inner"])  # type: ignore[attr-defined]
        self.refused_exchanges = int(snapshot["refused_exchanges"])  # type: ignore[arg-type]
        self.refused_opens = int(snapshot["refused_opens"])  # type: ignore[arg-type]

