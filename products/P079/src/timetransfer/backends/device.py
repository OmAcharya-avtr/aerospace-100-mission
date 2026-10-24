"""The device backend: the same HAL contract, over a driver shim.

What this is and is not
-----------------------
:class:`DeviceBackend` is the code path a real timestamping device would be
driven through. It holds a :class:`~timetransfer.hal.TimestampDriver`, reads
the host's monotonic clock for its own timebase, converts a driver-level
:class:`~timetransfer.errors.ExchangeLostError` into a lost
:class:`~timetransfer.hal.ExchangeOutcome`, and reports
:attr:`~timetransfer.hal.BackendInfo.is_hardware` from the driver's name.

**No driver in this repository is hardware.** The two that exist are a
loopback stub over the simulation engine and a replayer over a recording, and
:attr:`~timetransfer.hal.BackendInfo.is_hardware` is ``False`` for both. The
class is written as if a real driver existed because that is what makes the
parity check meaningful: the estimator path under test is the one a board
would exercise.

Timebase
--------
A device backend uses :class:`~timetransfer.timebase.MonotonicTimebase` by
default, which reads the host clock. Passing ``virtual_timebase=True`` makes it
use a :class:`~timetransfer.timebase.VirtualTimebase` instead, which is what
the parity check does: parity is a statement about the *data* path, and a
backend that reads a wall clock cannot be compared byte for byte against one
that does not.
"""

from __future__ import annotations

from ..errors import DeviceAbsentError, ExchangeLostError
from ..exchange import Exchange
from ..hal import BackendInfo, ExchangeOutcome, ExchangeTruth
from ..timebase import NS_PER_S, MonotonicTimebase, Timebase, VirtualTimebase

__all__ = ["DeviceBackend"]


class DeviceBackend:
    """HAL backend over a :class:`~timetransfer.hal.TimestampDriver`.

    Parameters
    ----------
    driver:
        The shim to drive.
    name:
        Label recorded in :attr:`~timetransfer.hal.BackendInfo.name`.
    period_s:
        Nominal exchange interval [s], used only to advance a virtual
        timebase; it does not pace a real driver, which paces itself.
    virtual_timebase:
        Use a deterministic timebase instead of the host clock. Required for a
        byte-for-byte parity comparison.
    resolution_ns:
        Resolution of the virtual timebase [ns], ignored when the host clock is
        used.
    """

    def __init__(
        self,
        driver: object,
        *,
        name: str = "device-link",
        period_s: float = 1.0,
        virtual_timebase: bool = False,
        resolution_ns: int = 1,
    ) -> None:
        self._driver = driver
        self._name = str(name)
        self._period_s = float(period_s)
        self._timebase: Timebase = (
            VirtualTimebase(resolution_ns=int(resolution_ns))
            if virtual_timebase
            else MonotonicTimebase()
        )
        self._open = False
        self.lost_count = 0

    @property
    def driver(self) -> object:
        """The driver shim."""
        return self._driver

    @property
    def info(self) -> BackendInfo:
        driver_name = str(getattr(self._driver, "name", "unknown"))
        detail = {"period_s": f"{self._period_s:.9g}"}
        if self._open:
            try:
                detail.update(self._driver.probe())  # type: ignore[attr-defined]
            except Exception as exc:  # pragma: no cover - a probe must not break info
                detail["probe_error"] = type(exc).__name__
        return BackendInfo(kind="device", name=self._name, driver=driver_name, detail=detail)

    @property
    def timebase(self) -> Timebase:
        return self._timebase

    @property
    def is_open(self) -> bool:
        return self._open

    def open(self) -> None:
        """Open the driver; propagates ``DeviceAbsentError`` if none answers."""
        self._driver.open()  # type: ignore[attr-defined]
        self._open = True

    def close(self) -> None:
        """Close the driver. Idempotent."""
        self._driver.close()  # type: ignore[attr-defined]
        self._open = False

    def next_exchange(self, seq: int) -> ExchangeOutcome:
        """Attempt exchange ``seq`` through the driver.

        A driver-level :class:`~timetransfer.errors.ExchangeLostError` becomes a
        lost outcome, because a loss is data. A
        :class:`~timetransfer.errors.DeviceAbsentError` propagates, because a
        missing device is a fault and must stop the run rather than be counted
        as link loss.
        """
        if not self._open:
            raise DeviceAbsentError(f"device backend {self._name!r} used before open()")
        if isinstance(self._timebase, VirtualTimebase):
            target_ns = round((int(seq) + 1) * self._period_s * NS_PER_S)
            delta = target_ns - self._timebase.now_ns()
            if delta > 0:
                self._timebase.advance_ns(delta)
        try:
            t1, t2, t3, t4 = self._driver.exchange(int(seq))  # type: ignore[attr-defined]
        except ExchangeLostError:
            self.lost_count += 1
            return ExchangeOutcome(seq=int(seq), exchange=None, lost=True, truth=None)
        truth: ExchangeTruth | None = getattr(self._driver, "last_truth", None)
        return ExchangeOutcome(
            seq=int(seq),
            exchange=Exchange(
                seq=int(seq), t1_ns=int(t1), t2_ns=int(t2), t3_ns=int(t3), t4_ns=int(t4)
            ),
            lost=False,
            truth=truth,
        )

    def snapshot(self) -> dict[str, object]:
        state: dict[str, object] = {"open": self._open, "lost_count": self.lost_count}
        snap = getattr(self._driver, "snapshot", None)
        if callable(snap):
            state["driver"] = snap()
        if isinstance(self._timebase, VirtualTimebase):
            state["timebase_ns"] = self._timebase.now_ns()
        return state

    def restore(self, snapshot: dict[str, object]) -> None:
        self._open = bool(snapshot["open"])
        self.lost_count = int(snapshot["lost_count"])  # type: ignore[arg-type]
        if "driver" in snapshot:
            self._driver.restore(snapshot["driver"])  # type: ignore[attr-defined]
        if "timebase_ns" in snapshot and isinstance(self._timebase, VirtualTimebase):
            self._timebase.set_ns(int(snapshot["timebase_ns"]))  # type: ignore[arg-type]
