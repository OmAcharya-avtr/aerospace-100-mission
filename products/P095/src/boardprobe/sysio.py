"""The single funnel for every device access, and the ledger that records it.

A dry-run mode that claims to touch no board is worth exactly as much as the
mechanism that proves it. Every read of a device node and every
``perf_event_open`` in this package goes through a function in this module,
each of which appends to a module-level :class:`IoLedger`. A dry run then
asserts the ledger recorded zero device accesses, which is a measurement
rather than a promise.

The funnel also keeps error text relative. ``read_integer`` is handed a root
and a locator and quotes the LOCATOR in its exception, never the joined
absolute path, because that text reaches committed validation output.
"""

from __future__ import annotations

import ctypes
import platform as _platform
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from .orin_nano import PERF_EVENT_OPEN_SYSCALL


class DeviceUnreachableError(OSError):
    """A declared device interface is not present or not permitted on this host.

    Raised with the relative locator, what was attempted, and what a reader
    should check. This is the failure path a board-absent host always takes and
    it is exercised by ``tests/test_sources_live.py``.
    """


@dataclass
class IoLedger:
    """Every device access this process attempted, in order."""

    events: list[tuple[str, str]] = field(default_factory=list)

    def record(self, kind: str, target: str) -> None:
        """Append one access. ``target`` must be relative, never absolute."""
        if target.startswith(("/", "\\")):
            raise ValueError(
                f"the io ledger records relative locators only; got {target!r}. An "
                "absolute path recorded here would reach committed validation output."
            )
        self.events.append((kind, target))

    def clear(self) -> None:
        self.events.clear()

    @property
    def device_reads(self) -> int:
        """How many device accesses have been recorded."""
        return len(self.events)

    def summary(self) -> tuple[tuple[str, int], ...]:
        """Access counts by kind, sorted by kind."""
        counts: dict[str, int] = {}
        for kind, _ in self.events:
            counts[kind] = counts.get(kind, 0) + 1
        return tuple(sorted(counts.items()))


#: The process-wide ledger. Module-level because the funnel is module-level;
#: the dry run snapshots and restores it rather than relying on a context
#: manager that a caller could forget.
LEDGER = IoLedger()


@contextmanager
def ledger_window() -> Iterator[IoLedger]:
    """Record accesses into a fresh ledger for the duration of the block."""
    global LEDGER
    previous = LEDGER
    window = IoLedger()
    LEDGER = window
    try:
        yield window
    finally:
        LEDGER = previous


def resolve_one(root: Path, locator: str) -> Path:
    """Resolve a locator that may contain one glob segment, to exactly one path.

    The INA3221 and thermal-zone locators contain a ``hwmon*`` segment whose
    index is assigned in probe order, so the locator is a pattern and the
    resolution is a run-time act. Raises :class:`DeviceUnreachableError`
    quoting the relative locator when it matches nothing, or when it matches
    more than one path -- an ambiguous device node is a configuration the
    caller must resolve, not one this package may pick from silently.
    """
    if "*" not in locator:
        return root / locator
    matches = sorted(root.glob(locator))
    if not matches:
        raise DeviceUnreachableError(
            f"no path matches the declared locator {locator!r} under the filesystem root; "
            "the device is not present on this host. On an NVIDIA Jetson Orin Nano this "
            "node is created by the driver named in the channel's reference; check that "
            "the driver is bound and that the pattern's variable segment (the hwmon or "
            "thermal-zone index, which is assigned in probe order) still matches."
        )
    if len(matches) > 1:
        raise DeviceUnreachableError(
            f"the declared locator {locator!r} matches {len(matches)} paths; a channel "
            "must resolve to exactly one device node. Narrow the declaration in "
            "boardprobe.orin_nano rather than letting this package choose."
        )
    return matches[0]


def read_integer(root: Path, locator: str) -> int:
    """Read one integer from a sysfs node. The only sysfs read in this package.

    Every sysfs-backed channel in ``boardprobe.orin_nano`` holds a single
    integer in a documented raw unit; the channel's ``scale`` converts it. A
    node that holds something else is a sign the declaration is wrong, and the
    error says so rather than returning a coerced value.
    """
    LEDGER.record("sysfs-read", locator)
    path = resolve_one(root, locator)
    try:
        text = path.read_text()
    except OSError as error:
        raise DeviceUnreachableError(
            f"cannot read the declared sysfs node {locator!r} relative to the filesystem "
            f"root: {error.strerror or error}. The board is not reachable from this host, "
            "or the driver that creates this node is not bound. See docs/RECOVERY.md, "
            "'board unreachable mid-run'."
        ) from error
    try:
        return int(text.strip())
    except ValueError as error:
        raise DeviceUnreachableError(
            f"the declared sysfs node {locator!r} does not hold an integer (got "
            f"{text.strip()[:40]!r}); the channel declaration in boardprobe.orin_nano is "
            "wrong for this kernel, or the node is not the one the reference describes"
        ) from error


class PerfEventAttr(ctypes.Structure):
    """``struct perf_event_attr`` as documented in ``perf_event_open(2)``.

    Field order and widths follow ``linux/perf_event.h``. The bitfield union
    of flags is carried as one 64-bit ``flags`` word, with the three bits this
    package sets named in :data:`PERF_FLAG_BITS`; the struct's ``size`` field
    is set to ``sizeof`` so the kernel can tell which version of the structure
    it was handed, which is the documented forward-compatibility mechanism.
    """

    _fields_ = (
        ("type", ctypes.c_uint32),
        ("size", ctypes.c_uint32),
        ("config", ctypes.c_uint64),
        ("sample_period", ctypes.c_uint64),
        ("sample_type", ctypes.c_uint64),
        ("read_format", ctypes.c_uint64),
        ("flags", ctypes.c_uint64),
        ("wakeup_events", ctypes.c_uint32),
        ("bp_type", ctypes.c_uint32),
        ("config1", ctypes.c_uint64),
        ("config2", ctypes.c_uint64),
        ("branch_sample_type", ctypes.c_uint64),
        ("sample_regs_user", ctypes.c_uint64),
        ("sample_stack_user", ctypes.c_uint32),
        ("clockid", ctypes.c_int32),
        ("sample_regs_intr", ctypes.c_uint64),
        ("aux_watermark", ctypes.c_uint32),
        ("sample_max_stack", ctypes.c_uint16),
        ("reserved_2", ctypes.c_uint16),
        ("aux_sample_size", ctypes.c_uint32),
        ("reserved_3", ctypes.c_uint32),
        ("sig_data", ctypes.c_uint64),
        ("config3", ctypes.c_uint64),
    )


#: ``disabled | inherit | pinned`` from ``struct perf_event_attr``'s bitfield,
#: in declaration order starting at bit 0 (``perf_event_open(2)``).
PERF_FLAG_BITS = {"disabled": 0, "inherit": 1, "pinned": 2}

#: Size of the structure above, on this build. Reported by the validation
#: scripts because a mismatch against the kernel's expectation is the first
#: thing to check if a live counter read fails with EINVAL.
PERF_EVENT_ATTR_SIZE = ctypes.sizeof(PerfEventAttr)


def open_perf_counter(perf_type: int, config: int) -> int:
    """Open a ``perf_event_open(2)`` counter for this process on any CPU.

    UNVERIFIED ON THE TARGET. This adapter was written from
    ``perf_event_open(2)`` and ``linux/perf_event.h`` and exercised only on the
    x86_64 build host, where the syscall returns ``-1`` because the virtual
    machine exposes no PMU. That failure path is the one this package tests.
    On an aarch64 Jetson the syscall number is 241 rather than 298; both are
    declared in ``boardprobe.orin_nano.PERF_EVENT_OPEN_SYSCALL`` and the
    aarch64 entry has never been executed.

    Returns
    -------
    int
        A file descriptor the caller must close.

    Raises
    ------
    DeviceUnreachableError
        With the C ``errno`` name and what to check. ``EACCES`` or ``EPERM``
        means ``/proc/sys/kernel/perf_event_paranoid`` is above 1; ``ENOENT``
        means the kernel has no PMU driver bound, which is what a virtual
        machine without a passed-through PMU reports.
    """
    import os

    LEDGER.record("perf-event-open", f"{perf_type}:{config}")
    machine = _platform.machine()
    try:
        number = PERF_EVENT_OPEN_SYSCALL[machine]
    except KeyError as error:
        raise DeviceUnreachableError(
            f"no perf_event_open syscall number is declared for machine {machine!r}; "
            "declared machines are "
            + ", ".join(sorted(PERF_EVENT_OPEN_SYSCALL))
            + ". Add the number from the kernel syscall table for this architecture to "
            "boardprobe.orin_nano.PERF_EVENT_OPEN_SYSCALL."
        ) from error
    libc = ctypes.CDLL(None, use_errno=True)
    attributes = PerfEventAttr()
    attributes.size = PERF_EVENT_ATTR_SIZE
    attributes.type = perf_type
    attributes.config = config
    attributes.flags = sum(1 << bit for bit in PERF_FLAG_BITS.values())
    ctypes.set_errno(0)
    descriptor = libc.syscall(
        ctypes.c_long(number),
        ctypes.byref(attributes),
        ctypes.c_int(0),
        ctypes.c_int(-1),
        ctypes.c_int(-1),
        ctypes.c_ulong(0),
    )
    if descriptor < 0:
        code = ctypes.get_errno()
        raise DeviceUnreachableError(
            f"perf_event_open(type={perf_type}, config={config}) failed with errno {code} "
            f"({os.strerror(code)}) on {machine}. ENOENT means the kernel has no PMU "
            "driver bound, which is what a virtual machine without a passed-through PMU "
            "reports; EACCES or EPERM means /proc/sys/kernel/perf_event_paranoid is above "
            "1. See docs/DEPLOYMENT.md, 'performance counters'."
        )
    return int(descriptor)
