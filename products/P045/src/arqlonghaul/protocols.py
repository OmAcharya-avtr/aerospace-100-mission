"""Slotted simulators for the three ARQ protocol state machines.

These are state machines, not throughput formulas.  Each keeps a send window, a
feedback pipeline of length N slots, and a sequence-number base that advances
only in order, which is what makes them reproduce effects the closed forms
cannot: head-of-line blocking, go-back storms, and the interaction between
burst length and retransmission spacing.

Timing convention, matching :mod:`closedform`:

    * time is discrete, one slot = one frame transmission time T_f,
    * a frame transmitted in slot ``t`` has its acknowledgement or negative
      acknowledgement usable at the *start* of slot ``t + N``,
    * the reverse channel is error free (see README limitations),
    * ``N = 1 + RTT/T_f`` so that stop-and-wait, which transmits once per cycle,
      occupies exactly N slots per attempt.

Each simulator takes a pre-generated boolean array ``errors`` of length
``n_slots``: ``errors[t]`` is whether a frame transmitted in slot ``t`` would
arrive in error.  Passing the array in rather than a channel object is what
allows the three protocols to be compared on the identical channel realisation.

Normalised throughput reported by all three is

    eta = frames delivered in order / slots elapsed

counted over the slots actually simulated.  A warm-up is discarded by default so
the measurement is of the stationary regime rather than of the pipeline filling.
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "SimResult",
    "simulate_stop_and_wait",
    "simulate_go_back_n",
    "simulate_selective_repeat",
    "simulate",
    "PROTOCOL_NAMES",
]

PROTOCOL_NAMES: tuple[str, ...] = (
    "stop_and_wait",
    "go_back_n",
    "selective_repeat",
)


@dataclass
class SimResult:
    """Outcome of one protocol run.

    Attributes:
        protocol: Protocol name.
        goodput: Frames delivered in order per slot, dimensionless in [0, 1].
        delivered: Frames delivered in order over the measured window, frames.
        transmissions: Frame transmissions attempted over the measured window.
        slots: Slots in the measured window.
        idle_slots: Measured slots in which nothing was transmitted.
        errored_transmissions: Transmissions that arrived in error.
        window: Send window used, frames.
        n: Slots per cycle N used.
        mean_transmissions_per_frame: ``transmissions / delivered``, or ``inf``.
        batch_goodput: Per-batch goodput, used for the standard error.
    """

    protocol: str
    goodput: float
    delivered: int
    transmissions: int
    slots: int
    idle_slots: int
    errored_transmissions: int
    window: int
    n: int
    mean_transmissions_per_frame: float
    batch_goodput: np.ndarray = field(default_factory=lambda: np.empty(0))

    @property
    def utilisation(self) -> float:
        """Fraction of measured slots carrying a transmission, dimensionless."""
        return 0.0 if self.slots == 0 else 1.0 - self.idle_slots / self.slots

    @property
    def goodput_stderr(self) -> float:
        """Standard error of :attr:`goodput` from the batch means, dimensionless.

        Batch means are the standard Monte Carlo estimator for a correlated
        time series: the run is cut into equal blocks long enough that block
        means are near independent, and the standard error is the sample
        standard deviation of the block means divided by sqrt(number of
        blocks).  Returns ``nan`` with fewer than two batches.
        """
        b = self.batch_goodput
        if b.size < 2:
            return float("nan")
        return float(np.std(b, ddof=1) / math.sqrt(b.size))

    def goodput_bps(self, payload_bits: int, frame_time_s: float) -> float:
        """Information throughput, bits per second.

        Args:
            payload_bits: Information bits per frame.
            frame_time_s: Slot duration, seconds.
        """
        if payload_bits <= 0 or frame_time_s <= 0:
            raise ValueError("payload_bits and frame_time_s must both be > 0")
        return self.goodput * payload_bits / frame_time_s

    def as_dict(self) -> dict[str, float | int | str]:
        """Flat dict of scalars, for printing and serialisation."""
        return {
            "protocol": self.protocol,
            "window": self.window,
            "N": self.n,
            "goodput": self.goodput,
            "goodput_stderr": self.goodput_stderr,
            "delivered": self.delivered,
            "transmissions": self.transmissions,
            "slots": self.slots,
            "utilisation": self.utilisation,
            "errored_transmissions": self.errored_transmissions,
            "mean_tx_per_frame": self.mean_transmissions_per_frame,
        }


def _prepare(errors: np.ndarray, n: int, window: int, warmup: int | None) -> tuple:
    """Validate arguments and return ``(errors, n_slots, n, window, warmup)``."""
    e = np.asarray(errors, dtype=bool)
    if e.ndim != 1:
        raise ValueError(f"errors must be 1-D, got shape {e.shape}")
    if e.size < 2:
        raise ValueError(f"errors must have at least 2 slots, got {e.size}")
    if int(n) != n or n < 1:
        raise ValueError(f"n must be a positive integer number of slots, got {n}")
    if int(window) != window or window < 1:
        raise ValueError(f"window must be a positive integer, got {window}")
    n = int(n)
    window = int(window)
    if warmup is None:
        warmup = min(10 * n, e.size // 4)
    if not 0 <= warmup < e.size:
        raise ValueError(f"warmup must be in [0, {e.size}), got {warmup}")
    return e, e.size, n, window, int(warmup)


def _batches(delivered_at: list[int], warmup: int, n_slots: int, n_batches: int) -> np.ndarray:
    """Per-batch goodput from a list of delivery slot indices."""
    if n_batches < 2 or n_slots - warmup < 2 * n_batches:
        return np.empty(0)
    edges = np.linspace(warmup, n_slots, n_batches + 1)
    counts, _ = np.histogram(np.asarray(delivered_at, dtype=float), bins=edges)
    widths = np.diff(edges)
    return counts / widths


def simulate_stop_and_wait(
    errors: np.ndarray,
    n: int,
    warmup: int | None = None,
    n_batches: int = 20,
) -> SimResult:
    """Simulate stop-and-wait (window 1) over a given error realisation.

    The sender transmits the frame at the base of the window, waits N-1 slots,
    reads the feedback, and either advances or retransmits.  Note that the N-1
    waiting slots still consume channel time, so a long burst that starts after
    the transmission slot is invisible to this protocol -- which is exactly why
    stop-and-wait throughput depends only on the marginal frame error rate.

    Args:
        errors: Boolean per-slot error flags.
        n: Slots per cycle, N >= 1.
        warmup: Slots to discard before measuring; defaults to ``min(10N, L/4)``.
        n_batches: Number of batches for the batch-means standard error.

    Returns:
        A :class:`SimResult`.
    """
    e, n_slots, n, _, warmup = _prepare(errors, n, 1, warmup)
    delivered = transmissions = errored = 0
    delivered_at: list[int] = []
    t = 0
    while t + n <= n_slots:
        measured = t >= warmup
        if measured:
            transmissions += 1
        if e[t]:
            if measured:
                errored += 1
        else:
            if measured:
                delivered += 1
                delivered_at.append(t + n)
        t += n
    measured_slots = max(1, t - warmup)
    idle = measured_slots - transmissions
    return SimResult(
        protocol="stop_and_wait",
        goodput=delivered / measured_slots,
        delivered=delivered,
        transmissions=transmissions,
        slots=measured_slots,
        idle_slots=max(0, idle),
        errored_transmissions=errored,
        window=1,
        n=n,
        mean_transmissions_per_frame=(
            transmissions / delivered if delivered else float("inf")
        ),
        batch_goodput=_batches(delivered_at, warmup, t, n_batches),
    )


@dataclass
class _Outstanding:
    seq: int
    send_slot: int
    err: bool


def simulate_go_back_n(
    errors: np.ndarray,
    n: int,
    window: int,
    warmup: int | None = None,
    n_batches: int = 20,
) -> SimResult:
    """Simulate go-back-N with cumulative acknowledgement.

    On learning that the frame at the window base was in error, the sender
    discards everything in flight and restarts transmission from the base.  The
    cost of one frame error is therefore up to N slots of pipeline, which is why
    go-back-N throughput is strongly non-linear in the frame error rate and why
    it is the protocol whose behaviour correlation changes most: a burst that
    kills several consecutive frames costs *one* go-back, where the same number
    of independent errors would cost several.

    Args:
        errors: Boolean per-slot error flags.
        n: Slots per cycle, N >= 1.
        window: Send window, frames.  Continuous transmission needs
            ``window >= ceil(N)``; a smaller window is simulated faithfully
            (the sender stalls) but the closed form no longer applies.
        warmup: Slots to discard before measuring.
        n_batches: Batches for the standard error.
    """
    e, n_slots, n, window, warmup = _prepare(errors, n, window, warmup)
    base = 0
    next_seq = 0
    outstanding: deque[_Outstanding] = deque()
    delivered = transmissions = errored = idle = 0
    delivered_at: list[int] = []
    for t in range(n_slots):
        measured = t >= warmup
        # 1. Read feedback that has become available at the start of this slot.
        while outstanding and outstanding[0].send_slot + n <= t:
            frame = outstanding[0]
            if frame.seq != base:
                outstanding.popleft()
                continue
            outstanding.popleft()
            if frame.err:
                outstanding.clear()
                next_seq = base
                break
            base += 1
            if measured:
                delivered += 1
                delivered_at.append(t)
        # 2. Transmit if the window permits.
        if next_seq < base + window:
            outstanding.append(_Outstanding(next_seq, t, bool(e[t])))
            next_seq += 1
            if measured:
                transmissions += 1
                if e[t]:
                    errored += 1
        elif measured:
            idle += 1
    measured_slots = max(1, n_slots - warmup)
    return SimResult(
        protocol="go_back_n",
        goodput=delivered / measured_slots,
        delivered=delivered,
        transmissions=transmissions,
        slots=measured_slots,
        idle_slots=idle,
        errored_transmissions=errored,
        window=window,
        n=n,
        mean_transmissions_per_frame=(
            transmissions / delivered if delivered else float("inf")
        ),
        batch_goodput=_batches(delivered_at, warmup, n_slots, n_batches),
    )


def simulate_selective_repeat(
    errors: np.ndarray,
    n: int,
    window: int,
    warmup: int | None = None,
    n_batches: int = 20,
) -> SimResult:
    """Simulate selective repeat with per-frame acknowledgement.

    Only errored frames are retransmitted, lowest sequence number first, and the
    window base advances in order as frames are acknowledged.  That last clause
    is the honest part: a real selective-repeat sender cannot release a sequence
    number until the frame at the base is acknowledged, so a frame that keeps
    failing fills the window and stalls the sender.  The ideal closed form
    ``eta = 1 - p`` assumes that never happens, which requires an unbounded
    window; the gap is measured in ``validation/validate_closed_form.py``.

    Args:
        errors: Boolean per-slot error flags.
        n: Slots per cycle, N >= 1.
        window: Send window, frames.
        warmup: Slots to discard before measuring.
        n_batches: Batches for the standard error.
    """
    e, n_slots, n, window, warmup = _prepare(errors, n, window, warmup)
    base = 0
    next_seq = 0
    pending: deque[_Outstanding] = deque()
    acked: set[int] = set()
    retx: list[int] = []
    delivered = transmissions = errored = idle = 0
    delivered_at: list[int] = []
    for t in range(n_slots):
        measured = t >= warmup
        while pending and pending[0].send_slot + n <= t:
            frame = pending.popleft()
            if frame.err:
                heapq.heappush(retx, frame.seq)
            else:
                acked.add(frame.seq)
        while base in acked:
            acked.discard(base)
            base += 1
            if measured:
                delivered += 1
                delivered_at.append(t)
        seq: int | None = None
        if retx:
            seq = heapq.heappop(retx)
        elif next_seq < base + window:
            seq = next_seq
            next_seq += 1
        if seq is None:
            if measured:
                idle += 1
        else:
            pending.append(_Outstanding(seq, t, bool(e[t])))
            if measured:
                transmissions += 1
                if e[t]:
                    errored += 1
    measured_slots = max(1, n_slots - warmup)
    return SimResult(
        protocol="selective_repeat",
        goodput=delivered / measured_slots,
        delivered=delivered,
        transmissions=transmissions,
        slots=measured_slots,
        idle_slots=idle,
        errored_transmissions=errored,
        window=window,
        n=n,
        mean_transmissions_per_frame=(
            transmissions / delivered if delivered else float("inf")
        ),
        batch_goodput=_batches(delivered_at, warmup, n_slots, n_batches),
    )


def simulate(
    protocol: str,
    errors: np.ndarray,
    n: int,
    window: int = 1,
    warmup: int | None = None,
    n_batches: int = 20,
) -> SimResult:
    """Dispatch to one of the three simulators by name.

    Args:
        protocol: One of :data:`PROTOCOL_NAMES`.
        errors: Boolean per-slot error flags.
        n: Slots per cycle N.
        window: Send window, frames (ignored by stop-and-wait).
        warmup: Slots to discard before measuring.
        n_batches: Batches for the standard error.

    Raises:
        ValueError: on an unknown protocol name.
    """
    if protocol == "stop_and_wait":
        return simulate_stop_and_wait(errors, n, warmup, n_batches)
    if protocol == "go_back_n":
        return simulate_go_back_n(errors, n, window, warmup, n_batches)
    if protocol == "selective_repeat":
        return simulate_selective_repeat(errors, n, window, warmup, n_batches)
    raise ValueError(f"unknown protocol {protocol!r}; expected one of {PROTOCOL_NAMES}")
