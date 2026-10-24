"""The seeded simulator: one transfer over one realisation of the link model.

What this is and is not
-----------------------
This is a *measurement harness*. It drives a declared retransmission
discipline against a seeded realisation of the link model in
:mod:`relfiledelivery.link` and reports what the transfer cost. It is **not a
CFDP implementation**: no PDU here is wire-conformant, there is no transaction
state machine, no store-and-forward, no filestore requests, no checksum
procedure from the standard. ``cfdp-py`` and ``spacepackets`` implement the
protocol; use them for that.

Declared modelling choices
--------------------------
Each of these is a choice, not a property of any protocol, and each one is a
place where a real transfer can differ from this measurement:

M1. A protocol data unit is never split across a contact boundary. If the
    remaining window time cannot carry it, it starts in the next window.
M2. A unit whose transmission interval intersects an outage is lost in full.
M3. The sender transmits during an outage, because a blind forward link has no
    way to know. This is the conservative case and it inflates the measured
    transmission count relative to an outage-aware sender.
M4. Control units -- end-of-file on the forward link, negative acknowledgement
    on the return link -- are never lost. Their time and bytes are charged in
    full, but no NAK timer and no control retransmission is modelled. A lost
    NAK in a real transfer costs a timer expiry that this harness does not
    measure.
M5. In immediate-NAK, the receiver reveals a gap at the arrival of the next
    successfully received unit, and names every outstanding gap in one NAK at
    that moment.
M6. Delivery time is the moment the *receiver* holds the whole file. The sender
    learns this one further turnaround later; that is reported separately as
    the confirmation time.
M7. The return link is gated by the same contact windows as the forward link,
    so the windows represent mutual visibility.

The completion invariant
------------------------
Every run that completes reassembles the received units and compares the
SHA-256 of the reassembly with the SHA-256 of the source. A mismatch raises
:class:`relfiledelivery.payload.CompletionInvariantError`. This is on by
default, it is checked inside the simulator rather than by a caller, and it is
a hard failure: a harness that reports a delivery-time distribution without
ever reconstructing the file would not notice an off-by-one in its own segment
accounting. :func:`simulate_transfer` also takes ``inject_corruption_at`` so
that the check can be shown to fail when it should, which is the only evidence
that an invariant that always passes is doing anything.
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .analytic import expected_total_transmissions
from .link import ContactSchedule, LinkConditions, OutageTrace
from .modes import TransferMode
from .payload import (
    CompletionInvariantError,
    FileTransferSpec,
    digest,
    reassemble,
    source_payload,
)


@dataclass(frozen=True)
class TransferResult:
    """What one simulated transfer cost.

    Attributes
    ----------
    mode, seed, pdu_count:
        What was run.
    delivered:
        True when the receiver held the whole file before the run stopped.
    delivery_time_s:
        Seconds from the first possible transmission instant to the moment the
        receiver held the file, or ``None`` when not delivered. The clock
        starts at the first contact-window start, so dead time before the first
        pass is not charged.
    confirmation_time_s:
        Seconds until the sender knew, one turnaround later (M6). ``None`` for
        unacknowledged or undelivered transfers.
    forward_transmissions:
        Data-unit transmissions on the forward link, including retransmissions.
    retransmissions:
        ``forward_transmissions`` minus the number of distinct units sent at
        least once. Cross-checked against a per-unit attempt count.
    attempts:
        Per-unit transmission counts, in sequence order. The independent count
        the aggregate is checked against.
    rounds:
        Data rounds for deferred-NAK; end-of-file cycles plus one for
        immediate-NAK; 1 for unacknowledged.
    forward_wire_bytes, return_wire_bytes:
        Bytes on each link, control units included.
    overhead_ratio:
        ``(forward + return wire bytes) / file_bytes``, dimensionless.
    forward_busy_s:
        Seconds the forward link spent transmitting data units.
    idle_s:
        ``delivery_time_s`` minus forward busy time minus forward control time,
        seconds: time lost to light time, turnaround, outages and contact gaps.
        ``None`` when not delivered.
    completion_contact_index:
        Zero-based index of the contact window in which the completing unit was
        transmitted; ``-1`` for continuous contact, ``None`` when not delivered.
    outage_seconds:
        Down time inside the span the transfer occupied, seconds.
    source_digest, delivered_digest:
        SHA-256 of the source and of the reassembly. Equal on every delivered
        run, or :class:`CompletionInvariantError` was raised.
    truncated, truncation_reason:
        Whether the run stopped at a declared limit, and which.
    """

    mode: TransferMode
    seed: int
    pdu_count: int
    delivered: bool
    delivery_time_s: float | None
    confirmation_time_s: float | None
    forward_transmissions: int
    retransmissions: int
    attempts: tuple[int, ...]
    rounds: int
    forward_wire_bytes: int
    return_wire_bytes: float
    overhead_ratio: float
    forward_busy_s: float
    idle_s: float | None
    completion_contact_index: int | None
    outage_seconds: float
    source_digest: str
    delivered_digest: str | None
    truncated: bool = False
    truncation_reason: str = ""
    nak_count: int = 0
    received_count: int = 0

    def describe(self) -> list[str]:
        """Human-readable lines for a report."""
        delivery = "not delivered" if not self.delivered else f"{self.delivery_time_s:.6f} s"
        return [
            f"mode                    : {self.mode.value}",
            f"seed                    : {self.seed}",
            f"delivered               : {self.delivered}"
            + (f" (truncated: {self.truncation_reason})" if self.truncated else ""),
            f"delivery time           : {delivery}",
            f"forward transmissions   : {self.forward_transmissions} for "
            f"{self.pdu_count} PDUs",
            f"retransmissions         : {self.retransmissions}",
            f"rounds                  : {self.rounds}",
            f"NAKs sent               : {self.nak_count}",
            f"overhead ratio          : {self.overhead_ratio:.6f}",
            f"forward busy            : {self.forward_busy_s:.6f} s",
            "idle                    : "
            + ("n/a" if self.idle_s is None else f"{self.idle_s:.6f} s"),
            f"outage time in span     : {self.outage_seconds:.6f} s",
            f"completing contact      : {self.completion_contact_index}",
            f"source digest           : {self.source_digest[:16]}...",
        ]


@dataclass
class _Counters:
    forward_transmissions: int = 0
    forward_wire_bytes: int = 0
    return_wire_bytes: float = 0.0
    nak_count: int = 0
    forward_busy_s: float = 0.0
    forward_control_s: float = 0.0
    attempts: list[int] = field(default_factory=list)


def default_transmission_cap(spec: FileTransferSpec, link: LinkConditions) -> int:
    """Default limit on forward data-unit transmissions, dimensionless.

    Twenty times the expected transmission count inflated by the outage duty
    cycle, floored at 256. A transfer that needs more than that has effectively
    failed, and the alternative to a cap is a run that does not terminate when
    the declared link cannot carry the file.
    """
    duty = max(link.outage.duty_cycle, 1e-3)
    scale = expected_total_transmissions(spec.pdu_count, link.pdu_loss_probability) / duty
    return int(max(256.0, 20.0 * scale))


def simulate_transfer(
    spec: FileTransferSpec,
    link: LinkConditions,
    mode: TransferMode,
    seed: int,
    *,
    verify: bool = True,
    horizon_s: float | None = None,
    max_forward_transmissions: int | None = None,
    payload_seed: int | None = None,
    inject_corruption_at: int | None = None,
) -> TransferResult:
    """Simulate one transfer and return what it cost.

    Parameters
    ----------
    spec:
        File and segmentation.
    link:
        Declared link conditions.
    mode:
        Retransmission discipline.
    seed:
        Integer seed. Two calls with equal arguments return equal results; this
        is asserted by test.
    verify:
        When True (the default) a delivered transfer is reassembled and its
        SHA-256 compared with the source. Setting it False skips the
        reassembly, which is only useful for a timing measurement of the
        simulator itself, and no number reported by this package is produced
        with it off.
    horizon_s:
        Wall-clock limit, seconds. Defaults to the end of the last declared
        contact window, or unlimited for continuous contact.
    max_forward_transmissions:
        Limit on forward data-unit transmissions. Defaults to
        :func:`default_transmission_cap`.
    payload_seed:
        Seed for the source bytes. Defaults to ``seed``, so changing the seed
        changes both the file and the link realisation; pass it explicitly to
        hold the file fixed across seeds.
    inject_corruption_at:
        Byte offset to flip in the reassembly *after* a successful transfer and
        *before* hashing. Exists so that the completion invariant can be shown
        to fail when it should; a delivered run with this set raises
        :class:`CompletionInvariantError`.

    Raises
    ------
    CompletionInvariantError
        When a delivered transfer does not reproduce the source bytes.
    """
    if not isinstance(mode, TransferMode):
        raise TypeError(f"mode must be a TransferMode; got {type(mode).__name__}")
    sequence = np.random.SeedSequence(int(seed))
    loss_rng, outage_rng = (np.random.default_rng(child) for child in sequence.spawn(2))

    source = source_payload(spec, seed if payload_seed is None else payload_seed)
    source_digest = digest(source)
    slices = spec.slices()

    schedule = ContactSchedule(link.windows)
    trace = OutageTrace(link.outage, outage_rng, schedule.start_time_s())
    t0 = schedule.start_time_s()
    limit_s = schedule.horizon_s if horizon_s is None else float(horizon_s)
    cap = (
        default_transmission_cap(spec, link)
        if max_forward_transmissions is None
        else int(max_forward_transmissions)
    )
    if cap < 1:
        raise ValueError(f"max_forward_transmissions must be at least 1; got {cap}")

    n = spec.pdu_count
    counters = _Counters(attempts=[0] * n)
    received: dict[int, bytes] = {}
    light = link.one_way_light_time_s
    loss_p = link.pdu_loss_probability

    state = {
        "t": t0,
        "truncated": False,
        "reason": "",
        "last_arrival": None,
        "last_window": None,
        "rounds": 0,
        "max_time": t0,
    }

    def transmit(seq: int, after_s: float) -> tuple[bool, float, float, int] | None:
        duration = 8.0 * spec.wire_bytes_of(seq) / link.forward_rate_bps
        placed = schedule.place(after_s, duration)
        if placed is None:
            state["truncated"] = True
            state["reason"] = "no remaining contact window can carry a PDU"
            return None
        start, end, window = placed
        if end > limit_s:
            state["truncated"] = True
            state["reason"] = f"horizon of {limit_s:.6g} s reached"
            return None
        if counters.forward_transmissions >= cap:
            state["truncated"] = True
            state["reason"] = f"transmission cap of {cap} reached"
            return None
        counters.forward_transmissions += 1
        counters.attempts[seq] += 1
        counters.forward_wire_bytes += spec.wire_bytes_of(seq)
        counters.forward_busy_s += duration
        draw = loss_rng.random()
        lost = trace.intersects_outage(start, end) or draw < loss_p
        state["t"] = end
        state["max_time"] = max(state["max_time"], end)
        if not lost:
            lo, hi = slices[seq]
            received[seq] = source[lo:hi]
            state["last_arrival"] = end + light
            state["last_window"] = window
        return lost, start, end, window

    def send_forward_control(nbytes: float, after_s: float) -> float | None:
        duration = 8.0 * nbytes / link.forward_rate_bps
        placed = schedule.place(after_s, duration)
        if placed is None or placed[1] > limit_s:
            state["truncated"] = True
            state["reason"] = (
                "no remaining contact window can carry an end-of-file PDU"
                if placed is None
                else f"horizon of {limit_s:.6g} s reached"
            )
            return None
        counters.forward_wire_bytes += int(nbytes)
        counters.forward_control_s += duration
        state["max_time"] = max(state["max_time"], placed[1])
        return placed[1]

    def send_return_control(nbytes: float, after_s: float) -> float | None:
        duration = 8.0 * nbytes / link.return_rate_bps
        placed = schedule.place(after_s, duration)
        if placed is None or placed[1] > limit_s:
            state["truncated"] = True
            state["reason"] = (
                "no remaining contact window can carry a NAK"
                if placed is None
                else f"horizon of {limit_s:.6g} s reached"
            )
            return None
        counters.return_wire_bytes += nbytes
        counters.nak_count += 1
        state["max_time"] = max(state["max_time"], placed[1])
        return placed[1]

    def turnaround(missing: int, after_s: float) -> float | None:
        """End-of-file, light time out, NAK, light time back. Returns sender time."""
        eof_end = send_forward_control(link.eof_bytes, after_s)
        if eof_end is None:
            return None
        nak_bytes = link.nak_bytes + link.nak_bytes_per_segment * missing
        nak_end = send_return_control(nak_bytes, eof_end + light)
        if nak_end is None:
            return None
        return nak_end + light

    if mode is TransferMode.UNACKNOWLEDGED:
        state["rounds"] = 1
        for seq in range(n):
            if transmit(seq, state["t"]) is None:
                break
    elif mode is TransferMode.DEFERRED_NAK:
        missing = list(range(n))
        while missing:
            state["rounds"] += 1
            still: list[int] = []
            aborted = False
            for seq in missing:
                outcome = transmit(seq, state["t"])
                if outcome is None:
                    aborted = True
                    break
                if outcome[0]:
                    still.append(seq)
            if aborted:
                break
            missing = still
            if not missing:
                break
            resume = turnaround(len(missing), state["t"])
            if resume is None:
                break
            state["t"] = resume
    elif mode is TransferMode.IMMEDIATE_NAK:
        state["rounds"] = 1
        queue: deque[int] = deque(range(n))
        pending: list[tuple[float, int]] = []
        awaiting_detection: list[int] = []
        while len(received) < n:
            while pending and pending[0][0] <= state["t"]:
                _, seq = heapq.heappop(pending)
                if seq not in received:
                    queue.append(seq)
            if not queue:
                if pending:
                    state["t"] = max(state["t"], pending[0][0])
                    continue
                outstanding = [seq for seq in range(n) if seq not in received]
                if not outstanding:
                    break
                resume = turnaround(len(outstanding), state["t"])
                if resume is None:
                    break
                state["t"] = resume
                state["rounds"] += 1
                awaiting_detection.clear()
                queue.extend(outstanding)
                continue
            seq = queue.popleft()
            if seq in received:
                continue
            outcome = transmit(seq, state["t"])
            if outcome is None:
                break
            lost = outcome[0]
            if lost:
                awaiting_detection.append(seq)
            elif awaiting_detection:
                # M5: the arrival of this unit reveals every outstanding gap,
                # and the receiver names them all in one NAK sent at once.
                arrival = outcome[2] + light
                nak_bytes = link.nak_bytes + link.nak_bytes_per_segment * len(awaiting_detection)
                nak_end = send_return_control(nak_bytes, arrival)
                if nak_end is None:
                    break
                available = nak_end + light
                for gap in awaiting_detection:
                    heapq.heappush(pending, (available, gap))
                awaiting_detection.clear()
    else:  # pragma: no cover - TransferMode is exhaustive
        raise ValueError(f"unhandled mode {mode!r}")

    delivered = len(received) == n
    delivery_time_s: float | None = None
    confirmation_time_s: float | None = None
    completion_contact_index: int | None = None
    idle_s: float | None = None
    delivered_digest: str | None = None

    if delivered:
        delivery_time_s = float(state["last_arrival"]) - t0
        completion_contact_index = int(state["last_window"])
        idle_s = delivery_time_s - counters.forward_busy_s - counters.forward_control_s
        if mode.uses_return_link:
            confirmation_time_s = (
                delivery_time_s
                + link.eof_seconds
                + link.nak_seconds(0)
                + 2.0 * light
            )
        if verify:
            data = reassemble(spec, received)
            if inject_corruption_at is not None:
                index = int(inject_corruption_at) % len(data)
                mutable = bytearray(data)
                mutable[index] ^= 0x01
                data = bytes(mutable)
            delivered_digest = digest(data)
            if delivered_digest != source_digest:
                raise CompletionInvariantError(
                    "completion invariant violated: the reassembled file does not match the "
                    f"source. mode={mode.value}, seed={seed}, pdu_count={n}, "
                    f"received={len(received)}, source={source_digest}, "
                    f"delivered={delivered_digest}"
                )
        else:
            delivered_digest = None

    distinct_sent = sum(1 for count in counters.attempts if count > 0)
    retransmissions = counters.forward_transmissions - distinct_sent
    total_wire = counters.forward_wire_bytes + counters.return_wire_bytes
    outage_seconds = trace.outage_seconds_in(t0, float(state["max_time"]))

    return TransferResult(
        mode=mode,
        seed=int(seed),
        pdu_count=n,
        delivered=delivered,
        delivery_time_s=delivery_time_s,
        confirmation_time_s=confirmation_time_s,
        forward_transmissions=counters.forward_transmissions,
        retransmissions=retransmissions,
        attempts=tuple(counters.attempts),
        rounds=int(state["rounds"]),
        forward_wire_bytes=counters.forward_wire_bytes,
        return_wire_bytes=counters.return_wire_bytes,
        overhead_ratio=total_wire / spec.file_bytes,
        forward_busy_s=counters.forward_busy_s,
        idle_s=idle_s,
        completion_contact_index=completion_contact_index,
        outage_seconds=outage_seconds,
        source_digest=source_digest,
        delivered_digest=delivered_digest,
        truncated=bool(state["truncated"]),
        truncation_reason=str(state["reason"]),
        nak_count=counters.nak_count,
        received_count=len(received),
    )


def independent_transmission_count(result: TransferResult) -> int:
    """Forward transmissions recomputed from the per-unit attempt counts.

    An accounting cross-check that does not reuse the aggregate counter:
    ``sum(result.attempts)`` must equal ``result.forward_transmissions``.
    """
    return int(sum(result.attempts))


def accounting_discrepancy(result: TransferResult) -> int:
    """Signed difference between the aggregate and the independent count.

    Zero on every run, and asserted zero by test. Reported rather than assumed
    because a transmission counter and a per-unit attempt array are two
    separate pieces of bookkeeping, and the cheapest way for a harness to
    report a wrong overhead is for them to disagree.
    """
    return result.forward_transmissions - independent_transmission_count(result)


def expected_wire_bytes_from_attempts(spec: FileTransferSpec, result: TransferResult) -> int:
    """Forward data-unit bytes recomputed from the attempt counts, bytes.

    Excludes control units, so it is compared against
    ``forward_wire_bytes`` minus the charged control bytes rather than against
    it directly.
    """
    return sum(
        count * spec.wire_bytes_of(seq) for seq, count in enumerate(result.attempts)
    )


def is_outage_free_continuous(link: LinkConditions) -> bool:
    """True when the link is in the regime the closed form is exact in."""
    return (
        link.outage.is_outage_free
        and link.is_continuous
        and link.nak_bytes_per_segment == 0.0
        and math.isfinite(link.one_way_light_time_s)
    )
