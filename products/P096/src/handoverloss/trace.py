"""The simulated handover trace: baseline 3's ground truth, vectorised over events.

WHAT THIS IS
------------

A discrete-slot tail-drop buffer driven by a bursty source and the declared
egress schedule, stepped slot by slot, with every dropped bit tagged by the
phase in which it was dropped. It is the ground truth the two analytic baselines
and the learned component are scored against, and it is deliberately NOT the
fluid model:

* arrivals carry per-slot multiplicative burstiness with unit mean, so a burst
  can overflow a buffer the fluid model leaves with headroom;
* the egress carries its own per-slot jitter with unit mean, so the realised
  ramp is not the declared straight line;
* time is discrete, so the overflow instant is resolved only to one slot.

Those three differences are the whole reason there is a ground truth separate
from baseline 2, and the size of each is a reported number in
``validation/validate_baselines_against_truth.py``.

THE RECURSION
-------------

Per slot of width ``dt``, for each event independently,

::

    arrival_n  = R_s * dt * burst_n                      (burst_n > 0, mean 1)
    service_n  = r(t_n) * dt * jitter_n                  (jitter_n > 0, mean 1)
    unconstrained = q_n + arrival_n - min(service_n, q_n + arrival_n)
    drop_n     = max(0, unconstrained - B)
    q_{n+1}    = min(unconstrained, B)

Service is applied to what is in the buffer after the arrival, which is the
cut-through convention: a slot's arrival may be served in the same slot. The
alternative (store-and-forward, service applied to ``q_n`` only) shifts every
delivery by one slot and changes the total by at most ``R_d * dt`` per event.
The convention is declared here because it is not inferable from the result.

``drop_n`` is accumulated into the phase bucket of slot ``n``, where the phase
is the one :meth:`~handoverloss.sequence.HandoverSequence.phase_of` gives for
the slot's START time. The phase of a slot is fixed before the simulation runs,
so the three buckets partition the window exactly and the sum of the buckets is
the total dropped volume up to the accumulation order of the sum.

WHY IT IS VECTORISED OVER EVENTS AND NOT OVER SLOTS
---------------------------------------------------

The recursion is sequential in slots and cannot be turned into a prefix sum for
a finite buffer with tail drop, because the clamp at ``B`` is a non-linear
reset. It IS independent across events, so the loop runs over slots and each
step is one numpy operation over all events at once. On the 2-core build
machine this is the difference between a corpus that takes minutes and one that
takes seconds; the measured figure is in ``validation/VALIDATION.md``.

UNITS
-----

Seconds, bit/s, bits. ``dt`` seconds. Burst and jitter factors dimensionless.

SOURCES
-------

* Finite-buffer fluid and discrete buffer formulations, and the tail-drop
  convention: Anick, D., Mitra, D. and Sondhi, M. M., "Stochastic theory of a
  data-handling system with multiple sources", *Bell System Technical Journal*,
  volume 61, number 8, pages 1871-1894, 1982.
* The reflected-random-walk form of the occupancy recursion, and the supremum
  representation the independent bound in :mod:`handoverloss.bounds` uses:
  Kleinrock, L., *Queueing Systems, Volume 1: Theory*, Wiley, 1975, chapter 2
  (Lindley's recursion), and Reich, E., "On the integrodifferential equation of
  Takacs I", *Annals of Mathematical Statistics*, volume 29, number 2, pages
  563-570, 1958.
* Lognormal multiplicative burstiness as a model for a positive, product-
  generated rate fluctuation: Aitchison, J. and Brown, J. A. C., *The Lognormal
  Distribution*, Cambridge University Press, 1957, chapter 1.

No numeric parameter value here is a measurement from any of those sources.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sequence import MECHANISMS, HandoverSequence


@dataclass(frozen=True)
class EventParameters:
    """Per-event sequence parameters as parallel float64 arrays, all length ``n``.

    Every field has the units of the matching
    :class:`~handoverloss.sequence.HandoverSequence` field. Validated at
    construction by the same bounds, applied elementwise.
    """

    source_rate_bps: np.ndarray
    downlink_rate_bps: np.ndarray
    buffer_capacity_bits: np.ndarray
    initial_occupancy_bits: np.ndarray
    acquisition_s: np.ndarray
    restart_s: np.ndarray
    ramp_s: np.ndarray
    horizon_s: float

    def __post_init__(self) -> None:
        fields = (
            "source_rate_bps",
            "downlink_rate_bps",
            "buffer_capacity_bits",
            "initial_occupancy_bits",
            "acquisition_s",
            "restart_s",
            "ramp_s",
        )
        lengths = {len(np.atleast_1d(getattr(self, name))) for name in fields}
        if len(lengths) != 1:
            raise ValueError(
                "every EventParameters array must have the same length, got lengths "
                f"{sorted(lengths)}"
            )
        for name in fields:
            array = np.asarray(getattr(self, name), dtype=float)
            if not np.isfinite(array).all():
                raise ValueError(f"EventParameters.{name} contains a non-finite value")
            object.__setattr__(self, name, array)
        if not float(self.horizon_s) > 0.0:
            raise ValueError(
                f"EventParameters.horizon_s must be positive, got {self.horizon_s!r}"
            )
        for name in ("source_rate_bps", "downlink_rate_bps", "buffer_capacity_bits"):
            if np.any(getattr(self, name) <= 0.0):
                raise ValueError(
                    f"EventParameters.{name} must be strictly positive everywhere"
                )
        for name in (
            "initial_occupancy_bits",
            "acquisition_s",
            "restart_s",
            "ramp_s",
        ):
            if np.any(getattr(self, name) < 0.0):
                raise ValueError(
                    f"EventParameters.{name} must be non-negative everywhere"
                )
        if np.any(self.initial_occupancy_bits > self.buffer_capacity_bits):
            raise ValueError(
                "EventParameters.initial_occupancy_bits exceeds "
                "buffer_capacity_bits for at least one event"
            )
        needed = self.acquisition_s + self.restart_s + self.ramp_s
        if np.any(float(self.horizon_s) < needed):
            worst = float(np.max(needed))
            raise ValueError(
                f"EventParameters.horizon_s ({float(self.horizon_s)!r}) is shorter than "
                f"the longest acquisition + restart + ramp ({worst!r}); the accounting "
                "window must contain the whole excursion for every event"
            )

    def __len__(self) -> int:
        return int(len(self.source_rate_bps))

    @classmethod
    def from_sequences(cls, sequences: list[HandoverSequence]) -> EventParameters:
        """Build from a list of validated :class:`HandoverSequence` objects.

        All sequences must share one ``horizon_s``; a corpus with differing
        windows would not have comparable drain-phase terms.
        """
        if not sequences:
            raise ValueError("at least one HandoverSequence is required")
        horizons = {s.horizon_s for s in sequences}
        if len(horizons) != 1:
            raise ValueError(
                "every HandoverSequence must share one horizon_s; got "
                f"{sorted(horizons)}"
            )
        return cls(
            source_rate_bps=np.array([s.source_rate_bps for s in sequences]),
            downlink_rate_bps=np.array([s.downlink_rate_bps for s in sequences]),
            buffer_capacity_bits=np.array([s.buffer_capacity_bits for s in sequences]),
            initial_occupancy_bits=np.array(
                [s.initial_occupancy_bits for s in sequences]
            ),
            acquisition_s=np.array([s.acquisition_s for s in sequences]),
            restart_s=np.array([s.restart_s for s in sequences]),
            ramp_s=np.array([s.ramp_s for s in sequences]),
            horizon_s=float(sequences[0].horizon_s),
        )


@dataclass(frozen=True)
class TraceResult:
    """Slot-level accounting over ``n`` simulated handovers. Bits throughout.

    Attributes
    ----------
    dropped_bits:
        ``(n, 3)`` array of dropped volume per mechanism, columns in
        :data:`MECHANISMS` order. This is the ground-truth decomposition.
    produced_bits:
        ``(n,)`` total arrivals offered over the window.
    delivered_bits:
        ``(n,)`` total volume served out of the buffer over the window.
    final_occupancy_bits:
        ``(n,)`` buffer occupancy at the end of the window.
    initial_occupancy_bits:
        ``(n,)`` buffer occupancy at the break, copied from the inputs so the
        conservation identity can be checked from this object alone.
    peak_occupancy_bits:
        ``(n,)`` largest occupancy reached.
    slots_full:
        ``(n,)`` count of slots that ended with the buffer at capacity.
    slot_seconds:
        The slot width ``dt`` used, seconds.
    slot_count:
        Number of slots stepped. Recorded because the accumulation error of the
        conservation sums scales with it, and
        :func:`~handoverloss.accounting.residual_tolerance_bits` needs it.
    """

    dropped_bits: np.ndarray
    produced_bits: np.ndarray
    delivered_bits: np.ndarray
    final_occupancy_bits: np.ndarray
    initial_occupancy_bits: np.ndarray
    peak_occupancy_bits: np.ndarray
    slots_full: np.ndarray
    slot_seconds: float
    slot_count: int

    def __len__(self) -> int:
        return int(len(self.produced_bits))

    @property
    def total_dropped_bits(self) -> np.ndarray:
        """``(n,)`` sum of the three mechanism buckets, bits."""
        return self.dropped_bits.sum(axis=1)

    @property
    def conservation_total_bits(self) -> np.ndarray:
        """``(n,)`` total loss from mass conservation alone, bits.

        ``produced - delivered - (q_end - q_0)``. This path never touches the
        per-phase buckets, so comparing it against :attr:`total_dropped_bits`
        is an independent check of the accounting and not a restatement of it.
        """
        return (
            self.produced_bits
            - self.delivered_bits
            - (self.final_occupancy_bits - self.initial_occupancy_bits)
        )

    @property
    def dominant_index(self) -> np.ndarray:
        """``(n,)`` int array: which mechanism dominates each event."""
        return np.argmax(self.dropped_bits, axis=1)

    def dominant_names(self) -> list[str]:
        """Mechanism name dominating each event, in event order."""
        return [MECHANISMS[int(i)] for i in self.dominant_index]


def simulate(
    parameters: EventParameters,
    rng: np.random.Generator,
    slot_seconds: float = 0.25,
    burst_log_sigma: float = 0.45,
    service_log_sigma: float = 0.12,
) -> TraceResult:
    """Step the tail-drop buffer over every event and return the accounting.

    Parameters
    ----------
    parameters:
        Per-event sequence parameters, already validated.
    rng:
        Source of the burst and jitter factors. A fixed-seed generator makes the
        whole result reproducible; ``tests/test_trace.py`` asserts that.
    slot_seconds:
        Slot width ``dt``, seconds, strictly positive. The discretisation error
        this introduces is bounded by ``R_s * dt`` per phase boundary and is
        measured in ``validation/validate_baselines_against_truth.py``.
    burst_log_sigma:
        Log-scale shape of the arrival burst factor. The factor is
        ``exp(sigma * z - sigma**2 / 2)`` so its mean is exactly 1 and the mean
        source rate is preserved; zero gives a constant-rate source.
    service_log_sigma:
        Log-scale shape of the service jitter factor, same unit-mean form.

    Returns
    -------
    TraceResult

    Raises
    ------
    ValueError
        On a non-positive ``slot_seconds``, a negative shape parameter, or a
        window that would need more than 2_000_000 slots, which is refused
        rather than run: the compute budget for this package is 60 s of CPU and
        a window that large would exceed it.
    """
    if not slot_seconds > 0.0:
        raise ValueError(f"slot_seconds must be strictly positive, got {slot_seconds!r}")
    if burst_log_sigma < 0.0 or service_log_sigma < 0.0:
        raise ValueError(
            "burst_log_sigma and service_log_sigma must be non-negative, got "
            f"{burst_log_sigma!r} and {service_log_sigma!r}"
        )
    count = len(parameters)
    horizon = float(parameters.horizon_s)
    slots = int(np.ceil(horizon / slot_seconds))
    if slots * max(count, 1) > 2_000_000_000 or slots > 2_000_000:
        raise ValueError(
            f"a window of {horizon} s at dt={slot_seconds} s needs {slots} slots, "
            "which exceeds this package's declared compute budget; widen slot_seconds "
            "or shorten horizon_s"
        )

    rate_s = parameters.source_rate_bps
    rate_d = parameters.downlink_rate_bps
    capacity = parameters.buffer_capacity_bits
    outage = parameters.acquisition_s + parameters.restart_s
    ramp = parameters.ramp_s

    occupancy = parameters.initial_occupancy_bits.astype(float).copy()
    stepped = 0
    dropped = np.zeros((count, 3), dtype=float)
    produced = np.zeros(count, dtype=float)
    delivered = np.zeros(count, dtype=float)
    peak = occupancy.copy()
    full = np.zeros(count, dtype=np.int64)

    burst_shift = 0.5 * burst_log_sigma * burst_log_sigma
    service_shift = 0.5 * service_log_sigma * service_log_sigma

    for index in range(slots):
        start = index * slot_seconds
        width = min(slot_seconds, horizon - start)
        if width <= 0.0:
            break
        stepped += 1

        burst = (
            np.exp(burst_log_sigma * rng.standard_normal(count) - burst_shift)
            if burst_log_sigma > 0.0
            else np.ones(count)
        )
        arrival = rate_s * width * burst
        produced += arrival

        # Declared egress schedule at the slot's START time, then jittered.
        elapsed = start - outage
        with np.errstate(divide="ignore", invalid="ignore"):
            fraction = np.where(ramp > 0.0, elapsed / np.where(ramp > 0.0, ramp, 1.0), 1.0)
        fraction = np.clip(fraction, 0.0, 1.0)
        nominal = np.where(elapsed < 0.0, 0.0, rate_d * fraction)
        jitter = (
            np.exp(service_log_sigma * rng.standard_normal(count) - service_shift)
            if service_log_sigma > 0.0
            else np.ones(count)
        )
        capability = nominal * width * jitter

        level = occupancy + arrival
        served = np.minimum(capability, level)
        delivered += served
        level -= served

        overflow = np.maximum(0.0, level - capacity)
        # Phase of this slot's start, as an index into the mechanism columns.
        phase = np.where(
            start < parameters.acquisition_s, 0, np.where(start < outage, 1, 2)
        )
        for column in range(3):
            dropped[:, column] += np.where(phase == column, overflow, 0.0)

        occupancy = np.minimum(level, capacity)
        peak = np.maximum(peak, occupancy)
        full += (occupancy >= capacity * (1.0 - 1e-12)).astype(np.int64)

    return TraceResult(
        dropped_bits=dropped,
        produced_bits=produced,
        delivered_bits=delivered,
        final_occupancy_bits=occupancy,
        initial_occupancy_bits=parameters.initial_occupancy_bits.astype(float).copy(),
        peak_occupancy_bits=peak,
        slots_full=full,
        slot_seconds=float(slot_seconds),
        slot_count=int(stepped),
    )
