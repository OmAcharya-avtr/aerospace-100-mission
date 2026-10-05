"""Frame synchroniser: acquisition, check, lock, flywheel, loss of sync.

The machine
-----------
A frame synchroniser does not trust a single marker hit. The classical
arrangement, described in CCSDS 130.1-G (TM Synchronization and Channel
Coding -- Summary of Concept and Rationale) and implemented in every ground
frame sync, has four states and two counters:

* ``SEARCH``   -- slide the correlator over every bit position. A marker hit
  anywhere, at search tolerance ``search_tolerance``, moves to CHECK and
  fixes a candidate frame phase.
* ``CHECK``    -- look only at the predicted next marker position, at
  tolerance ``lock_tolerance``. ``check_required`` consecutive hits promote
  to LOCK. One miss drops straight back to SEARCH, discarding the phase.
* ``LOCK``     -- deliver frames. A miss at the predicted position moves to
  FLYWHEEL without discarding the phase.
* ``FLYWHEEL`` -- keep predicting frame boundaries and keep delivering
  frames, on the assumption that the marker was corrupted rather than the
  phase lost. A hit returns to LOCK and clears the miss counter.
  ``flywheel_max`` consecutive misses declare loss of sync and return to
  SEARCH.

Design intent of the two tolerances: SEARCH is the state that is exposed to
the whole bit stream and therefore to false sync, so it runs at the
tightest tolerance; CHECK, LOCK and FLYWHEEL examine one position per frame
and can afford a looser tolerance to ride out channel bit errors.

What is asserted and what is not
--------------------------------
The four-state structure and the two counters are the standard
arrangement. The specific threshold *values* are configuration, not
standard values: CCSDS does not mandate them, and this package's defaults
(``check_required=1``, ``flywheel_max=4``) are this package's defaults and
nothing more. Any claim that a particular ground station uses particular
thresholds would be unverified, so none is made.

The machine is deterministic in its observation sequence, which is what
makes the hand trace in ``tests/test_sync.py`` and
``validation/validate_state_machine.py`` an exact reproduction test rather
than a statistical one.

Slip
----
A *slip* is a frame delivered at the wrong bit phase: the synchroniser still
reports lock, but its predicted boundaries are offset from the true ones, so
every delivered frame is a cut-and-shift of two real frames. It arises when
bits are inserted or deleted upstream of the synchroniser. ``analyse_slip``
drives the machine across a known insertion or deletion and reports how
many frames are delivered at the wrong phase before the flywheel gives up
and SEARCH re-acquires -- the re-acquisition latency, in frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from .asm import ASM_32_BITS, detect_asm, hamming_distances

__all__ = [
    "SyncState",
    "SyncConfig",
    "SyncEvent",
    "FrameSynchroniser",
    "analyse_slip",
    "SlipReport",
    "acquisition_offsets",
]


class SyncState(StrEnum):
    """States of the frame synchroniser."""

    SEARCH = "SEARCH"
    CHECK = "CHECK"
    LOCK = "LOCK"
    FLYWHEEL = "FLYWHEEL"


@dataclass(frozen=True)
class SyncConfig:
    """Synchroniser thresholds.

    Attributes
    ----------
    search_tolerance : int
        Hamming-distance tolerance, in bits, used while sliding in SEARCH.
    lock_tolerance : int
        Tolerance, in bits, at the predicted marker position in CHECK, LOCK
        and FLYWHEEL.
    check_required : int
        Consecutive hits required in CHECK before declaring LOCK, >= 1.
    flywheel_max : int
        Consecutive misses in FLYWHEEL before declaring loss of sync, >= 1.
    """

    search_tolerance: int = 0
    lock_tolerance: int = 3
    check_required: int = 1
    flywheel_max: int = 4

    def __post_init__(self) -> None:
        if self.search_tolerance < 0 or self.lock_tolerance < 0:
            raise ValueError("tolerances must be non-negative bit counts")
        if self.search_tolerance > self.lock_tolerance:
            raise ValueError(
                "search_tolerance must not exceed lock_tolerance: SEARCH is the state "
                "exposed to false sync and must be the tighter test"
            )
        if self.check_required < 1:
            raise ValueError(f"check_required must be >= 1, got {self.check_required}")
        if self.flywheel_max < 1:
            raise ValueError(f"flywheel_max must be >= 1, got {self.flywheel_max}")


@dataclass(frozen=True)
class SyncEvent:
    """One state transition. ``index`` counts observations, from 0."""

    index: int
    observation: str
    state_before: SyncState
    state_after: SyncState
    miss_count: int
    check_count: int


class FrameSynchroniser:
    """Deterministic acquisition / flywheel / loss state machine.

    The machine is driven observation by observation. Each observation is
    either ``"hit"`` (marker present within the applicable tolerance) or
    ``"miss"``. In SEARCH an observation is one *sliding* decision; in the
    other three states it is one *predicted frame boundary*.

    Examples
    --------
    >>> sm = FrameSynchroniser(SyncConfig(check_required=1, flywheel_max=2))
    >>> [sm.step(o).state_after.value for o in ("hit", "hit", "miss", "miss")]
    ['CHECK', 'LOCK', 'FLYWHEEL', 'SEARCH']
    """

    def __init__(self, config: SyncConfig | None = None) -> None:
        self.config = config or SyncConfig()
        self.state = SyncState.SEARCH
        self.miss_count = 0
        self.check_count = 0
        self.history: list[SyncEvent] = []

    def reset(self) -> None:
        """Return to SEARCH and clear both counters and the history."""
        self.state = SyncState.SEARCH
        self.miss_count = 0
        self.check_count = 0
        self.history = []

    def step(self, observation: str) -> SyncEvent:
        """Advance one observation. ``observation`` is ``"hit"`` or ``"miss"``."""
        if observation not in ("hit", "miss"):
            raise ValueError(f"observation must be 'hit' or 'miss', got {observation!r}")
        before = self.state
        hit = observation == "hit"
        cfg = self.config

        if self.state is SyncState.SEARCH:
            if hit:
                self.check_count = 1
                self.miss_count = 0
                self.state = SyncState.LOCK if cfg.check_required <= 1 else SyncState.CHECK
        elif self.state is SyncState.CHECK:
            if hit:
                self.check_count += 1
                if self.check_count >= cfg.check_required:
                    self.state = SyncState.LOCK
            else:
                self.check_count = 0
                self.state = SyncState.SEARCH
        elif self.state is SyncState.LOCK:
            if hit:
                self.miss_count = 0
            else:
                self.miss_count = 1
                self.state = SyncState.FLYWHEEL
                if cfg.flywheel_max <= 1:
                    self.miss_count = 0
                    self.check_count = 0
                    self.state = SyncState.SEARCH
        else:  # FLYWHEEL
            if hit:
                self.miss_count = 0
                self.state = SyncState.LOCK
            else:
                self.miss_count += 1
                if self.miss_count >= cfg.flywheel_max:
                    self.miss_count = 0
                    self.check_count = 0
                    self.state = SyncState.SEARCH

        event = SyncEvent(
            index=len(self.history),
            observation=observation,
            state_before=before,
            state_after=self.state,
            miss_count=self.miss_count,
            check_count=self.check_count,
        )
        self.history.append(event)
        return event

    def run(self, observations: list[str] | tuple[str, ...]) -> list[SyncEvent]:
        """Advance over a sequence of observations and return every event."""
        return [self.step(o) for o in observations]

    @property
    def delivering(self) -> bool:
        """True in LOCK and FLYWHEEL, where frames are passed downstream."""
        return self.state in (SyncState.LOCK, SyncState.FLYWHEEL)

    def states(self) -> list[SyncState]:
        """State after each observation so far, in order."""
        return [e.state_after for e in self.history]


@dataclass
class SlipReport:
    """Outcome of :func:`analyse_slip`."""

    slip_bits: int
    frame_bits: int
    frames_before_slip: int
    frames_delivered_at_wrong_phase: int
    reacquisition_frames: int | None
    final_state: SyncState
    observations: list[str] = field(default_factory=list)
    states: list[str] = field(default_factory=list)


def analyse_slip(
    *,
    frame_bits: int,
    n_frames: int,
    slip_at_frame: int,
    slip_bits: int,
    config: SyncConfig | None = None,
    pattern: np.ndarray | None = None,
    payload_seed: int = 20261005,
) -> SlipReport:
    """Drive the synchroniser across a known bit insertion or deletion.

    A stream of ``n_frames`` marker-prefixed frames is built with random
    payloads, then ``slip_bits`` bits are inserted (positive) or deleted
    (negative) immediately before frame ``slip_at_frame``. The synchroniser
    is driven at the *original* frame phase, so after the slip the predicted
    marker positions no longer land on markers. The report says how many
    frames are delivered at the wrong phase and how many frames pass before
    SEARCH re-acquires.

    Parameters
    ----------
    frame_bits : int
        Bits per frame, excluding the marker, positive.
    n_frames : int
        Number of frames in the stream, >= 3.
    slip_at_frame : int
        Index of the frame before which the slip is applied, 1 <= it < n_frames.
    slip_bits : int
        Bits inserted (> 0) or deleted (< 0). Non-zero.
    config : SyncConfig, optional
        Thresholds. Defaults to :class:`SyncConfig`.
    pattern : ndarray, optional
        Marker bits, defaults to the 32-bit CCSDS marker.
    payload_seed : int
        Seed for the random payload bits, for reproducibility.
    """
    cfg = config or SyncConfig()
    pat = ASM_32_BITS if pattern is None else np.asarray(pattern, dtype=np.uint8).ravel()
    if frame_bits <= 0:
        raise ValueError(f"frame_bits must be positive, got {frame_bits}")
    if n_frames < 3:
        raise ValueError(f"n_frames must be at least 3, got {n_frames}")
    if not 1 <= slip_at_frame < n_frames:
        raise ValueError(f"slip_at_frame must lie in [1, {n_frames - 1}], got {slip_at_frame}")
    if slip_bits == 0:
        raise ValueError("slip_bits must be non-zero; a zero slip is not a slip")

    rng = np.random.default_rng(payload_seed)
    period = pat.size + frame_bits
    blocks = [np.concatenate([pat, rng.integers(0, 2, frame_bits, dtype=np.uint8)])
              for _ in range(n_frames)]
    head = np.concatenate(blocks[:slip_at_frame])
    tail = np.concatenate(blocks[slip_at_frame:])
    if slip_bits > 0:
        stream = np.concatenate([head, rng.integers(0, 2, slip_bits, dtype=np.uint8), tail])
    else:
        stream = np.concatenate([head, tail[-slip_bits:]])

    dist = hamming_distances(stream, pat)
    sm = FrameSynchroniser(cfg)
    wrong_phase = 0
    frames_before = 0
    reacq: int | None = None

    # Predicted positions keep the pre-slip phase: position k*period. For
    # k < slip_at_frame those land on real markers; from k = slip_at_frame on
    # they are off by slip_bits, so any frame delivered there is at the wrong
    # phase. The loop runs two periods past the last frame so that a flywheel
    # that has not yet expired is still observed.
    for k in range(n_frames + 2):
        pos = k * period
        if pos >= dist.size:
            break
        tol = cfg.search_tolerance if sm.state is SyncState.SEARCH else cfg.lock_tolerance
        obs = "hit" if dist[pos] <= tol else "miss"
        was_delivering = sm.delivering
        sm.step(obs)
        if was_delivering or sm.delivering:
            if k < slip_at_frame:
                frames_before += 1
            else:
                wrong_phase += 1
        if k >= slip_at_frame and reacq is None and sm.state is SyncState.SEARCH:
            reacq = k - slip_at_frame + 1

    return SlipReport(
        slip_bits=int(slip_bits),
        frame_bits=int(frame_bits),
        frames_before_slip=int(frames_before),
        frames_delivered_at_wrong_phase=int(wrong_phase),
        reacquisition_frames=reacq,
        final_state=sm.state,
        observations=[e.observation for e in sm.history],
        states=[e.state_after.value for e in sm.history],
    )


def acquisition_offsets(
    stream: np.ndarray, frame_bits: int, tolerance: int = 0, pattern: np.ndarray | None = None
) -> np.ndarray:
    """Marker offsets in a stream, for building observation sequences.

    Thin wrapper over :func:`framesync.asm.detect_asm` that also checks the
    declared offsets are spaced by the frame period, which is the test a
    synchroniser applies before trusting a candidate phase.
    """
    pat = ASM_32_BITS if pattern is None else np.asarray(pattern, dtype=np.uint8).ravel()
    offsets = detect_asm(stream, tolerance, pat)
    period = pat.size + int(frame_bits)
    if offsets.size >= 2:
        spacing = np.diff(offsets)
        offsets = np.concatenate([offsets[:1], offsets[1:][spacing == period]])
    return offsets
