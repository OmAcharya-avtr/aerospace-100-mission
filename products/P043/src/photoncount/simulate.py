"""Event-level simulator: the ground truth this package is validated against.

Everything in :mod:`photoncount.deadtime` and :mod:`photoncount.afterpulse` is
a rate equation. A rate equation for dead time assumes afterpulsing is absent,
and a rate equation for afterpulsing assumes dead time is absent. Neither
assumption holds in a real SPAD, and there is no closed form for the two
together, so this module does what a lab would do: it generates the arrival
process event by event and gates it.

**Algorithm.** Primary detections arrive as a Poisson process of rate ``n`` over
a window ``T``. Events are processed in time order against a single state
variable ``blocked_until``:

* an event at ``t >= blocked_until`` is **registered**; ``blocked_until``
  becomes ``t + tau``; with probability ``p`` an afterpulse is scheduled at
  ``t + Exp(t_ap)``;
* an event at ``t < blocked_until`` is **lost**; for the paralyzable model
  ``blocked_until`` becomes ``t + tau`` anyway (the hallmark of an extending
  dead time), for the non-paralyzable model it is left alone.

Two things fall out of this that no rate equation gives:

1. An afterpulse landing inside the dead period is suppressed, so the effective
   afterpulse probability is below the nominal one --- equation (A3) of
   :mod:`photoncount.afterpulse` quantifies the pure case, and this simulator
   shows the composition with dead time, which (A3) does not cover.
2. Afterpulses *extend* a paralyzable detector's dead period, so afterpulsing
   makes paralyzable saturation worse rather than partly compensating for it.
   The two effects do not cancel and do not commute.

With ``p = 0`` the simulator must reproduce (D1) and (D3) exactly in
expectation, and with ``tau = 0`` it must reproduce (A2). Both are checked in
``validation/validate_simulator.py``; they are the only way to know the event
loop is right.

Determinism: every function takes a ``numpy.random.Generator`` and does nothing
else random. Same generator state, same output, on any machine.

Units: rates counts/s, times s, probabilities dimensionless.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np

from .deadtime import DeadTimeModel

__all__ = ["DetectorSpec", "RunResult", "simulate_run", "simulate_windows"]


@dataclass(frozen=True)
class DetectorSpec:
    """A photon-counting detector's non-idealities.

    Attributes
    ----------
    dead_time_s:
        ``tau >= 0``, s. Zero means an ideal counter.
    model:
        ``"paralyzable"`` or ``"nonparalyzable"``.
    afterpulse_probability:
        ``p`` in ``[0, 1)``, the probability that a registered count spawns an
        afterpulse (-).
    afterpulse_mean_delay_s:
        ``t_ap > 0``, mean of the exponential release delay, s.
    cascading:
        If True (default) an afterpulse that registers can itself spawn an
        afterpulse, matching (A2). If False, only primary detections spawn
        afterpulses, matching (A1).
    """

    dead_time_s: float = 0.0
    model: DeadTimeModel = "nonparalyzable"
    afterpulse_probability: float = 0.0
    afterpulse_mean_delay_s: float = 1e-7
    cascading: bool = True

    def __post_init__(self) -> None:
        if not np.isfinite(self.dead_time_s) or self.dead_time_s < 0.0:
            raise ValueError(f"dead_time_s must be finite and >= 0, got {self.dead_time_s!r}")
        if self.model not in ("paralyzable", "nonparalyzable"):
            raise ValueError(
                f"model must be 'paralyzable' or 'nonparalyzable', got {self.model!r}"
            )
        p = self.afterpulse_probability
        if not np.isfinite(p) or not 0.0 <= p < 1.0:
            raise ValueError(f"afterpulse_probability must be in [0, 1), got {p!r}")
        if not np.isfinite(self.afterpulse_mean_delay_s) or self.afterpulse_mean_delay_s <= 0.0:
            raise ValueError(
                f"afterpulse_mean_delay_s must be finite and > 0, "
                f"got {self.afterpulse_mean_delay_s!r}"
            )


@dataclass
class RunResult:
    """Outcome of one simulated acquisition window.

    Attributes
    ----------
    incident_rate_hz:
        The true rate the window was generated at, counts/s.
    duration_s:
        Window length, s.
    registered:
        Total counts the detector reported (primaries plus afterpulses).
    registered_primary:
        Of those, the ones caused by a real detection event.
    registered_afterpulse:
        Of those, the ones caused by an afterpulse.
    lost_primary:
        Primary events that fell in a dead period.
    lost_afterpulse:
        Afterpulses that fell in a dead period.
    primary_events:
        Primary events generated, equal to ``registered_primary +
        lost_primary``.
    timestamps:
        Registered event times, s, ascending. Empty array when
        ``keep_timestamps=False``.
    """

    incident_rate_hz: float
    duration_s: float
    registered: int = 0
    registered_primary: int = 0
    registered_afterpulse: int = 0
    lost_primary: int = 0
    lost_afterpulse: int = 0
    primary_events: int = 0
    timestamps: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))

    @property
    def observed_rate_hz(self) -> float:
        """Registered counts divided by wall-clock duration, counts/s."""
        return self.registered / self.duration_s

    @property
    def afterpulse_fraction(self) -> float:
        """Share of registered counts that were afterpulses (-), 0 if no counts."""
        return self.registered_afterpulse / self.registered if self.registered else 0.0


def simulate_run(
    incident_rate_hz: float,
    duration_s: float,
    spec: DetectorSpec,
    rng: np.random.Generator,
    keep_timestamps: bool = False,
) -> RunResult:
    """Simulate one acquisition window event by event.

    ``incident_rate_hz`` is the **true** detected-event rate before any
    non-ideality: the quantity the learned correction in
    :mod:`photoncount.correction` tries to recover. Cost is O(N log N) in the
    number of events, N ~ Poisson(incident_rate_hz * duration_s); keep the
    product below a few hundred thousand.
    """
    n = float(incident_rate_hz)
    t_end = float(duration_s)
    if not np.isfinite(n) or n < 0.0:
        raise ValueError(f"incident_rate_hz must be finite and >= 0, got {n!r}")
    if not np.isfinite(t_end) or t_end <= 0.0:
        raise ValueError(f"duration_s must be finite and > 0, got {t_end!r}")

    expected = n * t_end
    if expected > 5e6:
        raise ValueError(
            f"incident_rate_hz * duration_s = {expected:.3g} events is beyond the "
            "intended scale of this simulator (5e6); shorten the window"
        )
    n_primary = int(rng.poisson(expected))
    primaries = np.sort(rng.random(n_primary) * t_end)

    tau = spec.dead_time_s
    extending = spec.model == "paralyzable"
    p_ap = spec.afterpulse_probability
    t_ap = spec.afterpulse_mean_delay_s

    result = RunResult(
        incident_rate_hz=n, duration_s=t_end, primary_events=n_primary
    )
    stamps: list[float] = []
    pending: list[float] = []
    blocked_until = -np.inf
    idx = 0
    n_prim = primaries.size
    # Pre-draw the randomness the loop needs: one uniform per potential
    # registration decides whether an afterpulse is spawned, one exponential
    # gives its delay. Over-provisioned and refilled if exhausted.
    budget = max(64, int(n_prim * (1.0 / max(1.0 - p_ap, 1e-3)) + 64))
    spawn_u = rng.random(budget)
    delays = rng.exponential(t_ap, budget)
    draw = 0

    while idx < n_prim or pending:
        if pending and (idx >= n_prim or pending[0] <= primaries[idx]):
            t = heapq.heappop(pending)
            is_primary = False
        else:
            t = float(primaries[idx])
            idx += 1
            is_primary = True
        if t > t_end:
            continue
        if t < blocked_until:
            if is_primary:
                result.lost_primary += 1
            else:
                result.lost_afterpulse += 1
            if extending:
                blocked_until = t + tau
            continue
        result.registered += 1
        if is_primary:
            result.registered_primary += 1
        else:
            result.registered_afterpulse += 1
        if keep_timestamps:
            stamps.append(t)
        blocked_until = t + tau
        if p_ap > 0.0 and (is_primary or spec.cascading):
            if draw >= spawn_u.size:
                spawn_u = rng.random(budget)
                delays = rng.exponential(t_ap, budget)
                draw = 0
            if spawn_u[draw] < p_ap:
                nxt = t + float(delays[draw])
                if nxt <= t_end:
                    heapq.heappush(pending, nxt)
            draw += 1

    if keep_timestamps:
        result.timestamps = np.asarray(stamps, dtype=float)
    return result


def simulate_windows(
    incident_rate_hz: float,
    window_s: float,
    n_windows: int,
    spec: DetectorSpec,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Repeat :func:`simulate_run` over ``n_windows`` windows and summarise.

    Returns ``observed_rate_hz`` (pooled), ``count_mean``, ``count_variance``
    (sample variance over windows, ddof=1), ``fano_factor``,
    ``afterpulse_fraction``, ``n_windows``, ``window_s``. The Fano factor is the
    observable that distinguishes the two non-idealities: dead time pushes it
    below 1, afterpulsing above.

    Windows are simulated independently, so a cluster straddling a window edge
    is truncated rather than carried over. That biases the Fano factor slightly
    low for ``window_s`` comparable with ``t_ap``; keep ``window_s`` at least a
    few hundred times ``t_ap``.
    """
    w = int(n_windows)
    if w < 2:
        raise ValueError(f"n_windows must be >= 2, got {w!r}")
    counts = np.empty(w, dtype=float)
    ap = 0
    total = 0
    for i in range(w):
        run = simulate_run(incident_rate_hz, window_s, spec, rng)
        counts[i] = run.registered
        ap += run.registered_afterpulse
        total += run.registered
    mean = float(counts.mean())
    var = float(counts.var(ddof=1))
    return {
        "observed_rate_hz": mean / float(window_s),
        "count_mean": mean,
        "count_variance": var,
        "fano_factor": var / mean if mean > 0.0 else float("nan"),
        "afterpulse_fraction": ap / total if total else 0.0,
        "n_windows": float(w),
        "window_s": float(window_s),
    }
