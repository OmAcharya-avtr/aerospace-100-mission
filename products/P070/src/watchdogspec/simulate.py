"""Seeded discrete-event simulation of heartbeat arrivals and the supervisor.

This module exists to disagree with :mod:`watchdogspec.latency` if the closed
form is wrong. It does not reuse it. It generates heartbeat arrival times,
injects a fault at a continuous instant, runs the supervisor's deadline chain
and poll loop event by event, and reports the detection latency it actually
observed.

Two independent things are cross-checked:

1. **Worst-case detection latency.** Equation (3) of
   :mod:`watchdogspec.latency` must bound every simulated latency, and must be
   attained, to within floating-point rounding, by the adversarial placement
   of the fault instant and the poll phase.
2. **False-trip rate.** Equation (7) of :mod:`watchdogspec.falsetrip` must
   agree with the fraction of healthy heartbeat gaps on which the event loop
   declares a fault, to within a binomial confidence interval.

Event model
-----------
Heartbeats are receipts at the supervisor, so the generated deviation is the
deviation of the receipt-to-receipt gap from the nominal period, consistent
with the definition of ``J`` in :class:`watchdogspec.model.Task`.

* ``bounded`` jitter: gap ``= P + U(-J, +J)``. This is the model the worst-case
  arithmetic assumes; the uniform distribution inside the bound is a choice
  this simulation makes and the worst case does not depend on it.
* ``gaussian`` jitter: emission deviations ``d_n ~ N(0, sigma^2)`` and arrival
  ``n P + d_n``, which is the model equation (5) describes. Unbounded, so it
  violates ``J`` at the rate equation (7) gives.

The supervisor arms a deadline ``W`` after each receipt, re-arms ``W`` later on
each expiry, and declares on the ``k + 1``-th. It only *acts* at poll instants:
poll ``i`` is released at ``phase + i Q`` and executed at
``phase + i Q + u_i`` with ``u_i ~ U(0, Jq)``, and the declaration is issued
``O`` after the poll that observes the final expiry.

Compute budget
--------------
Everything here is a Python loop over scalar floats, by design: a vectorised
re-expression would share structure with the closed form and weaken the
cross-check. Measured on the 2-core machine this package was built on, the
whole validation set, including 40 000 detection trials and 2 000 000
simulated heartbeat gaps, runs in well under three minutes; the two
validation scripts report their own wall-clock times, which are committed in
``validation/``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .model import Supervisor, Task

#: Jitter models this module can generate.
JITTER_MODELS = ("bounded", "gaussian")


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """Summary over many trials. Times in ms."""

    trials: int
    seed: int
    max_latency_from_receipt_ms: float
    mean_latency_from_receipt_ms: float
    min_latency_from_receipt_ms: float
    max_latency_from_onset_ms: float
    bound_ms: float
    worst_overshoot_ms: float
    violations: int
    latencies_from_receipt_ms: tuple[float, ...] = ()
    """Every trial's latency, in ms, when ``collect_trials`` was set; else empty."""


#: Number of consecutive polls examined around an expiry. The poll grid is
#: anchored within one poll period of the expiry, so the observing poll is
#: always one of the first few; six covers a release jitter of several poll
#: periods, which is already outside the validity range of the model.
POLL_SCAN = 6


def _poll_observation_ms(
    expiry_ms: float,
    poll_base_ms: float,
    poll_period_ms: float,
    poll_jitters: np.ndarray,
) -> float:
    """Earliest poll execution instant at or after ``expiry_ms``, in ms.

    Poll ``n`` is released at ``poll_base + n Q`` and executed
    ``poll_jitters[n]`` later. Execution instants are not monotone in ``n``
    once the release jitter is non-zero, so every poll in the scan window is
    examined rather than the first one whose release is late enough.
    """
    best = math.inf
    for index in range(len(poll_jitters)):
        executed = poll_base_ms + index * poll_period_ms + float(poll_jitters[index])
        if executed >= expiry_ms and executed < best:
            best = executed
    return best


def simulate_detection(
    task: Task,
    supervisor: Supervisor,
    seed: int,
    trials: int = 2000,
    beats_before_fault: int = 24,
    jitter_model: str = "bounded",
    adversarial: bool = False,
    collect_trials: bool = False,
) -> DetectionResult:
    """Run ``trials`` seeded fault-detection experiments.

    Parameters
    ----------
    task, supervisor:
        Declared objects. Only their timing fields are read.
    seed:
        Passed to :func:`numpy.random.default_rng`. The result is a pure
        function of this seed and the arguments.
    trials:
        Number of independent experiments.
    beats_before_fault:
        How many healthy heartbeats precede the fault in each experiment. The
        detection latency does not depend on this in the model -- the deadline
        chain is armed from the last receipt -- and it is varied anyway so that
        a dependence introduced by a future change shows up.
    jitter_model:
        ``"bounded"`` or ``"gaussian"``.
    collect_trials:
        When ``True``, every trial's latency from the last receipt is returned
        in ``latencies_from_receipt_ms``. Off by default so that a large run
        does not hold a large array.
    adversarial:
        When ``True``, place the fault instant exactly at the last receipt and
        the poll phase so that the final expiry falls immediately after a poll
        execution, with the following poll taking its full jitter. This is the
        configuration equation (3) describes, and the simulated latency must
        then equal the closed form to within rounding.

    Returns
    -------
    DetectionResult
        ``violations`` counts trials whose latency from the last receipt
        exceeds the closed-form bound by more than ``1e-9`` ms. A non-zero
        count is a defect in one of the two models and is reported, not hidden.

    Raises
    ------
    ValueError
        On an unknown ``jitter_model``, a non-positive ``trials`` or
        ``beats_before_fault``, or a declared quantity outside the validity
        range of :mod:`watchdogspec.latency`.
    """
    if jitter_model not in JITTER_MODELS:
        raise ValueError(f"jitter_model must be one of {JITTER_MODELS}, got {jitter_model!r}")
    if trials <= 0:
        raise ValueError(f"trials must be >= 1, got {trials}")
    if beats_before_fault <= 0:
        raise ValueError(f"beats_before_fault must be >= 1, got {beats_before_fault}")

    period = float(task.heartbeat_period_ms)
    jitter = float(task.heartbeat_jitter_ms)
    sigma = float(task.jitter_sigma_ms)
    window = period + jitter
    k = int(task.missed_beats_tolerated)
    poll_period = float(supervisor.poll_period_ms)
    poll_jitter = float(supervisor.poll_jitter_ms)
    overhead = float(supervisor.declaration_overhead_ms)
    if period <= 0.0 or poll_period <= 0.0:
        raise ValueError("heartbeat_period_ms and poll_period_ms must be strictly positive")

    # The bound this simulation is checked against is recomputed here from the
    # same declared numbers, but the simulation never consults it while running.
    bound = (k + 1) * window + poll_period + poll_jitter + overhead

    rng = np.random.default_rng(seed)
    max_lat = -math.inf
    min_lat = math.inf
    max_onset = -math.inf
    total = 0.0
    violations = 0
    worst_overshoot = -math.inf
    collected: list[float] = []

    for _ in range(trials):
        beats = beats_before_fault
        if jitter_model == "bounded":
            gaps = period + rng.uniform(-jitter, jitter, size=beats)
            arrivals = np.cumsum(gaps)
        else:
            deviations = rng.normal(0.0, sigma, size=beats)
            arrivals = np.arange(1, beats + 1) * period + deviations
        last_receipt = float(arrivals[-1])

        if adversarial:
            onset = last_receipt
        else:
            onset = last_receipt + float(rng.uniform(0.0, period))

        # The deadline chain, stepped one expiry at a time. No beat arrives
        # after the fault onset, so every armed deadline expires.
        expiry = last_receipt
        expiries = 0
        while expiries <= k:
            expiry = expiry + window
            expiries += 1

        # The poll grid is anchored within one poll period of the expiry.
        # Anchoring relative to the expiry rather than to zero is equivalent
        # to a uniformly random absolute phase and keeps the scan bounded.
        if adversarial:
            # Poll 0 executes one ulp BEFORE the expiry with zero jitter, so it
            # does not see it; poll 1 takes its full release jitter. That is
            # the configuration equation (3) assumes, and the observation delay
            # is then Q + Jq less one ulp.
            poll_base = math.nextafter(expiry, -math.inf)
            poll_jitters = np.zeros(POLL_SCAN)
            poll_jitters[1] = poll_jitter
        else:
            poll_base = expiry - float(rng.uniform(0.0, poll_period))
            poll_jitters = rng.uniform(0.0, poll_jitter, size=POLL_SCAN)

        observed = _poll_observation_ms(expiry, poll_base, poll_period, poll_jitters)
        if not math.isfinite(observed):  # pragma: no cover - guarded by polls_needed
            raise RuntimeError("poll scan found no poll at or after the final expiry")
        declared = observed + overhead

        latency = declared - last_receipt
        onset_latency = declared - onset
        total += latency
        max_lat = max(max_lat, latency)
        min_lat = min(min_lat, latency)
        max_onset = max(max_onset, onset_latency)
        worst_overshoot = max(worst_overshoot, latency - bound)
        if collect_trials:
            collected.append(latency)
        if latency - bound > 1e-9:
            violations += 1

    return DetectionResult(
        trials=trials,
        seed=seed,
        max_latency_from_receipt_ms=max_lat,
        mean_latency_from_receipt_ms=total / trials,
        min_latency_from_receipt_ms=min_lat,
        max_latency_from_onset_ms=max_onset,
        bound_ms=bound,
        worst_overshoot_ms=worst_overshoot,
        violations=violations,
        latencies_from_receipt_ms=tuple(collected),
    )


@dataclass(frozen=True, slots=True)
class FalseTripResult:
    """Outcome of a healthy-operation run. No fault is injected."""

    gaps: int
    seed: int
    false_trips: int
    empirical_p: float
    simulated_hours: float
    empirical_per_hour: float


def simulate_false_trips(
    task: Task,
    seed: int,
    gaps: int = 200_000,
) -> FalseTripResult:
    """Count supervisor declarations over ``gaps`` healthy heartbeat gaps.

    The task never fails. Every declaration is a false trip. Jitter is the
    Gaussian model of equation (5); the deadline chain is stepped explicitly,
    one expiry at a time, so that the count is produced by the supervisor
    logic and not by re-evaluating the condition the closed form derives.

    Raises
    ------
    ValueError
        If ``gaps`` is below one, or ``jitter_sigma_ms`` is zero (there is
        nothing to sample: equation (7) is then exactly zero by declaration).
    """
    if gaps < 1:
        raise ValueError(f"gaps must be >= 1, got {gaps}")
    sigma = float(task.jitter_sigma_ms)
    if sigma <= 0.0:
        raise ValueError(
            "simulate_false_trips needs jitter_sigma_ms > 0; with sigma == 0 the "
            "declared false-trip rate is exactly zero and there is nothing to sample"
        )
    period = float(task.heartbeat_period_ms)
    window = period + float(task.heartbeat_jitter_ms)
    k = int(task.missed_beats_tolerated)

    rng = np.random.default_rng(seed)
    deviations = rng.normal(0.0, sigma, size=gaps + 1)
    arrivals = np.arange(gaps + 1, dtype=float) * period + deviations

    trips = 0
    previous = float(arrivals[0])
    for index in range(1, gaps + 1):
        current = float(arrivals[index])
        # Step the deadline chain from the previous receipt. A declaration
        # happens if k + 1 deadlines expire before the next receipt arrives.
        deadline = previous + window
        expiries = 0
        while current > deadline and expiries <= k:
            expiries += 1
            deadline = deadline + window
        if expiries > k:
            trips += 1
        previous = current

    simulated_hours = gaps * period / 3_600_000.0
    return FalseTripResult(
        gaps=gaps,
        seed=seed,
        false_trips=trips,
        empirical_p=trips / gaps,
        simulated_hours=simulated_hours,
        empirical_per_hour=trips / simulated_hours if simulated_hours > 0.0 else math.inf,
    )


def wilson_interval(
    successes: int, trials: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    ``z`` defaults to the two-sided 95 % normal quantile, 1.959963984540054.

    Wilson, E. B., "Probable inference, the law of succession, and statistical
    inference", *Journal of the American Statistical Association*, vol. 22,
    no. 158, 1927, pp. 209-212, doi:10.1080/01621459.1927.10502953. Chosen over
    the Wald interval because the proportions here are small and Wald is known
    to be badly behaved near zero.

    Raises
    ------
    ValueError
        If ``trials`` is below one or ``successes`` is outside ``0..trials``.
    """
    if trials < 1:
        raise ValueError(f"trials must be >= 1, got {trials}")
    if not 0 <= successes <= trials:
        raise ValueError(f"successes must be in 0..{trials}, got {successes}")
    phat = successes / trials
    denom = 1.0 + z * z / trials
    centre = (phat + z * z / (2.0 * trials)) / denom
    half = z * math.sqrt(phat * (1.0 - phat) / trials + z * z / (4.0 * trials * trials)) / denom
    return max(centre - half, 0.0), min(centre + half, 1.0)
