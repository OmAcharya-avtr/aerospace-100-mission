"""Synthetic hardware-in-the-loop runs: the declared trajectory and the timing trace.

Everything in this repository is measured against runs generated here. No
measurement of any real rig, flight computer, bus or test article appears
anywhere, and the generator's own equations are written out below so they can
be argued with rather than trusted.

The execution-time model
------------------------
A control iteration's execution time on a real target is a short, fairly tight
distribution with an occasional long tail: cache misses and branch variation
give the body, and bus arbitration, a DMA stall or a priority inversion give
the tail. The declared model is a lognormal body plus Bernoulli stalls:

```
d_k = T * f * exp(sigma Z_k)  +  S_k,     Z_k ~ Normal(0, 1)              (17)
S_k = B_k * E_k,   B_k ~ Bernoulli(p),  E_k ~ Exponential(mean = s T)     (18)
```

with ``f`` the declared mean execution fraction of the period, ``sigma`` the
declared log-jitter, ``p`` the stall rate and ``s`` the stall scale in periods.
A ``bursty`` profile additionally correlates ``B_k``: a stall starts a burst of
declared geometric length, which is what a bus holding a lock actually looks
like and is the case a per-iteration independence assumption gets wrong.

Equations (17)-(18) are a **declared fiction with an engineering shape**. What
they encode is the one qualitative fact that makes the product's question
interesting: overruns are mostly small and occasionally several periods long,
so a verdict procedure that must look at each one individually is dominated by
the cheap cases.

The clock model
---------------
The host and the device each timestamp every sample instant. The host series
is the reference and is exact by declaration. The device series carries a
constant relative rate error, a quantisation to its tick, additive jitter, and
zero or more resets:

```
device_k = o_j + (1 + rho)(host_k - host_{r_j}) + q(eps_k)                (19)
```

where ``rho`` is the fractional rate error (reported in parts per million),
``r_j`` is the index of the most recent reset, ``o_j`` the offset after it, and
``q`` rounds to the declared tick after adding Gaussian jitter. A reset sets
``o_j`` back to near zero, which is what a device whose counter restarts or
wraps does, and is the case that makes a measured duration meaningless rather
than merely biased.

The commanded manoeuvre
-----------------------
A doublet in commanded incidence, which is what a rig actually flies to excite
a short period: positive for the first 30 per cent of the run, negative to 60
per cent, then back to trim. The transients are where the command changes fast
and therefore where a held command is a large input error -- the structure the
whole verdict question turns on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .plant import LoopModel
from .resimulate import simulate
from .timeline import RunLog

__all__ = [
    "CAMPAIGNS",
    "CLOCK_CLEAN",
    "ClockSpec",
    "ExecutionModel",
    "TIMING_PROFILES",
    "generate_population",
    "generate_run",
]


@dataclass(frozen=True)
class ExecutionModel:
    """Declared execution-time distribution of equations (17)-(18).

    Attributes
    ----------
    mean_fraction
        ``f``: median execution time as a fraction of the period.
    log_jitter
        ``sigma``: standard deviation of the log execution time.
    stall_rate
        ``p``: per-iteration probability of starting a stall.
    stall_periods
        ``s``: mean stall length in periods.
    burst_continuation
        Probability that a stalled iteration is followed by another. Zero makes
        stalls independent; the ``bursty`` profile sets it above zero.
    name
        Declared label.
    """

    mean_fraction: float
    log_jitter: float
    stall_rate: float
    stall_periods: float
    burst_continuation: float
    name: str

    def __post_init__(self) -> None:
        if not 0.0 < self.mean_fraction < 2.0:
            raise ValueError(
                f"mean_fraction must lie in (0, 2), got {self.mean_fraction}"
            )
        if self.log_jitter < 0.0:
            raise ValueError(f"log_jitter must be non-negative, got {self.log_jitter}")
        if not 0.0 <= self.stall_rate <= 1.0:
            raise ValueError(f"stall_rate must lie in [0, 1], got {self.stall_rate}")
        if self.stall_periods < 0.0:
            raise ValueError(
                f"stall_periods must be non-negative, got {self.stall_periods}"
            )
        if not 0.0 <= self.burst_continuation < 1.0:
            raise ValueError(
                "burst_continuation must lie in [0, 1), got "
                f"{self.burst_continuation}"
            )

    def draw(
        self, steps: int, period_s: float, generator: np.random.Generator
    ) -> np.ndarray:
        """Draw ``steps`` execution times in seconds from (17)-(18)."""
        body = (
            period_s
            * self.mean_fraction
            * np.exp(self.log_jitter * generator.standard_normal(steps))
        )
        stalls = np.zeros(steps, dtype=float)
        in_burst = False
        for index in range(steps):
            if in_burst:
                starts = generator.random() < self.burst_continuation
            else:
                starts = generator.random() < self.stall_rate
            in_burst = bool(starts)
            if starts:
                stalls[index] = generator.exponential(self.stall_periods * period_s)
        return body + stalls


#: The declared timing profiles. ``nominal`` is a loop with margin,
#: ``loaded`` one running near its budget, ``bursty`` one whose stalls arrive
#: in correlated groups.
TIMING_PROFILES: dict[str, ExecutionModel] = {
    "nominal": ExecutionModel(
        mean_fraction=0.60,
        log_jitter=0.30,
        stall_rate=0.050,
        stall_periods=0.8,
        burst_continuation=0.0,
        name="nominal",
    ),
    "loaded": ExecutionModel(
        mean_fraction=0.82,
        log_jitter=0.16,
        stall_rate=0.035,
        stall_periods=1.1,
        burst_continuation=0.0,
        name="loaded",
    ),
    "bursty": ExecutionModel(
        mean_fraction=0.60,
        log_jitter=0.26,
        stall_rate=0.040,
        stall_periods=1.4,
        burst_continuation=0.50,
        name="bursty",
    ),
}


@dataclass(frozen=True)
class ClockSpec:
    """Declared device-clock behaviour of equation (19).

    Attributes
    ----------
    drift_ppm
        ``rho`` in parts per million. Positive means the device clock runs
        fast.
    jitter_s
        Standard deviation of the additive timestamp jitter, s.
    tick_s
        Device timer resolution, s. Timestamps are rounded to it.
    reset_fractions
        Fractions of the run at which the device clock resets. Empty means no
        reset.
    name
        Declared label.
    """

    drift_ppm: float
    jitter_s: float
    tick_s: float
    reset_fractions: tuple[float, ...]
    name: str

    def __post_init__(self) -> None:
        if self.jitter_s < 0.0:
            raise ValueError(f"jitter_s must be non-negative, got {self.jitter_s}")
        if self.tick_s <= 0.0:
            raise ValueError(f"tick_s must be positive, got {self.tick_s}")
        for fraction in self.reset_fractions:
            if not 0.0 < fraction < 1.0:
                raise ValueError(
                    f"reset fractions must lie in (0, 1), got {fraction}"
                )


#: A device clock with no drift worth the name and no reset.
CLOCK_CLEAN = ClockSpec(
    drift_ppm=0.0,
    jitter_s=2.0e-7,
    tick_s=1.0e-7,
    reset_fractions=(),
    name="clean",
)


def generate_clock(
    spec: ClockSpec,
    host_time_s: np.ndarray,
    generator: np.random.Generator,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Equation (19): the device timestamp series, and the reset indices used.

    Returns ``(device_time_s, reset_indices)``. The reset indices are returned
    so the recovery measurement in ``validation/validate_clock.py`` scores
    against the truth and not against its own detector.
    """
    host = np.asarray(host_time_s, dtype=float)
    samples = int(host.shape[0])
    resets = tuple(
        sorted({int(round(fraction * samples)) for fraction in spec.reset_fractions})
    )
    resets = tuple(index for index in resets if 0 < index < samples)
    rate = 1.0 + spec.drift_ppm * 1.0e-6
    device = np.empty(samples, dtype=float)
    anchor = 0
    offset = 0.0
    boundaries = (*resets, samples)
    position = 0
    for boundary in boundaries:
        indices = np.arange(position, boundary)
        device[indices] = offset + rate * (host[indices] - host[anchor])
        if boundary < samples:
            anchor = boundary
            offset = 0.0
        position = boundary
    noisy = device + spec.jitter_s * generator.standard_normal(samples)
    return np.round(noisy / spec.tick_s) * spec.tick_s, resets


#: Fractions of the run at which the commanded incidence switches. Six
#: switches rather than one, because a held command only matters when the
#: command was going to change, so a schedule with one transient would make
#: the whole study a study of one transient.
SWITCH_FRACTIONS: tuple[float, ...] = (0.10, 0.25, 0.40, 0.55, 0.70, 0.85)


def doublet_command(steps: int, amplitude_rad: float) -> np.ndarray:
    """The declared commanded-incidence schedule, rad, shape ``(steps,)``.

    A square wave through :data:`SWITCH_FRACTIONS`, alternating sign and
    returning to trim for the last segment.
    """
    fraction = np.arange(steps, dtype=float) / steps
    segment = np.searchsorted(np.asarray(SWITCH_FRACTIONS), fraction, side="right")
    levels = np.array([0.0, 1.0, -1.0, 1.0, -0.5, 0.5, 0.0]) * amplitude_rad
    return levels[segment].astype(float)


def generate_run(
    seed: int,
    profile: str = "nominal",
    amplitude_deg: float = 6.0,
    steps: int = 200,
    model: LoopModel | None = None,
    clock: ClockSpec = CLOCK_CLEAN,
    label: str | None = None,
) -> RunLog:
    """Generate one run log: a miss-free reference trajectory and a timing trace.

    Parameters
    ----------
    seed
        Declared seed. The same seed gives the same run on any machine with the
        same numpy generator, which is what makes every number here
        reproducible.
    profile
        Key into :data:`TIMING_PROFILES`.
    amplitude_deg
        Doublet amplitude in commanded incidence, deg.
    steps
        Controller iterations in the run.
    model
        The declared loop; defaults to :class:`hilverdict.plant.LoopModel`.
    clock
        Declared device-clock behaviour. Its drift scales the recorded
        durations, because they are measured on the device clock.
    label
        Report label; defaults to a string naming the seed and profile.

    Returns
    -------
    RunLog
    """
    if profile not in TIMING_PROFILES:
        raise ValueError(
            f"unknown timing profile {profile!r}; declared profiles are "
            f"{sorted(TIMING_PROFILES)}"
        )
    if steps < 2:
        raise ValueError(f"steps must be at least 2, got {steps}")
    loop = LoopModel() if model is None else model
    generator = np.random.default_rng(int(seed))
    command = doublet_command(steps, float(np.deg2rad(amplitude_deg)))
    trajectory = simulate(loop, command, None)
    durations = TIMING_PROFILES[profile].draw(
        steps, loop.sampling.period_s, generator
    )
    # The log's durations are DEVICE-MEASURED, so a device clock running fast
    # by rho reports every interval as (1 + rho) times its true length. This is
    # the coupling that makes the clock check part of the timing verdict rather
    # than a side report: the recorded durations, and therefore the set of
    # iterations that appear to have missed, depend on it.
    durations = durations * (1.0 + clock.drift_ppm * 1.0e-6)
    host = np.arange(steps + 1, dtype=float) * loop.sampling.period_s
    device, _resets = generate_clock(clock, host, generator)
    return RunLog(
        model=loop,
        command_rad=command,
        durations_s=durations,
        states=trajectory.states,
        computed_rad=trajectory.computed_rad,
        saturated=trajectory.saturated,
        host_time_s=host,
        device_time_s=device,
        label=label or f"{profile}-{amplitude_deg:g}deg-seed{seed}",
    )


#: The declared campaign shapes a population is drawn over. Three timing
#: profiles crossed with three manoeuvre amplitudes, all inside the actuator
#: limit and the declared incidence validity region.
CAMPAIGNS: tuple[tuple[str, float], ...] = (
    ("nominal", 3.0),
    ("nominal", 6.0),
    ("nominal", 8.0),
    ("loaded", 3.0),
    ("loaded", 6.0),
    ("loaded", 8.0),
    ("bursty", 3.0),
    ("bursty", 6.0),
    ("bursty", 8.0),
)


def generate_population(
    seeds: tuple[int, ...],
    steps: int = 200,
    model: LoopModel | None = None,
    clock: ClockSpec = CLOCK_CLEAN,
    campaigns: tuple[tuple[str, float], ...] = CAMPAIGNS,
) -> tuple[RunLog, ...]:
    """One run per seed, cycling through ``campaigns`` so every shape appears.

    The cycling is deliberate and matters for the learned component: a
    train/test split by seed then splits by RUN, never by miss, and misses
    within one run are not independent of each other.
    """
    if not seeds:
        raise ValueError("seeds must not be empty")
    runs = []
    for position, seed in enumerate(seeds):
        profile, amplitude = campaigns[position % len(campaigns)]
        runs.append(
            generate_run(
                seed=seed,
                profile=profile,
                amplitude_deg=amplitude,
                steps=steps,
                model=model,
                clock=clock,
            )
        )
    return tuple(runs)
