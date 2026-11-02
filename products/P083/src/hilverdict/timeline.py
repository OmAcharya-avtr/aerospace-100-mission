"""The run log, the miss model, and the deadline accounting.

What a run log is, exactly
--------------------------
A :class:`RunLog` carries two things that are measured in different places and
must not be confused:

- a **miss-free reference trajectory**: the state and command sequence the
  declared scenario produces with no timing fault at all. On a rig this is the
  software-in-the-loop twin of the same scenario, or the same scenario re-run
  offline; in this repository it is produced by
  :func:`hilverdict.resimulate.simulate` with an empty miss set.
- a **measured execution-time trace**: how long each controller iteration
  actually took on the device, together with the host and device timestamps
  that let the trace's own clock be checked.

The verdict is then a counterfactual question asked once per recorded overrun:
what would THIS overrun have done to that trajectory? That is the only
formulation in which "with and without the miss" is well defined per miss,
because with every miss already present in the trajectory there is no
single-miss counterfactual to compare against. The cost is that the analysis
is **one miss at a time**: interactions between two misses close together are
not covered by the per-miss answer, and are covered instead by the aggregate
superposition check of :func:`hilverdict.certificate.certify_run` and by the
all-misses re-simulation in ``validation/validate_resimulation.py``. That
limitation is real and is stated in the README rather than buried here.

The miss model, declared once
-----------------------------
An iteration whose measured execution time ``d_k`` exceeds the declared
deadline ``D`` has **missed**, with overrun ``tau_k = d_k - D``. The
consequence declared here is the hold-last-value dropout of the
networked-control literature (Hespanha, Naghshtabrizi & Xu 2007, "A survey of
recent results in networked control systems", *Proceedings of the IEEE* 95(1),
138-162, section III; Schenato, Sinopoli, Franceschetti, Poolla & Sastry 2007,
"Foundations of control and estimation over lossy networks", *Proceedings of
the IEEE* 95(1), 163-187): the actuator keeps the previous command for
``tau_k`` seconds past the sample instant, and from then on the freshest
command that exists is applied. With ``p = floor(tau_k / T)`` whole periods
inside the overrun, iterations ``k .. k+p-1`` compute commands that are
superseded, and iteration ``k+p``'s command is the one that lands.

**The model does not cascade.** A late iteration does not push the next one
late. :func:`overrun_counts` computes the cascade accounting as well, because
the two definitions disagree by a large factor on a loaded run and a reader
quoting an overrun count should know which one it is, but the certificate and
the re-simulation both use the non-cascading model and a run whose cascade
count greatly exceeds its direct count is outside what the verdict covers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .plant import LoopModel

__all__ = [
    "MissEvent",
    "OverrunCounts",
    "RunLog",
    "miss_events",
    "overrun_counts",
    "utilisation",
]


@dataclass(frozen=True)
class RunLog:
    """A miss-free reference trajectory plus a measured execution-time trace.

    Attributes
    ----------
    model
        The declared loop: plant, controller, sampling law, metric.
    command_rad
        Commanded incidence per iteration, rad, shape ``(steps,)``.
    durations_s
        Measured iteration execution time on the device, s, shape ``(steps,)``.
    states
        Reference state at each sample instant, rad and rad/s, shape
        ``(steps + 1, 2)``. Miss-free by construction.
    computed_rad
        Reference command of each iteration, rad, shape ``(steps,)``.
    saturated
        Whether the reference command hit the actuator limit, shape
        ``(steps,)``. A reference run that saturates is refused by the
        certificate: the declared linearisation has no saturation in it.
    prior_command_rad
        The command in force immediately before iteration 0, rad. Declared,
        because a miss on iteration 0 holds it and nothing in the log records
        it. Defaults to ``computed_rad[0]``, which is the rig already sitting
        in trim.
    host_time_s, device_time_s
        Paired timestamps at each sample instant, s, shape ``(steps + 1,)``.
        The device series carries whatever drift and resets the run had.
    label
        Declared label, carried into every report.
    """

    model: LoopModel
    command_rad: np.ndarray
    durations_s: np.ndarray
    states: np.ndarray
    computed_rad: np.ndarray
    saturated: np.ndarray
    host_time_s: np.ndarray
    device_time_s: np.ndarray
    prior_command_rad: float | None = None
    label: str = "run"

    def __post_init__(self) -> None:
        steps = int(np.asarray(self.durations_s).shape[0])
        if steps < 2:
            raise ValueError(f"a run log needs at least 2 iterations, got {steps}")
        for name in ("command_rad", "durations_s", "computed_rad", "saturated"):
            array = np.asarray(getattr(self, name))
            if array.shape != (steps,):
                raise ValueError(
                    f"{name} must have shape ({steps},), got {array.shape}"
                )
        if np.asarray(self.states).shape != (steps + 1, 2):
            raise ValueError(
                f"states must have shape ({steps + 1}, 2), got "
                f"{np.asarray(self.states).shape}"
            )
        for name in ("host_time_s", "device_time_s"):
            array = np.asarray(getattr(self, name))
            if array.shape != (steps + 1,):
                raise ValueError(
                    f"{name} must have shape ({steps + 1},), got {array.shape}"
                )
        if np.any(np.asarray(self.durations_s) < 0.0):
            raise ValueError("durations_s contains a negative execution time")
        if not np.all(np.isfinite(np.asarray(self.states))):
            raise ValueError("states contains a non-finite value")

    @property
    def steps(self) -> int:
        """Number of controller iterations in the run."""
        return int(np.asarray(self.durations_s).shape[0])

    @property
    def duration_s(self) -> float:
        """Nominal wall-clock length of the run, s."""
        return self.steps * self.model.sampling.period_s

    @property
    def prior_command(self) -> float:
        """The declared command in force before iteration 0, rad."""
        if self.prior_command_rad is None:
            return float(np.asarray(self.computed_rad, dtype=float)[0])
        return float(self.prior_command_rad)

    @property
    def missed(self) -> np.ndarray:
        """Boolean mask of iterations that exceeded the declared deadline."""
        return np.asarray(self.durations_s) > self.model.sampling.deadline_s

    @property
    def saturated_steps(self) -> int:
        """Number of reference iterations whose command hit the actuator limit."""
        return int(np.count_nonzero(np.asarray(self.saturated)))

    @property
    def peak_alpha_rad(self) -> float:
        """Largest absolute reference incidence, rad, checked against validity."""
        return float(np.max(np.abs(np.asarray(self.states)[:, 0])))


@dataclass(frozen=True)
class MissEvent:
    """One deadline miss, as the certificate and the re-simulation both see it.

    Attributes
    ----------
    index
        Iteration index ``k`` that overran.
    overrun_s
        ``tau = d_k - D``, s. Strictly positive by construction.
    hold_periods
        ``p = floor(tau / T)``: whole periods inside the overrun.
    remainder_s
        ``tau - p T``, s, in ``[0, T)``.
    period_s
        The sampling period the two above were split by, s. Carried so an
        event can be read without its loop model.
    stale_command_rad
        The command held through the overrun, rad.
    input_step_rad
        ``max_j |u_stale - u_reference[k+j]|`` over ``j = 0..p``, rad. The
        input-error magnitude the analytic bound is a function of.
    state_at_miss
        Reference state at the sample instant the overrun began, rad and rad/s.
    steps_remaining
        Sample instants left after the overrun ends: the horizon the divergence
        is maximised over.
    """

    index: int
    overrun_s: float
    hold_periods: int
    remainder_s: float
    period_s: float
    stale_command_rad: float
    input_step_rad: float
    state_at_miss: np.ndarray
    steps_remaining: int

    @property
    def overrun_periods(self) -> float:
        """Overrun in units of the sampling period; dimensionless."""
        return float(self.overrun_s / self.period_s)


def miss_events(log: RunLog) -> tuple[MissEvent, ...]:
    """Extract every deadline miss from ``log``, in iteration order.

    An overrun that would run past the end of the log is still returned with
    ``steps_remaining`` clipped at zero, so that both the certificate and the
    re-simulation report it over an empty horizon rather than dropping it.
    """
    period = log.model.sampling.period_s
    deadline = log.model.sampling.deadline_s
    durations = np.asarray(log.durations_s, dtype=float)
    computed = np.asarray(log.computed_rad, dtype=float)
    states = np.asarray(log.states, dtype=float)
    prior = log.prior_command
    events: list[MissEvent] = []
    for index in range(log.steps):
        overrun = float(durations[index] - deadline)
        if overrun <= 0.0:
            continue
        hold_periods = int(np.floor(overrun / period + 1e-12))
        remainder = max(float(overrun - hold_periods * period), 0.0)
        stale = prior if index == 0 else float(computed[index - 1])
        window = [
            float(computed[min(index + offset, log.steps - 1)])
            for offset in range(hold_periods + 1)
        ]
        step = max(abs(stale - value) for value in window)
        remaining = max(log.steps - (index + hold_periods + 1), 0)
        events.append(
            MissEvent(
                index=index,
                overrun_s=overrun,
                hold_periods=hold_periods,
                remainder_s=remainder,
                period_s=period,
                stale_command_rad=stale,
                input_step_rad=float(step),
                state_at_miss=states[index].copy(),
                steps_remaining=int(remaining),
            )
        )
    return tuple(events)


@dataclass(frozen=True)
class OverrunCounts:
    """Deadline overruns under both accounting definitions.

    ``direct`` counts ``d_k > D``. ``cascade`` propagates the finish time,
    ``c_k = max(k T, c_{k-1}) + d_k``, and counts ``c_k > k T + D``. The two
    answer different questions: the first asks whether an iteration took too
    long, the second whether it finished too late given everything before it.
    """

    direct: int
    cascade: int
    steps: int

    @property
    def direct_fraction(self) -> float:
        """Fraction of iterations that overran under the direct definition."""
        return self.direct / self.steps if self.steps else 0.0

    @property
    def cascade_fraction(self) -> float:
        """Fraction of iterations late under the cascade definition."""
        return self.cascade / self.steps if self.steps else 0.0

    @property
    def ratio(self) -> float:
        """``cascade / direct``, or ``inf`` when there is no direct overrun."""
        return self.cascade / self.direct if self.direct else float("inf")


def overrun_counts(log: RunLog) -> OverrunCounts:
    """Count overruns under both the direct and the cascade definition."""
    period = log.model.sampling.period_s
    deadline = log.model.sampling.deadline_s
    durations = np.asarray(log.durations_s, dtype=float)
    direct = int(np.count_nonzero(durations > deadline))
    finish = 0.0
    cascade = 0
    for index, duration in enumerate(durations):
        release = index * period
        finish = max(release, finish) + float(duration)
        if finish > release + deadline:
            cascade += 1
    return OverrunCounts(direct=direct, cascade=cascade, steps=log.steps)


def utilisation(log: RunLog) -> float:
    """Mean execution time divided by the period; dimensionless.

    At or above 1 the loop cannot keep up on average and no non-cascading
    analysis of it means anything.
    """
    return float(
        np.mean(np.asarray(log.durations_s, dtype=float)) / log.model.sampling.period_s
    )
