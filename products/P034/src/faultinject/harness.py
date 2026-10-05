"""Closed-loop harness: runs plant + wrapper + target and returns a trace.

The harness owns the random streams.  Two independent streams are derived from
the case seed so that changing the fault's stochastic behaviour cannot change
the plant's noise realisation:

    noise stream  = default_rng([seed, 1])   measurement and process noise
    fault stream  = default_rng([seed, 2])   handler draws

``numpy.random.default_rng`` with a PCG64 bit generator is reproducible for a
given seed, so a run is a pure function of ``(seed, injections, n_steps,
target class)``.  ``validation/validate_replay.py`` checks that by comparing
the IEEE 754 byte representation of every number in two traces.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .faults import Injection
from .target import N_STEPS, DoubleIntegratorPlant, GncController, reference
from .wrapper import Event, InjectionWrapper


@dataclass
class Trace:
    """Everything one run produced. All arrays have length ``n_steps``."""

    seed: int
    n_steps: int
    p_true: list[float] = field(default_factory=list)
    v_true: list[float] = field(default_factory=list)
    p_hat: list[float] = field(default_factory=list)
    v_hat: list[float] = field(default_factory=list)
    u_applied: list[float] = field(default_factory=list)
    ref: list[float] = field(default_factory=list)
    innovations: list[tuple[int, float, float]] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    monitor: dict[str, object] = field(default_factory=dict)
    updates: int = 0
    skipped: int = 0
    aborted_step: int | None = None

    @property
    def tracking_error(self) -> list[float]:
        """``p_true - ref`` per step, metres."""
        return [p - r for p, r in zip(self.p_true, self.ref, strict=True)]

    def finite(self) -> bool:
        """True if every logged plant and estimator number is finite."""
        for seq in (self.p_true, self.v_true, self.p_hat, self.v_hat, self.u_applied):
            for x in seq:
                if not math.isfinite(x):
                    return False
        return True

    def float_bytes(self) -> bytes:
        """Canonical IEEE 754 byte image of the trace, for bit-exact comparison."""
        vals: list[float] = []
        for seq in (self.p_true, self.v_true, self.p_hat, self.v_hat, self.u_applied, self.ref):
            vals.extend(seq)
        for k, ep, ev in self.innovations:
            vals.extend((float(k), ep, ev))
        arr = np.asarray(vals, dtype=np.float64)
        return arr.tobytes()


def noise_stream(seed: int, n_steps: int) -> np.ndarray:
    """``(n_steps, 4)`` array of standard normals: meas pos, meas vel, proc p, proc v."""
    rng = np.random.default_rng([int(seed), 1])
    return rng.standard_normal((n_steps, 4))


def fault_rng(seed: int) -> np.random.Generator:
    """Generator for handler draws."""
    return np.random.default_rng([int(seed), 2])


def run_case(
    injections: Sequence[Injection],
    seed: int,
    n_steps: int = N_STEPS,
    target: GncController | None = None,
    abort_on_nonfinite_state: bool = True,
) -> Trace:
    """Run one closed-loop case and return its trace.

    Parameters
    ----------
    injections
        Faults to inject. An empty sequence gives the nominal run.
    seed
        Case seed. Drives both noise and handler draws, through separate streams.
    n_steps
        Loop steps to run.
    target
        Target instance to wrap. A fresh :class:`~faultinject.target.GncController`
        is created if omitted. It is reset before the run.
    abort_on_nonfinite_state
        If true the loop stops once the plant state is no longer finite, and
        the remaining steps are filled with the last value plus the offending
        non-finite value, so a trace is never silently truncated. The step at
        which that happened is recorded in ``Trace.aborted_step``.

    Returns
    -------
    Trace
    """
    from .target import SIGMA_ACCEL, SIGMA_POS, SIGMA_VEL  # local: constants only

    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    ctrl = target if target is not None else GncController()
    ctrl.reset()
    plant = DoubleIntegratorPlant()
    plant.reset()
    wrapper = InjectionWrapper(ctrl, injections, fault_rng(seed))
    wrapper.reset()
    noise = noise_stream(seed, n_steps)
    dt = plant.dt
    sp = 0.5 * SIGMA_ACCEL * dt * dt
    sv = SIGMA_ACCEL * dt

    tr = Trace(seed=int(seed), n_steps=int(n_steps))
    for k in range(n_steps):
        meas = plant.measure(
            SIGMA_POS * float(noise[k, 0]), SIGMA_VEL * float(noise[k, 1])
        )
        cmd = wrapper.step(k, meas)
        u = float(cmd["u"])
        tr.p_true.append(plant.p)
        tr.v_true.append(plant.v)
        tr.p_hat.append(ctrl.p_hat)
        tr.v_hat.append(ctrl.v_hat)
        tr.u_applied.append(u)
        tr.ref.append(reference(k, dt))
        plant.advance(u, sp * float(noise[k, 2]), sv * float(noise[k, 3]))
        if abort_on_nonfinite_state and not (
            math.isfinite(plant.p) and math.isfinite(plant.v)
        ):
            tr.aborted_step = k
            pad = n_steps - (k + 1)
            tr.p_true.extend([plant.p] * pad)
            tr.v_true.extend([plant.v] * pad)
            tr.p_hat.extend([ctrl.p_hat] * pad)
            tr.v_hat.extend([ctrl.v_hat] * pad)
            tr.u_applied.extend([u] * pad)
            tr.ref.extend(reference(j, dt) for j in range(k + 1, n_steps))
            break

    tr.innovations = list(ctrl.innovations)
    tr.events = list(wrapper.events)
    tr.monitor = wrapper.monitor.report()
    tr.updates = ctrl.updates
    tr.skipped = ctrl.skipped
    return tr


_NOMINAL_CACHE: dict[tuple[int, int, str], Trace] = {}


def nominal_trace(seed: int, n_steps: int = N_STEPS, target_name: str = "GncController") -> Trace:
    """Fault-free trace for ``seed``, cached because severity needs it per case."""
    key = (int(seed), int(n_steps), target_name)
    if key not in _NOMINAL_CACHE:
        _NOMINAL_CACHE[key] = run_case((), seed, n_steps)
    return _NOMINAL_CACHE[key]
