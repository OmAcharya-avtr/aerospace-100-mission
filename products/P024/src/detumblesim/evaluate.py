"""Scenario scoring, the gain oracle, and the training-set builder.

Cost function
-------------
A detumble run is scored on two things a mission actually cares about: how
long it takes, and how much coil energy it burns while the battery is still
recovering from launch.  Both are normalised by the orbital period and by the
squared dipole limit so the score is dimensionless and comparable across
vehicles:

    cost = t_detumble / T_orbit
         + w * integral(|m|^2 dt) / (m_max^2 * T_orbit)

``integral |m|^2 dt`` is proportional to coil ohmic energy at fixed coil
resistance and turns-area.  ``w`` (``ENERGY_WEIGHT``) is a **design weight,
not a measured constant**; the sensitivity of every conclusion to it is
reported in ``validation/learned_vs_fixed_ci.py``.

A run that does not reach the rate threshold inside the simulated span is
scored with the span itself times ``FAILURE_PENALTY`` in the time term.  The
number of such runs is always reported separately, so the penalty cannot hide
a policy that simply fails.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .features import TelemetryWindow
from .policies import ScheduledGainPolicy, wrap_with_saturation_feedback
from .scenarios import Scenario
from .simulate import DetumbleResult, simulate_detumble

#: Weight on the actuation-energy term of the cost function [dimensionless].
ENERGY_WEIGHT: float = 0.5

#: Multiplier applied to the simulated span when a run never detumbles.
FAILURE_PENALTY: float = 2.0


@dataclass(frozen=True)
class RunScore:
    """Scalar summary of one detumble run.

    Attributes
    ----------
    cost : float
        Combined dimensionless cost, see the module docstring.
    detumble_time_s : float
        [s]; NaN if the threshold was never reached.
    time_orbits : float
        Detumble time in orbital periods, or the penalised span if it failed.
    energy_term : float
        ``w * integral |m|^2 dt / (m_max^2 T_orbit)``, dimensionless.
    actuation_cost_a2m4s : float
        ``integral |m|^2 dt`` [A^2 m^4 s].
    saturated_fraction : float
        Fraction of control steps with a clipped command.
    detumbled : bool
        Whether the rate threshold was reached.
    """

    cost: float
    detumble_time_s: float
    time_orbits: float
    energy_term: float
    actuation_cost_a2m4s: float
    saturated_fraction: float
    detumbled: bool


def score_run(
    result: DetumbleResult,
    scenario: Scenario,
    span_s: float,
    energy_weight: float = ENERGY_WEIGHT,
) -> RunScore:
    """Score one ``DetumbleResult`` against the cost function."""
    if energy_weight < 0.0:
        raise ValueError("energy_weight must be non-negative")
    period = scenario.orbit.period_s
    m_max = float(np.min(scenario.magnetorquer.max_dipole_am2))
    if result.detumbled:
        t_orb = result.detumble_time_s / period
    else:
        t_orb = FAILURE_PENALTY * span_s / period
    energy = energy_weight * result.actuation_cost_a2m4s / (m_max**2 * period)
    return RunScore(
        cost=float(t_orb + energy),
        detumble_time_s=float(result.detumble_time_s),
        time_orbits=float(t_orb),
        energy_term=float(energy),
        actuation_cost_a2m4s=float(result.actuation_cost_a2m4s),
        saturated_fraction=float(result.saturated_fraction),
        detumbled=bool(result.detumbled),
    )


def run_policy(
    scenario: Scenario,
    policy,
    duration_s: float = 23000.0,
    control_dt_s: float = 2.0,
    substeps: int = 2,
    mag_noise_t: float = 0.0,
    energy_weight: float = ENERGY_WEIGHT,
) -> tuple[DetumbleResult, RunScore]:
    """Simulate one scenario under one policy and score it.

    Policies that consume saturation feedback (``ScheduledGainPolicy``) are
    wrapped automatically so the loop is closed.
    """
    if hasattr(policy, "reset"):
        policy.reset()
    runner = (
        wrap_with_saturation_feedback(policy, scenario.magnetorquer)
        if isinstance(policy, ScheduledGainPolicy)
        else policy
    )
    cfg = scenario.to_config(
        duration_s=duration_s,
        control_dt_s=control_dt_s,
        substeps=substeps,
        mag_noise_t=mag_noise_t,
    )
    result = simulate_detumble(cfg, runner)
    return result, score_run(result, scenario, duration_s, energy_weight)


def oracle_gain(
    scenario: Scenario,
    gains: NDArray[np.float64],
    duration_s: float = 23000.0,
    control_dt_s: float = 2.0,
    substeps: int = 2,
    energy_weight: float = ENERGY_WEIGHT,
) -> tuple[float, float, NDArray[np.float64]]:
    """Best constant gain for one scenario by exhaustive grid search.

    Returns
    -------
    (best_gain, best_cost, costs)
        ``costs`` has one entry per gain in ``gains``.
    """
    from .policies import FixedGainPolicy

    g = np.asarray(gains, dtype=float)
    if g.ndim != 1 or g.size < 1:
        raise ValueError("gains must be a non-empty 1-D array")
    if np.any(g <= 0.0):
        raise ValueError("all gains must be positive")
    costs = np.empty(g.size)
    for i, k in enumerate(g):
        _, sc = run_policy(
            scenario,
            FixedGainPolicy(float(k)),
            duration_s=duration_s,
            control_dt_s=control_dt_s,
            substeps=substeps,
            energy_weight=energy_weight,
        )
        costs[i] = sc.cost
    j = int(np.argmin(costs))
    return float(g[j]), float(costs[j]), costs


def training_rows(
    scenario: Scenario,
    best_gain: float,
    base_gain: float,
    result: DetumbleResult,
    window_length: int = 60,
    stride: int = 30,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Feature rows and labels harvested from one scored training run.

    The run is replayed offline: the magnetometer history in ``result`` is
    pushed through a ``TelemetryWindow`` and a feature row is emitted every
    ``stride`` steps once the window is full.  Every row of a scenario carries
    the same label ``log10(best_gain / base_gain)``.
    """
    if not np.isfinite(best_gain) or best_gain <= 0.0:
        raise ValueError("best_gain must be positive")
    if not np.isfinite(base_gain) or base_gain <= 0.0:
        raise ValueError("base_gain must be positive")
    if int(stride) < 1:
        raise ValueError("stride must be >= 1")
    win = TelemetryWindow(window_length)
    b = result.b_body_t
    sat = result.saturated
    rows: list[NDArray[np.float64]] = []
    m_max = float(np.min(scenario.magnetorquer.max_dipole_am2))
    j_scale = scenario.inertia_scale_kgm2
    dt = float(result.t_s[1] - result.t_s[0]) if result.t_s.size > 1 else 1.0
    for i in range(1, b.shape[0]):
        b_dot = (b[i] - b[i - 1]) / dt
        win.push(float(i), b[i], b_dot, bool(sat[i - 1]))
        if len(win) >= window_length and i % stride == 0:
            rows.append(win.features(m_max, j_scale))
    if not rows:
        return np.empty((0, 8)), np.empty(0)
    x = np.vstack(rows)
    y = np.full(x.shape[0], float(np.log10(best_gain / base_gain)))
    return x, y


def fit_power_law_gain(
    scenarios: list[Scenario], oracle_gains: NDArray[np.float64]
) -> tuple[NDArray[np.float64], float]:
    """Least-squares fit of ``log10 k = a + b log10 m_max + c log10 j``.

    Parameters
    ----------
    scenarios : list of Scenario
    oracle_gains : ndarray (n,)
        Best constant gain for each scenario [A m^2 s T^-1], all positive.

    Returns
    -------
    (coefficients, rms_residual_dex)
        ``coefficients`` is ``(a, b, c)``; the residual is in log10 units.
    """
    g = np.asarray(oracle_gains, dtype=float).ravel()
    if g.size != len(scenarios):
        raise ValueError("oracle_gains must have one entry per scenario")
    if np.any(g <= 0.0):
        raise ValueError("oracle gains must be positive")
    if g.size < 3:
        raise ValueError("need at least three scenarios to fit three coefficients")
    m = np.array(
        [float(np.min(s.magnetorquer.max_dipole_am2)) for s in scenarios], dtype=float
    )
    j = np.array([s.inertia_scale_kgm2 for s in scenarios], dtype=float)
    design = np.column_stack([np.ones(g.size), np.log10(m), np.log10(j)])
    coef, *_ = np.linalg.lstsq(design, np.log10(g), rcond=None)
    resid = design @ coef - np.log10(g)
    return coef, float(np.sqrt(np.mean(resid**2)))
