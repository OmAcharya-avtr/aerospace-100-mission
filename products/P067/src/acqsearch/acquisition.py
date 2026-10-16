"""Expected acquisition time: the analytic model, its Monte Carlo twin, and the
continuous-geometry Monte Carlo that measures what the abstraction costs.

The analytic model
------------------
A1. The search region is partitioned into disjoint square cells with exact
    prior masses ``p_i`` (:class:`~acqsearch.geometry.CellGrid`); the prior
    mass outside the grid, ``mass_outside``, is never acquired.
A2. A dwell on cell ``i`` detects a target located in cell ``i`` with
    probability ``q_i``, independently of every other dwell. ``q_i`` is the
    prior-weighted average of the pointwise detection probability over the
    cell (:func:`cell_detection_probabilities`).
A3. A dwell on a cell that does not contain the target raises a false alarm
    with probability ``f``; a false alarm costs a fixed verification time
    ``t_v`` and the search then resumes at the same point in the pattern. The
    analytic model charges ``f * t_v`` to every dwell, which over-charges by at
    most one ``f * t_v`` per pass (the detecting dwell).
A4. The pattern repeats indefinitely until detection, with the same order and
    the same slew geometry on every pass.

Under A1-A4 the expected time, conditional on the target lying in cell visited
at position ``k`` of the order, is

    E[T | cell at position k] = T_k + T_period * (1 - q) / q          [s]

where ``T_k`` is the expected elapsed time at the end of dwell ``k`` and the
geometric term is the expected number of failed passes (Feller, W. (1968)
*An Introduction to Probability Theory and Its Applications*, Vol. 1, 3rd ed.,
Wiley, Ch. XIII: a geometric variable with success probability ``q`` has mean
``1 / q``). The deadline probability is the corresponding finite sum.

Everything in this module is exact *given* A1-A4. The cost of A1 and A2 -- the
cell abstraction -- is measured, not assumed, by
:func:`monte_carlo_acquisition_continuous`, which samples a continuous target
position and uses the pointwise detection probability instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .detection import PoissonDwellDetector
from .geometry import CellGrid, PointingUncertainty, TerminalGeometry


def cell_detection_probabilities(
    grid: CellGrid,
    geometry: TerminalGeometry,
    detector: PoissonDwellDetector,
    dwell_s: float,
    uncertainty: PointingUncertainty,
    quadrature_nodes: int = 8,
) -> np.ndarray:
    """Prior-weighted mean detection probability ``q_i`` for each cell.

    The target, given that it lies in cell ``i``, is distributed as the prior
    restricted to that square. ``q_i`` is the expectation of
    ``PoissonDwellDetector.detection_probability`` over that restricted
    distribution, evaluated with a tensor-product Gauss-Legendre rule of
    ``quadrature_nodes`` points per axis (Abramowitz & Stegun 1964, eq.
    25.4.29). Deterministic: no random numbers are used.

    Units: ``dwell_s`` seconds, result dimensionless in [0, 1], shape
    ``(n_cells,)``.
    """
    if int(quadrature_nodes) < 2:
        raise ValueError("quadrature_nodes must be >= 2")
    nodes, weights = np.polynomial.legendre.leggauss(int(quadrature_nodes))
    half = 0.5 * grid.side_rad
    offsets = nodes * half
    wx = weights * half
    centres = grid.centres_rad
    x = centres[:, 0][:, None, None] + offsets[None, :, None]
    y = centres[:, 1][:, None, None] + offsets[None, None, :]
    weight = (wx[None, :, None] * wx[None, None, :]) * np.exp(
        -(x**2 + y**2) / (2.0 * uncertainty.sigma_rad**2)
    )
    rho = np.hypot(x - centres[:, 0][:, None, None], y - centres[:, 1][:, None, None])
    pd = np.asarray(
        detector.detection_probability(dwell_s, rho, geometry.beam_half_angle_rad), dtype=float
    )
    numerator = (weight * pd).sum(axis=(1, 2))
    denominator = weight.sum(axis=(1, 2))
    return numerator / denominator


@dataclass(frozen=True)
class DwellSchedule:
    """Timing of one pass of a cell visit order.

    ``order`` is the permutation of cell indices. ``dwell_time_s`` is the
    per-dwell cost excluding the false-alarm charge (dwell + slew + settle),
    ``false_alarm_charge_s`` is the ``f * t_v`` charged to every dwell under
    assumption A3, and ``cumulative_time_s[k]`` is the expected elapsed time at
    the end of dwell ``k`` (0-based) including that charge. All in seconds.
    """

    order: np.ndarray
    dwell_time_s: np.ndarray
    false_alarm_charge_s: float
    cumulative_time_s: np.ndarray

    @property
    def n_dwells(self) -> int:
        """Number of dwells in one pass."""
        return int(self.order.size)

    @property
    def period_s(self) -> float:
        """Expected duration of one complete pass, seconds."""
        return float(self.cumulative_time_s[-1])

    def total_slew_rad(self, grid: CellGrid, start_rad: np.ndarray | None = None) -> float:
        """Total commanded angular travel of one pass, radians."""
        start = np.zeros(2) if start_rad is None else np.asarray(start_rad, dtype=float)
        points = np.vstack([start, grid.centres_rad[self.order]])
        steps = np.diff(points, axis=0)
        return float(np.hypot(steps[:, 0], steps[:, 1]).sum())


def build_schedule(
    grid: CellGrid,
    order: np.ndarray,
    geometry: TerminalGeometry,
    dwell_s: float,
    false_alarm_probability: float,
    verification_time_s: float,
    start_rad: np.ndarray | None = None,
) -> DwellSchedule:
    """Time one pass of ``order``, starting from ``start_rad`` (default boresight).

    Dwell ``k`` costs ``dwell_s + slew_time(distance from the previous
    position)``, plus the ``f * t_v`` false-alarm charge of assumption A3.
    Units: seconds.
    """
    order = np.asarray(order, dtype=int)
    if order.size == 0:
        raise ValueError("order must be non-empty")
    if np.unique(order).size != order.size:
        raise ValueError("order must not repeat a cell")
    if dwell_s <= 0.0 or not math.isfinite(dwell_s):
        raise ValueError(f"dwell_s must be finite and > 0, got {dwell_s!r}")
    if not 0.0 <= false_alarm_probability <= 1.0:
        raise ValueError("false_alarm_probability must lie in [0, 1]")
    if verification_time_s < 0.0 or not math.isfinite(verification_time_s):
        raise ValueError("verification_time_s must be finite and >= 0")
    start = np.zeros(2) if start_rad is None else np.asarray(start_rad, dtype=float)
    points = np.vstack([start, grid.centres_rad[order]])
    steps = np.diff(points, axis=0)
    distance = np.hypot(steps[:, 0], steps[:, 1])
    per_dwell = float(dwell_s) + np.asarray(geometry.slew_time_s(distance), dtype=float)
    charge = float(false_alarm_probability) * float(verification_time_s)
    cumulative = np.cumsum(per_dwell + charge)
    return DwellSchedule(
        order=order,
        dwell_time_s=per_dwell,
        false_alarm_charge_s=charge,
        cumulative_time_s=cumulative,
    )


@dataclass(frozen=True)
class AcquisitionEstimate:
    """Result of an expected-acquisition-time calculation.

    ``expected_time_s`` is conditional on the target lying inside the grid;
    ``mass_in_grid`` says how much prior mass that condition retains and
    ``mass_outside`` is the complement, which is never acquired. Reporting the
    conditional mean together with the mass it conditions on is the only
    honest way to quote a finite number here, because the unconditional mean
    is infinite whenever ``mass_outside > 0``.
    """

    expected_time_s: float
    mass_in_grid: float
    mass_outside: float
    period_s: float
    n_dwells: int


def expected_acquisition_time(
    grid: CellGrid,
    schedule: DwellSchedule,
    cell_detection: np.ndarray,
) -> AcquisitionEstimate:
    """Analytic E[T | target inside the grid] under assumptions A1-A4, seconds.

    ``cell_detection`` is ``q_i`` per cell, dimensionless in (0, 1]. A cell
    with ``q_i == 0`` can never be acquired and makes the mean infinite; this
    function raises instead of returning ``inf``.
    """
    q = np.asarray(cell_detection, dtype=float)
    if q.shape != (grid.n_cells,):
        raise ValueError(f"cell_detection must have shape ({grid.n_cells},), got {q.shape}")
    if np.any(q <= 0.0) or np.any(q > 1.0):
        raise ValueError("cell_detection entries must lie in (0, 1]")
    order = schedule.order
    p = grid.prior_mass[order]
    mass = float(p.sum())
    if mass <= 0.0:
        raise ValueError("the grid carries no prior mass")
    q_ordered = q[order]
    conditional = schedule.cumulative_time_s + schedule.period_s * (1.0 - q_ordered) / q_ordered
    return AcquisitionEstimate(
        expected_time_s=float((p * conditional).sum() / mass),
        mass_in_grid=mass,
        mass_outside=float(grid.mass_outside),
        period_s=schedule.period_s,
        n_dwells=schedule.n_dwells,
    )


def probability_acquired_by(
    grid: CellGrid,
    schedule: DwellSchedule,
    cell_detection: np.ndarray,
    deadline_s: float,
) -> float:
    """Analytic unconditional P(acquired by ``deadline_s``), dimensionless.

    ``sum_k p_k * sum_{m : T_k + m T_period <= D} q_k (1 - q_k)^m``. Prior mass
    outside the grid contributes zero, so the result is bounded above by
    ``1 - grid.mass_outside``. Exact under A1-A4, in which the elapsed time is
    the deterministic expected-time accounting of A3; the effect of making the
    false-alarm penalty stochastic is measured separately by
    :func:`monte_carlo_acquisition` with ``stochastic_false_alarms=True``.
    """
    if deadline_s < 0.0 or not math.isfinite(deadline_s):
        raise ValueError(f"deadline_s must be finite and >= 0, got {deadline_s!r}")
    q = np.asarray(cell_detection, dtype=float)[schedule.order]
    if np.any(q <= 0.0) or np.any(q > 1.0):
        raise ValueError("cell_detection entries must lie in (0, 1]")
    p = grid.prior_mass[schedule.order]
    period = schedule.period_s
    t_k = schedule.cumulative_time_s
    reachable = deadline_s - t_k
    passes = np.where(reachable >= 0.0, np.floor(reachable / period) + 1.0, 0.0)
    return float((p * (1.0 - (1.0 - q) ** passes)).sum())


@dataclass(frozen=True)
class MonteCarloResult:
    """Seeded Monte Carlo estimate of acquisition performance.

    ``mean_time_s`` is over the trials in which the target lay inside the grid.
    ``standard_error_s`` is the sample standard error of that mean.
    ``probability_by_deadline`` is unconditional over all trials.
    """

    mean_time_s: float
    standard_error_s: float
    probability_by_deadline: float
    n_trials: int
    n_inside: int


def monte_carlo_acquisition(
    grid: CellGrid,
    schedule: DwellSchedule,
    cell_detection: np.ndarray,
    n_trials: int = 20000,
    seed: int = 20261016,
    deadline_s: float | None = None,
    stochastic_false_alarms: bool = False,
    false_alarm_probability: float = 0.0,
    verification_time_s: float = 0.0,
    max_passes: int = 2000,
) -> MonteCarloResult:
    """Simulate assumptions A1-A4 directly and return the sample statistics.

    With ``stochastic_false_alarms=False`` this samples exactly the model that
    :func:`expected_acquisition_time` integrates, so the two must agree to
    within the Monte Carlo standard error; that agreement is the cross-check.
    With ``stochastic_false_alarms=True`` the number of false alarms is drawn
    binomially from the dwells actually performed instead of being charged as
    an expectation, which leaves the mean unchanged but changes the deadline
    probability.

    Units: seconds. Seeded with ``numpy.random.default_rng``.
    """
    if int(n_trials) < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials!r}")
    q_all = np.asarray(cell_detection, dtype=float)
    if np.any(q_all <= 0.0) or np.any(q_all > 1.0):
        raise ValueError("cell_detection entries must lie in (0, 1]")
    rng = np.random.default_rng(int(seed))
    order = schedule.order
    p = grid.prior_mass[order]
    weights = np.concatenate([p, [grid.mass_outside]])
    weights = weights / weights.sum()
    draw = rng.choice(weights.size, size=int(n_trials), p=weights)
    inside = draw < order.size
    positions = draw[inside]
    q = q_all[order][positions]
    failures = rng.geometric(q) - 1
    failures = np.minimum(failures, int(max_passes))
    base = schedule.cumulative_time_s[positions] + failures * schedule.period_s
    if stochastic_false_alarms:
        charge = schedule.false_alarm_charge_s
        dwells = failures * order.size + positions + 1
        base = base - dwells * charge
        alarms = rng.binomial(np.maximum(dwells - 1, 0), float(false_alarm_probability))
        base = base + alarms * float(verification_time_s)
    n_inside = int(inside.sum())
    if n_inside == 0:
        raise ValueError("no trial placed the target inside the grid; increase n_trials")
    mean = float(base.mean())
    se = float(base.std(ddof=1) / math.sqrt(n_inside)) if n_inside > 1 else float("nan")
    if deadline_s is None:
        probability = float("nan")
    else:
        probability = float((base <= deadline_s).sum()) / float(n_trials)
    return MonteCarloResult(
        mean_time_s=mean,
        standard_error_s=se,
        probability_by_deadline=probability,
        n_trials=int(n_trials),
        n_inside=n_inside,
    )


def monte_carlo_acquisition_continuous(
    grid: CellGrid,
    schedule: DwellSchedule,
    geometry: TerminalGeometry,
    detector: PoissonDwellDetector,
    uncertainty: PointingUncertainty,
    dwell_s: float,
    n_trials: int = 4000,
    seed: int = 20261016,
    max_dwells: int | None = None,
    false_alarm_probability: float = 0.0,
    verification_time_s: float = 0.0,
    restrict_to_grid: bool = False,
) -> MonteCarloResult:
    """Monte Carlo without the cell abstraction: continuous target, pointwise P_d.

    The target position is drawn from the continuous prior. At every dwell the
    detection probability is the pointwise ``P_d`` at the true angular offset
    from that dwell's boresight, so a target near a cell boundary can be
    acquired by a neighbouring dwell -- something assumption A2 cannot
    represent. ``P_d(rho)`` is evaluated by linear interpolation on a 1025-point
    table out to five beam half-widths; beyond that the value is taken as the
    false-alarm floor, which is correct to better than 1e-9 for the
    configurations used here.

    ``restrict_to_grid=True`` discards targets that fall outside every square
    cell, which reproduces the conditioning of assumption A1 and makes the
    comparison with :func:`monte_carlo_acquisition` like for like; with it
    ``False`` the remaining prior mass is included and some of it is acquired
    by beam spill past the grid edge, which the cell model cannot represent.

    The difference between this and :func:`monte_carlo_acquisition` is the cost
    of the cell abstraction and is reported, not hidden. Units: seconds.
    """
    if int(n_trials) < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials!r}")
    rng = np.random.default_rng(int(seed))
    targets = uncertainty.sample(int(n_trials), rng)
    if restrict_to_grid:
        nearest = grid.nearest_cell(targets)
        delta = np.abs(targets - grid.centres_rad[nearest])
        keep = np.all(delta <= 0.5 * grid.side_rad + 1e-15, axis=1)
        targets = targets[keep]
        if targets.shape[0] == 0:
            raise ValueError("no sampled target fell inside the grid")
    n_trials = targets.shape[0]
    w = geometry.beam_half_angle_rad
    table_x, table_y = detector.detection_probability_table(dwell_s, w, 5.0 * w, 1025)
    floor = detector.false_alarm_probability(dwell_s)
    limit = int(max_dwells) if max_dwells is not None else 20 * schedule.n_dwells
    centres = grid.centres_rad[schedule.order]
    per_dwell = schedule.dwell_time_s
    charge = float(false_alarm_probability) * float(verification_time_s)
    elapsed = np.zeros(int(n_trials))
    active = np.ones(int(n_trials), dtype=bool)
    acquired = np.zeros(int(n_trials), dtype=bool)
    time_at = np.full(int(n_trials), np.nan)
    for step in range(limit):
        k = step % schedule.n_dwells
        elapsed[active] += per_dwell[k] + charge
        offset = np.hypot(
            targets[active, 0] - centres[k, 0], targets[active, 1] - centres[k, 1]
        )
        pd = np.where(offset <= table_x[-1], np.interp(offset, table_x, table_y), floor)
        hit = rng.random(pd.size) < pd
        if hit.any():
            index = np.nonzero(active)[0][hit]
            acquired[index] = True
            time_at[index] = elapsed[index]
            active[index] = False
        if not active.any():
            break
    n_acquired = int(acquired.sum())
    if n_acquired == 0:
        raise ValueError("no trial acquired within the dwell limit")
    times = time_at[acquired]
    return MonteCarloResult(
        mean_time_s=float(times.mean()),
        standard_error_s=float(times.std(ddof=1) / math.sqrt(n_acquired)),
        probability_by_deadline=float(n_acquired) / float(n_trials),
        n_trials=int(n_trials),
        n_inside=n_acquired,
    )


@dataclass(frozen=True)
class DwellOptimum:
    """Result of the dwell-time trade.

    ``dwell_grid_s`` and ``expected_time_s`` are the swept curve;
    ``best_dwell_s`` is the grid minimum. ``detection_probability`` and
    ``false_alarm_probability`` are the mean cell detection probability and the
    per-dwell false-alarm probability along the same grid.
    """

    dwell_grid_s: np.ndarray
    expected_time_s: np.ndarray
    detection_probability: np.ndarray
    false_alarm_probability: np.ndarray
    best_dwell_s: float
    best_expected_time_s: float


def optimal_dwell_time(
    grid: CellGrid,
    order: np.ndarray,
    geometry: TerminalGeometry,
    detector: PoissonDwellDetector,
    uncertainty: PointingUncertainty,
    dwell_grid_s: np.ndarray,
    verification_time_s: float,
) -> DwellOptimum:
    """Sweep the dwell time and return the grid minimum of E[T].

    The trade is between three effects, all of them in the model above: a
    longer dwell raises ``q_i`` (more signal counts), raises ``f`` (more
    background counts past the same fixed threshold) and raises the per-dwell
    cost directly. The curve is evaluated on the supplied grid and the minimum
    is taken over it; no convexity is assumed, and none holds in general, so
    the returned optimum is a grid optimum and nothing more.

    Units: ``dwell_grid_s`` seconds, returned times seconds.
    """
    dwells = np.asarray(dwell_grid_s, dtype=float)
    if dwells.ndim != 1 or dwells.size < 1:
        raise ValueError("dwell_grid_s must be a non-empty 1-D array")
    if np.any(dwells <= 0.0):
        raise ValueError("dwell_grid_s entries must be > 0")
    expected = np.empty(dwells.size)
    mean_q = np.empty(dwells.size)
    fa = np.empty(dwells.size)
    for i, tau in enumerate(dwells):
        q = cell_detection_probabilities(grid, geometry, detector, float(tau), uncertainty)
        f = detector.false_alarm_probability(float(tau))
        schedule = build_schedule(grid, order, geometry, float(tau), f, verification_time_s)
        expected[i] = expected_acquisition_time(grid, schedule, q).expected_time_s
        mean_q[i] = float(np.average(q, weights=grid.prior_mass))
        fa[i] = f
    best = int(np.argmin(expected))
    return DwellOptimum(
        dwell_grid_s=dwells,
        expected_time_s=expected,
        detection_probability=mean_q,
        false_alarm_probability=fa,
        best_dwell_s=float(dwells[best]),
        best_expected_time_s=float(expected[best]),
    )
