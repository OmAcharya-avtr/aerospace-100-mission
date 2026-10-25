"""Seeded Monte Carlo simulation of a declared diagram.

What this is and is not for
---------------------------

For a declared diagram with declared distributions the system reliability has a
correct answer, and :mod:`reliablock.quantify` computes it. This module exists
to be **scored against** that answer, never to replace it. A simulation that
disagrees with the closed form is evidence of a defect in one of the two, and
the validation script reports the disagreement as measured rather than
preferring the simulation because it is "empirical".

What it is genuinely useful for is the structures whose closed forms are the
hardest to get right -- standby with a warm hazard fraction and an imperfect
switch, and shared load -- because the simulation of those follows the physical
story (the primary fails at ``u``, the switch is tried, the standby has been
ageing at a fraction of its rate) with no algebra in common with the closed
form.

How a lifetime is sampled
-------------------------

For any continuous lifetime ``L`` with cumulative hazard ``H``, the quantity
``H(L)`` is exactly ``Exponential(1)``-distributed. So a draw is

    ``E ~ Exponential(1)``,  ``L = H^-1(E)``

which is the inverse-transform method written in hazard space. It needs no
distribution-specific sampler and it is exact for every distribution in
:mod:`reliablock.distributions`, including
:class:`~reliablock.commoncause.ScaledHazard`, whose inverse is the base's
inverse at ``E / factor``.

Lifetimes are drawn in **effective-age** units, matching
:mod:`reliablock.mission`: a block is alive at the end of the mission when its
drawn lifetime exceeds its effective age.

Confidence interval
-------------------

The estimate is a binomial proportion. The interval reported is the Wilson
score interval, which stays inside ``[0, 1]`` and does not collapse to zero
width when no failure is observed -- both of which the normal-approximation
interval does, and both of which matter here because a redundant diagram's
unreliability is small by construction.

Wilson, E. B. (1927), "Probable Inference, the Law of Succession, and
Statistical Inference", Journal of the American Statistical Association 22,
209-212.

Sizing the budget
-----------------

:func:`trials_for_half_width` sizes the run from a target interval half-width
rather than from a round number, because a round number is not a statement
about the check's power. The relation is the normal-approximation sample size
``n = z^2 Q (1 - Q) / h^2``, evaluated at a pilot estimate of ``Q``; the
interval actually achieved is then reported, not the target.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .commoncause import ScaledHazard
from .diagram import (
    Block,
    Diagram,
    KOfN,
    Network,
    Node,
    Parallel,
    Series,
    SharedLoad,
    Standby,
)
from .distributions import Exponential, FailureDistribution
from .mission import MissionProfile
from .structure import switch_event_name

__all__ = [
    "MonteCarloResult",
    "sample_lifetimes",
    "simulate",
    "trials_for_half_width",
    "wilson_interval",
]

#: Normal quantile for a two-sided 95 percent interval.
Z_95 = 1.959963984540054


@dataclass(frozen=True)
class MonteCarloResult:
    """A Monte Carlo reliability estimate with its interval and its budget."""

    trials: int
    survivals: int
    seed: int
    #: Point estimate of the system reliability, dimensionless.
    reliability: float
    #: Wilson score interval on the reliability, at :attr:`confidence`.
    lower: float
    upper: float
    confidence: float = 0.95
    #: Wall-clock seconds for the run. Reported only in tables declared
    #: unstable, never in a prose conclusion.
    elapsed_s: float = 0.0

    @property
    def unreliability(self) -> float:
        """``1 - reliability``. The small quantity, so stated as such."""
        return 1.0 - self.reliability

    @property
    def half_width(self) -> float:
        """Half the width of the interval, dimensionless."""
        return 0.5 * (self.upper - self.lower)

    def brackets(self, value: float) -> bool:
        """True if ``value`` lies inside the interval."""
        return self.lower <= float(value) <= self.upper

    def standard_errors_from(self, value: float) -> float:
        """Distance from ``value`` in units of the binomial standard error.

        The quantity the validation script reports: an estimate two standard
        errors from the closed form is within its interval, one twenty standard
        errors away is a defect, and a verdict of "agrees" without this number
        says nothing about how hard the check was.
        """
        p = self.reliability
        if self.trials <= 0:
            return math.inf
        error = math.sqrt(max(p * (1.0 - p), 0.0) / self.trials)
        if error == 0.0:
            return 0.0 if p == float(value) else math.inf
        return abs(p - float(value)) / error

    def __str__(self) -> str:
        return (
            f"MC n={self.trials} seed={self.seed} R={self.reliability:.9f} "
            f"[{self.lower:.9f}, {self.upper:.9f}] half-width {self.half_width:.3e}"
        )


def wilson_interval(
    successes: int, trials: int, z: float = Z_95
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion, in ``[0, 1]``.

    ``centre = (k + z^2/2) / (n + z^2)``,
    ``half = z sqrt(k (n - k) / n + z^2 / 4) / (n + z^2)``.
    """
    if trials <= 0:
        raise ValueError(f"trials must be > 0, got {trials}")
    if not 0 <= successes <= trials:
        raise ValueError(f"successes must lie in 0..{trials}, got {successes}")
    n = float(trials)
    k = float(successes)
    z2 = z * z
    denominator = n + z2
    centre = (k + 0.5 * z2) / denominator
    half = z * math.sqrt(k * (n - k) / n + 0.25 * z2) / denominator
    return max(centre - half, 0.0), min(centre + half, 1.0)


def trials_for_half_width(
    unreliability: float, half_width: float, z: float = Z_95
) -> int:
    """Trials needed for an interval half-width of ``half_width`` at ``Q``.

    ``n = z^2 Q (1 - Q) / h^2``, the normal-approximation sample size,
    evaluated at a pilot estimate of the unreliability. The value returned is
    the ceiling, and the interval actually achieved by a run of that size is
    reported by :func:`simulate` rather than assumed.
    """
    q = float(unreliability)
    if not 0.0 < q < 1.0:
        raise ValueError(
            f"unreliability must lie strictly in (0, 1) to size a run, got {q}. "
            "A diagram with no failures in the pilot cannot be sized this way; "
            "state the budget directly and report the one-sided interval."
        )
    h = float(half_width)
    if h <= 0.0:
        raise ValueError(f"half_width must be > 0, got {h}")
    return int(math.ceil(z * z * q * (1.0 - q) / (h * h)))


def sample_lifetimes(
    distribution: FailureDistribution, rng: np.random.Generator, size: int
) -> np.ndarray:
    """Draw ``size`` lifetimes in effective-age hours by inverse transform.

    ``E ~ Exponential(1)``, ``L = H^-1(E)``. Vectorised in closed form for the
    exponential, Weibull and scaled-hazard cases; for any other distribution it
    falls back to a per-draw call of
    :meth:`~reliablock.distributions.FailureDistribution.inverse_cumulative_hazard`,
    which is correct and slower, and the cost is recorded in the timing table.
    """
    if size < 0:
        raise ValueError(f"size must be >= 0, got {size}")
    hazards = rng.exponential(1.0, size=size)
    return _invert(distribution, hazards)


def _invert(distribution: FailureDistribution, hazards: np.ndarray) -> np.ndarray:
    """Vectorised ``H^-1`` for the kinds that have one; per-element otherwise."""
    if isinstance(distribution, ScaledHazard):
        assert distribution.base is not None
        if distribution.factor == 0.0:
            return np.full(hazards.shape, np.inf)
        return _invert(distribution.base, hazards / distribution.factor)
    if isinstance(distribution, Exponential):
        if distribution.rate_per_hour == 0.0:
            return np.full(hazards.shape, np.inf)
        return hazards / distribution.rate_per_hour
    from .distributions import Weibull

    if isinstance(distribution, Weibull):
        return distribution.scale_hours * hazards ** (1.0 / distribution.shape)
    return np.array(
        [distribution.inverse_cumulative_hazard(float(h)) for h in hazards], dtype=float
    )


def simulate(
    diagram: Diagram,
    mission: MissionProfile | float,
    trials: int,
    *,
    seed: int = 0,
    confidence: float = 0.95,
    batch: int = 250_000,
) -> MonteCarloResult:
    """Estimate the reliability of ``diagram`` over ``mission`` by simulation.

    Runs in batches so that the peak memory is bounded by ``batch`` times the
    number of blocks rather than by ``trials`` times it: a 4-million-trial run
    over a twelve-block diagram would otherwise allocate 384 MB of float64 in
    one go.

    Parameters
    ----------
    diagram
        The declared diagram.
    mission
        Mission profile or duration in hours.
    trials
        Number of independent mission simulations, ``> 0``.
    seed
        Seed for ``numpy.random.default_rng``. Every number this package
        reports from a simulation is seeded, and the seed is in the result.
    confidence
        Two-sided confidence level for the Wilson interval.
    batch
        Trials per vectorised batch.

    Returns
    -------
    MonteCarloResult
    """
    import time

    if trials <= 0:
        raise ValueError(f"trials must be > 0, got {trials}")
    if batch <= 0:
        raise ValueError(f"batch must be > 0, got {batch}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie in (0, 1), got {confidence}")
    from scipy.stats import norm

    z = float(norm.isf(0.5 * (1.0 - confidence)))
    profile = (
        mission if isinstance(mission, MissionProfile) else MissionProfile.from_hours(mission)
    )
    ages = profile.effective_ages(diagram)
    switches = {
        id(item): switch_event_name(item, index)
        for index, item in enumerate(diagram.walk())
        if isinstance(item, Standby)
    }
    rng = np.random.default_rng(seed)
    survivals = 0
    started = time.perf_counter()
    remaining = trials
    while remaining > 0:
        size = min(batch, remaining)
        alive = _simulate_node(diagram.root, ages, rng, size, switches)
        survivals += int(np.count_nonzero(alive))
        remaining -= size
    elapsed = time.perf_counter() - started
    lower, upper = wilson_interval(survivals, trials, z)
    return MonteCarloResult(
        trials=trials,
        survivals=survivals,
        seed=seed,
        reliability=survivals / trials,
        lower=lower,
        upper=upper,
        confidence=confidence,
        elapsed_s=elapsed,
    )


def _simulate_node(
    item: Node | Block,
    ages: dict[str, float],
    rng: np.random.Generator,
    size: int,
    switches: dict[int, str],
) -> np.ndarray:
    """Boolean array of length ``size``: is this sub-structure alive at mission end?"""
    if isinstance(item, Block):
        lifetimes = sample_lifetimes(item.distribution, rng, size)
        return lifetimes > ages[item.name]
    if isinstance(item, Series):
        alive = np.ones(size, dtype=bool)
        for child in item.items:
            alive &= _simulate_node(child, ages, rng, size, switches)
        return alive
    if isinstance(item, Parallel):
        alive = np.zeros(size, dtype=bool)
        for child in item.items:
            alive |= _simulate_node(child, ages, rng, size, switches)
        return alive
    if isinstance(item, KOfN):
        count = np.zeros(size, dtype=np.int32)
        for child in item.items:
            count += _simulate_node(child, ages, rng, size, switches).astype(np.int32)
        return count >= item.k
    if isinstance(item, Standby):
        return _simulate_standby(item, ages, rng, size)
    if isinstance(item, SharedLoad):
        return _simulate_shared_load(item, ages, rng, size)
    if isinstance(item, Network):
        return _simulate_network(item, ages, rng, size, switches)
    raise TypeError(f"cannot simulate {type(item).__name__}")  # pragma: no cover


def _simulate_standby(
    item: Standby, ages: dict[str, float], rng: np.random.Generator, size: int
) -> np.ndarray:
    """Simulate a standby group by following the physical story.

    The primary runs until its drawn lifetime. If that exceeds the mission, the
    group survives. Otherwise the changeover is tried; a failed changeover ends
    the group there. A standby unit called upon at effective age ``u`` has been
    accruing hazard at a fraction ``f`` of its rate, so its effective age at
    mission end ``T`` is ``f u + (T - u)``, and it survives when its drawn
    lifetime exceeds that.

    Each unit's lifetime is drawn exactly once. Drawing the primary's lifetime
    twice -- once to decide whether the group survived on the primary alone and
    once to decide when the changeover happened -- would let the two disagree,
    and is the obvious way to get this wrong.

    There is no algebra here in common with the Erlang closed form or with the
    convolution integral in :mod:`reliablock.quantify`.
    """
    assert item.primary is not None
    if not isinstance(item.primary, Block) or not all(
        isinstance(s, Block) for s in item.standbys
    ):
        raise ValueError(
            "simulation of a standby group over nested structures is not "
            "implemented; declare the group over blocks"
        )
    fraction = float(item.standby_hazard_fraction)
    if fraction > 0.0 and len(item.standbys) > 1:
        raise ValueError(
            f"standby node {item.label or 'standby'!r} declares "
            f"{len(item.standbys)} units at a warm hazard fraction of "
            f"{fraction:g}. Tracking the accrued age of a third unit through two "
            "handovers at a partial rate is a different bookkeeping problem, and "
            "reliablock.quantify refuses the same case rather than approximating "
            "it; the simulator refuses it too so that the two always cover the "
            "same declarations."
        )
    duration = ages[item.primary.name]
    for unit in item.standbys:
        assert isinstance(unit, Block)
        if ages[unit.name] != duration:
            raise ValueError(
                f"standby node {item.label or 'standby'!r}: its members have "
                "different effective ages. A standby group shares one mission "
                "clock; a profile that gives its members different duty cycles "
                "is not a standby group."
            )

    primary_life = sample_lifetimes(item.primary.distribution, rng, size)
    survived = primary_life > duration
    # Effective age at which the next unit is called upon. Capped at the
    # mission duration, beyond which no changeover happens.
    called_at = np.minimum(primary_life, duration)
    switch_ok = np.ones(size, dtype=bool)
    for unit in item.standbys:
        assert isinstance(unit, Block)
        switch_ok = switch_ok & (rng.random(size) < item.switch_reliability)
        standby_life = sample_lifetimes(unit.distribution, rng, size)
        effective_age = fraction * called_at + (duration - called_at)
        unit_survives = standby_life > effective_age
        survived = survived | (switch_ok & unit_survives)
        # Cold standby only beyond the first unit (enforced above), so the
        # calendar time of this unit's failure is called_at + its lifetime.
        called_at = np.minimum(called_at + standby_life, duration)
    return survived


def _simulate_shared_load(
    item: SharedLoad, ages: dict[str, float], rng: np.random.Generator, size: int
) -> np.ndarray:
    """Simulate a load-sharing pair by following the two-stage physical story.

    While both units are alive each carries ``s lambda_i``. The first failure
    is at the minimum of two exponentials; the survivor then carries
    ``lambda_j`` and, being exponential and therefore memoryless, has a
    residual lifetime drawn fresh at that rate. No algebra in common with the
    four-state closed form.
    """
    first, second = item.units
    duration = ages[first.name]
    if ages[second.name] != duration:
        raise ValueError(
            f"shared_load node {item.label or 'shared_load'!r}: its units have "
            "different effective ages"
        )
    s = float(item.shared_rate_factor)
    rates = [first.distribution.rate_per_hour, second.distribution.rate_per_hour]  # type: ignore[union-attr]
    shared = [s * r for r in rates]
    draws = []
    for rate in shared:
        if rate == 0.0:
            draws.append(np.full(size, np.inf))
        else:
            draws.append(rng.exponential(1.0 / rate, size=size))
    first_event = np.minimum(draws[0], draws[1])
    failed_is_first = draws[0] <= draws[1]
    both_survive = first_event > duration
    survivor_rate = np.where(failed_is_first, rates[1], rates[0])
    residual = np.where(
        survivor_rate > 0.0,
        rng.exponential(1.0, size=size) / np.where(survivor_rate > 0.0, survivor_rate, 1.0),
        np.inf,
    )
    carried = (~both_survive) & (first_event + residual > duration)
    return both_survive | carried


def _simulate_network(
    item: Network,
    ages: dict[str, float],
    rng: np.random.Generator,
    size: int,
    switches: dict[int, str],
) -> np.ndarray:
    """Simulate a two-terminal network through its minimal path sets.

    The system is up when at least one minimal path has every edge alive. That
    is exact -- it is the definition of connectivity restricted to minimal
    paths -- and it vectorises, where a per-trial graph search does not.
    """
    from .structure import _network_paths  # internal: the same path enumeration

    alive_edges = {
        block.name: _simulate_node(block, ages, rng, size, switches)
        for _, _, block in item.edges
    }
    up = np.zeros(size, dtype=bool)
    for path in _network_paths(item):
        path_up = np.ones(size, dtype=bool)
        for event in sorted(path):
            path_up &= alive_edges[event]
        up |= path_up
    return up
