"""Classical desaturation schedulers, implemented and tuned before anything is learned.

The baseline here is the rule an ADCS engineer writes first and that most smallsats
actually fly: dump when the wheels pass an upper momentum threshold, stop when they fall
below a lower one. It is given the same safety override, the same simulator and the same
cost function as the learned scheduler, and its two thresholds are tuned by grid search
on the training episodes, so the comparison in ``validation/learned_vs_fixed_ci.py`` is
against a tuned baseline rather than a straw man.

Threshold desaturation with hysteresis is described in Wertz, *Spacecraft Attitude
Determination and Control*, and in Sidi, *Spacecraft Dynamics and Control*, as the
standard momentum-unloading logic.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from . import _validate as _v
from .episodes import Episode, EpisodeMetrics, rollout

__all__ = [
    "Decider",
    "FixedThresholdScheduler",
    "AlwaysOnScheduler",
    "NeverScheduler",
    "tune_fixed_threshold",
    "evaluate_policy",
]

Decider = Callable[[int, NDArray[np.float64]], tuple[bool, float]]
"""A scheduling decision function: ``(window index, features) -> (actuate, confidence)``."""


@dataclass(frozen=True)
class FixedThresholdScheduler:
    """Fixed-threshold momentum unloading with hysteresis.

    Dump when ``|h| / h_env`` rises to ``on_fraction``; keep dumping until it falls below
    ``off_fraction``. Both are dimensionless fractions of the wheel array's conservative
    body envelope.

    The rule uses one feature, ``h_fraction``, and ignores the field geometry entirely.
    That is the point: it is the incumbent, and whether knowing the geometry is worth
    anything is the question the learned scheduler has to answer.
    """

    on_fraction: float = 0.6
    off_fraction: float = 0.3

    def __post_init__(self) -> None:
        on = _v.in_range(self.on_fraction, "on_fraction", 0.0, 1.0)
        off = _v.in_range(self.off_fraction, "off_fraction", 0.0, 1.0)
        if off > on:
            raise ValueError(
                f"off_fraction ({off}) must not exceed on_fraction ({on}); with the "
                "reverse ordering the hysteresis loop never closes"
            )
        object.__setattr__(self, "on_fraction", on)
        object.__setattr__(self, "off_fraction", off)

    def decider(self) -> Decider:
        """Return a fresh stateful decision function for one episode."""
        active = [False]

        def decide(_k: int, features: NDArray[np.float64]) -> tuple[bool, float]:
            frac = float(features[0])
            if active[0]:
                active[0] = frac >= self.off_fraction
            else:
                active[0] = frac >= self.on_fraction
            return active[0], 1.0

        return decide


@dataclass(frozen=True)
class AlwaysOnScheduler:
    """Run the magnetorquers in every window. The upper bound on duty and the lower bound
    on time near saturation; useful as a reference, not as a policy."""

    def decider(self) -> Decider:
        """Return a decision function that always actuates."""
        return lambda _k, _f: (True, 1.0)


@dataclass(frozen=True)
class NeverScheduler:
    """Never actuate except through the safety override. The lower bound on duty."""

    def decider(self) -> Decider:
        """Return a decision function that never actuates."""
        return lambda _k, _f: (False, 1.0)


def evaluate_policy(
    policy: object, episodes: Sequence[Episode]
) -> list[EpisodeMetrics]:
    """Run a policy closed-loop on each episode and return its metrics.

    ``policy`` must expose ``decider()`` returning a fresh :data:`Decider`.
    """
    if not hasattr(policy, "decider"):
        raise TypeError(
            f"policy must expose a decider() method, got {type(policy).__name__}"
        )
    return [
        rollout(ep, policy.decider(), record_history=False).metrics  # type: ignore[attr-defined]
        for ep in episodes
    ]


def tune_fixed_threshold(
    episodes: Sequence[Episode],
    on_grid: Sequence[float] | None = None,
    off_ratio_grid: Sequence[float] | None = None,
) -> tuple[FixedThresholdScheduler, float, list[tuple[float, float, float]]]:
    """Grid-search the two thresholds on ``episodes``, minimising mean episode cost.

    Returns ``(best_scheduler, best_mean_cost, all_results)`` where ``all_results`` is a
    list of ``(on_fraction, off_fraction, mean_cost)``. The grid is coarse on purpose:
    the cost surface in these two parameters is flat enough that a finer grid changes the
    mean cost in the fourth decimal, which is shown in
    ``validation/learned_vs_fixed_ci.py``.
    """
    if len(episodes) == 0:
        raise ValueError("episodes must be non-empty")
    on_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] if on_grid is None else list(on_grid)
    ratios = [0.2, 0.4, 0.6, 0.8] if off_ratio_grid is None else list(off_ratio_grid)
    results: list[tuple[float, float, float]] = []
    best: tuple[float, FixedThresholdScheduler] | None = None
    for on in on_values:
        for ratio in ratios:
            off = on * ratio
            sched = FixedThresholdScheduler(on_fraction=on, off_fraction=off)
            mean_cost = float(np.mean([m.cost for m in evaluate_policy(sched, episodes)]))
            results.append((on, off, mean_cost))
            if best is None or mean_cost < best[0]:
                best = (mean_cost, sched)
    assert best is not None
    return best[1], best[0], results
