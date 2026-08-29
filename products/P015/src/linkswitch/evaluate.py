"""Seeded Monte Carlo comparison of switching policies, with confidence intervals.

Every policy is scored on the *same* set of seeded traces, so the comparison is
paired: for each seed the per-trial difference between two policies is computed
on an identical channel realisation. Paired intervals are far tighter than
independent ones and are the primary evidence reported.

Interval construction
---------------------
For ``n`` independent seeded trials with per-trial statistic ``x_i``:

    mean +/- t_{1-alpha/2, n-1} * s / sqrt(n)

where ``s`` is the sample standard deviation and ``t`` the Student-t quantile
(Student, "The probable error of a mean", *Biometrika* 6(1), 1908). This is a
normal-theory interval on the mean over trials; it is valid because trials are
independent by construction (disjoint seeds) even though samples *within* a
trial are strongly autocorrelated. It says nothing about the accuracy of the
underlying channel model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.stats import t as student_t

from .metrics import PolicyMetrics, evaluate_selection
from .policies import Policy
from .scenario import HybridLinkScenario, simulate_trace

__all__ = [
    "ConfidenceInterval",
    "PolicyRun",
    "mean_ci",
    "paired_diff_ci",
    "run_monte_carlo",
    "summarise",
]

_METRIC_FIELDS = (
    "throughput_bps",
    "outage_fraction",
    "n_switches",
    "switches_per_s",
    "optical_fraction",
    "guard_fraction",
)


@dataclass(frozen=True)
class ConfidenceInterval:
    """Mean of a per-trial statistic with a two-sided Student-t interval."""

    mean: float
    low: float
    high: float
    n: int
    confidence: float

    @property
    def half_width(self) -> float:
        """Half-width of the interval, in the units of the statistic."""
        return 0.5 * (self.high - self.low)

    def excludes_zero(self) -> bool:
        """True when the interval lies entirely above or entirely below zero."""
        return (self.low > 0.0) or (self.high < 0.0)


@dataclass(frozen=True)
class PolicyRun:
    """Per-trial metrics for one policy over a set of seeds."""

    name: str
    seeds: tuple[int, ...]
    metrics: tuple[PolicyMetrics, ...]

    def field(self, key: str) -> NDArray[np.float64]:
        """Per-trial values of one metric field as a float array."""
        if key not in _METRIC_FIELDS:
            raise KeyError(f"unknown metric {key!r}; available: {_METRIC_FIELDS}")
        return np.array([float(getattr(m, key)) for m in self.metrics], dtype=float)


def mean_ci(x: Sequence[float] | NDArray, confidence: float = 0.95) -> ConfidenceInterval:
    """Student-t confidence interval for the mean of independent trial statistics."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError("need at least 2 trial values to form an interval")
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    n = arr.size
    m = float(arr.mean())
    s = float(arr.std(ddof=1))
    half = float(student_t.ppf(0.5 + confidence / 2.0, n - 1)) * s / np.sqrt(n)
    return ConfidenceInterval(mean=m, low=m - half, high=m + half, n=n, confidence=confidence)


def paired_diff_ci(
    a: Sequence[float] | NDArray, b: Sequence[float] | NDArray, confidence: float = 0.95
) -> ConfidenceInterval:
    """Student-t interval for the mean paired difference ``a - b``.

    ``a`` and ``b`` must be aligned per trial (same seed at the same index).
    """
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if aa.shape != bb.shape:
        raise ValueError(f"paired arrays must have the same shape, got {aa.shape} and {bb.shape}")
    return mean_ci(aa - bb, confidence=confidence)


def run_monte_carlo(
    scenario: HybridLinkScenario,
    policies: Mapping[str, Policy],
    seeds: Iterable[int],
) -> dict[str, PolicyRun]:
    """Score every policy on every seeded trace of ``scenario``.

    Traces are generated once per seed and shared by all policies, guaranteeing
    a paired comparison.
    """
    if not policies:
        raise ValueError("policies must be a non-empty mapping")
    seed_tuple = tuple(int(s) for s in seeds)
    if len(seed_tuple) < 2:
        raise ValueError("need at least 2 seeds for interval estimation")
    if len(set(seed_tuple)) != len(seed_tuple):
        raise ValueError("seeds must be unique")
    collected: dict[str, list[PolicyMetrics]] = {k: [] for k in policies}
    for s in seed_tuple:
        trace = simulate_trace(scenario, s)
        for key, pol in policies.items():
            sel = pol.select(trace)
            collected[key].append(
                evaluate_selection(
                    sel,
                    trace.optical_up,
                    trace.rf_up,
                    rate_optical_bps=scenario.rate_optical_bps,
                    rate_rf_bps=scenario.rate_rf_bps,
                    dt_s=scenario.dt_s,
                    switch_penalty_steps=scenario.switch_penalty_steps,
                )
            )
    return {k: PolicyRun(name=k, seeds=seed_tuple, metrics=tuple(v)) for k, v in collected.items()}


def summarise(
    runs: Mapping[str, PolicyRun], key: str = "throughput_bps", confidence: float = 0.95
) -> dict[str, ConfidenceInterval]:
    """Confidence interval on one metric for every policy."""
    return {name: mean_ci(run.field(key), confidence) for name, run in runs.items()}
