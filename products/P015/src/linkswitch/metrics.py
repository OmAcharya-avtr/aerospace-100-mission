"""Aggregate Monte Carlo run metrics with confidence intervals."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import stats

from .simulate import RunMetrics

__all__ = ["Aggregate", "mean_ci", "aggregate_runs", "compare_policies"]


@dataclass(frozen=True)
class Aggregate:
    """Mean and a two-sided confidence interval on the mean."""

    mean: float
    ci_low: float
    ci_high: float
    n: int
    ci_level: float


def mean_ci(values: np.ndarray, ci_level: float = 0.95) -> Aggregate:
    """Two-sided CI on the mean via Student-t (unknown variance, small n).

    half_width = t_(1 - alpha/2, n-1) * sample_std(ddof=1) / sqrt(n)

    Standard interval for the mean of i.i.d. (here: independently seeded)
    replicates. For n=1 the interval collapses to a point (returned as
    ``(mean, mean)``, since a single observation carries no variance
    estimate) — callers requiring a real interval should use n >= 2.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.shape[0] < 1:
        raise ValueError("values must be a non-empty 1-D array")
    if not (0.0 < ci_level < 1.0):
        raise ValueError(f"ci_level must be in (0, 1), got {ci_level!r}")
    n = values.shape[0]
    mean = float(np.mean(values))
    if n == 1:
        return Aggregate(mean=mean, ci_low=mean, ci_high=mean, n=1, ci_level=ci_level)
    std = float(np.std(values, ddof=1))
    if std == 0.0:
        return Aggregate(mean=mean, ci_low=mean, ci_high=mean, n=n, ci_level=ci_level)
    t_crit = float(stats.t.ppf(0.5 + ci_level / 2.0, df=n - 1))
    half_width = t_crit * std / math.sqrt(n)
    return Aggregate(mean=mean, ci_low=mean - half_width, ci_high=mean + half_width,
                      n=n, ci_level=ci_level)


def aggregate_runs(runs: list[RunMetrics], ci_level: float = 0.95) -> dict[str, Aggregate]:
    """Aggregate throughput / outage_fraction / switch_count across runs."""
    if not runs:
        raise ValueError("runs must be non-empty")
    throughput = np.array([r.throughput_mbps for r in runs])
    outage = np.array([r.outage_fraction for r in runs])
    switches = np.array([float(r.switch_count) for r in runs])
    return {
        "throughput_mbps": mean_ci(throughput, ci_level),
        "outage_fraction": mean_ci(outage, ci_level),
        "switch_count": mean_ci(switches, ci_level),
    }


def compare_policies(
    config,
    policy_factories: dict[str, Callable],
    n_steps: int,
    n_reps: int,
    seed0: int,
    ci_level: float = 0.95,
) -> dict[str, dict[str, Aggregate]]:
    """Paired Monte Carlo comparison: same telemetry per rep, every policy.

    Using the *same* telemetry realisation for every policy at a given rep
    (a paired design) removes telemetry-to-telemetry variance from the
    policy-vs-policy comparison, tightening the CIs relative to an
    unpaired comparison at the same ``n_reps``.

    Parameters
    ----------
    policy_factories : mapping name -> zero-argument callable returning a
        fresh policy instance (a ``select_channels(telemetry)`` object).
    """
    from .scenario import generate_telemetry
    from .simulate import simulate_policy

    if not policy_factories:
        raise ValueError("policy_factories must be non-empty")
    if not isinstance(n_reps, (int, np.integer)) or isinstance(n_reps, bool) or n_reps < 1:
        raise ValueError(f"n_reps must be a positive integer, got {n_reps!r}")

    runs: dict[str, list] = {name: [] for name in policy_factories}
    for i in range(n_reps):
        telemetry = generate_telemetry(config, n_steps, seed=seed0 + i)
        for name, factory in policy_factories.items():
            policy = factory()
            select_optical = policy.select_channels(telemetry)
            runs[name].append(simulate_policy(telemetry, select_optical, config))

    return {name: aggregate_runs(r, ci_level=ci_level) for name, r in runs.items()}
