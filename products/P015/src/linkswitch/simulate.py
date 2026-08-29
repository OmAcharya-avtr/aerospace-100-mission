"""Driving loop: apply a policy's channel-selection sequence to telemetry and
score delivered throughput, outage time, and switch count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from .scenario import ScenarioConfig, Telemetry, generate_telemetry

__all__ = ["RunMetrics", "simulate_policy", "run_monte_carlo", "PolicyLike"]


class PolicyLike(Protocol):
    def select_channels(self, telemetry: Telemetry) -> np.ndarray: ...


@dataclass(frozen=True)
class RunMetrics:
    """Per-run scored outcome.

    throughput_mbps : mean delivered rate over the run (Mb/s), averaged over
        all steps including switch-downtime and physical-outage steps
        (i.e. it already reflects both channel outages and switching cost).
    outage_steps : number of steps with zero delivered throughput (either
        the selected channel was physically unavailable, or the step fell
        inside a post-switch downtime window).
    outage_fraction : outage_steps / n_steps.
    switch_count : number of channel changes over the run.
    n_steps : episode length.
    """

    throughput_mbps: float
    outage_steps: int
    outage_fraction: float
    switch_count: int
    n_steps: int


def simulate_policy(
    telemetry: Telemetry, select_optical: np.ndarray, config: ScenarioConfig
) -> RunMetrics:
    """Score one policy decision sequence against one telemetry realisation.

    ``select_optical[t]`` is the policy's *intended* channel at step t
    (True=optical). Actual delivered rate accounts for: (1) physical channel
    availability at that step, and (2) a fixed downtime window of
    ``config.switch_cost.downtime_steps`` steps of zero throughput
    immediately following every channel change.
    """
    select_optical = np.asarray(select_optical, dtype=bool)
    n = telemetry.n_steps
    if select_optical.shape != (n,):
        raise ValueError(
            f"select_optical must have shape ({n},) matching telemetry, got "
            f"{select_optical.shape}"
        )

    opt_rate = config.optical.rate_mbps
    rf_rate = config.rf.rate_mbps
    downtime = int(config.switch_cost.downtime_steps)

    total_rate = 0.0
    outage_steps = 0
    switch_count = 0
    active = select_optical[0]
    downtime_remaining = 0

    for t in range(n):
        desired = select_optical[t]
        if t > 0 and desired != active:
            switch_count += 1
            downtime_remaining = downtime
            active = desired

        if downtime_remaining > 0:
            rate = 0.0
            downtime_remaining -= 1
        elif active:
            rate = opt_rate if telemetry.opt_available[t] else 0.0
        else:
            rate = rf_rate if telemetry.rf_available[t] else 0.0

        total_rate += rate
        if rate == 0.0:
            outage_steps += 1

    return RunMetrics(
        throughput_mbps=total_rate / n,
        outage_steps=outage_steps,
        outage_fraction=outage_steps / n,
        switch_count=switch_count,
        n_steps=n,
    )


def run_monte_carlo(
    config: ScenarioConfig,
    policy_factory: Callable[[], PolicyLike],
    n_steps: int,
    n_reps: int,
    seed0: int,
) -> list[RunMetrics]:
    """Run ``n_reps`` independent seeded episodes and score one policy on each.

    ``policy_factory()`` is called once per rep to obtain a fresh policy
    instance (needed because ``LearnedPolicy`` carries no per-episode state,
    but the factory pattern keeps this function usable for any policy that
    might, e.g. one that re-trains per rep — none currently do).
    """
    if not isinstance(n_reps, (int, np.integer)) or isinstance(n_reps, bool) or n_reps < 1:
        raise ValueError(f"n_reps must be a positive integer, got {n_reps!r}")
    results = []
    for i in range(n_reps):
        telemetry = generate_telemetry(config, n_steps, seed=seed0 + i)
        policy = policy_factory()
        select_optical = policy.select_channels(telemetry)
        results.append(simulate_policy(telemetry, select_optical, config))
    return results
