"""Head-to-head comparison of the fixed patterns, Bayesian-greedy and the learned policy.

The comparison protocol, stated before any number is produced:

* Scenarios are split by ``scenario_id`` into a training set and a held-out
  test set. No test scenario contributes a single training row.
* Training rows come from episodes rolled out under an epsilon-greedy
  Bayesian-greedy behaviour policy restricted to the same shortlist the
  learned policy uses at decision time.
* Every policy is evaluated on exactly the same test scenarios, with exactly
  the same drawn target cells and exactly the same per-episode random seed, so
  the comparison is paired and the difference between two policies is measured
  on identical draws rather than on independent samples.
* The headline statistic is the mean episode time with non-acquisition charged
  at the scenario deadline. It is bounded, it is defined for every episode and
  it penalises a policy that acquires quickly when it acquires at all but
  often does not. The acquisition rate and the median are reported beside it
  because the mean alone can hide either failure mode.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .policy import (
    BayesianGreedyPolicy,
    EpisodeResult,
    FixedPatternPolicy,
    LearnedScanPolicy,
    acquisition_rate,
    collect_training_rows,
    episode_time_standard_error,
    mean_censored_time,
    median_time,
    run_episode,
    sample_target_cells,
)
from .scenarios import Scenario, build_environment

DEFAULT_PATTERN = "spiral"


@dataclass(frozen=True)
class PolicyScore:
    """Aggregate performance of one policy over the held-out episodes.

    ``mean_time_s`` charges a censored episode at its scenario deadline.
    ``standard_error_s`` is the standard error of that mean over episodes.
    ``acquisition_rate`` is the fraction of episodes that acquired before the
    deadline, dimensionless.
    """

    name: str
    mean_time_s: float
    standard_error_s: float
    median_time_s: float
    acquisition_rate: float
    mean_dwells: float
    n_episodes: int


@dataclass(frozen=True)
class PairedDifference:
    """Paired difference between two policies on identical episodes.

    ``mean_difference_s`` is ``mean(a - b)`` over paired episodes, so a
    negative value means policy ``a`` is faster. ``standard_error_s`` is the
    standard error of the paired difference, which is the statistic that
    decides whether a gap is real; an unpaired comparison would be dominated
    by the scenario-to-scenario spread.
    """

    policy_a: str
    policy_b: str
    mean_difference_s: float
    standard_error_s: float
    n_pairs: int

    @property
    def z_score(self) -> float:
        """Paired difference in units of its own standard error, dimensionless."""
        if self.standard_error_s == 0.0 or math.isnan(self.standard_error_s):
            return float("nan")
        return self.mean_difference_s / self.standard_error_s


def _episode_times(results: list[EpisodeResult]) -> np.ndarray:
    return np.asarray([r.time_s for r in results], dtype=float)


def score_policy(name: str, results: list[EpisodeResult]) -> PolicyScore:
    """Aggregate a list of episode results into a :class:`PolicyScore`."""
    return PolicyScore(
        name=name,
        mean_time_s=mean_censored_time(results),
        standard_error_s=episode_time_standard_error(results),
        median_time_s=median_time(results),
        acquisition_rate=acquisition_rate(results),
        mean_dwells=float(np.mean([r.dwells for r in results])),
        n_episodes=len(results),
    )


def paired_difference(
    name_a: str, results_a: list[EpisodeResult], name_b: str, results_b: list[EpisodeResult]
) -> PairedDifference:
    """Paired mean difference ``a - b`` over episodes in the same order."""
    if len(results_a) != len(results_b):
        raise ValueError("paired comparison requires the same number of episodes")
    delta = _episode_times(results_a) - _episode_times(results_b)
    se = float(delta.std(ddof=1) / math.sqrt(delta.size)) if delta.size > 1 else float("nan")
    return PairedDifference(
        policy_a=name_a,
        policy_b=name_b,
        mean_difference_s=float(delta.mean()),
        standard_error_s=se,
        n_pairs=int(delta.size),
    )


def build_training_set(
    scenarios: list[Scenario],
    episodes_per_scenario: int = 2,
    epsilon: float = 0.35,
    shortlist: int = 12,
    seed: int = 20261016,
    max_rows: int = 12000,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll out behaviour episodes over the training scenarios and return the rows.

    ``max_rows`` caps the fit so the two-core compute budget is met; when the
    cap binds, rows are subsampled with a fixed seed, which is recorded in
    MODEL_CARD.md together with the row count actually used.
    """
    rng = np.random.default_rng(int(seed))
    blocks_x: list[np.ndarray] = []
    blocks_y: list[np.ndarray] = []
    for scenario in scenarios:
        environment, _orders = build_environment(scenario)
        behaviour = BayesianGreedyPolicy(
            epsilon=float(epsilon),
            rng=np.random.default_rng(int(seed) + 7919 * scenario.scenario_id),
            explore_shortlist=int(shortlist),
        )
        targets = sample_target_cells(
            environment.grid,
            int(episodes_per_scenario),
            np.random.default_rng(int(seed) + 104729 * scenario.scenario_id),
        )
        x, y = collect_training_rows(
            environment,
            behaviour,
            targets,
            np.random.default_rng(int(seed) + 1299709 * scenario.scenario_id),
        )
        if x.shape[0]:
            blocks_x.append(x)
            blocks_y.append(y)
    if not blocks_x:
        raise ValueError("no training rows were produced")
    features = np.vstack(blocks_x)
    target = np.concatenate(blocks_y)
    if features.shape[0] > int(max_rows):
        keep = rng.choice(features.shape[0], size=int(max_rows), replace=False)
        keep.sort()
        features = features[keep]
        target = target[keep]
    return features, target


def evaluate_policies(
    scenarios: list[Scenario],
    learned: LearnedScanPolicy | None,
    targets_per_scenario: int = 6,
    seed: int = 20261017,
    pattern: str = DEFAULT_PATTERN,
    include_all_patterns: bool = False,
) -> dict[str, list[EpisodeResult]]:
    """Run every policy on the same episodes and return the per-policy results.

    The returned lists are aligned: index ``i`` of every list is the same
    scenario and the same target cell, simulated with the same per-episode
    seed, so :func:`paired_difference` is meaningful.
    """
    results: dict[str, list[EpisodeResult]] = {}
    for scenario in scenarios:
        environment, orders = build_environment(scenario)
        if pattern not in orders:
            raise ValueError(f"unknown pattern {pattern!r}; expected one of {sorted(orders)}")
        targets = sample_target_cells(
            environment.grid,
            int(targets_per_scenario),
            np.random.default_rng(int(seed) + 104729 * scenario.scenario_id),
        )
        policies: dict[str, object] = {}
        if include_all_patterns:
            for name, order in orders.items():
                policies[f"fixed_{name}"] = FixedPatternPolicy(order, name=f"fixed_{name}")
        else:
            policies[f"fixed_{pattern}"] = FixedPatternPolicy(
                orders[pattern], name=f"fixed_{pattern}"
            )
        policies["bayesian_greedy"] = BayesianGreedyPolicy()
        if learned is not None:
            policies["learned"] = learned
        for index, target in enumerate(targets):
            for name, policy in policies.items():
                rng = np.random.default_rng(
                    int(seed) + 15485863 * scenario.scenario_id + 31 * index
                )
                outcome = run_episode(environment, policy, int(target), rng)
                results.setdefault(name, []).append(outcome)
    return results


def summarise(results: dict[str, list[EpisodeResult]]) -> dict[str, PolicyScore]:
    """Score every policy in a result dictionary."""
    return {name: score_policy(name, value) for name, value in results.items()}


def winner(scores: dict[str, PolicyScore]) -> str:
    """Name of the policy with the smallest mean censored time."""
    if not scores:
        raise ValueError("scores must be non-empty")
    return min(scores.values(), key=lambda s: s.mean_time_s).name
