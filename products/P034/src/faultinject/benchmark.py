"""Same-budget benchmark of the search strategies, with confidence intervals.

Protocol
--------
``n_pools`` independent case pools are built (each covering every coverage cell
``replicates`` times) and the severity of **every** case in every pool is
computed once, so all strategies see the same ground truth and the comparison
costs no extra target executions.  Each strategy is then run ``n_seeds`` times
per pool with distinct strategy seeds.  A run's score is the number of severe
cases (severity >= ``SEVERE_THRESHOLD``) among its ``budget`` executions.

Intervals
---------
Two intervals are reported, because they answer different questions.

* A **percentile bootstrap** 95 % interval on each strategy's mean score,
  resampling runs with replacement.  This is the interval the mission's
  decision rule refers to: if the uniform-random interval contains the learned
  strategy's mean score, the verdict is ``no measurable advantage``.
* A **paired** bootstrap 95 % interval on the per-run difference
  ``learned - uniform_random``, which is tighter because every run of the two
  strategies shares a pool and a seed.  Both are printed; neither is allowed to
  override the other, and the verdict function uses the first.

Nothing here retunes a strategy after seeing its score.  The parameters below
are the ones the benchmark was first run with.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from . import search as search_mod
from .campaign import FaultCase, build_pool, evaluate_pool
from .search import SearchResult
from .severity import SEVERE_THRESHOLD

DEFAULT_BUDGET = 60
DEFAULT_WARMUP = 32
DEFAULT_N_POOLS = 3
DEFAULT_N_SEEDS = 8
DEFAULT_REPLICATES = 2
BOOTSTRAP_RESAMPLES = 5000


@dataclass
class StrategyStats:
    """Scores of one strategy over all runs, with its bootstrap interval."""

    strategy: str
    scores: list[float]
    mean: float
    lo: float
    hi: float
    mean_severity: float
    coverage: float

    def to_dict(self) -> dict[str, object]:
        return {
            "strategy": self.strategy,
            "n_runs": len(self.scores),
            "mean_severe_found": self.mean,
            "ci95": [self.lo, self.hi],
            "mean_severity_of_executed": self.mean_severity,
            "mean_final_coverage": self.coverage,
        }


@dataclass
class BenchmarkReport:
    """Everything the benchmark produced."""

    budget: int
    pool_size: int
    severe_in_pool: float
    stats: dict[str, StrategyStats]
    paired: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    calibration: tuple[float, float] = (float("nan"), float("nan"))
    importances: dict[str, float] = field(default_factory=dict)

    def verdict(self, learned: str = "learned", baseline: str = "uniform_random") -> str:
        """``no measurable advantage`` or ``measurable advantage``.

        The rule, verbatim from the specification: if the baseline's confidence
        interval contains the learned strategy's mean score, the result is
        ``no measurable advantage``.  The comparison is also made in the
        opposite direction, so a baseline that *beats* the learned model is
        reported as such rather than as a tie.
        """
        a = self.stats[learned]
        b = self.stats[baseline]
        if b.lo <= a.mean <= b.hi:
            return "no measurable advantage"
        if a.mean > b.hi:
            return "measurable advantage"
        return f"baseline {baseline} outperforms {learned}"

    def to_dict(self) -> dict[str, object]:
        return {
            "budget": self.budget,
            "pool_size": self.pool_size,
            "severe_fraction_in_pool": self.severe_in_pool,
            "severe_threshold": SEVERE_THRESHOLD,
            "strategies": {k: v.to_dict() for k, v in self.stats.items()},
            "paired_differences": {
                k: {"mean": v[0], "ci95": [v[1], v[2]]} for k, v in self.paired.items()
            },
            "uncertainty_interval_coverage": self.calibration[0],
            "uncertainty_interval_width": self.calibration[1],
            "verdict_vs_uniform_random": self.verdict(),
        }


def bootstrap_ci(
    values: Sequence[float], resamples: int = BOOTSTRAP_RESAMPLES, seed: int = 20261005
) -> tuple[float, float, float]:
    """``(mean, 2.5th, 97.5th)`` percentile bootstrap of the mean of ``values``."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("cannot bootstrap an empty sample")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(resamples, arr.size))
    means = arr[idx].mean(axis=1)
    return float(arr.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def run_benchmark(
    budget: int = DEFAULT_BUDGET,
    n_pools: int = DEFAULT_N_POOLS,
    n_seeds: int = DEFAULT_N_SEEDS,
    replicates: int = DEFAULT_REPLICATES,
    warmup: int = DEFAULT_WARMUP,
    kappa: float = 0.0,
    progress: bool = False,
) -> BenchmarkReport:
    """Run the full comparison and return the report.

    Compute: ``n_pools * replicates * 248`` target executions for the oracle
    tables plus the strategy runs, which are table lookups. With the defaults
    that is 1488 executions, about 3 s on one core.
    """
    pools: list[tuple[FaultCase, ...]] = []
    oracles: list[tuple[float, ...]] = []
    for p in range(n_pools):
        pool = build_pool(pool_seed=p + 1, replicates=replicates)
        pools.append(pool)
        oracles.append(evaluate_pool(pool))
        if progress:  # pragma: no cover - diagnostic only
            print(f"  pool {p + 1}/{n_pools} evaluated ({len(pool)} cases)", flush=True)

    runs: dict[str, list[SearchResult]] = {s: [] for s in search_mod.STRATEGIES}
    last_model = None
    for p, (pool, sev) in enumerate(zip(pools, oracles, strict=True)):

        def oracle(i: int, _sev: tuple[float, ...] = sev) -> float:
            return _sev[i]

        for s in range(n_seeds):
            rng_seed = 1000 * (p + 1) + s
            runs["uniform_random"].append(
                search_mod.uniform_random(pool, oracle, budget, np.random.default_rng(rng_seed))
            )
            runs["coverage_greedy"].append(
                search_mod.coverage_greedy(pool, oracle, budget, np.random.default_rng(rng_seed))
            )
            runs["kind_mean"].append(
                search_mod.kind_mean(
                    pool, oracle, budget, np.random.default_rng(rng_seed), warmup=warmup
                )
            )
            res, model = search_mod.learned(
                pool,
                oracle,
                budget,
                np.random.default_rng(rng_seed),
                warmup=warmup,
                kappa=kappa,
            )
            runs["learned"].append(res)
            last_model = model
        if progress:  # pragma: no cover - diagnostic only
            print(f"  pool {p + 1}/{n_pools} searched", flush=True)

    stats: dict[str, StrategyStats] = {}
    for name, rs in runs.items():
        scores = [float(r.n_severe) for r in rs]
        mean, lo, hi = bootstrap_ci(scores)
        stats[name] = StrategyStats(
            strategy=name,
            scores=scores,
            mean=mean,
            lo=lo,
            hi=hi,
            mean_severity=float(np.mean([r.mean_severity for r in rs])),
            coverage=float(np.mean([r.coverage_curve[-1] for r in rs])),
        )

    paired = {}
    base = [float(r.n_severe) for r in runs["uniform_random"]]
    for name in ("coverage_greedy", "kind_mean", "learned"):
        diffs = [a - b for a, b in zip(
            [float(r.n_severe) for r in runs[name]], base, strict=True
        )]
        paired[f"{name} - uniform_random"] = bootstrap_ci(diffs)

    all_sev = [s for sev in oracles for s in sev]
    severe_frac = float(np.mean([1.0 if s >= SEVERE_THRESHOLD else 0.0 for s in all_sev]))

    calibration = (float("nan"), float("nan"))
    importances: dict[str, float] = {}
    if last_model is not None:
        calibration = last_model.interval_coverage(list(pools[-1]), list(oracles[-1]))
        importances = {k: float(v) for k, v in list(last_model.feature_importances().items())[:8]}

    return BenchmarkReport(
        budget=budget,
        pool_size=len(pools[0]),
        severe_in_pool=severe_frac,
        stats=stats,
        paired=paired,
        calibration=calibration,
        importances=importances,
    )
