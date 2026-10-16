"""Head-to-head benchmark: coverage achieved per scenario executed, with the spread.

The question the benchmark answers is the only one that decides whether the
learned gap finder is worth running: starting from a partial campaign, if you
are allowed ``m`` more scenarios drawn from the same candidate pool, how much
t-way coverage does each strategy buy per scenario?

Protocol, fixed and seeded
--------------------------
For each seed:

1. draw a starting campaign of ``n_start`` scenarios uniformly without
   replacement from the executed campaign (the same draw for every strategy at
   that seed, so the comparison is paired);
2. draw a candidate pool of ``pool_size`` scenarios uniformly without
   replacement from the supplied pool (again shared across strategies);
3. let each strategy choose ``n_steps`` scenarios in order from that pool;
4. record t-way coverage after each addition, giving a curve of length
   ``n_steps + 1``.

Reported per strategy: the mean curve, the standard deviation across seeds, and
the minimum and maximum curves. **The spread is reported because the mean alone
is not a result**: on a small number of seeds two strategies whose mean curves
differ by less than a standard deviation have not been separated by the
experiment, and the tables and plots in this repository therefore always carry
the spread next to the mean.

The headline number is the **coverage gain per scenario**

    gain = (C_t after n_steps - C_t at start) / n_steps                  (6)

dimensionless, in units of coverage fraction per scenario executed. Paired
differences between strategies are reported at the same seeds, since step 1 and
step 2 are shared.

Units: all coverage figures are dimensionless fractions in [0, 1].
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .combinatorial import count_tway_tuples
from .space import ScenarioSpace
from .strategies import SelectionContext, Strategy

__all__ = [
    "BenchmarkResult",
    "StrategyCurves",
    "coverage_benchmark",
    "incremental_coverage_curve",
    "incremental_occupancy_curve",
]


def _covered_keys(space: ScenarioSpace, encoded: np.ndarray, t: int) -> set[tuple[int, ...]]:
    from itertools import combinations

    k = len(space)
    keys: set[tuple[int, ...]] = set()
    for s_index, subset in enumerate(combinations(range(k), t)):
        for row in encoded:
            keys.add((s_index, *(int(row[i]) for i in subset)))
    return keys


def incremental_coverage_curve(
    space: ScenarioSpace,
    start_encoded: np.ndarray,
    added_encoded: np.ndarray,
    t: int = 2,
) -> np.ndarray:
    """t-way coverage fraction after 0, 1, ... len(added) additions.

    Args:
        space: the declared space.
        start_encoded: ``(n_start, k)`` integer level indices already executed.
        added_encoded: ``(m, k)`` integer level indices in the order a strategy
            chose them.
        t: interaction strength.

    Returns:
        ``(m + 1,)`` array of coverage fractions, dimensionless in [0, 1]. The
        covered counts behind it are exact integers from set arithmetic.
    """
    from itertools import combinations

    total = count_tway_tuples(space.level_counts, t)
    subsets = list(combinations(range(len(space)), t))
    covered = _covered_keys(space, start_encoded, t)
    curve = np.empty(len(added_encoded) + 1, dtype=float)
    curve[0] = len(covered) / total
    for step, row in enumerate(added_encoded, start=1):
        for s_index, subset in enumerate(subsets):
            covered.add((s_index, *(int(row[i]) for i in subset)))
        curve[step] = len(covered) / total
    return curve


def incremental_occupancy_curve(
    space: ScenarioSpace,
    start_encoded: np.ndarray,
    added_encoded: np.ndarray,
    parameters: Sequence[str] | None = None,
) -> np.ndarray:
    """Grid-cell occupancy fraction after 0, 1, ... len(added) additions.

    The second metric, reported because t-way coverage is exactly what the
    coverage-greedy reference optimises, and a density model has no reason to
    be good at it. Occupancy over the continuous parameters is the measure the
    gap finder's own view of the space is closest to, so running the benchmark
    on both metrics is the fair comparison rather than the convenient one.

    Args:
        space: the declared space.
        start_encoded: ``(n_start, k)`` integer level indices.
        added_encoded: ``(m, k)`` integer level indices in selection order.
        parameters: grid parameters; defaults to the continuous ones.

    Returns:
        ``(m + 1,)`` array of occupancy fractions, dimensionless in [0, 1].
    """
    if parameters is None:
        chosen = tuple(p.name for p in space.parameters if p.kind == "continuous")
        if not chosen:
            raise ValueError("the space declares no continuous parameters")
    else:
        chosen = tuple(parameters)
    columns = [space.index_of(name) for name in chosen]
    n_cells = math.prod(space.parameters[c].n_levels for c in columns)
    cells = {tuple(int(row[c]) for c in columns) for row in start_encoded}
    curve = np.empty(len(added_encoded) + 1, dtype=float)
    curve[0] = len(cells) / n_cells
    for step, row in enumerate(added_encoded, start=1):
        cells.add(tuple(int(row[c]) for c in columns))
        curve[step] = len(cells) / n_cells
    return curve


@dataclass(frozen=True)
class StrategyCurves:
    """Per-strategy benchmark curves over seeds.

    Attributes:
        name: strategy name.
        curves: ``(n_seeds, n_steps + 1)`` coverage fractions.
        seconds: total wall-clock seconds spent in this strategy's selections.
    """

    name: str
    curves: np.ndarray
    seconds: float

    @property
    def mean(self) -> np.ndarray:
        """Mean curve over seeds."""
        return self.curves.mean(axis=0)

    @property
    def sd(self) -> np.ndarray:
        """Sample standard deviation over seeds (ddof=1); zeros for one seed."""
        if self.curves.shape[0] < 2:
            return np.zeros(self.curves.shape[1])
        return self.curves.std(axis=0, ddof=1)

    @property
    def minimum(self) -> np.ndarray:
        """Per-step minimum over seeds."""
        return self.curves.min(axis=0)

    @property
    def maximum(self) -> np.ndarray:
        """Per-step maximum over seeds."""
        return self.curves.max(axis=0)

    @property
    def final(self) -> np.ndarray:
        """Final coverage per seed, ``(n_seeds,)``."""
        return self.curves[:, -1]

    @property
    def gain_per_scenario(self) -> np.ndarray:
        """Equation (6) per seed, ``(n_seeds,)``, coverage fraction per scenario."""
        steps = self.curves.shape[1] - 1
        return (self.curves[:, -1] - self.curves[:, 0]) / steps

    def summary(self) -> dict[str, float]:
        """Mean, sd, min and max of the final coverage and of equation (6)."""
        gain = self.gain_per_scenario
        return {
            "final_mean": float(self.final.mean()),
            "final_sd": float(self.final.std(ddof=1)) if len(self.final) > 1 else 0.0,
            "final_min": float(self.final.min()),
            "final_max": float(self.final.max()),
            "gain_mean": float(gain.mean()),
            "gain_sd": float(gain.std(ddof=1)) if len(gain) > 1 else 0.0,
            "seconds": float(self.seconds),
        }


@dataclass(frozen=True)
class BenchmarkResult:
    """Result of :func:`coverage_benchmark`.

    Attributes:
        t: interaction strength the curves measure.
        n_start: starting campaign size per seed.
        n_steps: scenarios added per seed.
        pool_size: candidate pool size per seed.
        seeds: the seeds used.
        by_strategy: name -> :class:`StrategyCurves`.
    """

    t: int
    n_start: int
    n_steps: int
    pool_size: int
    seeds: tuple[int, ...]
    by_strategy: Mapping[str, StrategyCurves]

    def ranking(self) -> tuple[tuple[str, float, float], ...]:
        """``(name, mean gain, sd of gain)`` ordered by mean gain, best first."""
        rows = [
            (name, float(c.gain_per_scenario.mean()), float(c.gain_per_scenario.std(ddof=1)))
            if c.curves.shape[0] > 1
            else (name, float(c.gain_per_scenario.mean()), 0.0)
            for name, c in self.by_strategy.items()
        ]
        return tuple(sorted(rows, key=lambda r: -r[1]))

    def paired_difference(self, left: str, right: str) -> dict[str, float]:
        """Paired per-seed difference in final coverage, ``left - right``.

        Returns the mean and standard deviation of the per-seed difference and
        the win/tie/loss counts. No p-value is quoted: the per-seed
        differences here are not independent of the shared starting campaign
        and the shared pool in any way a two-sample test would model, and the
        win/tie/loss split plus the spread says what the experiment supports.
        A mean difference smaller than its own standard deviation has not
        separated the two strategies, whatever its sign.
        """
        a = self.by_strategy[left].final
        b = self.by_strategy[right].final
        diff = a - b
        return {
            "mean": float(diff.mean()),
            "sd": float(diff.std(ddof=1)) if len(diff) > 1 else 0.0,
            "left_ahead_on": int(np.count_nonzero(diff > 0.0)),
            "tied_on": int(np.count_nonzero(diff == 0.0)),
            "right_ahead_on": int(np.count_nonzero(diff < 0.0)),
            "n_seeds": int(len(diff)),
        }


def coverage_benchmark(
    space: ScenarioSpace,
    executed: Sequence[Mapping[str, Any]],
    pool: Sequence[Mapping[str, Any]],
    strategies: Sequence[Strategy],
    n_start: int = 40,
    n_steps: int = 40,
    pool_size: int = 200,
    seeds: Sequence[int] = tuple(range(12)),
    t: int = 2,
    metric: str = "tway",
) -> BenchmarkResult:
    """Run the paired, seeded coverage-per-scenario benchmark.

    Args:
        space: the declared space.
        executed: the finished campaign, used as the pot the starting campaign
            is drawn from.
        pool: candidate scenarios that could be executed next.
        strategies: the strategies to compare.
        n_start: starting campaign size per seed.
        n_steps: scenarios each strategy may add.
        pool_size: candidate pool size per seed.
        seeds: the seeds; each gives one paired replicate.
        t: interaction strength for the coverage metric.
        metric: ``"tway"`` for t-way combinatorial coverage (the default) or
            ``"occupancy"`` for grid-cell occupancy over the continuous
            parameters. Both are reported in ``validation/``.

    Returns:
        A :class:`BenchmarkResult` holding one curve per strategy per seed.
    """
    import time

    if n_start < 2:
        raise ValueError("n_start must be at least 2, since the density model needs two points")
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")
    if pool_size < n_steps:
        raise ValueError("pool_size must be at least n_steps")
    if len(executed) < n_start:
        raise ValueError(f"executed campaign has {len(executed)} scenarios, need {n_start}")
    if len(pool) < pool_size:
        raise ValueError(f"candidate pool has {len(pool)} scenarios, need {pool_size}")
    if not strategies:
        raise ValueError("pass at least one strategy")
    if metric not in ("tway", "occupancy"):
        raise ValueError("metric must be 'tway' or 'occupancy'")

    curves: dict[str, list[np.ndarray]] = {s.name: [] for s in strategies}
    seconds: dict[str, float] = {s.name: 0.0 for s in strategies}

    for seed in seeds:
        rng = np.random.default_rng(seed)
        start_idx = rng.choice(len(executed), size=n_start, replace=False)
        pool_idx = rng.choice(len(pool), size=pool_size, replace=False)
        start_records = [executed[int(i)] for i in start_idx]
        pool_records = [pool[int(i)] for i in pool_idx]
        context = SelectionContext(
            space=space,
            executed_encoded=space.encode_many(start_records),
            executed_norm=space.normalise(start_records),
            candidate_encoded=space.encode_many(pool_records),
            candidate_norm=space.normalise(pool_records),
        )
        for strategy in strategies:
            strategy_rng = np.random.default_rng(seed * 1000 + 7)
            t0 = time.perf_counter()
            picks = strategy.select(context, n_steps, strategy_rng)
            seconds[strategy.name] += time.perf_counter() - t0
            added = context.candidate_encoded[list(picks)]
            if metric == "tway":
                curve = incremental_coverage_curve(space, context.executed_encoded, added, t=t)
            else:
                curve = incremental_occupancy_curve(space, context.executed_encoded, added)
            curves[strategy.name].append(curve)

    return BenchmarkResult(
        t=t,
        n_start=n_start,
        n_steps=n_steps,
        pool_size=pool_size,
        seeds=tuple(int(s) for s in seeds),
        by_strategy={
            name: StrategyCurves(name=name, curves=np.vstack(rows), seconds=seconds[name])
            for name, rows in curves.items()
        },
    )
