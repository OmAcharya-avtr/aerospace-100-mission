r"""Running every strategy over every instance at a common budget, and the
per-instance accounting that goes with it.

The comparison rule
-------------------
For a comparison between strategies to mean anything, all of them must see the
same instances at the same budget with the same number of repeats, and the
``r``-th repeat of every strategy on every instance must use the **same seed**,
derived as ``base_seed + r``. Using a different seed per strategy would make the
comparison noisier for no reason; using the same seed does **not** couple the
strategies, because each consumes its own independent generator.

What is reported, and in which order
------------------------------------
Per instance first, aggregate second -- deliberately. An aggregate hides exactly
the thing a falsification user needs to know, which is whether the method that
wins on average is the one that loses on their instance.
:meth:`BenchmarkReport.baseline_wins` returns the instances where the baseline
beat a given strategy, and the README is required to quote it.

Compute budget
--------------
``instances x strategies x repeats`` searches, each of at most ``budget``
simulations at about 0.25 ms per simulation plus, for the surrogate, a forest
refit every ``refit_every`` simulations. The measured wall clock of the shipped
configuration is recorded in ``validation/validate_benchmark_output.txt`` along
with the container's core count. It is a wall clock on a shared 2-core
container, not a hardware characteristic of anything.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from .curves import (
    DEFAULT_BOOTSTRAP,
    aggregate_curve,
    area_under_curve,
    bootstrap_aggregate_band,
    bootstrap_band,
    efficiency_curve,
    median_first_violation,
    success_rate,
)
from .instances import Instance, suite
from .search import BASELINE, STRATEGIES, SearchResult, strategy


@dataclass(frozen=True)
class CellSummary:
    """Summary of one (instance, strategy) cell of the benchmark.

    Attributes
    ----------
    instance_id, strategy_name:
        The cell's coordinates.
    budget, repeats:
        Simulations allowed per run, and runs performed.
    first_violations:
        One entry per run; ``None`` for a run that found nothing.
    success_rate:
        Fraction of runs that found a violation within the budget.
    median_simulations:
        Median simulations to the first violation, or ``None`` when censored.
    mean_curve_probability:
        Mean of the efficiency curve over ``1..budget``; a compact ordering
        statistic, not a classification metric.
    best_robustness:
        Lowest robustness any run on this cell reached.
    """

    instance_id: str
    strategy_name: str
    budget: int
    repeats: int
    first_violations: tuple[int | None, ...]
    success_rate: float
    median_simulations: float | None
    mean_curve_probability: float
    best_robustness: float

    def curve(self) -> np.ndarray:
        """Efficiency curve for this cell, shape ``(budget,)``."""
        return efficiency_curve(self.first_violations, self.budget)

    def band(
        self, n_boot: int = DEFAULT_BOOTSTRAP, alpha: float = 0.05, seed: int = 0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pointwise bootstrap band for this cell's curve."""
        return bootstrap_band(self.first_violations, self.budget, n_boot, alpha, seed)


@dataclass(frozen=True)
class BenchmarkReport:
    """Every cell of one benchmark run, plus the aggregate views.

    Attributes
    ----------
    budget, repeats, base_seed:
        The configuration the whole report was produced under.
    instance_ids, strategy_names:
        Rows and columns, in the order they were run.
    cells:
        ``(instance_id, strategy_name) -> CellSummary``.
    wall_clock_seconds:
        Measured wall clock of the whole run, on a shared 2-core container.
        Reported because the compute budget is part of the result, and
        explicitly **not** a hardware characteristic.
    """

    budget: int
    repeats: int
    base_seed: int
    instance_ids: tuple[str, ...]
    strategy_names: tuple[str, ...]
    cells: Mapping[tuple[str, str], CellSummary]
    wall_clock_seconds: float = field(default=0.0)

    def cell(self, instance_id: str, strategy_name: str) -> CellSummary:
        """Look up one cell.

        Raises
        ------
        KeyError
            If that cell was not run.
        """
        try:
            return self.cells[(instance_id, strategy_name)]
        except KeyError:
            raise KeyError(
                f"no cell for instance {instance_id!r} and strategy {strategy_name!r}; "
                f"this report covers {list(self.instance_ids)} x {list(self.strategy_names)}"
            ) from None

    def per_instance(self, strategy_name: str) -> Mapping[str, tuple[int | None, ...]]:
        """First-violation lists for one strategy, keyed by instance."""
        return {
            iid: self.cell(iid, strategy_name).first_violations for iid in self.instance_ids
        }

    def aggregate_curve(self, strategy_name: str) -> np.ndarray:
        """Unweighted mean of this strategy's per-instance curves."""
        return aggregate_curve(self.per_instance(strategy_name), self.budget)

    def aggregate_band(
        self, strategy_name: str, n_boot: int = DEFAULT_BOOTSTRAP, alpha: float = 0.05,
        seed: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Stratified bootstrap band for this strategy's aggregate curve."""
        return bootstrap_aggregate_band(
            self.per_instance(strategy_name), self.budget, n_boot, alpha, seed
        )

    def aggregate_mean_probability(self, strategy_name: str) -> float:
        """Mean of this strategy's aggregate curve over ``1..budget``."""
        return float(np.mean(self.aggregate_curve(strategy_name)))

    def baseline_wins(
        self, strategy_name: str, baseline: str = BASELINE
    ) -> tuple[str, ...]:
        """Instances where ``baseline`` beat ``strategy_name``.

        "Beat" is measured on the mean curve probability -- the average
        probability of having found a violation over the whole budget -- because
        it uses the entire curve rather than one point on it. Ties are not wins.
        """
        out = []
        for iid in self.instance_ids:
            mine = self.cell(iid, strategy_name).mean_curve_probability
            theirs = self.cell(iid, baseline).mean_curve_probability
            if theirs > mine:
                out.append(iid)
        return tuple(out)

    def hardest_instance(self, baseline: str = BASELINE) -> str:
        """The instance the baseline does worst on, by mean curve probability."""
        return min(
            self.instance_ids,
            key=lambda iid: self.cell(iid, baseline).mean_curve_probability,
        )


def run_cell(
    instance: Instance,
    strategy_name: str,
    budget: int,
    repeats: int,
    base_seed: int,
    options: Mapping[str, object] | None = None,
) -> tuple[CellSummary, tuple[SearchResult, ...]]:
    """Run one (instance, strategy) cell for ``repeats`` seeds.

    Parameters
    ----------
    instance:
        The instance to falsify.
    strategy_name:
        A key of :data:`falsifyloop.search.STRATEGIES`.
    budget:
        Simulations per run, at least 1.
    repeats:
        Independent seeded runs, at least 2 (a single run has no spread and a
        curve from it is a step function, which would be reported as if it were
        an estimate).
    base_seed:
        Seeds used are ``base_seed .. base_seed + repeats - 1``.
    options:
        Extra keyword arguments passed to the strategy.

    Returns
    -------
    tuple
        The summary and the individual :class:`SearchResult` objects, so a
        caller that wants the raw histories is not forced to re-run.
    """
    if repeats < 2:
        raise ValueError(f"repeats must be at least 2 for a curve to be an estimate, got {repeats}")
    if budget < 1:
        raise ValueError(f"budget must be at least 1, got {budget}")
    fn = strategy(strategy_name)
    kwargs = dict(options or {})
    results = tuple(
        fn(instance, budget, base_seed + r, **kwargs) for r in range(int(repeats))
    )
    first = tuple(r.first_violation for r in results)
    summary = CellSummary(
        instance_id=instance.identifier,
        strategy_name=strategy_name,
        budget=int(budget),
        repeats=int(repeats),
        first_violations=first,
        success_rate=success_rate(first, budget),
        median_simulations=median_first_violation(first, budget),
        mean_curve_probability=area_under_curve(first, budget),
        best_robustness=float(min(r.best_robustness for r in results)),
    )
    return summary, results


def run_benchmark(
    instances: Sequence[Instance] | None = None,
    strategy_names: Sequence[str] | None = None,
    budget: int = 100,
    repeats: int = 30,
    base_seed: int = 1000,
    options: Mapping[str, Mapping[str, object]] | None = None,
) -> BenchmarkReport:
    """Run the full instance-by-strategy grid and summarise it.

    Parameters
    ----------
    instances:
        Instances to run; the whole shipped suite if omitted.
    strategy_names:
        Strategies to run; all registered ones if omitted, baseline first.
    budget:
        Simulations per run.
    repeats:
        Seeded runs per cell.
    base_seed:
        First seed; repeat ``r`` of every cell uses ``base_seed + r``.
    options:
        ``strategy_name -> kwargs`` passed through to that strategy.

    Returns
    -------
    BenchmarkReport
        With ``wall_clock_seconds`` measured over the whole grid.
    """
    import time

    chosen = tuple(instances) if instances is not None else suite()
    names = tuple(strategy_names) if strategy_names is not None else tuple(STRATEGIES)
    if not chosen:
        raise ValueError("need at least one instance")
    if not names:
        raise ValueError("need at least one strategy")
    unknown = [n for n in names if n not in STRATEGIES]
    if unknown:
        raise KeyError(f"unknown strategies {unknown}; registered are {list(STRATEGIES)}")

    per_strategy_options = dict(options or {})
    cells: dict[tuple[str, str], CellSummary] = {}
    started = time.perf_counter()
    for inst in chosen:
        for name in names:
            summary, _ = run_cell(
                inst,
                name,
                budget,
                repeats,
                base_seed,
                per_strategy_options.get(name),
            )
            cells[(inst.identifier, name)] = summary
    elapsed = time.perf_counter() - started
    return BenchmarkReport(
        budget=int(budget),
        repeats=int(repeats),
        base_seed=int(base_seed),
        instance_ids=tuple(i.identifier for i in chosen),
        strategy_names=names,
        cells=cells,
        wall_clock_seconds=float(elapsed),
    )
