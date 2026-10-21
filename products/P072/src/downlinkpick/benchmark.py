"""Scoring harness: value sources and size estimators, through the same solver.

The design rule of this module is that only one thing changes between arms. A
value-source arm changes the value mapping and nothing else -- same inventory,
same plan, same sizes, same solver -- so a difference in realised value is
attributable to the value source. A size-estimator arm changes the size mapping
and nothing else. Without that discipline an "AI beats baseline" number is a
comparison of two whole pipelines and says nothing about the model.

How an arm is scored. Each arm produces values, the solver selects against
them, and the resulting selection is scored against the *ground-truth* values
through :meth:`downlinkpick.selection.SelectionProblem.score`. The oracle arm
selects against the ground truth itself, so its score is the best achievable
under that solver, and

    regret = oracle score - arm score                                       (8)

in value units, with the fractional form ``regret / oracle score``. Regret, not
regression error, is the decision-relevant metric: an estimator can have a large
mean absolute error and zero regret if the error never crosses a selection
boundary, and the whole point of measuring end to end is to find out which one
this is.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .contacts import feasibility_slack_bits
from .dataset import GeneratedInstance
from .selection import Selection, SelectionProblem, sizes_from_ratios
from .valuemodel import spearman

Selector = Callable[[SelectionProblem], Selection]

#: A value-source arm: name, and a function from instance to value mapping.
ValueSource = Callable[[GeneratedInstance], Mapping[str, float]]

#: A size arm: name, and a function from instance to compression ratios.
RatioSource = Callable[[GeneratedInstance], Mapping[str, float]]


@dataclass(frozen=True)
class ArmResult:
    """One arm's aggregate result over an instance family.

    Attributes
    ----------
    name:
        Arm label.
    n_instances:
        Instances scored.
    mae, rmse:
        Error of the arm's value estimates against the ground truth, pooled
        over every product of every instance, in value units.
    spearman:
        Rank correlation of estimate against truth, pooled, dimensionless.
    realised_value:
        Mean ground-truth value of the arm's selection, in value units.
    oracle_value:
        Mean ground-truth value of the oracle's selection, in value units.
    regret, regret_fraction:
        Equation (8), mean over instances, and the mean of the per-instance
        fraction.
    worst_regret_fraction:
        Largest per-instance ``regret / oracle`` over the family.
    """

    name: str
    n_instances: int
    mae: float
    rmse: float
    spearman: float
    realised_value: float
    oracle_value: float
    regret: float
    regret_fraction: float
    worst_regret_fraction: float


def _problem(
    instance: GeneratedInstance,
    values: Mapping[str, float],
    ratios: Mapping[str, float],
    source: str,
) -> SelectionProblem:
    return SelectionProblem(
        inventory=instance.inventory,
        plan=instance.plan,
        values=dict(values),
        sizes_bits=sizes_from_ratios(instance.inventory, ratios),
        value_source=source,
    )


def benchmark_value_sources(
    instances: Sequence[GeneratedInstance],
    arms: Mapping[str, ValueSource],
    *,
    selector: Selector,
) -> tuple[ArmResult, ...]:
    """Score value-source arms on an instance family through one selector.

    Ground-truth compression ratios are used for every arm, so the only thing
    varying is the value estimate. Returns one :class:`ArmResult` per arm in the
    order the mapping iterates, with the oracle arm (ground-truth values) added
    under the name ``"oracle"`` if not supplied.
    """
    if not instances:
        raise ValueError("benchmark_value_sources needs at least one instance")
    all_arms: dict[str, ValueSource] = dict(arms)
    all_arms.setdefault("oracle", lambda instance: instance.true_values)

    oracle_values: list[float] = []
    for instance in instances:
        problem = _problem(instance, instance.true_values, instance.true_ratios, "oracle")
        oracle_values.append(selector(problem).total_value)

    results: list[ArmResult] = []
    for name, arm in all_arms.items():
        errors: list[float] = []
        truths: list[float] = []
        estimates: list[float] = []
        realised: list[float] = []
        fractions: list[float] = []
        for index, instance in enumerate(instances):
            values = dict(arm(instance))
            problem = _problem(instance, values, instance.true_ratios, name)
            selection = selector(problem)
            truth_problem = _problem(
                instance, instance.true_values, instance.true_ratios, "oracle"
            )
            score = truth_problem.score(selection)
            realised.append(score)
            best = oracle_values[index]
            fractions.append((best - score) / best if best > 0.0 else 0.0)
            for product in instance.inventory.names:
                truths.append(instance.true_values[product])
                estimates.append(float(values[product]))
                errors.append(float(values[product]) - instance.true_values[product])
        error_array = np.asarray(errors)
        results.append(
            ArmResult(
                name=name,
                n_instances=len(instances),
                mae=float(np.abs(error_array).mean()),
                rmse=float(np.sqrt((error_array**2).mean())),
                spearman=spearman(estimates, truths),
                realised_value=float(np.mean(realised)),
                oracle_value=float(np.mean(oracle_values)),
                regret=float(np.mean(oracle_values) - np.mean(realised)),
                regret_fraction=float(np.mean(fractions)),
                worst_regret_fraction=float(np.max(fractions)),
            )
        )
    return tuple(results)


@dataclass(frozen=True)
class RatioArmResult:
    """One size-estimator arm's result over an instance family.

    Attributes
    ----------
    name:
        Arm label.
    n_instances:
        Instances scored.
    mae, bias:
        Error of the ratio estimate against the ground truth, pooled,
        dimensionless. ``bias`` is the signed mean, so a positive bias means the
        estimator over-reserves capacity.
    overflow_rate:
        Fraction of windows whose committed true size exceeds capacity, over all
        windows of all instances. The quantity an operator sees.
    mean_overflow_bits:
        Mean overflow over the windows that overflowed, in bits. ``0.0`` when
        none did.
    realised_value, oracle_value:
        Mean value actually delivered -- counting only the products that fit
        once true sizes are known -- and the oracle's, in value units.
    delivered_fraction:
        ``realised_value / oracle_value``.
    unspent_fraction:
        Mean fraction of plan capacity left unused by the plan made with the
        arm's estimates. The cost of a conservative estimator.
    """

    name: str
    n_instances: int
    mae: float
    bias: float
    overflow_rate: float
    mean_overflow_bits: float
    realised_value: float
    oracle_value: float
    delivered_fraction: float
    unspent_fraction: float


def _delivered_value(
    instance: GeneratedInstance, selection: Selection, problem: SelectionProblem
) -> tuple[float, dict[str, float]]:
    """Value actually delivered once the true sizes are known, per window.

    Products are transmitted in the order the selection lists them within a
    window, true size is charged for each, and the first product that does not
    fit and everything after it in that window is lost. This is the behaviour of
    a pass that simply runs out of time, and it is the mechanism by which an
    under-estimated size costs value rather than merely being wrong.
    """
    capacity = problem.plan.capacity_bits()
    spent: dict[str, float] = dict.fromkeys(problem.plan.names, 0.0)
    stalled: set[str] = set()
    delivered = 0.0
    for assignment in selection.assignments:
        window = assignment.window
        if window in stalled:
            continue
        true_bits = (
            instance.inventory[assignment.product].raw_bits
            * instance.true_ratios[assignment.product]
        )
        if spent[window] + true_bits > capacity[window] + feasibility_slack_bits(
            capacity[window]
        ):
            stalled.add(window)
            continue
        spent[window] += true_bits
        delivered += instance.true_values[assignment.product] * problem.plan[window].reliability
    return delivered, spent


def benchmark_ratio_sources(
    instances: Sequence[GeneratedInstance],
    arms: Mapping[str, RatioSource],
    *,
    selector: Selector,
) -> tuple[RatioArmResult, ...]:
    """Score size-estimator arms, charging true sizes at transmission time.

    Ground-truth values are used for every arm so that the only thing varying is
    the size estimate. The oracle arm (ground-truth ratios) is added under
    ``"oracle"`` if not supplied.
    """
    if not instances:
        raise ValueError("benchmark_ratio_sources needs at least one instance")
    all_arms: dict[str, RatioSource] = dict(arms)
    all_arms.setdefault("oracle", lambda instance: instance.true_ratios)

    oracle_delivered: list[float] = []
    for instance in instances:
        problem = _problem(instance, instance.true_values, instance.true_ratios, "oracle")
        selection = selector(problem)
        delivered, _ = _delivered_value(instance, selection, problem)
        oracle_delivered.append(delivered)

    results: list[RatioArmResult] = []
    for name, arm in all_arms.items():
        errors: list[float] = []
        delivered_values: list[float] = []
        overflow_flags: list[int] = []
        overflow_sizes: list[float] = []
        unspent: list[float] = []
        for instance in instances:
            ratios = dict(arm(instance))
            problem = _problem(instance, instance.true_values, ratios, "oracle")
            selection = selector(problem)
            capacity = problem.plan.capacity_bits()
            true_committed: dict[str, float] = dict.fromkeys(problem.plan.names, 0.0)
            for assignment in selection.assignments:
                true_committed[assignment.window] += (
                    instance.inventory[assignment.product].raw_bits
                    * instance.true_ratios[assignment.product]
                )
            for window, committed in true_committed.items():
                excess = committed - capacity[window]
                over = excess > feasibility_slack_bits(capacity[window])
                overflow_flags.append(1 if over else 0)
                if over:
                    overflow_sizes.append(excess)
            delivered, spent = _delivered_value(instance, selection, problem)
            delivered_values.append(delivered)
            total_capacity = sum(capacity.values())
            unspent.append(
                max(total_capacity - sum(spent.values()), 0.0) / total_capacity
            )
            for product in instance.inventory.names:
                errors.append(float(ratios[product]) - instance.true_ratios[product])
        error_array = np.asarray(errors)
        oracle_mean = float(np.mean(oracle_delivered))
        realised_mean = float(np.mean(delivered_values))
        results.append(
            RatioArmResult(
                name=name,
                n_instances=len(instances),
                mae=float(np.abs(error_array).mean()),
                bias=float(error_array.mean()),
                overflow_rate=float(np.mean(overflow_flags)),
                mean_overflow_bits=float(np.mean(overflow_sizes)) if overflow_sizes else 0.0,
                realised_value=realised_mean,
                oracle_value=oracle_mean,
                delivered_fraction=realised_mean / oracle_mean if oracle_mean > 0.0 else 0.0,
                unspent_fraction=float(np.mean(unspent)),
            )
        )
    return tuple(results)


@dataclass(frozen=True)
class GapStatistics:
    """Distribution of the greedy optimality gap over an instance family.

    The gap of an instance is ``(optimum - greedy) / optimum``, dimensionless,
    and zero when greedy is optimal. The whole distribution is carried because a
    mean gap of a few per cent can be a family in which greedy is optimal most
    of the time and catastrophic occasionally, and those are different products.
    """

    n_instances: int
    n_optimal: int
    mean: float
    median: float
    quantiles: tuple[tuple[float, float], ...]
    maximum: float
    worst_seed: int
    histogram_edges: tuple[float, ...]
    histogram_counts: tuple[int, ...]


def gap_statistics(
    gaps: Sequence[float],
    seeds: Sequence[int],
    *,
    quantile_levels: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99),
    histogram_edges: Sequence[float] = (0.0, 1e-12, 0.01, 0.02, 0.05, 0.1, 0.2, 1.0),
    optimal_tolerance: float = 1.0e-9,
) -> GapStatistics:
    """Summarise a gap sample, including its tail and its histogram.

    ``optimal_tolerance`` is the relative tolerance within which greedy counts
    as having matched the optimum. It is a tolerance and not an equality test:
    the two objectives are independent floating-point sums over the same terms
    in different orders, so they agree to within rounding rather than exactly.
    """
    array = np.asarray(gaps, dtype=float)
    if array.size == 0:
        raise ValueError("gap_statistics needs at least one gap")
    if len(seeds) != array.size:
        raise ValueError(f"{len(seeds)} seeds against {array.size} gaps")
    counts, edges = np.histogram(array, bins=np.asarray(histogram_edges, dtype=float))
    worst = int(np.argmax(array))
    return GapStatistics(
        n_instances=int(array.size),
        n_optimal=int(np.sum(array <= optimal_tolerance)),
        mean=float(array.mean()),
        median=float(np.median(array)),
        quantiles=tuple(
            (float(level), float(np.quantile(array, level))) for level in quantile_levels
        ),
        maximum=float(array.max()),
        worst_seed=int(seeds[worst]),
        histogram_edges=tuple(float(edge) for edge in edges),
        histogram_counts=tuple(int(count) for count in counts),
    )
