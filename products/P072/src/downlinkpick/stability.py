"""Selection stability under perturbed value estimates.

An optimal selector that changes its answer when a value estimate moves by one
per cent is unusable, whatever its optimality gap. An operator will not accept a
plan that is different every time the value table is refreshed, and more
importantly a selection that is sensitive to the estimate is a selection whose
*value* is sensitive to the estimate, which means the optimality was bought
against numbers nobody believes to that precision.

What is measured here, for a sequence of relative noise levels ``sigma``:

* **Jaccard similarity** of the selected set against the unperturbed selection.
  ``|A and B| / |A or B|``, dimensionless, 1.0 for an identical selection.
* **Value retained.** The perturbed selection scored against the *unperturbed*
  values, divided by the unperturbed selection's value. This is the number that
  matters: a selection can swap two products of equal value and lose nothing,
  in which case the Jaccard falls and the retained value does not. Reporting
  only the Jaccard would overstate the harm, and reporting only the value would
  hide churn that an operator has to review.
* **Flip count**, the symmetric difference size, which is what an operator
  actually has to re-check by hand.

The perturbation is multiplicative and log-normal: ``v -> v * exp(N(0, sigma))``,
which keeps values non-negative and makes ``sigma`` a relative error rather than
an absolute one. An additive perturbation would be dominated by the largest
values and would say nothing about the small ones, where the selection boundary
usually sits.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .selection import Selection, SelectionProblem

Selector = Callable[[SelectionProblem], Selection]


def perturb_values(
    values: Mapping[str, float], sigma: float, rng: np.random.Generator
) -> dict[str, float]:
    """Multiplicative log-normal perturbation of a value mapping.

    Parameters
    ----------
    values:
        Value per product name, in value units.
    sigma:
        Standard deviation of the perturbation in log space, dimensionless.
        ``0.0`` returns the values unchanged.
    rng:
        Generator; pass a seeded one for reproducibility.
    """
    if sigma < 0.0:
        raise ValueError(f"sigma must be non-negative, got {sigma!r}")
    if sigma == 0.0:
        return {name: float(value) for name, value in values.items()}
    return {
        name: float(value) * float(np.exp(rng.normal(0.0, sigma)))
        for name, value in values.items()
    }


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    """Jaccard similarity of two sets, dimensionless in ``[0, 1]``.

    Two empty sets are defined as similarity 1.0: nothing was selected either
    time, which is agreement, not disagreement.
    """
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


@dataclass(frozen=True)
class StabilityPoint:
    """Stability statistics at one noise level.

    Attributes
    ----------
    sigma:
        Relative perturbation standard deviation in log space.
    n_trials:
        Number of perturbed selections drawn.
    mean_jaccard, p05_jaccard, min_jaccard:
        Set-similarity statistics against the unperturbed selection.
    mean_value_retained, p05_value_retained:
        Perturbed selection's value under the unperturbed value table, divided
        by the unperturbed selection's value, dimensionless.
    mean_flips:
        Mean symmetric-difference size, in products.
    """

    sigma: float
    n_trials: int
    mean_jaccard: float
    p05_jaccard: float
    min_jaccard: float
    mean_value_retained: float
    p05_value_retained: float
    mean_flips: float


def stability_curve(
    problem: SelectionProblem,
    sigmas: Sequence[float],
    *,
    selector: Selector,
    n_trials: int = 40,
    seed: int = 20261021,
) -> tuple[StabilityPoint, ...]:
    """Measure selection stability against perturbed values.

    Parameters
    ----------
    problem:
        The problem, whose ``values`` are treated as the truth.
    sigmas:
        Relative noise levels to test.
    selector:
        The selector under test.
    n_trials:
        Perturbed draws per noise level.
    seed:
        Seed for the perturbation stream. The same seed, sigmas and trial count
        reproduce the curve exactly.

    Returns
    -------
    tuple of StabilityPoint
        One per noise level, in the order given.
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials!r}")
    nominal = selector(problem)
    nominal_set = nominal.selected_set
    nominal_value = nominal.total_value
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x5721]))

    points: list[StabilityPoint] = []
    for sigma in sigmas:
        similarities: list[float] = []
        retained: list[float] = []
        flips: list[int] = []
        for _ in range(n_trials):
            perturbed = perturb_values(problem.values, float(sigma), rng)
            selection = selector(problem.with_values(perturbed, value_source="perturbed"))
            similarities.append(jaccard(nominal_set, selection.selected_set))
            scored = problem.score(selection)
            retained.append(scored / nominal_value if nominal_value > 0.0 else 1.0)
            flips.append(len(nominal_set ^ selection.selected_set))
        points.append(
            StabilityPoint(
                sigma=float(sigma),
                n_trials=n_trials,
                mean_jaccard=float(np.mean(similarities)),
                p05_jaccard=float(np.quantile(similarities, 0.05)),
                min_jaccard=float(np.min(similarities)),
                mean_value_retained=float(np.mean(retained)),
                p05_value_retained=float(np.quantile(retained, 0.05)),
                mean_flips=float(np.mean(flips)),
            )
        )
    return tuple(points)
