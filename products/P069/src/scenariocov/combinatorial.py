"""t-way combinatorial coverage of a finished campaign, with the gaps enumerated.

Definition used, and its source
-------------------------------
For a space of ``k`` parameters whose parameter ``i`` has ``v_i`` declared
levels, the set of **t-way value tuples** is

    T_t = union over all C(k, t) parameter subsets S of size t
          of  { (S, assignment) : assignment in prod_{i in S} {0 .. v_i - 1} }

so that

    |T_t| = sum over subsets S of size t of  prod_{i in S} v_i          (1)

A campaign *covers* a t-way value tuple when at least one executed scenario
takes those t values simultaneously. **Simple t-way coverage** is

    C_t = (number of covered tuples) / |T_t|                            (2)

Both numerator and denominator in (2) are exact integers obtained by set
arithmetic over integer level indices; no floating-point comparison enters the
decision of whether a tuple is covered. The ratio itself is an ordinary float
division of two exact integers.

Source: D. R. Kuhn, R. N. Kacker and Y. Lei, *Practical Combinatorial Testing*,
NIST Special Publication 800-142, National Institute of Standards and
Technology, October 2010, doi:10.6028/NIST.SP.800-142 -- section 2 defines
t-way (combinatorial) coverage and the combination coverage measures this
module computes; the term "simple t-way combination coverage" and the
per-combination variant below are from that document.

Units: all quantities here are counts (dimensionless) or ratios of counts
(dimensionless, in [0, 1]).

Validity range and cost
-----------------------
The enumeration is exhaustive in ``|T_t|``, which by (1) grows as ``C(k, t)``
times the product of ``t`` level counts. This module refuses to enumerate more
than ``max_tuples`` tuples (default 2_000_000) rather than hang: for the
fixture space in this repository (level counts 4,4,3,3,3,3,3) ``|T_2| = 226``
and ``|T_3| = 1230``, both trivial, while ``|T_4| = 4005`` and ``|T_7| = 3888``
remain small only because ``k`` is 7. Raising ``k`` to 20 with 4 levels each
gives ``|T_3| = 72960``, still fine, and ``|T_6| = 158 x 10^6``, which is not.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations, product

import numpy as np

from .space import ScenarioSpace

__all__ = [
    "CombinationCoverage",
    "TwayCoverage",
    "UncoveredTuple",
    "count_tway_tuples",
    "tway_coverage",
]

DEFAULT_MAX_TUPLES = 2_000_000


@dataclass(frozen=True)
class UncoveredTuple:
    """One t-way value tuple that no executed scenario realised.

    Attributes:
        parameters: the ``t`` parameter names, in declaration order.
        levels: the corresponding level indices.
        labels: the corresponding level labels.
    """

    parameters: tuple[str, ...]
    levels: tuple[int, ...]
    labels: tuple[str, ...]

    def as_text(self) -> str:
        """One-line ``name=label`` rendering, used by the reports and the CLI."""
        pairs = (f"{n}={v}" for n, v in zip(self.parameters, self.labels, strict=True))
        return " & ".join(pairs)

    def matches(self, record: dict[str, object], space: ScenarioSpace) -> bool:
        """True when a raw scenario record realises this tuple."""
        return all(
            space.parameter(name).level_index(record[name]) == level
            for name, level in zip(self.parameters, self.levels, strict=True)
        )


@dataclass(frozen=True)
class CombinationCoverage:
    """Coverage of the value tuples belonging to one parameter subset.

    Attributes:
        parameters: the parameter names in this subset.
        total: number of value tuples for this subset, ``prod v_i``.
        covered: how many of them the campaign realised.
    """

    parameters: tuple[str, ...]
    total: int
    covered: int

    @property
    def fraction(self) -> float:
        """``covered / total``, dimensionless in [0, 1]."""
        return self.covered / self.total


@dataclass(frozen=True)
class TwayCoverage:
    """Result of :func:`tway_coverage`.

    Attributes:
        t: interaction strength.
        n_scenarios: number of executed scenarios the result was computed from.
        total: ``|T_t|`` from equation (1), an exact integer.
        covered: number of covered tuples, an exact integer.
        uncovered: every uncovered tuple, enumerated. Not a sample.
        per_combination: one entry per parameter subset, worst first.
    """

    t: int
    n_scenarios: int
    total: int
    covered: int
    uncovered: tuple[UncoveredTuple, ...]
    per_combination: tuple[CombinationCoverage, ...]

    @property
    def fraction(self) -> float:
        """Simple t-way coverage, equation (2). Dimensionless, in [0, 1]."""
        return self.covered / self.total

    @property
    def n_uncovered(self) -> int:
        """``total - covered``; equal to ``len(self.uncovered)`` by construction."""
        return self.total - self.covered

    def worst_combinations(self, limit: int = 5) -> tuple[CombinationCoverage, ...]:
        """The ``limit`` parameter subsets with the lowest coverage fraction."""
        if limit < 0:
            raise ValueError("limit must be non-negative")
        return tuple(sorted(self.per_combination, key=lambda c: (c.fraction, c.parameters))[:limit])

    def uncovered_involving(self, parameter: str) -> tuple[UncoveredTuple, ...]:
        """Uncovered tuples that constrain a given parameter."""
        return tuple(u for u in self.uncovered if parameter in u.parameters)


def count_tway_tuples(level_counts: Sequence[int], t: int) -> int:
    """``|T_t|`` from equation (1), without enumerating anything.

    Args:
        level_counts: number of levels per parameter.
        t: interaction strength, ``1 <= t <= len(level_counts)``.

    Returns:
        Exact integer count of t-way value tuples.
    """
    k = len(level_counts)
    if t < 1:
        raise ValueError("t must be at least 1")
    if t > k:
        raise ValueError(f"t={t} exceeds the number of parameters ({k})")
    if any(int(v) < 1 for v in level_counts):
        raise ValueError("every parameter needs at least one level")
    return sum(math.prod(subset) for subset in combinations([int(v) for v in level_counts], t))


def tway_coverage(
    space: ScenarioSpace,
    encoded: np.ndarray,
    t: int = 2,
    max_tuples: int = DEFAULT_MAX_TUPLES,
) -> TwayCoverage:
    """Compute t-way coverage and enumerate every uncovered tuple.

    Args:
        space: the declared scenario space.
        encoded: ``(n_scenarios, n_parameters)`` integer level indices, as
            produced by :meth:`ScenarioSpace.encode_many`.
        t: interaction strength. ``t=1`` is per-level coverage.
        max_tuples: refuse to enumerate a larger ``|T_t|`` than this.

    Returns:
        A :class:`TwayCoverage`. ``covered`` and ``total`` are exact integers;
        the only floating-point value is their ratio.

    Raises:
        ValueError: on a malformed array, an out-of-range level index, or an
            enumeration larger than ``max_tuples``.
    """
    counts = space.level_counts
    k = len(counts)
    if t < 1 or t > k:
        raise ValueError(f"t must satisfy 1 <= t <= {k}, got {t}")
    array = np.asarray(encoded)
    if array.ndim != 2 or array.shape[1] != k:
        raise ValueError(f"encoded must have shape (n_scenarios, {k}), got {array.shape}")
    if array.size and not np.issubdtype(array.dtype, np.integer):
        raise TypeError("encoded must hold integer level indices; use ScenarioSpace.encode_many")
    if array.size:
        if int(array.min()) < 0:
            raise ValueError("encoded contains a negative level index")
        for col, limit in enumerate(counts):
            if int(array[:, col].max()) >= limit:
                raise ValueError(
                    f"parameter {space.names[col]!r} has {limit} levels but encoded "
                    f"holds index {int(array[:, col].max())}"
                )

    total = count_tway_tuples(counts, t)
    if total > max_tuples:
        raise ValueError(
            f"|T_{t}| = {total} exceeds max_tuples={max_tuples}; raise max_tuples "
            "deliberately or reduce t, the parameter count or the level counts"
        )

    names = space.names
    uncovered: list[UncoveredTuple] = []
    per_combination: list[CombinationCoverage] = []
    covered_total = 0
    rows = [tuple(int(v) for v in row) for row in array]

    for subset in combinations(range(k), t):
        subset_total = math.prod(counts[i] for i in subset)
        present = {tuple(row[i] for i in subset) for row in rows}
        covered_total += len(present)
        per_combination.append(
            CombinationCoverage(
                parameters=tuple(names[i] for i in subset),
                total=subset_total,
                covered=len(present),
            )
        )
        if len(present) == subset_total:
            continue
        for values in product(*(range(counts[i]) for i in subset)):
            if values in present:
                continue
            uncovered.append(
                UncoveredTuple(
                    parameters=tuple(names[i] for i in subset),
                    levels=values,
                    labels=tuple(
                        space.parameters[i].levels[v]
                        for i, v in zip(subset, values, strict=True)
                    ),
                )
            )

    return TwayCoverage(
        t=t,
        n_scenarios=int(array.shape[0]),
        total=total,
        covered=covered_total,
        uncovered=tuple(uncovered),
        per_combination=tuple(per_combination),
    )
