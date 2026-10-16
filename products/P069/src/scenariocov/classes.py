"""Equivalence-class and boundary coverage against classes the user declares.

Equivalence partitioning and boundary value analysis are test-design
techniques, not properties this package infers. The classes and the boundaries
are **declared by the person who wrote the verification plan**; this module
only reports which of them the finished campaign actually touched.

Source: G. J. Myers, *The Art of Software Testing*, Wiley, 1979, chapter 4
("Test-Case Design"), which introduces equivalence partitioning and boundary
value analysis in the form used here: partition each input domain into classes
that the implementation is expected to treat alike, and test at and immediately
either side of the class boundaries, because that is where off-by-one and
comparison-operator faults live. The second edition (Myers, Sandler, Badgett,
2011) carries the same two sections.

Units: a class or boundary is declared in the parameter's own units; the counts
reported here are dimensionless.

Floating-point note, which matters for boundaries
-------------------------------------------------
Equivalence-class membership for a continuous parameter is an interval
comparison, and boundary coverage is the predicate ``|x - b| <= tol``. Both are
strict comparisons on floating-point values, so a scenario that sits within
rounding distance of a class edge or exactly ``tol`` away from a boundary can
fall either way. Every result in this module therefore reports the **margin**
alongside the boolean: :attr:`BoundaryHit.nearest_distance` is the smallest
``|x - b|`` observed, so a campaign that only just hit (or only just missed) a
boundary is visible as such instead of being reduced to True or False.
Categorical class membership is a set-membership test on labels and has no
such caveat.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "BoundaryHit",
    "BoundarySpec",
    "ClassCoverage",
    "ClassHit",
    "EquivalenceClass",
    "boundary_coverage",
    "class_coverage",
]


@dataclass(frozen=True)
class EquivalenceClass:
    """One declared equivalence class on one parameter.

    A continuous class is the interval ``[lower, upper]`` if ``closed_upper``
    else ``[lower, upper)``. A categorical class is the declared set of labels.

    Attributes:
        parameter: parameter the class constrains.
        name: class name, unique per parameter.
        lower: interval lower bound, in the parameter's units. Continuous only.
        upper: interval upper bound, in the parameter's units. Continuous only.
        closed_upper: whether ``upper`` belongs to the class.
        labels: the member labels. Categorical only.
        expected: whether the class is a valid-input class (True) or a declared
            invalid/off-nominal class (False). Reporting only.
    """

    parameter: str
    name: str
    lower: float | None = None
    upper: float | None = None
    closed_upper: bool = False
    labels: tuple[str, ...] = ()
    expected: bool = True

    def __post_init__(self) -> None:
        interval = self.lower is not None and self.upper is not None
        if interval == bool(self.labels):
            raise ValueError(
                f"{self.parameter}/{self.name}: declare either (lower, upper) or labels, not both"
            )
        if interval:
            if not (math.isfinite(float(self.lower)) and math.isfinite(float(self.upper))):
                raise ValueError(f"{self.parameter}/{self.name}: bounds must be finite")
            if float(self.lower) >= float(self.upper):
                raise ValueError(f"{self.parameter}/{self.name}: need lower < upper")

    @property
    def is_interval(self) -> bool:
        """True for a continuous (interval) class."""
        return self.lower is not None and self.upper is not None

    def contains(self, value: Any) -> bool:
        """Membership test. Interval classes use a floating-point comparison."""
        if self.is_interval:
            x = float(value)
            upper_ok = x <= float(self.upper) if self.closed_upper else x < float(self.upper)
            return float(self.lower) <= x and upper_ok
        return str(value) in self.labels

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form."""
        payload: dict[str, Any] = {
            "parameter": self.parameter,
            "name": self.name,
            "expected": self.expected,
        }
        if self.is_interval:
            payload["lower"] = float(self.lower)
            payload["upper"] = float(self.upper)
            payload["closed_upper"] = self.closed_upper
        else:
            payload["labels"] = list(self.labels)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> EquivalenceClass:
        """Inverse of :meth:`to_dict`."""
        return cls(
            parameter=payload["parameter"],
            name=payload["name"],
            lower=payload.get("lower"),
            upper=payload.get("upper"),
            closed_upper=bool(payload.get("closed_upper", False)),
            labels=tuple(payload.get("labels", ())),
            expected=bool(payload.get("expected", True)),
        )


@dataclass(frozen=True)
class BoundarySpec:
    """A declared boundary value and the tolerance that counts as hitting it.

    Attributes:
        parameter: parameter the boundary is on.
        name: boundary name.
        value: the boundary, in the parameter's units.
        tolerance: half-width of the window that counts as a hit, same units,
            strictly positive.
    """

    parameter: str
    name: str
    value: float
    tolerance: float

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.value)):
            raise ValueError(f"{self.parameter}/{self.name}: boundary value must be finite")
        if not (float(self.tolerance) > 0.0 and math.isfinite(float(self.tolerance))):
            raise ValueError(f"{self.parameter}/{self.name}: tolerance must be positive and finite")

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form."""
        return {
            "parameter": self.parameter,
            "name": self.name,
            "value": float(self.value),
            "tolerance": float(self.tolerance),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> BoundarySpec:
        """Inverse of :meth:`to_dict`."""
        return cls(
            parameter=payload["parameter"],
            name=payload["name"],
            value=float(payload["value"]),
            tolerance=float(payload["tolerance"]),
        )


@dataclass(frozen=True)
class ClassHit:
    """How many executed scenarios landed in one declared class."""

    cls: EquivalenceClass
    count: int

    @property
    def covered(self) -> bool:
        """True when at least one scenario landed in the class (exact integer test)."""
        return self.count > 0


@dataclass(frozen=True)
class BoundaryHit:
    """How close the campaign came to one declared boundary.

    Attributes:
        spec: the declared boundary.
        count: number of scenarios within ``spec.tolerance`` of it.
        nearest_distance: smallest ``|x - value|`` over the campaign, in the
            parameter's units, or ``inf`` for an empty campaign. Reported so a
            near-tie against the tolerance is visible.
    """

    spec: BoundarySpec
    count: int
    nearest_distance: float

    @property
    def covered(self) -> bool:
        """True when ``count > 0``. Derived from a floating-point comparison."""
        return self.count > 0

    @property
    def margin(self) -> float:
        """``tolerance - nearest_distance``: positive when hit, in the parameter's units."""
        return float(self.spec.tolerance) - self.nearest_distance


@dataclass(frozen=True)
class ClassCoverage:
    """Result of :func:`class_coverage`."""

    hits: tuple[ClassHit, ...]
    n_scenarios: int

    @property
    def total(self) -> int:
        """Number of declared classes."""
        return len(self.hits)

    @property
    def covered(self) -> int:
        """Number of classes with at least one scenario in them."""
        return sum(1 for h in self.hits if h.covered)

    @property
    def fraction(self) -> float:
        """``covered / total``, dimensionless in [0, 1]."""
        if not self.hits:
            raise ValueError("no classes declared, so class coverage is undefined")
        return self.covered / self.total

    @property
    def uncovered(self) -> tuple[EquivalenceClass, ...]:
        """The declared classes no scenario reached."""
        return tuple(h.cls for h in self.hits if not h.covered)


def class_coverage(
    classes: Sequence[EquivalenceClass],
    records: Sequence[Mapping[str, Any]],
) -> ClassCoverage:
    """Count how many executed scenarios fall in each declared equivalence class.

    Args:
        classes: declared classes; several may overlap, and a scenario counts
            once in each class it satisfies.
        records: the executed campaign, as raw scenario dictionaries.

    Returns:
        A :class:`ClassCoverage`.

    Raises:
        ValueError: when a declared class names a parameter absent from a
            record, or when ``classes`` is empty.
    """
    if not classes:
        raise ValueError("declare at least one equivalence class")
    hits: list[ClassHit] = []
    for cls in classes:
        count = 0
        for i, record in enumerate(records):
            if cls.parameter not in record:
                raise ValueError(
                    f"record {i} has no parameter {cls.parameter!r} required by "
                    f"class {cls.name!r}"
                )
            if cls.contains(record[cls.parameter]):
                count += 1
        hits.append(ClassHit(cls=cls, count=count))
    return ClassCoverage(hits=tuple(hits), n_scenarios=len(records))


def boundary_coverage(
    boundaries: Sequence[BoundarySpec],
    records: Sequence[Mapping[str, Any]],
) -> tuple[BoundaryHit, ...]:
    """Measure how close the campaign came to each declared boundary.

    Args:
        boundaries: declared boundary values with tolerances.
        records: the executed campaign.

    Returns:
        One :class:`BoundaryHit` per boundary, in the declared order. The hit
        count uses the floating-point predicate ``|x - value| <= tolerance``;
        ``nearest_distance`` is reported alongside so that a result sitting on
        the tolerance is not hidden behind a boolean.
    """
    out: list[BoundaryHit] = []
    for spec in boundaries:
        count = 0
        nearest = math.inf
        for i, record in enumerate(records):
            if spec.parameter not in record:
                raise ValueError(
                    f"record {i} has no parameter {spec.parameter!r} required by "
                    f"boundary {spec.name!r}"
                )
            distance = abs(float(record[spec.parameter]) - float(spec.value))
            nearest = min(nearest, distance)
            if distance <= float(spec.tolerance):
                count += 1
        out.append(BoundaryHit(spec=spec, count=count, nearest_distance=nearest))
    return tuple(out)
