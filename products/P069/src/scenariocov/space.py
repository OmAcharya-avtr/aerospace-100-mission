"""Declaration of a scenario space and the discretisation applied to it.

A scenario space is an ordered list of parameters. A parameter is either

* **categorical** -- an unordered finite set of declared levels, for example
  ``lighting in {full_sun, terminator, eclipse}``; or
* **continuous** -- a real interval with declared **bin edges**. The bins are
  the discretisation, and they are an *assumption imposed by this package*, not
  a property of the campaign. Every combinatorial and occupancy number reported
  by ``scenariocov`` is a number about the binned space, and changing the edges
  changes the numbers. This is stated again in README.md and in every report
  this package prints, because it is the single easiest thing to forget.

Binning convention (fixed, and relied on by the tests):

    bin i covers [edges[i], edges[i + 1])  for i < n_bins - 1
    the last bin covers [edges[-2], edges[-1]]  (closed at the top)

so the declared interval is exactly ``[edges[0], edges[-1]]`` and every value
inside it lands in exactly one bin.

**Floating-point caveat, stated because it is load-bearing.** The bin index of
a value that sits on or within rounding distance of an internal edge is decided
by a strict comparison on floating-point numbers and is therefore not something
this package can promise. Nothing in this package asserts a bin assignment for
such a value. ``ScenarioSpace.edge_ambiguities`` reports every record that
sits within a declared relative tolerance of an internal edge so that the
ambiguity is visible rather than silently resolved.

The normalised embedding used by the distance-based and density-based
strategies (``ScenarioSpace.normalise``) is also an assumption:

* a continuous parameter is mapped linearly onto [0, 1] over its declared
  interval, so the metric treats one declared range as one unit regardless of
  the physical units involved;
* a categorical parameter with L levels is one-hot encoded and scaled by
  ``1 / sqrt(2)``, so the Euclidean distance between two different levels of
  one categorical parameter is exactly 1 and the distance between equal levels
  is 0, matching the span of a normalised continuous parameter.

There is no claim that this embedding is the right metric for any particular
campaign; it is a declared, documented default.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "EdgeAmbiguity",
    "Parameter",
    "ScenarioSpace",
]


@dataclass(frozen=True)
class Parameter:
    """One declared scenario parameter.

    Attributes:
        name: parameter name, unique within a space.
        kind: ``"categorical"`` or ``"continuous"``.
        levels: level labels. For a continuous parameter these are the
            generated interval labels, one per bin.
        edges: bin edges in the parameter's own units, increasing, length
            ``n_levels + 1``. Empty for a categorical parameter.
        units: free-text unit string, for reporting only. Empty for a
            categorical parameter.
    """

    name: str
    kind: str
    levels: tuple[str, ...]
    edges: tuple[float, ...] = ()
    units: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("parameter name must be a non-empty string")
        if self.kind not in ("categorical", "continuous"):
            raise ValueError(f"{self.name}: kind must be 'categorical' or 'continuous'")
        if len(self.levels) < 2:
            raise ValueError(f"{self.name}: a parameter needs at least 2 levels")
        if len(set(self.levels)) != len(self.levels):
            raise ValueError(f"{self.name}: level labels must be unique")
        if self.kind == "continuous":
            if len(self.edges) != len(self.levels) + 1:
                raise ValueError(
                    f"{self.name}: continuous parameter needs len(edges) == len(levels) + 1"
                )
            pairs = zip(self.edges, self.edges[1:], strict=False)
            if any(not math.isfinite(e) or e >= f for e, f in pairs):
                raise ValueError(f"{self.name}: edges must be finite and strictly increasing")
        elif self.edges:
            raise ValueError(f"{self.name}: a categorical parameter must not declare edges")

    @property
    def n_levels(self) -> int:
        """Number of declared levels (bins, for a continuous parameter)."""
        return len(self.levels)

    @property
    def lower(self) -> float:
        """Lower end of the declared interval. Continuous parameters only."""
        if self.kind != "continuous":
            raise TypeError(f"{self.name}: lower is defined for continuous parameters only")
        return self.edges[0]

    @property
    def upper(self) -> float:
        """Upper end of the declared interval. Continuous parameters only."""
        if self.kind != "continuous":
            raise TypeError(f"{self.name}: upper is defined for continuous parameters only")
        return self.edges[-1]

    @classmethod
    def categorical(cls, name: str, levels: Sequence[str]) -> Parameter:
        """Declare an unordered categorical parameter."""
        return cls(name=name, kind="categorical", levels=tuple(str(level) for level in levels))

    @classmethod
    def continuous(
        cls,
        name: str,
        edges: Sequence[float],
        units: str = "",
        labels: Sequence[str] | None = None,
    ) -> Parameter:
        """Declare a continuous parameter with explicit bin edges.

        Args:
            name: parameter name.
            edges: strictly increasing bin edges in the parameter's units.
            units: unit string, reporting only.
            labels: optional bin labels; defaults to ``"[lo,hi)"`` strings.
        """
        edge_tuple = tuple(float(e) for e in edges)
        if len(edge_tuple) < 3:
            raise ValueError(f"{name}: need at least 3 edges to form 2 bins")
        if labels is None:
            built = [
                f"[{edge_tuple[i]:g},{edge_tuple[i + 1]:g})"
                for i in range(len(edge_tuple) - 2)
            ]
            built.append(f"[{edge_tuple[-2]:g},{edge_tuple[-1]:g}]")
            labels = built
        return cls(
            name=name,
            kind="continuous",
            levels=tuple(str(label) for label in labels),
            edges=edge_tuple,
            units=str(units),
        )

    def level_index(self, value: Any) -> int:
        """Map a raw parameter value onto its declared level index.

        Categorical: exact label match. Continuous: half-open bins as described
        in the module docstring, with the top bin closed.

        Raises:
            ValueError: on an unknown category or a value outside the declared
                interval.
        """
        if self.kind == "categorical":
            try:
                return self.levels.index(str(value))
            except ValueError as exc:
                raise ValueError(
                    f"{self.name}: value {value!r} is not a declared level "
                    f"(declared: {', '.join(self.levels)})"
                ) from exc
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"{self.name}: value must be finite, got {value!r}")
        if numeric < self.edges[0] or numeric > self.edges[-1]:
            raise ValueError(
                f"{self.name}: value {numeric!r} is outside the declared interval "
                f"[{self.edges[0]}, {self.edges[-1]}]"
            )
        internal = np.asarray(self.edges[1:-1], dtype=float)
        index = int(np.searchsorted(internal, numeric, side="right"))
        return min(index, self.n_levels - 1)

    def unit_position(self, value: Any, level: int | None = None) -> float:
        """Position of a value in [0, 1] for the normalised embedding.

        Continuous: linear over the declared interval. Categorical: the level
        index divided by ``n_levels - 1``, used only for plotting; the
        distance metric uses the one-hot columns instead.
        """
        if self.kind == "continuous":
            span = self.edges[-1] - self.edges[0]
            return (float(value) - self.edges[0]) / span
        idx = self.level_index(value) if level is None else level
        return idx / (self.n_levels - 1)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form, round-tripped by :meth:`from_dict`."""
        payload: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind,
            "levels": list(self.levels),
        }
        if self.kind == "continuous":
            payload["edges"] = list(self.edges)
            payload["units"] = self.units
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Parameter:
        """Inverse of :meth:`to_dict`."""
        kind = payload.get("kind")
        if kind == "categorical":
            return cls.categorical(payload["name"], payload["levels"])
        if kind == "continuous":
            return cls.continuous(
                payload["name"],
                payload["edges"],
                units=payload.get("units", ""),
                labels=payload.get("levels"),
            )
        raise ValueError(f"unknown parameter kind {kind!r}")


@dataclass(frozen=True)
class EdgeAmbiguity:
    """A record whose value lies within tolerance of an internal bin edge.

    Attributes:
        record_index: index of the record in the campaign.
        parameter: parameter name.
        value: the raw value, in the parameter's units.
        edge: the internal edge it is close to, same units.
        distance: ``abs(value - edge)``, same units.
        assigned_level: the level index this package assigned, which a
            different but mathematically equivalent binning could change.
    """

    record_index: int
    parameter: str
    value: float
    edge: float
    distance: float
    assigned_level: int


class ScenarioSpace:
    """An ordered collection of declared parameters plus the encodings on it."""

    def __init__(self, parameters: Sequence[Parameter]) -> None:
        if not parameters:
            raise ValueError("a scenario space needs at least one parameter")
        names = [p.name for p in parameters]
        if len(set(names)) != len(names):
            raise ValueError("parameter names must be unique")
        self._parameters = tuple(parameters)

    def __len__(self) -> int:
        return len(self._parameters)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ScenarioSpace({len(self)} parameters, {self.n_combinations} combinations)"

    @property
    def parameters(self) -> tuple[Parameter, ...]:
        """The declared parameters, in declaration order."""
        return self._parameters

    @property
    def names(self) -> tuple[str, ...]:
        """Parameter names, in declaration order."""
        return tuple(p.name for p in self._parameters)

    @property
    def level_counts(self) -> tuple[int, ...]:
        """Number of levels per parameter, in declaration order."""
        return tuple(p.n_levels for p in self._parameters)

    @property
    def n_combinations(self) -> int:
        """Size of the fully crossed discretised space (exact integer)."""
        return math.prod(self.level_counts)

    def parameter(self, name: str) -> Parameter:
        """Look a parameter up by name."""
        for p in self._parameters:
            if p.name == name:
                return p
        raise KeyError(f"no parameter named {name!r}; declared: {', '.join(self.names)}")

    def index_of(self, name: str) -> int:
        """Position of a parameter in declaration order."""
        try:
            return self.names.index(name)
        except ValueError as exc:
            raise KeyError(f"no parameter named {name!r}") from exc

    def encode(self, record: Mapping[str, Any]) -> tuple[int, ...]:
        """Encode one scenario record as a tuple of level indices."""
        missing = [n for n in self.names if n not in record]
        if missing:
            raise ValueError(f"scenario is missing parameter(s): {', '.join(missing)}")
        return tuple(p.level_index(record[p.name]) for p in self._parameters)

    def encode_many(self, records: Iterable[Mapping[str, Any]]) -> np.ndarray:
        """Encode a campaign as an ``(n_records, n_parameters)`` int array."""
        rows = [self.encode(r) for r in records]
        if not rows:
            return np.zeros((0, len(self)), dtype=np.int64)
        return np.asarray(rows, dtype=np.int64)

    def normalise(self, records: Sequence[Mapping[str, Any]]) -> np.ndarray:
        """Normalised embedding of a campaign, shape ``(n_records, n_columns)``.

        One column per continuous parameter (linear on [0, 1] over the declared
        interval) and ``n_levels`` columns per categorical parameter (one-hot
        scaled by ``1 / sqrt(2)``). See the module docstring for the assumption
        this encodes. Units: dimensionless.
        """
        n_cols = self.n_embedding_columns
        out = np.zeros((len(records), n_cols), dtype=float)
        for row, record in enumerate(records):
            col = 0
            for p in self._parameters:
                if p.kind == "continuous":
                    out[row, col] = p.unit_position(record[p.name])
                    col += 1
                else:
                    out[row, col + p.level_index(record[p.name])] = 1.0 / math.sqrt(2.0)
                    col += p.n_levels
        return out

    @property
    def n_embedding_columns(self) -> int:
        """Width of the :meth:`normalise` embedding."""
        return sum(1 if p.kind == "continuous" else p.n_levels for p in self._parameters)

    @property
    def embedding_columns(self) -> tuple[str, ...]:
        """Column labels of the :meth:`normalise` embedding."""
        labels: list[str] = []
        for p in self._parameters:
            if p.kind == "continuous":
                labels.append(p.name)
            else:
                labels.extend(f"{p.name}={level}" for level in p.levels)
        return tuple(labels)

    def decode(self, levels: Sequence[int]) -> dict[str, str]:
        """Map a tuple of level indices back to level labels."""
        if len(levels) != len(self):
            raise ValueError(f"expected {len(self)} level indices, got {len(levels)}")
        return {p.name: p.levels[int(i)] for p, i in zip(self._parameters, levels, strict=True)}

    def edge_ambiguities(
        self, records: Sequence[Mapping[str, Any]], rtol: float = 1e-9
    ) -> tuple[EdgeAmbiguity, ...]:
        """Records sitting within ``rtol`` of a bin width of an internal edge.

        The bin assignment of such a record is decided by a strict comparison on
        floating-point values and is not guaranteed by this package. The caller
        decides what to do about it; this method only makes it visible.

        Args:
            records: the campaign.
            rtol: tolerance as a fraction of the width of the narrower adjacent
                bin. Dimensionless, must be positive.
        """
        if rtol <= 0:
            raise ValueError("rtol must be positive")
        found: list[EdgeAmbiguity] = []
        for i, record in enumerate(records):
            for p in self._parameters:
                if p.kind != "continuous" or p.name not in record:
                    continue
                value = float(record[p.name])
                for j in range(1, len(p.edges) - 1):
                    edge = p.edges[j]
                    width = min(edge - p.edges[j - 1], p.edges[j + 1] - edge)
                    distance = abs(value - edge)
                    if distance <= rtol * width:
                        found.append(
                            EdgeAmbiguity(
                                record_index=i,
                                parameter=p.name,
                                value=value,
                                edge=edge,
                                distance=distance,
                                assigned_level=p.level_index(value),
                            )
                        )
        return tuple(found)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form, round-tripped by :meth:`from_dict`."""
        return {"parameters": [p.to_dict() for p in self._parameters]}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ScenarioSpace:
        """Inverse of :meth:`to_dict`."""
        params = payload.get("parameters")
        if not isinstance(params, list) or not params:
            raise ValueError("space payload needs a non-empty 'parameters' list")
        return cls([Parameter.from_dict(p) for p in params])
