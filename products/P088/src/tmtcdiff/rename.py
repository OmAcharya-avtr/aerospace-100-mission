"""Rename detection by structural similarity, with a measurable false-match rate.

A rename is the one change class that cannot be read off the two documents.
Every other class compares two records that share a key; a rename has to be
*inferred* by pairing a removed name with an added name, and an inference has
a false-match rate. This module keeps that rate measurable by making every
component of the score explicit and every acceptance decision a single
threshold on one number.

Score
-----

For a removed item ``r`` and an added item ``a``::

    s(r, a) = w_struct * structural(r, a) + w_name * name(r, a) + w_desc * desc(r, a)

with weights summing to one.

- ``structural`` is the weighted fraction of agreeing structural fields. The
  field weights are in :data:`FIELD_WEIGHTS`. A parameter's structural
  signature excludes every name-bearing field, so the structural term cannot
  be inflated by a similar name.
- ``name`` is ``difflib.SequenceMatcher.ratio``, the Ratcliff/Obershelp
  gestalt-pattern-matching similarity (Ratcliff and Metzener, *Pattern
  Matching: the Gestalt Approach*, Dr. Dobb's Journal, July 1988), computed on
  the upper-cased name with separators normalised. Range 0 to 1.
- ``desc`` is the same ratio on the descriptions, 0 when either is empty.

Matching
--------

Two strategies, both deterministic, and the simpler one is the baseline:

``greedy``
    Score every removed-added pair, sort descending, accept the best available
    pair that clears the threshold, remove both from the pool, repeat. This is
    the obvious implementation and it is the baseline the optimal matcher is
    measured against.

``optimal``
    Maximise the total score over one-to-one assignments with
    ``scipy.optimize.linear_sum_assignment`` (the rectangular assignment
    algorithm of Crouse, *On implementing 2D rectangular assignment
    algorithms*, IEEE Transactions on Aerospace and Electronic Systems 52(4),
    2016, which scipy documents as its implementation; the problem is the
    assignment problem of Kuhn, *The Hungarian method for the assignment
    problem*, Naval Research Logistics Quarterly 2, 1955), then apply the
    threshold to each assigned pair.

Neither strategy is assumed better. ``validation/validate_rename_sweep.py``
measures both across a grid of threshold and dictionary size and reports the
result as found, including where the baseline wins.

Ambiguity
---------

A pair whose score lands within :data:`AMBIGUOUS_BAND` of the threshold is
accepted or rejected by the threshold as usual *and* recorded as ambiguous. An
ambiguous count is not a warning in prose: it is carried into the coverage
verdict as something the tool cannot determine.
"""

from __future__ import annotations

import difflib
import math
import re
from dataclasses import dataclass
from typing import Any

from .model import Command, Parameter

#: Structural field weights. ``scale`` and ``offset`` are weighted below the
#: layout fields deliberately: a rescale is a change this package expects to
#: see *across* a rename, so weighting it heavily would make a renamed-and-
#: rescaled parameter unmatchable, which is the case that matters most.
FIELD_WEIGHTS: dict[str, float] = {
    "data_type": 2.0,
    "bit_offset": 1.0,
    "bit_width": 2.0,
    "byte_order": 1.0,
    "scale": 1.0,
    "offset": 1.0,
    "unit": 2.0,
    "enum": 2.0,
    "limits": 0.5,
    "packet": 1.5,
}

_FIELD_ORDER = (
    "data_type",
    "bit_offset",
    "bit_width",
    "byte_order",
    "scale",
    "offset",
    "unit",
    "enum",
    "low_limit",
    "high_limit",
    "packet",
)

#: Default score weights. Structural dominates, by design: the name is the
#: thing that changed.
DEFAULT_WEIGHTS: dict[str, float] = {"structural": 0.55, "name": 0.30, "description": 0.15}

#: Half-width of the band around the threshold inside which an accepted or
#: rejected pair is also reported as ambiguous.
AMBIGUOUS_BAND = 0.05

_SEPARATORS = re.compile(r"[\s_\-./]+")


@dataclass(frozen=True)
class RenameCandidate:
    """A scored removed-to-added pairing."""

    old_name: str
    new_name: str
    score: float
    structural: float
    name_similarity: float
    description_similarity: float
    accepted: bool
    ambiguous: bool

    def __post_init__(self) -> None:
        for field_name in ("score", "structural", "name_similarity", "description_similarity"):
            value = getattr(self, field_name)
            if not math.isfinite(value) or not -1e-9 <= value <= 1.0 + 1e-9:
                raise ValueError(f"RenameCandidate.{field_name} must lie in [0, 1], got {value!r}")


def normalise_name(name: str) -> str:
    """Upper-case and collapse separators, so ``T_BAT_1`` and ``t.bat.1`` compare equal."""
    return _SEPARATORS.sub("_", name.strip().upper())


def text_similarity(left: str, right: str) -> float:
    """Ratcliff/Obershelp ratio between two strings. 0.0 when either is empty."""
    if not left or not right:
        return 0.0
    return difflib.SequenceMatcher(None, left, right).ratio()


def _field_values(item: Parameter | Command) -> dict[str, Any]:
    if isinstance(item, Parameter):
        return dict(zip(_FIELD_ORDER, item.structural_signature(), strict=True))
    return {"arguments": item.structural_signature()}


def structural_similarity(left: Parameter | Command, right: Parameter | Command) -> float:
    """Weighted fraction of agreeing structural fields, in [0, 1].

    For commands the structural term is the Ratcliff/Obershelp ratio over the
    argument signature rendered as text, because a command's structure is a
    sequence and a field-by-field fraction would not express a partial match of
    an argument list.
    """
    if isinstance(left, Command) or isinstance(right, Command):
        if not isinstance(left, Command) or not isinstance(right, Command):
            raise TypeError("structural_similarity compares a parameter with a parameter")
        return text_similarity(
            repr(left.structural_signature()), repr(right.structural_signature())
        )

    left_fields = _field_values(left)
    right_fields = _field_values(right)
    total = 0.0
    agreeing = 0.0
    for field_name, weight in FIELD_WEIGHTS.items():
        total += weight
        if field_name == "limits":
            same = (left.low_limit, left.high_limit) == (right.low_limit, right.high_limit)
        else:
            same = left_fields[field_name] == right_fields[field_name]
        if same:
            agreeing += weight
    return agreeing / total if total else 0.0


def pair_score(
    left: Parameter | Command,
    right: Parameter | Command,
    weights: dict[str, float] | None = None,
) -> tuple[float, float, float, float]:
    """Return ``(score, structural, name, description)`` for one candidate pair."""
    active = dict(DEFAULT_WEIGHTS if weights is None else weights)
    missing = {"structural", "name", "description"} - set(active)
    if missing:
        raise ValueError(f"weights must define {sorted(missing)}")
    total = sum(active.values())
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(f"weights must sum to 1.0, got {total!r}")
    if any(value < 0.0 for value in active.values()):
        raise ValueError(f"weights must be non-negative, got {active!r}")
    structural = structural_similarity(left, right)
    name = text_similarity(normalise_name(left.name), normalise_name(right.name))
    description = text_similarity(left.description, right.description)
    score = (
        active["structural"] * structural
        + active["name"] * name
        + active["description"] * description
    )
    return (min(1.0, max(0.0, score)), structural, name, description)


def score_matrix(
    removed: list[Parameter] | list[Command],
    added: list[Parameter] | list[Command],
    weights: dict[str, float] | None = None,
) -> list[list[tuple[float, float, float, float]]]:
    """Full score matrix, rows indexed by ``removed`` and columns by ``added``."""
    return [[pair_score(old, new, weights) for new in added] for old in removed]


def _greedy(matrix: list[list[tuple[float, float, float, float]]]) -> list[tuple[int, int]]:
    """Greedy one-to-one pairing, best score first. Ties broken by index, so deterministic."""
    entries = sorted(
        (
            (-matrix[i][j][0], i, j)
            for i in range(len(matrix))
            for j in range(len(matrix[i]) if matrix else 0)
        ),
    )
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _, i, j in entries:
        if i in used_rows or j in used_cols:
            continue
        used_rows.add(i)
        used_cols.add(j)
        pairs.append((i, j))
    return pairs


def _optimal(matrix: list[list[tuple[float, float, float, float]]]) -> list[tuple[int, int]]:
    """Maximum-total-score one-to-one pairing via rectangular assignment."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    scores = np.array([[cell[0] for cell in row] for row in matrix], dtype=float)
    rows, cols = linear_sum_assignment(-scores)
    return [(int(i), int(j)) for i, j in zip(rows, cols, strict=True)]


#: Matching strategies. ``greedy`` is the baseline.
STRATEGIES: tuple[str, ...] = ("greedy", "optimal")


def match_renames(
    removed: list[Parameter] | list[Command],
    added: list[Parameter] | list[Command],
    *,
    threshold: float,
    strategy: str = "optimal",
    weights: dict[str, float] | None = None,
    ambiguous_band: float = AMBIGUOUS_BAND,
) -> tuple[RenameCandidate, ...]:
    """Pair removed items with added items and accept the pairs clearing ``threshold``.

    Returns every pair the strategy produced, accepted or not, so that a caller
    measuring a false-match rate can see the rejected pairs as well. Pairs are
    returned in descending score order.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must lie in [0, 1], got {threshold!r}")
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {', '.join(STRATEGIES)}, got {strategy!r}")
    if ambiguous_band < 0.0:
        raise ValueError(f"ambiguous_band must be >= 0, got {ambiguous_band!r}")
    if not removed or not added:
        return ()

    matrix = score_matrix(removed, added, weights)
    pairs = _greedy(matrix) if strategy == "greedy" else _optimal(matrix)

    candidates: list[RenameCandidate] = []
    for i, j in pairs:
        score, structural, name, description = matrix[i][j]
        candidates.append(
            RenameCandidate(
                old_name=removed[i].name,
                new_name=added[j].name,
                score=score,
                structural=structural,
                name_similarity=name,
                description_similarity=description,
                accepted=score >= threshold,
                ambiguous=abs(score - threshold) <= ambiguous_band,
            )
        )
    return tuple(sorted(candidates, key=lambda c: (-c.score, c.old_name, c.new_name)))


def signature_collision_bound(items: list[Parameter]) -> tuple[int, int, float]:
    """An upper bound on structurally indistinguishable pairs, by counting alone.

    Returns ``(colliding_items, distinct_signatures, collision_fraction)``.

    This is derived from the multiset of exact structural signatures and from
    nothing the matcher computes: if two items share an exact signature, no
    structural term can separate them, so every false match the structural
    term alone can make lies inside this set. It is the independently derived
    bound the matcher's measured false-match rate is checked against in
    ``validation/validate_bounds.py``. The counting argument is the classical
    occupancy argument of Feller, *An Introduction to Probability Theory and
    Its Applications*, Volume I, 3rd edition, 1968, chapter II.
    """
    signatures: dict[tuple[Any, ...], int] = {}
    for item in items:
        key = item.structural_signature()
        signatures[key] = signatures.get(key, 0) + 1
    colliding = sum(count for count in signatures.values() if count > 1)
    distinct = len(signatures)
    fraction = colliding / len(items) if items else 0.0
    return (colliding, distinct, fraction)
