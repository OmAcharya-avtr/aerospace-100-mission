"""The provenance lattice: the algebra every derived figure in this package obeys.

Provenance is not a logging field here. It is the one piece of information that
decides whether a number may be called a measurement, and it is carried in the
type of every value this package produces.

The lattice
-----------
Three elements, totally ordered by how much they entitle a reader to believe:

``SIMULATED`` < ``RECORDED`` < ``LIVE``

``LIVE`` is a reading taken from a device that responded now. ``RECORDED`` is a
replay of a reading some device gave earlier. ``SIMULATED`` is a number a
generator made up. The order is the order of entitlement, not of usefulness: a
simulated trace is often the more useful thing to develop against, and it is
still not evidence.

A chain of three elements is a lattice, and :func:`meet` is its greatest lower
bound. Every derived quantity in this package carries the **meet** of the
provenance of its inputs, which is the formal statement of the only rule that
matters: *a derivation is no better than its worst input.* ``LIVE ∧ SIMULATED =
SIMULATED``, so one simulated reading in an otherwise live set makes the whole
aggregate simulated, and no amount of live data dilutes it.

The laws, stated so they can be tested rather than believed
-----------------------------------------------------------
``meet`` on this set is a commutative, associative, idempotent, monotone
operation with identity ``LIVE`` and absorbing element ``SIMULATED``; that is,
``(Provenance, meet)`` is a bounded meet-semilattice and a commutative
idempotent monoid. ``join`` is its dual. Because the order is total the lattice
is a chain, hence distributive, and the absorption laws hold. Every one of
those laws is checked by Hypothesis in ``tests/test_provenance_properties.py``
and exhaustively over all 3^3 triples in
``validation/validate_provenance_lattice.py``.

Reference for the algebra: B. A. Davey and H. A. Priestley, *Introduction to
Lattices and Order*, 2nd edition, Cambridge University Press 2002, chapter 2
(definitions 2.1--2.8: order, lattice, meet, join, bounded, distributive).
Nothing here is novel mathematics; what is engineering is that the package
refuses to let a figure escape without it.

The empty derivation
--------------------
``LIVE`` is the identity of ``meet``, so the meet over an EMPTY collection is
``LIVE`` by the usual convention for a monoid fold. That convention is a trap
in this product: it would let a derivation with no inputs at all claim the
strongest provenance in the lattice and be labelled a measurement.
:func:`meet_all` therefore **raises** on an empty collection rather than
returning the identity, and the identity law is tested in the form
``meet(IDENTITY, x) == x`` instead. The trap is one of the routes the
adversarial check in ``validation/validate_adversarial_label.py`` attacks.
"""

from __future__ import annotations

import enum
from collections.abc import Iterable


class Provenance(enum.Enum):
    """Where a reading came from. Ordered by entitlement, weakest first."""

    SIMULATED = "simulated"
    RECORDED = "recorded"
    LIVE = "live"

    def __str__(self) -> str:
        return self.value


#: Integer rank of each element, weakest first. This encoding is what
#: :func:`meet` uses. It is checked against an order relation derived by
#: transitive closure of the Hasse diagram -- mathematics this module does not
#: use -- in ``validation/validate_independent_bound.py``.
_RANK: dict[Provenance, int] = {
    Provenance.SIMULATED: 0,
    Provenance.RECORDED: 1,
    Provenance.LIVE: 2,
}

#: Identity of :func:`meet`: the top of the lattice. ``meet(IDENTITY, x) == x``.
IDENTITY = Provenance.LIVE

#: Absorbing element of :func:`meet`: the bottom. ``meet(ABSORBING, x) == ABSORBING``.
ABSORBING = Provenance.SIMULATED

#: The chain, weakest first. Order is part of the contract, not an accident of
#: enum declaration order, so it is written down.
ORDER: tuple[Provenance, ...] = (
    Provenance.SIMULATED,
    Provenance.RECORDED,
    Provenance.LIVE,
)


class ProvenanceError(ValueError):
    """Raised when provenance cannot be established, rather than downgraded."""


def _checked(value: object, argument: str) -> Provenance:
    """Return ``value`` as a :class:`Provenance`, or raise naming ``argument``.

    An exact type check, not ``isinstance``. A subclass of an ``enum.Enum``
    member is not constructible, but an arbitrary object carrying a ``value``
    attribute is, and the rank lookup would then raise an opaque ``KeyError``
    instead of saying what was wrong.
    """
    if type(value) is not Provenance:
        raise ProvenanceError(
            f"{argument} must be a boardprobe.Provenance member "
            f"(one of {', '.join(p.value for p in ORDER)}); got "
            f"{type(value).__name__} {value!r}"
        )
    return value


def rank(value: Provenance) -> int:
    """Position of ``value`` in the chain: 0 simulated, 1 recorded, 2 live."""
    return _RANK[_checked(value, "value")]


def leq(left: Provenance, right: Provenance) -> bool:
    """``True`` when ``left`` is no stronger than ``right`` in the chain."""
    return rank(left) <= rank(right)


def meet(left: Provenance, right: Provenance) -> Provenance:
    """Greatest lower bound: the weaker of two provenances.

    This is the whole propagation rule. ``meet(LIVE, SIMULATED)`` is
    ``SIMULATED``, because a figure derived from a live reading and a simulated
    one is not a measurement of anything.
    """
    return left if rank(left) <= rank(right) else right


def join(left: Provenance, right: Provenance) -> Provenance:
    """Least upper bound: the stronger of two provenances.

    Provided because the lattice laws are only statable with both operations
    (the absorption and distributivity laws relate them), and because the dual
    is what a *selection* of the best available source would use. **No
    derivation in this package ever joins provenances**; joining is how a
    simulated reading would get promoted, and there is deliberately no caller.
    """
    return left if rank(left) >= rank(right) else right


def meet_all(values: Iterable[Provenance]) -> Provenance:
    """Meet over a non-empty collection. Raises on empty; see the module docstring.

    Returning the identity ``LIVE`` for an empty collection is the ordinary
    monoid convention and is refused here, because a derivation with no inputs
    has no provenance and must not be allowed to claim the strongest one.
    """
    collected = tuple(values)
    if not collected:
        raise ProvenanceError(
            "provenance cannot be established from an empty collection of "
            "readings: the meet of nothing is the lattice identity (live), "
            "which would let a derivation with no inputs claim to be a "
            "measurement. Supply at least one reading, or abort the run."
        )
    result = _checked(collected[0], "values[0]")
    for index, value in enumerate(collected[1:], start=1):
        result = meet(result, _checked(value, f"values[{index}]"))
    return result
