"""The five-component bridge network: a closed form derived here, not quoted.

What this module is for
-----------------------

The bridge is the smallest reliability structure that is **not**
series-parallel: no nesting of series and parallel groups reproduces it, so a
package that can only nest those two cannot express it at all, and a
general-network implementation that silently decomposes it as series-parallel
is wrong in a way no series-parallel test case can catch. It is therefore the
right check on :mod:`reliablock.quantify`'s pivotal decomposition.

The network
-----------

Four vertices ``s``, ``a``, ``b``, ``t`` and five edges::

    e1: s - a      e2: s - b      e3: a - b (the bridge)
    e4: a - t      e5: b - t

The system succeeds when surviving edges connect ``s`` to ``t``.

Minimal path sets: ``{e1, e4}``, ``{e2, e5}``, ``{e1, e3, e5}``,
``{e2, e3, e4}``.
Minimal cut sets: ``{e1, e2}``, ``{e4, e5}``, ``{e1, e3, e5}``,
``{e2, e3, e4}``.

The closed form, derived
------------------------

Pivotal (conditional-probability) decomposition on the bridge edge ``e3``,
which is the classical method for this structure:

    ``R = p_3 R(e3 works) + (1 - p_3) R(e3 failed)``

With ``e3`` working, vertices ``a`` and ``b`` are one node. Then ``s`` reaches
it through ``e1`` or ``e2`` in parallel, and it reaches ``t`` through ``e4`` or
``e5`` in parallel, and the two parallel groups are in series. With equal edge
reliability ``p``:

    ``R(e3 works) = (1 - (1 - p)^2)^2 = (2p - p^2)^2
                  = 4p^2 - 4p^3 + p^4``

With ``e3`` failed, only the two disjoint paths ``{e1, e4}`` and ``{e2, e5}``
remain, in parallel:

    ``R(e3 failed) = 1 - (1 - p^2)^2 = 2p^2 - p^4``

Substituting and expanding::

    R = p (4p^2 - 4p^3 + p^4) + (1 - p)(2p^2 - p^4)
      = 4p^3 - 4p^4 + p^5 + 2p^2 - p^4 - 2p^3 + p^5
      = 2p^5 - 5p^4 + 2p^3 + 2p^2

which is :func:`bridge_reliability_equal`. At ``p = 0.9`` it gives
``0.978480``.

What is cited and what is derived
---------------------------------

**Derived here, in full, above:** the closed form
``R = 2p^5 - 5p^4 + 2p^3 + 2p^2``, and the value at ``p = 0.9``.

**Cited for the method, not for a number:** the bridge network and the
conditional-probability (pivotal) decomposition of it are standard material in
the reliability-engineering literature. Barlow, R. E. and Proschan, F. (1975),
*Statistical Theory of Reliability and Life Testing*, Holt, Rinehart and
Winston, chapter 1, gives the pivotal decomposition of a coherent structure.
Billinton, R. and Allan, R. N., *Reliability Evaluation of Engineering
Systems*, 2nd edition, Plenum Press, 1992, treats the bridge network and the
conditional-probability method applied to it.

**No page number and no published figure is quoted for the bridge result**, by
deliberate choice: this build had no access to either volume, and a page
number that cannot be checked is worse than none. The closed form above stands
on its own derivation, and it is verified three further independent ways in
``validation/validate_bridge.py``:

1. exhaustive enumeration of all ``2**5 = 32`` states against the structure
   function, with no algebra in common with the polynomial;
2. inclusion-exclusion over the four minimal path sets;
3. the package's own pivotal decomposition in
   :mod:`reliablock.quantify`, over the declared ``network`` node.

Four routes to one polynomial is a stronger statement than a citation would
have been, and it is reproducible from this repository alone.
"""

from __future__ import annotations

from .diagram import Block, Diagram, Network
from .distributions import Exponential, FixedProbability
from .numerics import hazard_from_unreliability

__all__ = [
    "BRIDGE_CUT_SETS",
    "BRIDGE_PATH_SETS",
    "bridge_diagram",
    "bridge_reliability_equal",
    "bridge_reliability_inclusion_exclusion",
    "bridge_reliability_pivotal",
]

#: Minimal path sets of the five-edge bridge, as edge-name sets.
BRIDGE_PATH_SETS: tuple[frozenset[str], ...] = (
    frozenset({"e1", "e4"}),
    frozenset({"e2", "e5"}),
    frozenset({"e1", "e3", "e5"}),
    frozenset({"e2", "e3", "e4"}),
)

#: Minimal cut sets of the five-edge bridge, as edge-name sets.
BRIDGE_CUT_SETS: tuple[frozenset[str], ...] = (
    frozenset({"e1", "e2"}),
    frozenset({"e4", "e5"}),
    frozenset({"e1", "e3", "e5"}),
    frozenset({"e2", "e3", "e4"}),
)


def bridge_reliability_equal(p: float) -> float:
    """``R = 2p^5 - 5p^4 + 2p^3 + 2p^2`` for equal edge reliability ``p``.

    The closed form derived in the module docstring. Dimensionless, and in
    ``[0, 1]`` for ``p`` in ``[0, 1]``.

    Evaluated by Horner's rule on the polynomial
    ``p^2 (2 + 2p - 5p^2 + 2p^3)``, which keeps the five terms from being
    summed at different magnitudes.
    """
    value = float(p)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"p must lie in [0, 1], got {value}")
    inner = 2.0 + value * (2.0 + value * (-5.0 + 2.0 * value))
    return value * value * inner


def bridge_reliability_pivotal(p1: float, p2: float, p3: float, p4: float, p5: float) -> float:
    """Bridge reliability for non-identical edges, by pivotal decomposition on ``e3``.

    ``R = p3 [1 - (1-p1)(1-p2)][1 - (1-p4)(1-p5)]
          + (1 - p3) [1 - (1 - p1 p4)(1 - p2 p5)]``

    The same decomposition as in the module docstring, written out without
    assuming the edges are identical. With all five equal this reduces to
    :func:`bridge_reliability_equal`, which is asserted as a property test over
    random ``p``.
    """
    probabilities = (p1, p2, p3, p4, p5)
    for index, value in enumerate(probabilities, start=1):
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"p{index} must lie in [0, 1], got {value}")
    merged = (1.0 - (1.0 - p1) * (1.0 - p2)) * (1.0 - (1.0 - p4) * (1.0 - p5))
    split = 1.0 - (1.0 - p1 * p4) * (1.0 - p2 * p5)
    return p3 * merged + (1.0 - p3) * split


def bridge_reliability_inclusion_exclusion(reliabilities: dict[str, float]) -> float:
    """Bridge reliability by inclusion-exclusion over the four minimal path sets.

    ``R = sum_j P(path j up) - sum_{j<k} P(paths j and k up) + ...``, where
    ``P(a set of paths all up)`` is the product of the reliabilities of the
    union of their edges. Sixteen terms with alternating signs; an independent
    route to the same number as :func:`bridge_reliability_pivotal` with no
    algebra in common, and the one that would expose a wrong path set.
    """
    from itertools import combinations

    missing = {e for path in BRIDGE_PATH_SETS for e in path} - set(reliabilities)
    if missing:
        raise KeyError(f"missing edge reliabilities for {sorted(missing)}")
    total = 0.0
    paths = BRIDGE_PATH_SETS
    for size in range(1, len(paths) + 1):
        sign = 1.0 if size % 2 else -1.0
        for chosen in combinations(paths, size):
            union: frozenset[str] = frozenset()
            for path in chosen:
                union = union | path
            product = 1.0
            for edge in sorted(union):
                product *= reliabilities[edge]
            total += sign * product
    return total


def bridge_diagram(
    p: float | None = None,
    *,
    edge_unreliabilities: dict[str, float] | None = None,
    mission_hours: float = 1.0,
) -> Diagram:
    """A declared ``network`` diagram for the five-edge bridge.

    With ``p`` given, every edge is a one-shot block at failure probability
    ``1 - p``, so the diagram's reliability at any mission duration is the
    closed form at ``p``. With ``edge_unreliabilities`` given, each edge takes
    its own exponential rate chosen to reach that unreliability at
    ``mission_hours``, which is how the validation script drives the five
    edges independently.
    """
    if (p is None) == (edge_unreliabilities is None):
        raise ValueError("give exactly one of p or edge_unreliabilities")
    edges: list[tuple[str, str, Block]] = []
    layout = (
        ("s", "a", "e1"),
        ("s", "b", "e2"),
        ("a", "b", "e3"),
        ("a", "t", "e4"),
        ("b", "t", "e5"),
    )
    for tail, head, name in layout:
        if p is not None:
            block = Block(name=name, distribution=FixedProbability(1.0 - float(p)))
        else:
            assert edge_unreliabilities is not None
            if name not in edge_unreliabilities:
                raise KeyError(f"edge_unreliabilities is missing {name!r}")
            hazard = hazard_from_unreliability(edge_unreliabilities[name])
            rate = hazard / float(mission_hours) if mission_hours > 0.0 else 0.0
            block = Block(name=name, distribution=Exponential(rate))
        edges.append((tail, head, block))
    return Diagram(
        name="textbook five-edge bridge network",
        root=Network(edges=tuple(edges), source="s", sink="t", label="bridge"),
        description=(
            "The classical five-component bridge. Not series-parallel: its "
            "closed form R = 2p^5 - 5p^4 + 2p^3 + 2p^2 is derived in "
            "reliablock.bridge and verified four independent ways in "
            "validation/validate_bridge.py."
        ),
        notes=(
            "Minimal path sets: {e1,e4}, {e2,e5}, {e1,e3,e5}, {e2,e3,e4}.",
            "Minimal cut sets: {e1,e2}, {e4,e5}, {e1,e3,e5}, {e2,e3,e4}.",
        ),
    )
