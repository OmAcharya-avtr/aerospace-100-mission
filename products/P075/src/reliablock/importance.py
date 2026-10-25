"""Importance measures: Birnbaum, criticality and Fussell-Vesely.

Definitions, and what each one answers
--------------------------------------

Let ``Q_sys`` be the system unreliability and ``q_i`` the unreliability of
basic event ``i``.

**Birnbaum.** ``I_B(i) = dQ_sys / dq_i = Q_sys(q_i = 1) - Q_sys(q_i = 0)``.
The rate at which the system unreliability responds to a change in event
``i``. It does not involve ``q_i`` itself, so it answers "how much does this
position in the diagram matter", not "how much does this item contribute now".
Birnbaum, Z. W. (1969), "On the importance of different components in a
multicomponent system", in *Multivariate Analysis II*, Academic Press,
pp. 581-592.

**Criticality.** ``I_C(i) = I_B(i) q_i / Q_sys``. The probability that event
``i`` has occurred and is critical for the system, given that the system has
failed. This is the one that answers "where is my unreliability coming from",
and it sums over all events to a number that exceeds 1 only where cut sets
overlap.

**Fussell-Vesely.** ``I_FV(i) = P(at least one minimal cut set containing i has
occurred) / Q_sys``. The fraction of system unreliability to which event ``i``
contributes through any cut set it appears in, which is the measure risk
assessments usually quote. Fussell, J. B. (1975), "How to hand-calculate system
reliability and safety characteristics", IEEE Transactions on Reliability
R-24(3), 169-174. Computed here **exactly**, by enumerating the states of the
events appearing in the relevant cut sets, not by the rare-event approximation
``sum of cut set probabilities``, which overestimates and is the usual source
of a Fussell-Vesely figure above 1.

Exact relations this module asserts
-----------------------------------

``I_C(i) = (Q_sys - Q_sys(q_i = 0)) / Q_sys`` is an identity, from the pivotal
decomposition ``Q_sys = q_i Q_sys(q_i = 1) + (1 - q_i) Q_sys(q_i = 0)``:

    ``Q_sys - Q_sys(q_i = 0) = q_i [Q_sys(q_i = 1) - Q_sys(q_i = 0)]
                             = q_i I_B(i)``

and ``I_FV(i) >= I_C(i)`` always, with equality when every cut set containing
``i`` becomes critical only through ``i``. Both are checked in
``validation/validate_importance.py``, and the Birnbaum derivative is checked
against a central finite difference in the same script -- the finite difference
being the only one of the three that shares no expression with the
implementation.

Scope
-----

Importance measures are defined over a diagram whose quantification *is* the
independent-event arithmetic over its Boolean structure function, which means
every node is one of ``series``, ``parallel``, ``k_of_n`` or ``network``. A
``standby`` or ``shared_load`` node has an exact Boolean structure function but
a dependent quantification, so ``dQ_sys / dq_i`` is not the derivative of the
number the quantifier returns. :func:`importance_table` raises on such a
diagram and says so, rather than returning a figure whose meaning differs from
its name.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from .diagram import Diagram
from .mission import MissionProfile
from .numerics import MAX_HAZARD, hazard_from_unreliability
from .quantify import quantify
from .structure import MAX_ENUMERATION_EVENTS, minimal_cut_sets, structure_function

__all__ = [
    "Importance",
    "boolean_system_unreliability",
    "importance_table",
    "pivot_unreliability",
    "system_unreliability_at",
]


@dataclass(frozen=True)
class Importance:
    """Importance measures for one basic event."""

    event: str
    #: Event unreliability at the mission point, dimensionless.
    unreliability: float
    #: ``dQ_sys / dq_i``, dimensionless.
    birnbaum: float
    #: ``I_B q_i / Q_sys``, dimensionless, in ``[0, 1]``.
    criticality: float
    #: ``P(some minimal cut set containing i occurred) / Q_sys``, in ``[0, 1]``.
    fussell_vesely: float
    #: Number of minimal cut sets this event appears in.
    cut_sets: int

    def __str__(self) -> str:
        return (
            f"{self.event:<28s} q={self.unreliability:.6e} "
            f"B={self.birnbaum:.6e} C={self.criticality:.6f} "
            f"FV={self.fussell_vesely:.6f} cuts={self.cut_sets}"
        )


def pivot_unreliability(
    diagram: Diagram,
    mission: MissionProfile | float,
    event: str,
    value: float,
    *,
    base_unreliabilities: dict[str, float] | None = None,
) -> float:
    """System unreliability with one event's unreliability pinned to ``value``.

    The pin is applied in unreliability space and converted to an effective
    cumulative hazard for the quantifier, so that pinning ``value = 0.0``
    really does give a block that cannot fail rather than one with a very small
    rate.
    """
    unreliabilities = (
        dict(base_unreliabilities)
        if base_unreliabilities is not None
        else _event_unreliabilities(diagram, mission)
    )
    if event not in unreliabilities:
        raise KeyError(f"unknown basic event {event!r}; known events are {sorted(unreliabilities)}")
    unreliabilities[event] = float(value)
    return _quantify_from_unreliabilities(diagram, unreliabilities)


def _event_unreliabilities(
    diagram: Diagram, mission: MissionProfile | float
) -> dict[str, float]:
    """Per-block unreliability at the mission point."""
    profile = mission if isinstance(mission, MissionProfile) else MissionProfile.from_hours(mission)
    ages = profile.effective_ages(diagram)
    return {b.name: b.distribution.unreliability(ages[b.name]) for b in diagram.blocks()}


def _quantify_from_unreliabilities(diagram: Diagram, unreliabilities: dict[str, float]) -> float:
    """System unreliability given per-block unreliabilities, through the quantifier.

    Each block's unreliability is converted to the cumulative hazard that
    produces it and handed to :func:`~reliablock.quantify.quantify` as an
    effective age against a unit-rate interpretation. The conversion is exact:
    a block's quantity depends on its distribution only through
    ``cumulative_hazard(age)``, so driving the age is driving the hazard.
    """
    # Rebuild the diagram over unit-rate exponential blocks, whose cumulative
    # hazard at age a is exactly a, then drive the ages. The structure is
    # untouched, so this is the same structural arithmetic at declared
    # component unreliabilities.
    # Clamped at MAX_HAZARD: a pinned q of exactly 1.0 maps to an infinite
    # hazard, and MAX_HAZARD already gives Q = 1.0 and R = 0.0 in float64
    # without propagating an infinity through a product.
    ages = {
        name: min(hazard_from_unreliability(q), MAX_HAZARD)
        for name, q in unreliabilities.items()
    }
    unit = Diagram(name=diagram.name, root=_rewrite_to_unit_rate(diagram.root))
    return quantify(unit, 0.0, block_ages=ages).unreliability


def _rewrite_to_unit_rate(item):  # type: ignore[no-untyped-def]
    """Replace every leaf distribution with a unit-rate exponential."""
    from dataclasses import replace

    from .diagram import Block, KOfN, Network, Parallel, Series
    from .distributions import Exponential

    if isinstance(item, Block):
        return replace(item, distribution=Exponential(1.0))
    if isinstance(item, Series):
        return replace(item, items=tuple(_rewrite_to_unit_rate(c) for c in item.items))
    if isinstance(item, Parallel):
        return replace(item, items=tuple(_rewrite_to_unit_rate(c) for c in item.items))
    if isinstance(item, KOfN):
        return replace(item, items=tuple(_rewrite_to_unit_rate(c) for c in item.items))
    if isinstance(item, Network):
        return replace(
            item, edges=tuple((a, b, _rewrite_to_unit_rate(blk)) for a, b, blk in item.edges)
        )
    raise ValueError(
        f"importance measures are not defined over a {type(item).__name__} node: "
        "its quantification is not the independent-event arithmetic over its "
        "Boolean structure function. See the module docstring."
    )


def importance_table(
    diagram: Diagram, mission: MissionProfile | float
) -> tuple[Importance, ...]:
    """Importance measures for every basic event, most critical first.

    Raises
    ------
    ValueError
        If the diagram contains a ``standby`` or ``shared_load`` node, whose
        quantification is not the independent-event arithmetic over its
        structure function.
    """
    if not diagram.is_boolean:
        offending = sorted(
            {
                n.kind
                for n in diagram.walk()
                if hasattr(n, "kind") and n.kind in {"standby", "shared_load"}
            }
        )
        raise ValueError(
            f"importance measures are not defined for this diagram: it contains "
            f"{offending} node(s). Those have an exact Boolean structure function "
            "but a dependent quantification, so dQ_sys/dq_i is not the derivative "
            "of the number quantify() returns. Returning a figure whose meaning "
            "differs from its name would be worse than refusing."
        )
    base = _event_unreliabilities(diagram, mission)
    q_system = _quantify_from_unreliabilities(diagram, base)
    cuts = minimal_cut_sets(diagram)
    out: list[Importance] = []
    for event in diagram.block_names():
        high = dict(base)
        high[event] = 1.0
        low = dict(base)
        low[event] = 0.0
        birnbaum = _quantify_from_unreliabilities(diagram, high) - _quantify_from_unreliabilities(
            diagram, low
        )
        criticality = birnbaum * base[event] / q_system if q_system > 0.0 else 0.0
        relevant = tuple(cut for cut in cuts if event in cut)
        fv = (
            _union_probability(relevant, base) / q_system
            if q_system > 0.0 and relevant
            else 0.0
        )
        out.append(
            Importance(
                event=event,
                unreliability=base[event],
                birnbaum=birnbaum,
                criticality=criticality,
                fussell_vesely=fv,
                cut_sets=len(relevant),
            )
        )
    return tuple(sorted(out, key=lambda i: (-i.fussell_vesely, i.event)))


def _union_probability(
    cut_sets: tuple[frozenset[str], ...], unreliabilities: dict[str, float]
) -> float:
    """Exact probability that at least one of ``cut_sets`` has fully failed.

    Enumerated over the states of the events appearing in ``cut_sets``, which
    is exact and avoids both the rare-event approximation and the
    ``2**len(cut_sets)`` inclusion-exclusion sum with its alternating signs.
    Raises ``ValueError`` above :data:`~reliablock.structure.MAX_ENUMERATION_EVENTS`
    involved events rather than running for an unbounded time.
    """
    events = sorted({event for cut in cut_sets for event in cut})
    n = len(events)
    if n > MAX_ENUMERATION_EVENTS:
        raise ValueError(
            f"exact Fussell-Vesely refused: {n} events appear in the relevant cut "
            f"sets, above the declared cap of {MAX_ENUMERATION_EVENTS}. The "
            "rare-event approximation would fit, but it overestimates and is the "
            "usual source of a Fussell-Vesely figure above 1, so it is not "
            "substituted silently."
        )
    index = {event: i for i, event in enumerate(events)}
    total = 0.0
    for state in product((False, True), repeat=n):
        # True means failed.
        if not any(all(state[index[e]] for e in cut) for cut in cut_sets):
            continue
        probability = 1.0
        for i, failed in enumerate(state):
            q = unreliabilities[events[i]]
            probability *= q if failed else 1.0 - q
            if probability == 0.0:
                break
        total += probability
    return total


def system_unreliability_at(
    diagram: Diagram, unreliabilities: dict[str, float]
) -> float:
    """System unreliability for declared per-event unreliabilities.

    Exposed because the validation scripts drive structures at declared
    component reliabilities rather than through a distribution and a mission,
    and because the finite-difference check needs exactly this map.
    """
    return _quantify_from_unreliabilities(diagram, unreliabilities)


def boolean_system_unreliability(diagram: Diagram, unreliabilities: dict[str, float]) -> float:
    """System unreliability by exhaustive state enumeration over the structure function.

    An independent route to the same number as :func:`system_unreliability_at`,
    sharing no code with the structural quantifier: it sums the probability of
    every state the structure function calls a failure. Exponential in the
    number of events, and capped at
    :data:`~reliablock.structure.MAX_ENUMERATION_EVENTS`.
    """
    from .structure import basic_events

    events = basic_events(diagram)
    n = len(events)
    if n > MAX_ENUMERATION_EVENTS:
        raise ValueError(
            f"exhaustive enumeration refused for {n} basic events, above the "
            f"declared cap of {MAX_ENUMERATION_EVENTS}"
        )
    total = 0.0
    for state in product((False, True), repeat=n):
        alive = [events[i] for i, failed in enumerate(state) if not failed]
        if structure_function(diagram, alive):
            continue
        probability = 1.0
        for i, failed in enumerate(state):
            q = unreliabilities[events[i]]
            probability *= q if failed else 1.0 - q
            if probability == 0.0:
                break
        total += probability
    return total
