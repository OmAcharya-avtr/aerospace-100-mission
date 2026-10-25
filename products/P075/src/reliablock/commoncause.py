"""Common cause, as a declared beta factor, applied by rewriting the diagram.

Why this is the headline
------------------------

Independence is the assumption that fails in flight hardware. Two identical
units from one lot, qualified by one procedure, mounted on one panel, powered
from one bus, exposed to one thermal cycle and inspected by one technician are
not independent, and the arithmetic that treats them as independent produces
a number that is wrong in a known direction by an unknown amount. Most quoted
redundancy figures do not say what was assumed. This module makes the
assumption a declared number and reports what it is worth.

The beta-factor model
---------------------

A component's total failure rate is split into an independent part and a part
attributed to a cause shared with the rest of its redundant group:

    ``lambda_total = lambda_independent + lambda_common``,
    ``lambda_common = beta * lambda_total``

so ``lambda_independent = (1 - beta) * lambda_total``. The shared cause is
modelled as a single additional basic event in series with the whole redundant
group: when it occurs, every member of the group fails together.

For a group of ``n`` members each with cumulative hazard ``H`` and a structure
function ``phi``:

    ``R_group = exp(-beta H) * R_phi((1 - beta) H)``

Both limits are exact and are asserted as property tests. At ``beta = 0`` the
first factor is 1 and the second is the independent result. At ``beta = 1``
the second factor is ``R_phi(0) = 1`` and the group reduces to ``exp(-H)``,
the reliability of a single member -- which is the correct answer, because a
group in which every failure is shared has no redundancy at all.

References for the model, not for a number
------------------------------------------

Fleming, K. N. (1975), "A reliability model for common mode failures in
redundant safety systems", General Atomic report GA-A13284, is the origin of
the beta factor. Mosleh, A., Fleming, K. N., Parry, G. W., Paula, H. M.,
Worledge, D. H. and Rasmuson, D. M. (1988), "Procedures for Treating Common
Cause Failures in Safety and Reliability Studies", NUREG/CR-4780 / EPRI
NP-5613, is the standard treatment and sets the beta factor in the context of
the wider class of parametric models (multiple Greek letter, alpha factor,
binomial failure rate). IEC 61508-6:2010, Annex D, gives the beta-factor
quantification used in functional-safety practice.

**No numerical beta in this repository comes from any of those sources.** Every
beta used in a validation script, an example or the README is a declared input
chosen to span a range, and the README says so. A beta factor for real hardware
comes from the plant-specific or programme-specific data the sources above
describe how to collect, and this package has no such data and does not
pretend to.

Why identical members are required
----------------------------------

The beta-factor model is symmetric: one shared event fails the whole group, and
``lambda_common = beta lambda`` needs one ``lambda``. For a group of
non-identical members there is no single shared rate, and the usual fixes --
taking the mean, the minimum or the maximum -- are modelling choices with
different answers and no principled basis. This module therefore **refuses** a
declared beta on a group whose direct members do not share one distribution,
and names the alternative (an alpha-factor or multiple-Greek-letter model,
which this package does not implement) rather than picking a convention
silently.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .diagram import (
    Block,
    Diagram,
    KOfN,
    Network,
    Node,
    Parallel,
    Series,
    SharedLoad,
    Standby,
)
from .distributions import (
    Exponential,
    FailureDistribution,
    FixedProbability,
    Lognormal,
    Weibull,
)

__all__ = [
    "GroupBetaEffect",
    "ScaledHazard",
    "apply_common_cause",
    "common_cause_event_name",
    "group_beta_effects",
    "strip_common_cause",
]


def common_cause_event_name(node: Node, index: int) -> str:
    """Name of the synthetic block carrying a group's shared-cause event."""
    label = node.label or f"{node.kind}#{index}"
    return f"{label}::common_cause"


@dataclass(frozen=True)
class ScaledHazard(FailureDistribution):
    """A distribution whose cumulative hazard is ``factor`` times another's.

    The beta split multiplies a cumulative hazard by ``1 - beta`` for the
    independent part and by ``beta`` for the shared part. Scaling the hazard
    rather than the distribution's parameters keeps the split exact for every
    distribution kind: for an exponential it is equivalent to scaling the rate,
    for a Weibull it is equivalent to scaling ``eta`` by
    ``factor**(-1/shape)``, and for a lognormal there is no equivalent
    parameter change at all, which is precisely why the hazard is the thing
    that gets scaled.
    """

    base: FailureDistribution | None = None
    factor: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.base, FailureDistribution):
            raise TypeError(
                f"ScaledHazard needs a FailureDistribution base, got {type(self.base).__name__}"
            )
        factor = float(self.factor)
        if not 0.0 <= factor <= 1.0:
            raise ValueError(f"ScaledHazard factor must lie in [0, 1], got {self.factor!r}")
        object.__setattr__(self, "factor", factor)

    @property
    def ages_with_time(self) -> bool:  # type: ignore[override]
        """Inherited from the base distribution."""
        assert self.base is not None
        return self.base.ages_with_time

    def cumulative_hazard(self, t_hours: float) -> float:
        """``factor * H_base(t)``, dimensionless."""
        assert self.base is not None
        return self.factor * self.base.cumulative_hazard(t_hours)

    def density(self, t_hours: float) -> float:
        """``factor * h_base(t) * R(t)``, units 1/hour.

        The scaled distribution's hazard *rate* is ``factor`` times the base's,
        so its density is that rate times its own reliability -- not
        ``factor`` times the base's density, which would use the wrong
        survival term.
        """
        assert self.base is not None
        base_rate = _hazard_rate(self.base, t_hours)
        return self.factor * base_rate * self.reliability(t_hours)

    def describe(self) -> str:
        assert self.base is not None
        return f"{self.factor:.6g} x {self.base.describe()}"

    def to_mapping(self) -> dict[str, Any]:
        assert self.base is not None
        return {"kind": "scaled_hazard", "factor": self.factor, "base": self.base.to_mapping()}


def _hazard_rate(distribution: FailureDistribution, t_hours: float) -> float:
    """Instantaneous hazard rate ``h(t) = f(t) / R(t)``, units 1/hour."""
    if isinstance(distribution, Exponential):
        return distribution.rate_per_hour
    reliability = distribution.reliability(t_hours)
    if reliability <= 0.0:
        return 0.0
    return distribution.density(t_hours) / reliability


def _same_distribution(a: FailureDistribution, b: FailureDistribution) -> bool:
    """True when two distributions are the same kind with the same parameters."""
    if type(a) is not type(b):
        return False
    if isinstance(a, Exponential) and isinstance(b, Exponential):
        return a.rate_per_hour == b.rate_per_hour
    if isinstance(a, Weibull) and isinstance(b, Weibull):
        return a.shape == b.shape and a.scale_hours == b.scale_hours
    if isinstance(a, Lognormal) and isinstance(b, Lognormal):
        return a.log_median == b.log_median and a.log_sigma == b.log_sigma
    if isinstance(a, FixedProbability) and isinstance(b, FixedProbability):
        return a.probability == b.probability
    if isinstance(a, ScaledHazard) and isinstance(b, ScaledHazard):
        assert a.base is not None and b.base is not None
        return a.factor == b.factor and _same_distribution(a.base, b.base)
    return a.to_mapping() == b.to_mapping()  # pragma: no cover - future kinds


def apply_common_cause(diagram: Diagram) -> Diagram:
    """Rewrite ``diagram`` so that every declared beta factor is in the structure.

    A redundant node with ``beta > 0`` becomes a :class:`~reliablock.diagram.Series`
    of two things: a single synthetic block carrying the shared-cause event at
    cumulative hazard ``beta H``, and the original node with ``beta`` cleared
    and each member's hazard scaled by ``1 - beta``.

    Returns a new :class:`~reliablock.diagram.Diagram`; the input is not
    modified. A diagram with no declared beta is returned with an equal
    structure, so a caller can always run both and compare.

    Raises
    ------
    ValueError
        If a beta is declared on a group whose direct members do not share one
        distribution, or on a group with a nested structure as a member. Both
        are refusals rather than conventions: see the module docstring.
    """
    rewritten = _rewrite(diagram.root, 0)[0]
    notes = tuple(diagram.notes)
    betas = diagram.declared_betas
    if betas:
        rendered = ", ".join(f"{k}={v:g}" for k, v in sorted(betas.items()))
        notes = notes + (
            f"Common cause applied: {rendered}. Each group carries a synthetic "
            "block named '<label>::common_cause' at beta times its members' "
            "cumulative hazard, and each member's hazard is scaled by 1 - beta.",
        )
    return Diagram(
        name=f"{diagram.name} (with declared common cause)" if betas else diagram.name,
        root=rewritten,
        description=diagram.description,
        notes=notes,
    )


def strip_common_cause(diagram: Diagram) -> Diagram:
    """Return ``diagram`` with every declared beta factor set to zero.

    The independence-assuming half of the headline comparison. Every other
    declaration is preserved, so the two diagrams differ in exactly the
    assumption under test.
    """
    return Diagram(
        name=f"{diagram.name} (independence assumed)" if diagram.declared_betas else diagram.name,
        root=_clear_beta(diagram.root),
        description=diagram.description,
        notes=tuple(diagram.notes)
        + (
            (
                "Common cause suppressed: every declared beta factor set to zero. "
                "Every member of every redundant group is treated as failing "
                "independently.",
            )
            if diagram.declared_betas
            else ()
        ),
    )


def _clear_beta(item: Node | Block) -> Node | Block:
    """Recursively set every ``beta_factor`` to zero."""
    if isinstance(item, Block):
        return item
    if isinstance(item, Series):
        return replace(item, items=tuple(_clear_beta(c) for c in item.items))
    if isinstance(item, Parallel):
        return replace(item, items=tuple(_clear_beta(c) for c in item.items), beta_factor=0.0)
    if isinstance(item, KOfN):
        return replace(item, items=tuple(_clear_beta(c) for c in item.items), beta_factor=0.0)
    if isinstance(item, Standby):
        return replace(
            item,
            primary=None if item.primary is None else _clear_beta(item.primary),
            standbys=tuple(_clear_beta(c) for c in item.standbys),
        )
    if isinstance(item, SharedLoad):
        return item
    if isinstance(item, Network):
        return item
    raise TypeError(f"cannot clear beta on {type(item).__name__}")  # pragma: no cover


def _rewrite(item: Node | Block, index: int) -> tuple[Node | Block, int]:
    """Recursively apply beta factors. Returns the new item and the next index."""
    if isinstance(item, Block):
        return item, index + 1
    own_index = index
    index += 1
    children: list[Node | Block] = []
    for child in item.children:
        new_child, index = _rewrite(child, index)
        children.append(new_child)
    rebuilt = _rebuild(item, children)
    beta = item.beta
    if beta <= 0.0:
        return rebuilt, index
    members = _beta_members(item, rebuilt)
    shared = Block(
        name=common_cause_event_name(item, own_index),
        distribution=ScaledHazard(base=members[0].distribution, factor=beta),
        description=(
            f"shared-cause event for the {len(members)}-member group "
            f"{item.label or item.kind!r}, at beta = {beta:g} of a member's "
            "cumulative hazard"
        ),
    )
    independent = _scale_members(rebuilt, 1.0 - beta)
    return (
        Series(
            items=(shared, independent),
            label=f"{item.label or item.kind}::with_common_cause",
        ),
        index,
    )


def _rebuild(item: Node, children: list[Node | Block]) -> Node:
    """Rebuild ``item`` with new children, preserving every other field."""
    if isinstance(item, Series):
        return replace(item, items=tuple(children))
    if isinstance(item, Parallel):
        return replace(item, items=tuple(children))
    if isinstance(item, KOfN):
        return replace(item, items=tuple(children))
    if isinstance(item, Standby):
        return replace(item, primary=children[0], standbys=tuple(children[1:]))
    if isinstance(item, SharedLoad):
        units = tuple(c for c in children if isinstance(c, Block))
        if len(units) != len(children):  # pragma: no cover - guarded at validation
            raise ValueError("shared_load units must be blocks")
        return replace(item, units=units)
    if isinstance(item, Network):
        return item
    raise TypeError(f"cannot rebuild {type(item).__name__}")  # pragma: no cover


def _beta_members(item: Node, rebuilt: Node) -> tuple[Block, ...]:
    """The group's direct members, checked to be identical blocks.

    Raises ``ValueError`` naming the alternative model rather than adopting a
    convention for a non-identical group.
    """
    where = item.label or item.kind
    members = rebuilt.children
    if not all(isinstance(m, Block) for m in members):
        raise ValueError(
            f"node {where!r} declares beta_factor={item.beta:g} but at least one "
            "member is a nested structure. The beta-factor model fails a whole "
            "group through one shared event; for a group of sub-structures, "
            "declare the beta on the redundant group of blocks that actually "
            "shares the cause."
        )
    blocks: tuple[Block, ...] = tuple(m for m in members if isinstance(m, Block))
    first = blocks[0].distribution
    different = [b.name for b in blocks[1:] if not _same_distribution(b.distribution, first)]
    if different:
        raise ValueError(
            f"node {where!r} declares beta_factor={item.beta:g} over members that do "
            f"not share one distribution: {different} differ from {blocks[0].name!r}. "
            "The beta-factor model is symmetric -- one shared event fails the whole "
            "group, and lambda_common = beta lambda needs one lambda. Taking the "
            "mean, minimum or maximum of the members' rates gives three different "
            "answers with no principled basis. A non-identical group needs an "
            "alpha-factor or multiple-Greek-letter model, which this package does "
            "not implement (see NUREG/CR-4780); it is refused here rather than "
            "approximated silently."
        )
    return blocks


def _scale_members(node: Node, factor: float) -> Node:
    """Scale every direct member block's cumulative hazard by ``factor``."""
    scaled: list[Node | Block] = []
    for child in node.children:
        if not isinstance(child, Block):  # pragma: no cover - guarded above
            raise ValueError("only a group of blocks can carry a beta factor")
        scaled.append(
            replace(
                child,
                distribution=ScaledHazard(base=child.distribution, factor=factor),
                beta_group=node.label or node.kind,
            )
        )
    rebuilt = _rebuild(node, scaled)
    if isinstance(rebuilt, (Parallel, KOfN)):
        rebuilt = replace(rebuilt, beta_factor=0.0)
    return rebuilt


@dataclass(frozen=True)
class GroupBetaEffect:
    """What a declared beta factor does to one redundant group, measured.

    ``conservative`` is True when declaring the beta factor **increases** the
    group's unreliability, which is what a reader expects a correlation model to
    do. It is False when the beta factor *decreases* it, and that case is real,
    not a defect: see :func:`group_beta_effects`.
    """

    label: str
    kind: str
    beta: float
    members: int
    k: int
    #: Group unreliability with the beta factor suppressed.
    unreliability_independent: float
    #: Group unreliability with the beta factor applied.
    unreliability_with_common_cause: float

    @property
    def ratio(self) -> float:
        """``Q_with_cc / Q_independent``; ``inf`` when the independent figure is 0."""
        if self.unreliability_independent == 0.0:
            return float("inf")
        return self.unreliability_with_common_cause / self.unreliability_independent

    @property
    def conservative(self) -> bool:
        """True when the declared beta factor increases this group's unreliability."""
        return self.unreliability_with_common_cause >= self.unreliability_independent

    def __str__(self) -> str:
        verdict = "increases Q" if self.conservative else "DECREASES Q"
        return (
            f"{self.label:<22s} {self.kind:<9s} beta={self.beta:<6.3g} "
            f"{self.k}-of-{self.members}  Q {self.unreliability_independent:.6e} -> "
            f"{self.unreliability_with_common_cause:.6e}  ratio {self.ratio:.4f}  {verdict}"
        )


def group_beta_effects(diagram: Diagram, mission: Any) -> tuple[GroupBetaEffect, ...]:
    """Per-group effect of each declared beta factor, measured group by group.

    Why this function exists
    ------------------------

    The beta-factor model has **exactly one** common-cause mode: the event that
    fails every member of the group at once. For a parallel group (``k = 1``)
    that mode is the only way the group can fail anyway, so moving failure
    probability into it always increases the group's unreliability, and the
    model is conservative.

    For a ``k``-of-``n`` group with ``k >= 2`` the group fails once
    ``n - k + 1`` members have failed, which is fewer than all of them. The
    single common-cause mode fails all ``n``, which is more failures than the
    group needs, so moving probability from the independent part into it can
    *reduce* the computed group unreliability. Measured on a 3-of-4 group at
    beta = 0.1, the ratio is 2.5162 at a member cumulative hazard of 0.01 and
    0.9442 at 0.3 -- it crosses one. The locus of the crossing is mapped in
    ``validation/validate_commoncause_output.txt``.

    This is a known inadequacy of the beta-factor model for groups larger than
    two, and the reason NUREG/CR-4780 introduces the multiple-Greek-letter and
    alpha-factor models, which carry a separate parameter for each size of
    partial common-cause failure. Those are **not implemented here.** This
    function exists so that the limitation is reported on the actual
    declaration rather than left in a document: any group for which
    ``conservative`` is False is a group whose beta factor is making the answer
    look better, and the report says so.
    """
    from .quantify import quantify

    effects: list[GroupBetaEffect] = []
    for index, item in enumerate(diagram.walk()):
        if not isinstance(item, Node) or item.beta <= 0.0:
            continue
        label = item.label or f"{item.kind}#{index}"
        group = Diagram(name=label, root=item)
        independent = quantify(strip_common_cause(group), mission).unreliability
        with_cc = quantify(apply_common_cause(group), mission).unreliability
        effects.append(
            GroupBetaEffect(
                label=label,
                kind=item.kind,
                beta=item.beta,
                members=len(item.children),
                k=getattr(item, "k", 1),
                unreliability_independent=independent,
                unreliability_with_common_cause=with_cc,
            )
        )
    return tuple(effects)
