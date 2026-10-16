"""Typed model of a hierarchical engineering budget.

Vocabulary
----------
All quantities in one budget share a single unit, declared once on the budget
(kg, W, Mbit, dB). The model keeps two allowances strictly apart, because
conflating them is the arithmetic error this package exists to make visible:

**contingency** (dimensionless fraction)
    A deterministic allowance added to a nominal estimate to cover expected
    growth of that item: ``predicted = nominal * (1 + contingency)``. It is a
    bias allowance, not a random variable, so contingencies **always combine
    linearly**. Root-sum-squaring a set of contingencies is not a
    conservatism trade; it is a category error, and this package never does
    it. Maturity-keyed contingency schedules are standard practice in space
    systems mass properties control; see ANSI/AIAA S-120A-2015, *Mass
    Properties Control for Space Systems*, and NASA/SP-2016-6105 Rev 2,
    *NASA Systems Engineering Handbook*, section 6.5. This package supplies no
    schedule of its own: the policy is declared by the user and reproduced in
    the report with its stated source.

**uncertainty** (same unit as the budget, 1 sigma)
    A random dispersion about the predicted value. Uncertainties combine by
    the law of propagation of uncertainty (JCGM 100:2008, *Evaluation of
    measurement data - Guide to the expression of uncertainty in measurement*,
    section 5.1.2 for uncorrelated inputs and section 5.2.2 for correlated
    inputs). For a sum of terms with unit sensitivity coefficients that law
    reduces to ``u = sqrt(w^T R w)``, which is root-sum-square when ``R`` is
    the identity and the linear sum when ``R`` is all ones.

Validity range
--------------
The aggregation implemented here is exact for quantities that combine
**additively** in the declared unit: masses in kg, orbit-average powers in W,
data volumes in Mbit, and the log-domain terms of a link budget in dB. It is
not valid for quantities that combine multiplicatively in the declared unit
(efficiencies, ratios in linear units); convert those to a log domain first,
where they are additive, or use a general propagation library. There is no
distributional assumption anywhere: a 1 sigma value is treated as a standard
deviation and nothing is assumed about the shape of the distribution, so a
k-sigma figure here is a multiple of a standard deviation and **not** a
confidence level.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .errors import DeclarationError

#: The three implemented combination rules for uncertainty at an internal node.
Combination = Literal["linear", "rss", "correlated"]

COMBINATIONS: tuple[str, ...] = ("linear", "rss", "correlated")

#: Severity ordering used by the checks and the reports.
SEVERITIES: tuple[str, ...] = ("error", "warning", "info")


@dataclass(frozen=True)
class MarginPolicy:
    """A declared maturity-keyed contingency schedule.

    Parameters
    ----------
    name:
        Identifier referenced by a budget.
    description:
        Free text reproduced in every report that uses the policy.
    source:
        Where the numbers come from. Reproduced verbatim in reports. This
        package ships no schedule of its own and asserts nothing about the
        correctness of any schedule declared through it.
    levels:
        Maturity level name to contingency fraction (dimensionless, >= 0).
        ``0.20`` means a twenty per cent allowance on the nominal value.
    """

    name: str
    description: str
    source: str
    levels: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.levels:
            raise DeclarationError(
                f"margin policy {self.name!r} declares no maturity levels; "
                "at least one level is required"
            )
        for level, fraction in self.levels.items():
            if not isinstance(fraction, (int, float)) or isinstance(fraction, bool):
                raise DeclarationError(
                    f"margin policy {self.name!r} level {level!r}: contingency must be a "
                    f"number, got {type(fraction).__name__}"
                )
            if fraction < 0.0:
                raise DeclarationError(
                    f"margin policy {self.name!r} level {level!r}: contingency must be "
                    f">= 0, got {fraction}"
                )

    def contingency_for(self, maturity: str) -> float:
        """Return the contingency fraction for ``maturity`` (dimensionless).

        Raises
        ------
        DeclarationError
            If the maturity level is not declared by this policy. The message
            lists the levels that are declared.
        """
        try:
            return float(self.levels[maturity])
        except KeyError:
            known = ", ".join(sorted(self.levels))
            raise DeclarationError(
                f"maturity {maturity!r} is not declared by margin policy "
                f"{self.name!r}; declared levels are: {known}"
            ) from None

    @property
    def most_mature_level(self) -> str:
        """The level with the smallest contingency; ties broken by name."""
        return min(sorted(self.levels), key=lambda lvl: self.levels[lvl])


@dataclass(frozen=True)
class Correlation:
    """A declared correlation between two leaf items.

    Parameters
    ----------
    a, b:
        Leaf identifiers. Order is irrelevant; the matrix is symmetric.
    rho:
        Pearson correlation coefficient in [-1, 1], dimensionless.
    rationale:
        Why these two items are correlated. Required, and reproduced in the
        report, because an undocumented correlation is no more defensible
        than an undocumented independence assumption.
    """

    a: str
    b: str
    rho: float
    rationale: str

    def __post_init__(self) -> None:
        if self.a == self.b:
            raise DeclarationError(
                f"correlation declares item {self.a!r} with itself; the diagonal is "
                "fixed at 1.0 and cannot be declared"
            )
        if not -1.0 <= self.rho <= 1.0:
            raise DeclarationError(
                f"correlation {self.a!r}-{self.b!r}: rho must lie in [-1, 1], "
                f"got {self.rho}"
            )
        if not self.rationale.strip():
            raise DeclarationError(
                f"correlation {self.a!r}-{self.b!r}: a non-empty rationale is required"
            )


@dataclass(frozen=True)
class Item:
    """A leaf of a budget: one line of the spreadsheet.

    Parameters
    ----------
    id:
        Unique within the budget.
    name:
        Human-readable label.
    value:
        Nominal estimate in the budget's unit. May be negative when the unit
        is a log-domain quantity such as dB, where losses are entered
        negative.
    maturity:
        Key into the budget's margin policy. ``None`` only when
        ``contingency`` is given explicitly.
    contingency:
        Explicit override of the policy value, dimensionless and >= 0. When
        both this and ``maturity`` are given, the override is used and check
        BR005 reports the divergence.
    uncertainty:
        1 sigma dispersion in the budget's unit, >= 0.
    common_driver:
        Optional tag naming a shared physical or programmatic driver (a
        supplier, a converter efficiency, a thermal environment). Two leaves
        with the same tag and no declared correlation raise check BR010.
    """

    id: str
    name: str
    value: float
    maturity: str | None = None
    contingency: float | None = None
    uncertainty: float = 0.0
    common_driver: str | None = None

    def __post_init__(self) -> None:
        if self.maturity is None and self.contingency is None:
            raise DeclarationError(
                f"item {self.id!r}: one of 'maturity' or 'contingency' must be given"
            )
        if self.contingency is not None and self.contingency < 0.0:
            raise DeclarationError(
                f"item {self.id!r}: contingency must be >= 0, got {self.contingency}"
            )
        if self.uncertainty < 0.0:
            raise DeclarationError(
                f"item {self.id!r}: uncertainty is a 1 sigma magnitude and must be "
                f">= 0, got {self.uncertainty}"
            )

    @property
    def is_leaf(self) -> bool:
        """True. Present so that items and nodes share one interface."""
        return True


@dataclass(frozen=True)
class Node:
    """An internal node of a budget: a subtotal.

    Parameters
    ----------
    id, name:
        As for :class:`Item`.
    combination:
        How the children's uncertainties combine at this node. One of
        ``linear`` (arithmetic sum, the fully correlated upper bound),
        ``rss`` (root-sum-square, valid only under mutual independence), or
        ``correlated`` (the quadratic form over this node's descendant leaves
        using the budget's declared correlation matrix).
    children:
        At least one :class:`Item` or :class:`Node`.
    allocation:
        Optional cap in the budget's unit. When present, margin is reported
        at this node.
    independence_rationale:
        Why the children may be treated as independent. Required reading for
        a reviewer of an ``rss`` node and reproduced beside every such node
        in the report.
    """

    id: str
    name: str
    combination: str
    children: tuple[Item | Node, ...]
    allocation: float | None = None
    independence_rationale: str = ""

    def __post_init__(self) -> None:
        if self.combination not in COMBINATIONS:
            raise DeclarationError(
                f"node {self.id!r}: combination must be one of "
                f"{', '.join(COMBINATIONS)}; got {self.combination!r}"
            )
        if not self.children:
            raise DeclarationError(f"node {self.id!r}: a node must have at least one child")

    @property
    def is_leaf(self) -> bool:
        """False. Present so that items and nodes share one interface."""
        return False


@dataclass(frozen=True)
class Budget:
    """A complete budget declaration.

    Parameters
    ----------
    id, name:
        Identifiers for the budget as a whole.
    unit:
        The single unit shared by every value, uncertainty and allocation in
        the tree. Reproduced in every report. Not interpreted: this package
        does no unit conversion and no dimensional analysis. Use ``pint`` if
        that is what you need.
    quantity:
        Free-text label for what is being budgeted (mass, power, data, link).
    policy:
        The declared margin policy.
    root:
        The root node.
    correlations:
        Declared leaf-to-leaf correlations. Every pair not listed is treated
        as independent, and that treatment is a declaration, not a default
        the user is unaware of: check BR001 reports every ``rss`` node whose
        descendants carry a declared non-zero correlation.
    k_sigma:
        The sigma multiplier used when reporting margin under uncertainty.
        Dimensionless. Not a confidence level; see the module docstring.
    """

    id: str
    name: str
    unit: str
    quantity: str
    policy: MarginPolicy
    root: Node
    correlations: tuple[Correlation, ...] = ()
    k_sigma: float = 1.0

    def __post_init__(self) -> None:
        if self.k_sigma < 0.0:
            raise DeclarationError(f"k_sigma must be >= 0, got {self.k_sigma}")
        seen: set[str] = set()
        for node in walk(self.root):
            if node.id in seen:
                raise DeclarationError(
                    f"duplicate identifier {node.id!r}; identifiers must be unique "
                    "across the whole budget tree"
                )
            seen.add(node.id)
        leaf_ids = {item.id for item in leaves(self.root)}
        for corr in self.correlations:
            for ident in (corr.a, corr.b):
                if ident not in leaf_ids:
                    raise DeclarationError(
                        f"correlation {corr.a!r}-{corr.b!r} references {ident!r}, which "
                        "is not a leaf item of this budget; correlations are declared "
                        "between leaves only"
                    )
        pairs: set[frozenset[str]] = set()
        for corr in self.correlations:
            key = frozenset((corr.a, corr.b))
            if key in pairs:
                raise DeclarationError(
                    f"correlation {corr.a!r}-{corr.b!r} is declared more than once"
                )
            pairs.add(key)


def walk(node: Item | Node) -> list[Item | Node]:
    """Return ``node`` and all its descendants in pre-order (depth first)."""
    out: list[Item | Node] = [node]
    if isinstance(node, Node):
        for child in node.children:
            out.extend(walk(child))
    return out


def leaves(node: Item | Node) -> list[Item]:
    """Return the leaf items under ``node``, left to right."""
    return [n for n in walk(node) if isinstance(n, Item)]


def depths(root: Item | Node) -> dict[str, int]:
    """Map every identifier under ``root`` to its depth, with the root at 0."""
    out: dict[str, int] = {}

    def visit(node: Item | Node, depth: int) -> None:
        out[node.id] = depth
        if isinstance(node, Node):
            for child in node.children:
                visit(child, depth + 1)

    visit(root, 0)
    return out


def parents(root: Item | Node) -> dict[str, str | None]:
    """Map every identifier under ``root`` to its parent identifier."""
    out: dict[str, str | None] = {root.id: None}

    def visit(node: Item | Node) -> None:
        if isinstance(node, Node):
            for child in node.children:
                out[child.id] = node.id
                visit(child)

    visit(root)
    return out
