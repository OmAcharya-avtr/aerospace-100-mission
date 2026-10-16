"""Rollup of a declared budget: values, contingency, uncertainty and margin.

Equations
---------
For a node with children indexed by ``c``:

    nominal    N = sum_c N_c                         [budget unit]
    allowance  A = sum_c A_c                         [budget unit]
    predicted  P = N + A                             [budget unit]

with the leaf allowance ``A_i = N_i * c_i`` for contingency fraction ``c_i``.
Contingency is a deterministic growth allowance and is therefore summed, never
root-sum-squared (see ``model`` module docstring).

Uncertainty at a node follows the node's declared combination rule:

    linear      U = sum_c U_c
    rss         U = sqrt( sum_c U_c^2 )
    correlated  U = sqrt( w^T R w ) over this node's descendant leaves

Source for the uncertainty rules: JCGM 100:2008 (the GUM), sections 5.1.2 and
5.2.2. Units: budget unit throughout; the correlation matrix is dimensionless.
Assumptions: additive combination with unit sensitivity coefficients; ``rss``
additionally assumes the children are mutually independent, which is what
check BR001 tests against the declared correlation structure. Validity range:
any finite real values; no distributional assumption is made, so ``k_sigma``
is a multiple of a standard deviation, not a confidence level.

Margin at a node carrying an allocation ``L``:

    margin_absolute = L - P                          [budget unit]
    margin_fraction = (L - P) / L                    [dimensionless, L != 0]
    margin_at_k     = L - (P + k * U)                [budget unit]

Note that a hierarchical rollup with mixed rules is **not** in general equal
to the exact correlated form evaluated over all descendant leaves at once.
That inequality is not a defect of the implementation; it is a property of
the declaration, and ``NodeResult.exact_uncertainty`` records the exact value
beside the declared one so that check BR002 can report the divergence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from .correlation import budget_correlation_matrix, quadratic_form
from .errors import AggregationError
from .model import Budget, Item, Node, depths, leaves, parents, walk


@dataclass(frozen=True)
class NodeResult:
    """The rolled-up state of one node or item.

    All magnitudes are in the budget's unit except ``contingency``,
    ``margin_fraction`` and ``consumption_of_parent_allocation``, which are
    dimensionless.
    """

    id: str
    name: str
    depth: int
    is_leaf: bool
    combination: str
    nominal: float
    allowance: float
    predicted: float
    contingency: float | None
    uncertainty: float
    exact_uncertainty: float
    allocation: float | None
    margin_absolute: float | None
    margin_fraction: float | None
    margin_at_k: float | None
    consumption_of_parent_allocation: float | None
    leaf_ids: tuple[str, ...]


@dataclass(frozen=True)
class Rollup:
    """The result of rolling up a whole budget."""

    budget_id: str
    unit: str
    quantity: str
    k_sigma: float
    results: tuple[NodeResult, ...]

    @property
    def root(self) -> NodeResult:
        """The root node's result."""
        return self.results[0]

    def by_id(self, ident: str) -> NodeResult:
        """Return the result for ``ident``.

        Raises
        ------
        KeyError
            If no node or item in the budget carries that identifier.
        """
        for result in self.results:
            if result.id == ident:
                return result
        raise KeyError(f"no node or item with id {ident!r} in this rollup")

    def leaf_results(self) -> tuple[NodeResult, ...]:
        """Every leaf result, left to right."""
        return tuple(r for r in self.results if r.is_leaf)


def leaf_contingency(item: Item, budget: Budget) -> float:
    """Return the contingency fraction applied to ``item`` (dimensionless).

    An explicit ``contingency`` on the item overrides the policy value for
    its maturity. Check BR005 reports the override when it diverges.
    """
    if item.contingency is not None:
        return float(item.contingency)
    assert item.maturity is not None  # enforced by Item.__post_init__
    return budget.policy.contingency_for(item.maturity)


def combine(rule: str, children: list[float]) -> float:
    """Combine child uncertainties by ``rule``; units are the budget's unit.

    ``correlated`` is not handled here, because it is defined over the
    descendant leaves of a node rather than over its immediate children;
    :func:`rollup` evaluates it directly.
    """
    if rule == "linear":
        return math.fsum(children)
    if rule == "rss":
        return math.sqrt(math.fsum(u * u for u in children))
    raise AggregationError(
        f"combination rule {rule!r} cannot be evaluated over immediate children; "
        "expected 'linear' or 'rss'"
    )


def rollup(budget: Budget) -> Rollup:
    """Roll ``budget`` up from the leaves and return every node's state.

    The returned results are ordered pre-order from the root, so
    ``results[0]`` is the root and a parent always precedes its children.
    """
    leaf_ids, matrix = budget_correlation_matrix(budget)
    leaf_index = {ident: i for i, ident in enumerate(leaf_ids)}
    leaf_sigma = {
        item.id: float(item.uncertainty) for item in leaves(budget.root)
    }
    node_depths = depths(budget.root)
    node_parents = parents(budget.root)
    allocations = {
        node.id: node.allocation for node in walk(budget.root) if isinstance(node, Node)
    }

    computed: dict[str, NodeResult] = {}

    def exact_for(ids: tuple[str, ...]) -> float:
        rows = [leaf_index[i] for i in ids]
        sub = matrix[np.ix_(rows, rows)]
        return quadratic_form([leaf_sigma[i] for i in ids], sub)

    def visit(node: Item | Node) -> NodeResult:
        if isinstance(node, Item):
            cont = leaf_contingency(node, budget)
            nominal = float(node.value)
            allowance = nominal * cont
            sigma = float(node.uncertainty)
            result = NodeResult(
                id=node.id,
                name=node.name,
                depth=node_depths[node.id],
                is_leaf=True,
                combination="leaf",
                nominal=nominal,
                allowance=allowance,
                predicted=nominal + allowance,
                contingency=cont,
                uncertainty=sigma,
                exact_uncertainty=sigma,
                allocation=None,
                margin_absolute=None,
                margin_fraction=None,
                margin_at_k=None,
                consumption_of_parent_allocation=None,
                leaf_ids=(node.id,),
            )
            computed[node.id] = result
            return result

        child_results = [visit(child) for child in node.children]
        nominal = math.fsum(c.nominal for c in child_results)
        allowance = math.fsum(c.allowance for c in child_results)
        predicted = nominal + allowance
        descendant_leaves = tuple(
            ident for child in child_results for ident in child.leaf_ids
        )
        if node.combination == "correlated":
            uncertainty = exact_for(descendant_leaves)
        else:
            uncertainty = combine(node.combination, [c.uncertainty for c in child_results])
        exact = exact_for(descendant_leaves)

        allocation = allocations.get(node.id)
        if allocation is None:
            margin_abs = margin_frac = margin_k = None
        else:
            margin_abs = allocation - predicted
            margin_frac = margin_abs / allocation if allocation != 0.0 else None
            margin_k = allocation - (predicted + budget.k_sigma * uncertainty)

        result = NodeResult(
            id=node.id,
            name=node.name,
            depth=node_depths[node.id],
            is_leaf=False,
            combination=node.combination,
            nominal=nominal,
            allowance=allowance,
            predicted=predicted,
            contingency=(allowance / nominal) if nominal != 0.0 else None,
            uncertainty=uncertainty,
            exact_uncertainty=exact,
            allocation=allocation,
            margin_absolute=margin_abs,
            margin_fraction=margin_frac,
            margin_at_k=margin_k,
            consumption_of_parent_allocation=None,
            leaf_ids=descendant_leaves,
        )
        computed[node.id] = result
        return result

    visit(budget.root)

    def nearest_allocation(ident: str) -> float | None:
        current = node_parents.get(ident)
        while current is not None:
            alloc = allocations.get(current)
            if alloc is not None:
                return alloc
            current = node_parents.get(current)
        return None

    ordered: list[NodeResult] = []
    for node in walk(budget.root):
        result = computed[node.id]
        alloc = nearest_allocation(node.id)
        if alloc is not None and alloc != 0.0:
            result = replace(
                result, consumption_of_parent_allocation=result.predicted / alloc
            )
        ordered.append(result)

    return Rollup(
        budget_id=budget.id,
        unit=budget.unit,
        quantity=budget.quantity,
        k_sigma=budget.k_sigma,
        results=tuple(ordered),
    )


def linear_rollup_uncertainty(budget: Budget) -> float:
    """Root uncertainty if every node combined linearly: ``sum_i u_i``.

    This is the fully correlated upper bound and the reference point for the
    RSS-never-exceeds-linear inequality.
    """
    return math.fsum(item.uncertainty for item in leaves(budget.root))


def rss_rollup_uncertainty(budget: Budget) -> float:
    """Root uncertainty if every leaf were independent: ``sqrt(sum_i u_i^2)``.

    Equal to the hierarchical RSS rollup of the same leaves in exact
    arithmetic, since root-sum-square is associative; the two agree to within
    rounding in floating point, which is the property tested by
    ``test_properties.py::test_rss_associativity``.
    """
    return math.sqrt(math.fsum(item.uncertainty**2 for item in leaves(budget.root)))
