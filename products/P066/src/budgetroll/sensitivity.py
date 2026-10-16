"""Which leaf moves the top-level margin most, and by how much.

Three distinct questions are answered separately, because the spreadsheet
habit of answering them with one column is where the mistake starts.

1. **Value share.** ``p_i / P_root``: the fraction of the predicted root
   value this leaf accounts for, contingency included. Dimensionless.
   Shares sum to 1 when every nominal value is non-negative.

2. **Variance share.** The leaf's share of the root variance under the
   declared correlation structure:

       s_i = u_i * (R u)_i / (u^T R u)             [dimensionless]

   This is the standard decomposition of a quadratic form; the shares sum to
   1 exactly in exact arithmetic and to within rounding in floating point.
   Source: JCGM 100:2008 section 5.2.2, rearranged. With ``R = I`` it
   reduces to ``u_i^2 / sum_j u_j^2``, the familiar RSS contribution.
   A share may be **negative** when a declared correlation is negative, which
   is correct: a negatively correlated item reduces the total dispersion.

3. **Margin elasticity.** The measured change in top-level margin when one
   leaf's nominal value is perturbed by a declared relative step, everything
   else held fixed, with the full rollup recomputed:

       d_i = margin(x_i * (1 + eps)) - margin(x_i)   [budget unit]

   This is the practitioner's question and it is a finite difference, not a
   derivative: it captures the contingency factor, any uncertainty declared
   as a fraction of the value by the caller, and the effect of the node's
   combination rule. For the model implemented here the response is exactly
   linear in ``eps``, so the finite difference equals ``eps * x_i * (1 + c_i)``
   in exact arithmetic for a budget whose leaf uncertainties do not depend on
   the leaf value; the two are compared in
   ``validation/validate_sensitivity.py``.

Validity range: all three measures assume the additive model of the
``aggregate`` module. None of them is a probability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from .aggregate import Rollup, rollup
from .correlation import budget_correlation_matrix
from .errors import DeclarationError
from .model import Budget, Item, Node, leaves

#: Default relative perturbation used by :func:`margin_elasticity`.
DEFAULT_STEP = 0.10


@dataclass(frozen=True)
class LeafSensitivity:
    """Sensitivity of the top-level margin to one leaf."""

    id: str
    name: str
    nominal: float
    predicted: float
    uncertainty: float
    value_share: float | None
    variance_share: float | None
    margin_delta: float | None
    step: float


def value_shares(roll: Rollup) -> dict[str, float | None]:
    """Fraction of the predicted root value held by each leaf (dimensionless).

    Returns ``None`` for every leaf when the predicted root value is zero,
    rather than dividing by zero or reporting a misleading 100 per cent.
    """
    total = roll.root.predicted
    if total == 0.0:
        return {r.id: None for r in roll.leaf_results()}
    return {r.id: r.predicted / total for r in roll.leaf_results()}


def variance_shares(budget: Budget) -> dict[str, float | None]:
    """Share of the root variance held by each leaf under the declared ``R``.

    Returns ``None`` for every leaf when the total variance is zero (every
    uncertainty zero, or a perfectly cancelling negative correlation), since
    no share is defined in that case.
    """
    leaf_ids, matrix = budget_correlation_matrix(budget)
    sigma = np.array([item.uncertainty for item in leaves(budget.root)], dtype=float)
    variance = float(sigma @ matrix @ sigma)
    if variance <= 0.0:
        return dict.fromkeys(leaf_ids, None)
    contributions = sigma * (matrix @ sigma)
    return {ident: float(c / variance) for ident, c in zip(leaf_ids, contributions, strict=True)}


def _scaled_leaf(budget: Budget, leaf_id: str, factor: float) -> Budget:
    """Return a copy of ``budget`` with one leaf's nominal value scaled."""

    def rebuild(node: Item | Node) -> Item | Node:
        if isinstance(node, Item):
            if node.id == leaf_id:
                return replace(node, value=node.value * factor)
            return node
        return replace(node, children=tuple(rebuild(c) for c in node.children))

    new_root = rebuild(budget.root)
    assert isinstance(new_root, Node)
    return replace(budget, root=new_root)


def margin_elasticity(budget: Budget, step: float = DEFAULT_STEP) -> dict[str, float | None]:
    """Change in root margin when each leaf's nominal value grows by ``step``.

    Parameters
    ----------
    step:
        Relative perturbation, dimensionless. Must be finite and non-zero.

    Returns
    -------
    dict
        Leaf identifier to the change in root ``margin_absolute`` in the
        budget's unit, or ``None`` for every leaf when the root declares no
        allocation and therefore has no margin. The value is negative for a
        leaf whose growth consumes margin, which is the usual sign.
    """
    if step == 0.0 or not math.isfinite(step):
        raise DeclarationError(f"step must be a finite non-zero fraction, got {step}")
    base = rollup(budget)
    if base.root.margin_absolute is None:
        return {item.id: None for item in leaves(budget.root)}
    baseline = base.root.margin_absolute
    out: dict[str, float | None] = {}
    for item in leaves(budget.root):
        perturbed = rollup(_scaled_leaf(budget, item.id, 1.0 + step))
        assert perturbed.root.margin_absolute is not None
        out[item.id] = perturbed.root.margin_absolute - baseline
    return out


def sensitivity(budget: Budget, step: float = DEFAULT_STEP) -> list[LeafSensitivity]:
    """Rank every leaf by the magnitude of its effect on the root margin.

    Ordering is by ``abs(margin_delta)`` descending, falling back to
    ``abs(value_share)`` when the root declares no allocation, and finally to
    the declaration order so that the result is deterministic.
    """
    roll = rollup(budget)
    shares = value_shares(roll)
    variances = variance_shares(budget)
    deltas = margin_elasticity(budget, step=step)
    rows = [
        LeafSensitivity(
            id=r.id,
            name=r.name,
            nominal=r.nominal,
            predicted=r.predicted,
            uncertainty=r.uncertainty,
            value_share=shares[r.id],
            variance_share=variances[r.id],
            margin_delta=deltas[r.id],
            step=step,
        )
        for r in roll.leaf_results()
    ]
    order = {item.id: i for i, item in enumerate(leaves(budget.root))}

    def key(row: LeafSensitivity) -> tuple[float, int]:
        if row.margin_delta is not None:
            return (-abs(row.margin_delta), order[row.id])
        if row.value_share is not None:
            return (-abs(row.value_share), order[row.id])
        return (0.0, order[row.id])

    return sorted(rows, key=key)
