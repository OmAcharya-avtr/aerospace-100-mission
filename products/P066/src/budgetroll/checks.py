"""Ten checks on a declared budget, BR001 to BR010.

Every check names the objects it is about. A check reports what the
declaration says, never what the system is: a clean run means the arithmetic
and the stated assumptions are mutually consistent, and it is not a statement
that any budget is correct, that any estimate is good, or that any margin is
adequate.

Severities
----------
``error``
    The declaration is internally contradictory, or an allocation is already
    exceeded. Exit code 1 from the CLI.
``warning``
    The declaration is admissible but a reviewer should see it.
``info``
    A declared rule that has no effect.
"""

from __future__ import annotations

from dataclasses import dataclass

from .aggregate import Rollup, leaf_contingency, rollup
from .correlation import (
    budget_correlation_matrix,
    declared_pairs,
    is_positive_semidefinite,
)
from .model import Budget, Item, Node, leaves, walk

#: Relative tolerance for the margin sign in BR003 and BR004. A margin whose
#: magnitude is below ``MARGIN_RTOL * max(|allocation|, 1)`` is within
#: floating-point rounding of zero, and its SIGN is not a property the
#: arithmetic can promise: three items of 0.1 kg against an allocation of
#: 0.3 kg sum to 0.30000000000000004 and give a margin of -5.55e-17. Such a
#: case is reported as a warning that the budget closes exactly, never as an
#: error that it is exceeded. The boundary case is pinned in
#: ``tests/test_regression.py``.
MARGIN_RTOL = 1e-12

#: Relative tolerance for BR002: the hierarchical uncertainty and the exact
#: correlated form are reported as divergent only beyond this, so that
#: ordinary floating-point rounding is never reported as a modelling defect.
DIVERGENCE_RTOL = 1e-9

CODES: dict[str, str] = {
    "BR001": "RSS node whose descendants carry a declared non-zero correlation",
    "BR002": "hierarchical uncertainty diverges from the exact correlated form",
    "BR003": "allocation exceeded, or met exactly to within rounding",
    "BR004": "margin positive deterministically but negative at k sigma",
    "BR005": "explicit contingency override diverges from the declared policy",
    "BR006": "item with zero uncertainty at a maturity that is not the most mature",
    "BR007": "combination rule declared on a node with a single child has no effect",
    "BR008": "declared correlation matrix is not positive semi-definite",
    "BR009": "node declares 'correlated' but no correlation applies to it",
    "BR010": "items share a common driver and no correlation is declared between them",
}

_SEVERITY_ORDER = {"error": 0, "warning": 1, "info": 2}


@dataclass(frozen=True)
class Finding:
    """One finding against a budget declaration."""

    code: str
    severity: str
    subject: str
    message: str

    def __str__(self) -> str:
        return f"[{self.severity.upper():7s}] {self.code} {self.subject}: {self.message}"


def _fmt(value: float, unit: str) -> str:
    return f"{value:.6g} {unit}"


def check_br001(budget: Budget, roll: Rollup) -> list[Finding]:
    """RSS node whose descendant leaves carry a declared non-zero correlation.

    Root-sum-square is the law of propagation of uncertainty specialised to
    mutually independent inputs (JCGM 100:2008 section 5.1.2). A node that
    declares ``rss`` while the budget declares a non-zero correlation among
    its descendants is asserting independence that the same document denies.
    """
    out: list[Finding] = []
    for node in walk(budget.root):
        if not isinstance(node, Node) or node.combination != "rss":
            continue
        result = roll.by_id(node.id)
        pairs = declared_pairs(budget, result.leaf_ids)
        if not pairs:
            continue
        listed = ", ".join(f"{c.a}-{c.b} rho={c.rho:+.3g}" for c in pairs)
        out.append(
            Finding(
                "BR001",
                "error",
                node.id,
                f"declares 'rss', which asserts mutual independence, but the budget "
                f"declares {len(pairs)} non-zero correlation(s) among its descendants "
                f"({listed}); declared uncertainty "
                f"{_fmt(result.uncertainty, roll.unit)} against exact correlated "
                f"{_fmt(result.exact_uncertainty, roll.unit)}",
            )
        )
    return out


def check_br002(budget: Budget, roll: Rollup) -> list[Finding]:
    """Hierarchical uncertainty diverges from the exact correlated form.

    Reported only beyond :data:`DIVERGENCE_RTOL` relative, so that
    floating-point rounding is never reported as a modelling defect.

    The comparison is two-sided and the two directions are graded
    differently, because they are not the same mistake. A hierarchical value
    **smaller** than the exact form understates the dispersion and is graded
    ``error``: the budget claims a tighter number than its own declared
    correlations support. A hierarchical value **larger** than the exact form
    overstates it and is graded ``info``: that is what a deliberately
    conservative ``linear`` node inside an otherwise ``rss`` tree does, it is
    a legitimate choice, and the reviewer is told the size of the
    conservatism rather than being asked to change it.
    """
    out: list[Finding] = []
    for node in walk(budget.root):
        if not isinstance(node, Node):
            continue
        result = roll.by_id(node.id)
        declared, exact = result.uncertainty, result.exact_uncertainty
        scale = max(abs(declared), abs(exact), 1e-300)
        if abs(declared - exact) <= DIVERGENCE_RTOL * scale:
            continue
        understates = declared < exact
        direction = "understates" if understates else "overstates"
        out.append(
            Finding(
                "BR002",
                "error" if understates else "info",
                node.id,
                f"combination '{node.combination}' gives "
                f"{_fmt(declared, roll.unit)} and {direction} the exact correlated "
                f"form over its {len(result.leaf_ids)} descendant leaves, which gives "
                f"{_fmt(exact, roll.unit)} "
                f"(relative difference {abs(declared - exact) / scale:.4g})",
            )
        )
    return out


def check_br003(budget: Budget, roll: Rollup) -> list[Finding]:
    """Allocation exceeded by the predicted value, before any uncertainty.

    Graded ``error`` only when the overrun exceeds
    ``MARGIN_RTOL * max(|allocation|, 1)``. Inside that band the predicted
    value equals the allocation to within rounding, the sign of the margin
    depends on the order the terms were summed in, and the finding is graded
    ``warning`` and worded as closed exactly. Asserting the sign of a
    floating-point difference as if it were a fact about the budget is the
    defect class this grading exists to avoid.
    """
    out: list[Finding] = []
    for result in roll.results:
        if result.allocation is None or result.margin_absolute is None:
            continue
        band = MARGIN_RTOL * max(abs(result.allocation), 1.0)
        if result.margin_absolute >= band:
            continue
        if result.margin_absolute > -band:
            out.append(
                Finding(
                    "BR003",
                    "warning",
                    result.id,
                    f"predicted {_fmt(result.predicted, roll.unit)} equals the declared "
                    f"allocation {_fmt(result.allocation, roll.unit)} to within "
                    f"floating-point rounding (margin "
                    f"{result.margin_absolute:+.3e} {roll.unit}); the sign of a margin "
                    "this small is a property of the summation order, not of the "
                    "budget, so it is reported as closed exactly rather than exceeded",
                )
            )
            continue
        out.append(
            Finding(
                "BR003",
                "error",
                result.id,
                f"predicted {_fmt(result.predicted, roll.unit)} exceeds the declared "
                f"allocation {_fmt(result.allocation, roll.unit)} by "
                f"{_fmt(-result.margin_absolute, roll.unit)}; this is the nominal plus "
                f"contingency figure and carries no uncertainty term",
            )
        )
    return out


def check_br004(budget: Budget, roll: Rollup) -> list[Finding]:
    """Margin non-negative deterministically but negative at k sigma.

    The budget closes only by ignoring the dispersion it declares. Both
    comparisons use the same rounding band as BR003, so a k-sigma margin that
    is zero to within rounding is not reported as negative.
    """
    out: list[Finding] = []
    for result in roll.results:
        if result.margin_absolute is None or result.margin_at_k is None:
            continue
        band = MARGIN_RTOL * max(abs(result.allocation or 0.0), 1.0)
        if result.margin_absolute < -band or result.margin_at_k >= -band:
            continue
        out.append(
            Finding(
                "BR004",
                "warning",
                result.id,
                f"margin is {_fmt(result.margin_absolute, roll.unit)} on predicted values "
                f"but {_fmt(result.margin_at_k, roll.unit)} at k={roll.k_sigma:g} sigma "
                f"(uncertainty {_fmt(result.uncertainty, roll.unit)}); the budget closes "
                "only if the declared dispersion is ignored",
            )
        )
    return out


def check_br005(budget: Budget, roll: Rollup) -> list[Finding]:
    """Explicit contingency override diverges from the declared policy."""
    out: list[Finding] = []
    for item in leaves(budget.root):
        if item.contingency is None or item.maturity is None:
            continue
        policy_value = budget.policy.contingency_for(item.maturity)
        if item.contingency == policy_value:
            continue
        out.append(
            Finding(
                "BR005",
                "warning",
                item.id,
                f"carries an explicit contingency of {item.contingency:.4g} while policy "
                f"{budget.policy.name!r} gives {policy_value:.4g} for maturity "
                f"{item.maturity!r}; the override is applied and the policy is not",
            )
        )
    return out


def check_br006(budget: Budget, roll: Rollup) -> list[Finding]:
    """Zero uncertainty declared at a maturity that is not the most mature.

    An item nobody has built has a dispersion. Declaring it as exactly zero
    removes it from every RSS at every level above, silently.
    """
    out: list[Finding] = []
    most_mature = budget.policy.most_mature_level
    for item in leaves(budget.root):
        if item.uncertainty != 0.0:
            continue
        if item.maturity is None or item.maturity == most_mature:
            continue
        out.append(
            Finding(
                "BR006",
                "warning",
                item.id,
                f"declares uncertainty 0 {roll.unit} at maturity {item.maturity!r}, which "
                f"is not the most mature level {most_mature!r} of policy "
                f"{budget.policy.name!r}; a zero contributes nothing to any "
                "root-sum-square above it",
            )
        )
    return out


def check_br007(budget: Budget, roll: Rollup) -> list[Finding]:
    """A combination rule on a single-child node has no effect."""
    out: list[Finding] = []
    for node in walk(budget.root):
        if not isinstance(node, Node) or len(node.children) != 1:
            continue
        if node.combination == "linear":
            continue
        out.append(
            Finding(
                "BR007",
                "info",
                node.id,
                f"declares combination {node.combination!r} over a single child "
                f"({node.children[0].id!r}); every rule gives the same result here, so "
                "the declaration states an assumption it does not use",
            )
        )
    return out


def check_br008(budget: Budget, roll: Rollup) -> list[Finding]:
    """The declared correlation matrix is not positive semi-definite.

    A symmetric matrix with unit diagonal that has a negative eigenvalue is
    not the correlation matrix of any joint distribution, so the quadratic
    form it defines can return a negative variance. The declaration is
    impossible rather than merely pessimistic.
    """
    leaf_ids, matrix = budget_correlation_matrix(budget)
    ok, smallest = is_positive_semidefinite(matrix)
    if ok:
        return []
    return [
        Finding(
            "BR008",
            "error",
            budget.id,
            f"the declared correlation matrix over {len(leaf_ids)} leaves has smallest "
            f"eigenvalue {smallest:.6g} and is therefore not positive semi-definite; no "
            "joint distribution has these correlations and any variance computed from "
            "them may be negative",
        )
    ]


def check_br009(budget: Budget, roll: Rollup) -> list[Finding]:
    """A node declares ``correlated`` but no correlation applies to it."""
    out: list[Finding] = []
    for node in walk(budget.root):
        if not isinstance(node, Node) or node.combination != "correlated":
            continue
        result = roll.by_id(node.id)
        if declared_pairs(budget, result.leaf_ids):
            continue
        out.append(
            Finding(
                "BR009",
                "error",
                node.id,
                "declares combination 'correlated' but no non-zero correlation is "
                "declared among its descendant leaves, so the rule evaluates to plain "
                "root-sum-square; the declaration claims a correlation structure the "
                "budget does not contain",
            )
        )
    return out


def check_br010(budget: Budget, roll: Rollup) -> list[Finding]:
    """Items sharing a common driver with no declared correlation between them."""
    groups: dict[str, list[Item]] = {}
    for item in leaves(budget.root):
        if item.common_driver:
            groups.setdefault(item.common_driver, []).append(item)
    declared = {frozenset((c.a, c.b)) for c in budget.correlations if c.rho != 0.0}
    out: list[Finding] = []
    for driver, items in sorted(groups.items()):
        for i, left in enumerate(items):
            for right in items[i + 1 :]:
                if frozenset((left.id, right.id)) in declared:
                    continue
                out.append(
                    Finding(
                        "BR010",
                        "warning",
                        f"{left.id},{right.id}",
                        f"both declare common driver {driver!r} and no correlation is "
                        "declared between them, so every root-sum-square above them "
                        "treats them as independent",
                    )
                )
    return out


#: Every check, in code order.
CHECKS = (
    check_br001,
    check_br002,
    check_br003,
    check_br004,
    check_br005,
    check_br006,
    check_br007,
    check_br008,
    check_br009,
    check_br010,
)


def run_checks(budget: Budget, roll: Rollup | None = None) -> list[Finding]:
    """Run every check and return the findings sorted by severity then code.

    Parameters
    ----------
    roll:
        A rollup of ``budget``. Computed if not supplied.
    """
    roll = rollup(budget) if roll is None else roll
    findings: list[Finding] = []
    for check in CHECKS:
        findings.extend(check(budget, roll))
    return sorted(findings, key=lambda f: (_SEVERITY_ORDER[f.severity], f.code, f.subject))


def has_blocking(findings: list[Finding]) -> bool:
    """True when at least one finding has severity ``error``."""
    return any(f.severity == "error" for f in findings)


def severity_counts(findings: list[Finding]) -> dict[str, int]:
    """Count findings by severity, with every severity present as a key."""
    counts = dict.fromkeys(_SEVERITY_ORDER, 0)
    for finding in findings:
        counts[finding.severity] += 1
    return counts


__all__ = [
    "CHECKS",
    "CODES",
    "DIVERGENCE_RTOL",
    "MARGIN_RTOL",
    "Finding",
    "has_blocking",
    "leaf_contingency",
    "run_checks",
    "severity_counts",
]
